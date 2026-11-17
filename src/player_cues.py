"""Players-only features for the ball-missing stretch (session after F340: box inflow, heading
persistence, who starts moving, braking, quadrant residuals).

Why: on BAR-ATH the ball is missing on ~41% of graded frames, and that is where v7 still calls the
wrong team or invents a turnover (long balls, set pieces, a switch in the middle of a spell). v7's
shape group (src/shape.py: width, press, goalside, stretch, work, vx) is silent whenever the ball is
seen; its flight cue is silent on a smoothed or missing ball. These features ask other questions of
the same players: is everybody running at one box (a counter), do both teams keep running the same
way (no turnover), who accelerates first, does everybody brake at once (a stoppage, not a turnover),
and where do a team's players stand beyond what its centroid already says.

Conventions (as src/shape.py):
  - each team in its own attack frame (shape.attack_sign, label-free per period): it attacks +x, its
    own goal at x' = 0, a 180-degree rotation so velocities rotate with it;
  - goalkeeper = the deepest seen player; the rest are outfielders; a team's value is NaN where it
    has fewer than shape.MIN_OUTFIELD (7) seen outfielders (positions), or fewer than 7 outfielders
    with a velocity for the moving features;
  - "seen" excludes SkillCorner 2024/25's estimated (is_detected false) players: the loader fills
    off-camera players with SkillCorner's estimates, which move slower and fake "who moved"
    (switch_cue.seen_mask). Every other source leaves NaN where a player isn't tracked, as BAR-ATH;
  - per-player velocity = displacement over VEL_SPAN_S centred (frame noise cancels, C5); players
    faster than shape.MAX_SPEED_MPS are dropped (glitches, slot swaps);
  - sampled every `step` frames (RATE_HZ). Label-free and ball-free throughout.

team_values(match) returns (idx, {name: (m, 2)}) with column 0 home, 1 away. pair_values() turns
them into home - away differences (and the two-team quantities: counter, both-brake, stable).
"""

from __future__ import annotations

import warnings

import numpy as np

from src import shape
from src.schema import AWAY, HOME

RATE_HZ = 10.0
VEL_SPAN_S = 0.4
BOX_DEPTH_M, BOX_HALF_W_M = 16.5, 20.16
SPOT_M = 11.0  # penalty spot: 11 m out, on the centre line
SPOT_R_M = 20.0
BACK_N, FRONT_N = 4, 3
ACC_FRONT_N = 4
ACC_W_S = 0.5  # forward acceleration: mean vx over (t, t+W] minus over [t-W, t)
HEADING_MIN_MPS = 1.0  # team mean velocity below this: no heading
HEADING_W_S = (1.0, 2.0)  # heading change across [t - W/2, t + W/2]
BRAKE_DROP_MPS, BRAKE_AHEAD_S = 1.0, 0.5
SMOOTH_S = 1.0  # centred nan-mean for the position-type features

TEAM = ("opp_box", "own_box", "opp_spot20", "own_spot20", "v_opp_spot", "v_own_spot",
        "back4_speed", "back4_width", "front3_speed", "back_front", "coherence", "heading",
        "dhead1", "dhead2", "acc_all", "acc_front", "brake", "q_own_l", "q_own_r", "q_opp_l",
        "q_opp_r", "cx", "cy")
SMOOTHED = ("opp_box", "own_box", "opp_spot20", "own_spot20", "v_opp_spot", "v_own_spot",
            "back4_speed", "back4_width", "front3_speed", "back_front", "coherence",
            "q_own_l", "q_own_r", "q_opp_l", "q_opp_r", "cx", "cy")
QUADS = ("q_own_l", "q_own_r", "q_opp_l", "q_opp_r")


def seen_players(match) -> np.ndarray:
    """(n, k, 2) positions with SkillCorner 2024/25's estimated players set to NaN; the match's own
    array (no copy) on every other source."""
    from src.models.switch_cue import seen_mask

    detected = seen_mask(match)
    if detected is None:
        return match.players
    return np.where(detected[..., None], match.players, np.nan)


def _nanmean_win(values: np.ndarray, half: int) -> np.ndarray:
    """Centred nan-mean over +/- half samples along axis 0 (NaN where no sample is known)."""
    if half <= 0:
        return values
    known = ~np.isnan(values)
    v = np.where(known, values, 0.0)
    pad = [(1, 0)] + [(0, 0)] * (values.ndim - 1)
    cs = np.pad(np.cumsum(v, axis=0), pad)
    cn = np.pad(np.cumsum(known, axis=0), pad)
    m = len(values)
    lo = np.clip(np.arange(m) - half, 0, m)
    hi = np.clip(np.arange(m) + half + 1, 0, m)
    s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _shift(values: np.ndarray, k: int) -> np.ndarray:
    """values[t + k] (NaN past the ends)."""
    out = np.full_like(values, np.nan)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def team_values(match, rate_hz: float = RATE_HZ, players: np.ndarray | None = None):
    """(idx (m,), {name: (m, 2)}): every step-th frame, each team's TEAM values in its own frame.
    `players` overrides the positions read (default: seen_players(match))."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _team_values(match, rate_hz, seen_players(match) if players is None else players)


def _team_values(match, rate_hz, players):
    step = max(1, int(round(match.fps / rate_hz)))
    rate = match.fps / step
    idx = np.arange(0, match.n_frames, step)
    m = len(idx)
    out = {name: np.full((m, 2), np.nan) for name in TEAM}
    if not m:
        return idx, out
    length, width = match.pitch
    sign = shape.attack_sign(match)
    h = max(1, int(round(VEL_SPAN_S * match.fps / 2)))
    lo, hi, same = shape._shifted(match, idx, h)
    dt = 2 * h / match.fps
    ahead = max(1, int(round(BRAKE_AHEAD_S * rate)))
    for t in (HOME, AWAY):
        cols = match.slot_team == t
        if not np.any(cols):
            continue
        s = sign[idx, t].astype(np.float64)
        f = {name: np.full(m, np.nan) for name in TEAM}
        speed_all = np.full((m, int(cols.sum())), np.nan)
        for start in range(0, m, shape.CHUNK):
            sl = slice(start, start + shape.CHUNK)
            ss = s[sl, None]
            p = players[idx[sl]][:, cols].copy()
            disp = players[hi[sl]][:, cols] - players[lo[sl]][:, cols]
            # own frame: x' = L - x, y' = W - y where the team attacks -x; velocity rotates too
            px = np.where(ss < 0, length - p[..., 0], p[..., 0])
            py = np.where(ss < 0, width - p[..., 1], p[..., 1])
            vx, vy = disp[..., 0] * ss / dt, disp[..., 1] * ss / dt
            sp = np.hypot(vx, vy)
            okv = (sp <= shape.MAX_SPEED_MPS) & same[sl, None]
            vx, vy, sp = (np.where(okv, a, np.nan) for a in (vx, vy, sp))
            seen = ~np.isnan(px) & (ss != 0)
            count = seen.sum(axis=1)
            order = np.argsort(np.where(seen, px, np.inf), axis=1, kind="stable")
            take = lambda a: np.take_along_axis(a, order, axis=1)  # noqa: E731
            X, Y, VX, VY, SP = take(px), take(py), take(vx), take(vy), take(sp)
            pos = np.arange(X.shape[1])[None, :]
            out_f = (pos >= 1) & (pos < count[:, None])  # outfielders: all seen but the deepest
            n_out = out_f.sum(axis=1)
            okp = n_out >= shape.MIN_OUTFIELD
            hasv = out_f & ~np.isnan(VX)
            n_v = hasv.sum(axis=1)
            okm = n_v >= shape.MIN_OUTFIELD
            Xo, Yo = np.where(out_f, X, np.nan), np.where(out_f, Y, np.nan)
            VXo, VYo, SPo = (np.where(hasv, a, np.nan) for a in (VX, VY, SP))
            inbox_opp = (Xo > length - BOX_DEPTH_M) & (np.abs(Yo - width / 2) <= BOX_HALF_W_M)
            inbox_own = (Xo < BOX_DEPTH_M) & (np.abs(Yo - width / 2) <= BOX_HALF_W_M)
            g = {}
            g["opp_box"] = np.where(okp, inbox_opp.sum(axis=1), np.nan)
            g["own_box"] = np.where(okp, inbox_own.sum(axis=1), np.nan)
            for name, sx in (("opp", length - SPOT_M), ("own", SPOT_M)):
                dx, dy = sx - Xo, width / 2 - Yo
                d = np.hypot(dx, dy)
                g[name + "_spot20"] = np.where(okp, np.sum(d <= SPOT_R_M, axis=1), np.nan)
                toward = (VXo * dx + VYo * dy) / np.maximum(d, 1e-6)
                g["v_{}_spot".format(name)] = np.where(okm, np.nanmean(toward, axis=1), np.nan)
            # back 4 / front 3 among outfielders (positions sorted by x' ascending)
            back = out_f & (pos <= BACK_N)
            front = out_f & (pos >= count[:, None] - FRONT_N)
            g["back4_speed"] = np.where(okm, np.nanmean(np.where(back, SPo, np.nan), axis=1), np.nan)
            g["front3_speed"] = np.where(okm, np.nanmean(np.where(front, SPo, np.nan), axis=1), np.nan)
            yb = np.where(back, Yo, np.nan)
            g["back4_width"] = np.where(okp, np.nanmax(yb, axis=1) - np.nanmin(yb, axis=1), np.nan)
            g["back_front"] = g["back4_speed"] - g["front3_speed"]
            mvx, mvy = np.nanmean(VXo, axis=1), np.nanmean(VYo, axis=1)
            msp = np.nanmean(SPo, axis=1)
            res = np.hypot(mvx, mvy)
            g["coherence"] = np.where(okm & (msp >= 0.5), res / np.maximum(msp, 1e-6), np.nan)
            g["heading"] = np.where(okm & (res >= HEADING_MIN_MPS), np.arctan2(mvy, mvx), np.nan)
            g["acc_all"] = np.where(okm, mvx, np.nan)  # vx means for now; differenced below
            accf = out_f & hasv & (pos >= count[:, None] - ACC_FRONT_N)
            g["acc_front"] = np.where(okm, np.nanmean(np.where(accf, VXo, np.nan), axis=1), np.nan)
            half_x, half_y = length / 2, width / 2
            for name, cond in (("q_own_l", (Xo < half_x) & (Yo < half_y)),
                               ("q_own_r", (Xo < half_x) & (Yo >= half_y)),
                               ("q_opp_l", (Xo >= half_x) & (Yo < half_y)),
                               ("q_opp_r", (Xo >= half_x) & (Yo >= half_y))):
                g[name] = np.where(okp, np.sum(cond, axis=1), np.nan)
            g["cx"] = np.where(okp, np.nanmean(Xo, axis=1), np.nan)
            g["cy"] = np.where(okp, np.nanmean(Yo, axis=1), np.nan)
            for name, value in g.items():
                f[name][sl] = value
            # per-slot speeds (outfielders only) for the brake share; slot order, not sorted
            slot_out = np.zeros_like(seen)
            np.put_along_axis(slot_out, order, out_f, axis=1)
            speed_all[sl] = np.where(slot_out, sp, np.nan)
        # moving features that need neighbouring samples
        w = max(1, int(round(ACC_W_S * rate)))
        for name in ("acc_all", "acc_front"):
            v = f[name]
            after = np.nanmean(np.stack([_shift(v, k) for k in range(1, w + 1)]), axis=0)
            before = np.nanmean(np.stack([_shift(v, -k) for k in range(0, w)]), axis=0)
            f[name] = after - before
        for name, win in (("dhead1", HEADING_W_S[0]), ("dhead2", HEADING_W_S[1])):
            k = max(1, int(round(win * rate / 2)))
            f[name] = np.abs(_wrap(_shift(f["heading"], k) - _shift(f["heading"], -k)))
        later = _shift(speed_all, ahead)
        drop = later - speed_all <= -BRAKE_DROP_MPS
        both = ~np.isnan(later) & ~np.isnan(speed_all)
        nb = both.sum(axis=1)
        f["brake"] = np.where(nb >= shape.MIN_OUTFIELD, np.sum(drop & both, axis=1) / np.maximum(nb, 1),
                              np.nan)
        half = int(round(SMOOTH_S * rate / 2))
        for name in SMOOTHED:
            f[name] = _nanmean_win(f[name], half)
        for name in TEAM:
            out[name][:, t] = f[name]
    return idx, out


def quad_residuals(vals: dict) -> dict:
    """{q: (m, 2)} quadrant counts minus what the team's own centroid (cx, cy) explains: a
    least-squares fit count ~ 1 + cx + cy per quadrant, pooled over both teams' known samples of
    this match (label-free)."""
    out = {}
    cx, cy = vals["cx"].ravel(), vals["cy"].ravel()
    for q in QUADS:
        c = vals[q].ravel()
        ok = ~np.isnan(c) & ~np.isnan(cx) & ~np.isnan(cy)
        res = np.full(c.shape, np.nan)
        if ok.sum() > 10:
            design = np.column_stack([np.ones(ok.sum()), cx[ok], cy[ok]])
            beta, *_ = np.linalg.lstsq(design, c[ok], rcond=None)
            res[ok] = c[ok] - design @ beta
        out["res_" + q] = res.reshape(vals[q].shape)
    return out


def pair_values(vals: dict) -> dict:
    """{name: (m,)} home - away for every team value (plus quadrant residuals), and the two-team
    quantities. Each is antisymmetric (swapping the teams negates it) unless it says otherwise:
      counter     both teams' speed toward the box home attacks minus toward the box away attacks
                  (home's v_opp_spot + away's v_own_spot - home's v_own_spot - away's v_opp_spot)
      brake_both  min of the two teams' brake shares (symmetric: a stoppage-like brake)
      stable1/2   min coherence and max heading change over 1 / 2 s (symmetric)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _pair_values(vals)


def _pair_values(vals: dict) -> dict:
    diff = {name: v[:, HOME] - v[:, AWAY] for name, v in vals.items() if name != "heading"}
    for name, v in quad_residuals(vals).items():
        diff[name] = v[:, HOME] - v[:, AWAY]
    h, a = HOME, AWAY
    diff["counter"] = (vals["v_opp_spot"][:, h] + vals["v_own_spot"][:, a]
                       - vals["v_own_spot"][:, h] - vals["v_opp_spot"][:, a])
    diff["brake_both"] = np.fmin(vals["brake"][:, h], vals["brake"][:, a])
    with np.errstate(invalid="ignore"):
        both_known = ~np.isnan(vals["brake"]).any(axis=1)
    diff["brake_both"] = np.where(both_known, diff["brake_both"], np.nan)
    diff["coh_min"] = np.where(~np.isnan(vals["coherence"]).any(axis=1),
                               np.nanmin(vals["coherence"], axis=1), np.nan)
    for k in ("1", "2"):
        d = vals["dhead" + k]
        diff["dhead{}_max".format(k)] = np.where(~np.isnan(d).any(axis=1), np.nanmax(d, axis=1), np.nan)
    return diff
