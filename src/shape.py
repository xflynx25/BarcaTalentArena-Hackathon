"""Team-shape features (backlog B23): what each team's layout says about who has the ball.

Why: every `learned` feature is about the ball, and on BAR-ATH the ball is missing on ~41% of graded
frames [official DATA.md], while both teams' players are tracked on nearly every frame. A team in
possession tends to spread out and push up; the other team drops behind the ball and closes space.
All features here are label-free and ball-free: only players and the pitch.

Conventions:
  - Attacking direction is found label-free, per period: the team whose players' median x is lower
    attacks +x (`attack_sign`). Each team's coordinates are then rotated into its own frame, in which
    it attacks +x and its own goal is at x = 0 (x' = L - x, y' = W - y when it attacks -x; a 180
    degree rotation, so left and right stay the team's own left and right).
  - The goalkeeper is the team's deepest player on that frame (lowest x'), not a fixed slot: slots
    are tracklets on BAR-ATH (src/schema.py), so per-slot roles would break. The rest are outfield.
  - Per team, at most MAX_PLAYERS players per frame are used (the deepest ones if a tracklet overlap
    shows more). A team's features are NaN on frames with fewer than MIN_OUTFIELD outfielders seen.
  - `team_features` gives (m, 2) arrays (column 0 home, 1 away), each value in that team's own frame.
    `differences` gives home minus away, the symmetric form models should use: swapping the teams
    negates it (tests/test_shape.py), so a model can't learn "home has it more" from it.
  - Derivatives are centred over SPAN_S (displacement across the span, so frame noise cancels, C5).

Features (team t, in t's frame; "outfield" = all but the deepest player):
  width     10-90 percentile spread of outfield y (m)
  length    10-90 percentile spread of outfield x' (m)
  back      mean x' of the deepest 4 outfielders: back-line height (m)
  front     mean x' of the 3 most advanced outfielders: front-line height (m)
  stretch   front - back: distance between the lines (m)
  cx        outfield centroid x' (m)
  opp_half  outfielders past the halfway line (count)
  hull      convex-hull area of the outfielders (m^2)
  compact   mean distance of outfielders to their centroid (m)
  fb_x      mean x' of the widest outfielder on each side: full-back height (m)
  fb_w      y distance between those two widest outfielders: full width (m)
  vx        mean player velocity toward the opponent goal over SPAN_S (m/s)
  dwidth    d width / dt over SPAN_S (m/s)
  dhull     d hull / dt over SPAN_S (m^2/s)
  -- not on the backlog's starting list (why, in docs/findings.md F88):
  last_line outfielders within LAST_LINE_M of, or past, the opponent's deepest outfielder (count):
            runners on the opponent's back line. (A "centroid relative to the opponent's" feature
            cancels exactly in home - away, so it isn't here.)
  goalside  share of outfielders nearer their own goal (lower x') than their nearest opponent
  marking   mean distance from each outfielder to the nearest opponent (m): attackers are freer
  press     share of the opponent's outfielders with one of t's players within PRESS_M
  recovery  share of t's players running toward their own goal faster than RECOVERY_MPS
  work      mean player speed over SPAN_S (m/s)
  gk        goalkeeper (deepest player) x' (m)
  control   share of the opponent's half closer to one of t's players than to any opponent
            (a Voronoi share on a CONTROL_GRID of points)
"""

from __future__ import annotations

import warnings

import numpy as np

from src.schema import AWAY, HOME, Match

MAX_PLAYERS = 12
MIN_OUTFIELD = 7
SPAN_S = 1.0
PRESS_M = 5.0
LAST_LINE_M = 5.0
RECOVERY_MPS = 3.0
MAX_SPEED_MPS = 12.0  # tracking glitches and slot swaps
CONTROL_GRID = (8, 10)  # points along x' and y in the opponent's half
CHUNK = 4096  # frames per vectorised block (memory)

STATIC = ("width", "length", "back", "front", "stretch", "cx", "opp_half", "last_line", "hull",
          "compact", "fb_x", "fb_w", "goalside", "marking", "press", "gk", "control")
MOVING = ("vx", "dwidth", "dhull", "recovery", "work")
FEATURES = STATIC + MOVING


# -- direction ------------------------------------------------------------------------------------

def attack_sign(match: Match) -> np.ndarray:
    """(n, 2) int8: +1 if that team (column HOME / AWAY) attacks +x on the frame, -1 if -x, 0 outside
    play or where it can't be told. Label-free: per period, the team with the lower median player x
    defends the x = 0 goal."""
    sign = np.zeros((match.n_frames, 2), dtype=np.int8)
    x = match.players[:, :, 0]
    for p in np.unique(match.period[match.period > 0]):
        rows = match.period == p
        med = [np.nanmedian(x[rows][:, match.slot_team == team]) if np.any(match.slot_team == team)
               else np.nan for team in (HOME, AWAY)]
        if np.isnan(med).any() or med[0] == med[1]:
            continue
        home = 1 if med[0] < med[1] else -1
        sign[rows, HOME], sign[rows, AWAY] = home, -home
    return sign


def _own_frame(points: np.ndarray, sign: np.ndarray, pitch) -> np.ndarray:
    """(m, k, 2) absolute -> the team's own frame (rotated 180 degrees where sign < 0). NaN where
    sign is 0."""
    length, width = pitch
    out = points.copy()
    flip = sign < 0
    out[flip, :, 0] = length - out[flip, :, 0]
    out[flip, :, 1] = width - out[flip, :, 1]
    out[sign == 0] = np.nan
    return out


def _sorted_by_x(points: np.ndarray) -> np.ndarray:
    """(m, k, 2) -> (m, MAX_PLAYERS, 2) sorted by x ascending, NaN rows last, padded with NaN."""
    key = np.where(np.isnan(points[..., 0]), np.inf, points[..., 0])
    order = np.argsort(key, axis=1, kind="stable")[:, :MAX_PLAYERS]
    out = np.take_along_axis(points, order[..., None], axis=1)
    if out.shape[1] < MAX_PLAYERS:
        pad = np.full((len(out), MAX_PLAYERS - out.shape[1], 2), np.nan)
        out = np.concatenate([out, pad], axis=1)
    return out


# -- small vectorised helpers ---------------------------------------------------------------------

def _quantile_sorted(values: np.ndarray, count: np.ndarray, q: float) -> np.ndarray:
    """q-quantile (linear) of each row of `values`, sorted ascending with its `count` valid entries
    first. NaN where count is 0."""
    pos = q * np.maximum(count - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    a = np.take_along_axis(values, lo[:, None], axis=1)[:, 0]
    b = np.take_along_axis(values, hi[:, None], axis=1)[:, 0]
    out = a + (b - a) * (pos - lo)
    return np.where(count > 0, out, np.nan)


def hull_area(points: np.ndarray) -> np.ndarray:
    """(m, k, 2) with NaN rows ignored -> (m,) convex-hull area. Vectorised (no scipy): a directed
    edge i -> j is a counter-clockwise hull edge when every other point is strictly to its left, and
    the area is the shoelace sum over those edges. A tiny fixed perturbation by point index breaks
    exact ties (duplicate or collinear points). NaN with fewer than 3 points."""
    m, k, _ = points.shape
    j = np.arange(k, dtype=float)
    p = points + np.stack([1e-6 * j, 1e-6 * j * j / k], axis=-1)[None]
    valid = ~np.isnan(p[..., 0])
    out = np.full(m, np.nan)
    eye = np.eye(k, dtype=bool)
    for s in range(0, m, CHUNK):
        q, v = p[s:s + CHUNK], valid[s:s + CHUNK]
        qx, qy = np.nan_to_num(q[..., 0]), np.nan_to_num(q[..., 1])
        ex = qx[:, None, :] - qx[:, :, None]  # (c, i, j): x_j - x_i, also used as (c, i, m)
        ey = qy[:, None, :] - qy[:, :, None]
        cross = ex[:, :, :, None] * ey[:, :, None, :] - ey[:, :, :, None] * ex[:, :, None, :]
        ignore = ~v[:, None, None, :] | eye[None, :, None, :] | eye[None, None, :, :]
        left = np.all((cross > 0) | ignore, axis=3)
        edge = left & v[:, :, None] & v[:, None, :] & ~eye[None]
        term = qx[:, :, None] * qy[:, None, :] - qx[:, None, :] * qy[:, :, None]
        area = 0.5 * np.sum(np.where(edge, term, 0.0), axis=(1, 2))
        out[s:s + CHUNK] = np.where(v.sum(axis=1) >= 3, area, np.nan)
    return out


def _nearest(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point of a (m, ka, 2): distance to and x of the nearest point of b (m, kb, 2).
    NaN where a's point is missing or b has none."""
    d = np.hypot(a[:, :, None, 0] - b[:, None, :, 0], a[:, :, None, 1] - b[:, None, :, 1])
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=2)
    dist = np.take_along_axis(d, k[..., None], axis=2)[..., 0]
    bx = np.take_along_axis(np.broadcast_to(b[:, None, :, 0], d.shape), k[..., None], axis=2)[..., 0]
    bad = ~np.isfinite(dist)
    return np.where(bad, np.nan, dist), np.where(bad, np.nan, bx)


def _control(own: np.ndarray, opp: np.ndarray, pitch) -> np.ndarray:
    """(m,) share of the grid in the opponent's half (x' > L/2, own frame) nearer an own player."""
    length, width = pitch
    nx, ny = CONTROL_GRID
    gx = length / 2 + (np.arange(nx) + 0.5) * (length / 2) / nx
    gy = (np.arange(ny) + 0.5) * width / ny
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    out = np.full(len(own), np.nan)
    for s in range(0, len(own), CHUNK):
        best = []
        for pts in (own[s:s + CHUNK], opp[s:s + CHUNK]):
            d = np.hypot(grid[None, :, None, 0] - pts[:, None, :, 0],
                         grid[None, :, None, 1] - pts[:, None, :, 1])
            best.append(np.min(np.where(np.isnan(d), np.inf, d), axis=2))
        ok = np.isfinite(best[0]).any(axis=1) & np.isfinite(best[1]).any(axis=1)
        out[s:s + CHUNK] = np.where(ok, np.mean(best[0] < best[1], axis=1), np.nan)
    return out


# -- features -------------------------------------------------------------------------------------

def _static(match: Match, idx: np.ndarray, sign: np.ndarray, wanted=STATIC) -> dict:
    """{name: (m, 2)} for the STATIC features on frames idx. The two costly ones (`hull`,
    `control`) are only computed if wanted; the others are cheap and always there."""
    teams = [_sorted_by_x(_own_frame(match.players[idx][:, match.slot_team == t], sign[idx, t],
                                     match.pitch)) for t in (HOME, AWAY)]
    length, width = match.pitch
    # the opponent in each team's frame: rotate the other team's own-frame points by 180 degrees
    opps = [np.stack([length - teams[1 - t][..., 0], width - teams[1 - t][..., 1]], axis=-1)
            for t in (HOME, AWAY)]
    out = {name: np.full((len(idx), 2), np.nan) for name in STATIC}
    near_d = [None, None]
    for t in (HOME, AWAY):
        own, opp = teams[t], opps[t]
        outfield = own[:, 1:]
        x, y = outfield[..., 0], outfield[..., 1]
        count = np.sum(~np.isnan(x), axis=1)
        ok = count >= MIN_OUTFIELD
        ysort = np.sort(y, axis=1)  # NaN last
        f = {}
        f["width"] = _quantile_sorted(ysort, count, 0.9) - _quantile_sorted(ysort, count, 0.1)
        f["length"] = _quantile_sorted(x, count, 0.9) - _quantile_sorted(x, count, 0.1)
        f["back"] = np.nanmean(x[:, :4], axis=1)
        tail = np.maximum(count - 3, 0)[:, None] + np.arange(3)[None]
        f["front"] = np.mean(np.take_along_axis(x, np.minimum(tail, x.shape[1] - 1), axis=1), axis=1)
        f["stretch"] = f["front"] - f["back"]
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        f["cx"] = cx
        # opp keeps the opponent's own sort: row 0 is its goalkeeper, row 1 its deepest outfielder
        f["last_line"] = np.sum(x > opp[:, 1:2, 0] - LAST_LINE_M, axis=1).astype(float)
        f["opp_half"] = np.sum(x > length / 2, axis=1).astype(float)
        f["hull"] = hull_area(outfield) if "hull" in wanted else np.full(len(idx), np.nan)
        f["compact"] = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), axis=1)
        wide = np.where(np.isnan(y), np.inf, y)
        lo = np.argmin(wide, axis=1)
        hi = np.argmax(np.where(np.isnan(y), -np.inf, y), axis=1)
        take = lambda a, i: np.take_along_axis(a, i[:, None], axis=1)[:, 0]  # noqa: E731
        f["fb_x"] = 0.5 * (take(x, lo) + take(x, hi))
        f["fb_w"] = take(y, hi) - take(y, lo)
        dist, nx = _nearest(outfield, opp)
        near_d[t] = dist
        valid = ~np.isnan(dist)
        with np.errstate(invalid="ignore"):
            f["goalside"] = np.sum((x < nx) & valid, axis=1) / np.maximum(valid.sum(axis=1), 1)
        f["marking"] = np.nanmean(dist, axis=1)
        f["gk"] = own[:, 0, 0]
        f["control"] = (_control(own, opp, match.pitch) if "control" in wanted
                        else np.full(len(idx), np.nan))
        for name, value in f.items():
            out[name][:, t] = np.where(ok, value, np.nan)
    for t in (HOME, AWAY):  # press: the opponent's outfielders with one of t's players near them
        d = near_d[1 - t]
        valid = ~np.isnan(d)
        share = np.sum((d < PRESS_M) & valid, axis=1) / np.maximum(valid.sum(axis=1), 1)
        out["press"][:, t] = np.where(np.isnan(out["width"][:, t]) | (valid.sum(axis=1) == 0),
                                      np.nan, share)
    return out


def _shifted(match: Match, idx: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """idx - h, idx + h (clipped) and whether both lie in the same period as idx."""
    n = match.n_frames
    lo, hi = np.clip(idx - h, 0, n - 1), np.clip(idx + h, 0, n - 1)
    same = ((idx - h >= 0) & (idx + h < n) & (match.period[lo] == match.period[idx])
            & (match.period[hi] == match.period[idx]) & (match.period[idx] > 0))
    return lo, hi, same


def team_features(match: Match, step: int = 1, span_s: float = SPAN_S,
                  features=FEATURES) -> tuple[np.ndarray, dict]:
    """Frames idx = 0, step, 2*step, ... and {name: (len(idx), 2)} (column HOME, AWAY), each team in
    its own frame. NaN outside play, where the direction is unknown, or with too few players."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: NaN is the answer
        return _team_features(match, step, span_s, features)


def _team_features(match, step, span_s, features):
    idx = np.arange(0, match.n_frames, max(1, int(step)))
    sign = attack_sign(match)
    h = max(1, int(round(span_s * match.fps / 2)))
    lo, hi, same = _shifted(match, idx, h)
    dt = 2 * h / match.fps
    wanted = set(features)
    need_static = wanted & set(STATIC) or wanted & {"dwidth", "dhull"}
    out = {}
    if need_static:
        if wanted & {"dwidth", "dhull"}:
            frames, inverse = np.unique(np.concatenate([idx, lo, hi]), return_inverse=True)
            stat = _static(match, frames, sign, wanted | {"hull"} if "dhull" in wanted else wanted)
            at = lambda i: {k: v[inverse[i * len(idx):(i + 1) * len(idx)]]  # noqa: E731
                            for k, v in stat.items()}
            now, before, after = at(0), at(1), at(2)
        else:
            now = _static(match, idx, sign, wanted)
            before = after = None
        out.update({k: v for k, v in now.items() if k in wanted})
        for name, base in (("dwidth", "width"), ("dhull", "hull")):
            if name in wanted:
                out[name] = np.where(same[:, None], (after[base] - before[base]) / dt, np.nan)
    if wanted & {"vx", "recovery", "work"}:
        disp = match.players[hi] - match.players[lo]  # (m, k, 2), absolute
        speed = np.hypot(disp[..., 0], disp[..., 1]) / dt
        with np.errstate(invalid="ignore"):
            disp = np.where((speed <= MAX_SPEED_MPS)[..., None], disp, np.nan)
        for t in (HOME, AWAY):
            cols = match.slot_team == t
            vx = disp[:, cols, 0] * sign[idx, t][:, None] / dt
            sp = speed[:, cols]
            seen = np.sum(~np.isnan(vx), axis=1)
            ok = same & (seen >= MIN_OUTFIELD) & (sign[idx, t] != 0)
            with np.errstate(invalid="ignore"):
                vals = {"vx": np.nanmean(vx, axis=1),
                        "recovery": np.sum(vx < -RECOVERY_MPS, axis=1) / np.maximum(seen, 1),
                        "work": np.nanmean(np.where(np.isnan(vx), np.nan, sp), axis=1)}
            for name, value in vals.items():
                if name in wanted:
                    out.setdefault(name, np.full((len(idx), 2), np.nan))[:, t] = np.where(
                        ok, value, np.nan)
    return idx, {name: out[name] for name in features}


def differences(feats: dict) -> dict:
    """{name: (m,) home minus away} from team_features' output. Symmetric: swapping the teams
    negates every value."""
    return {name: v[:, HOME] - v[:, AWAY] for name, v in feats.items()}


def zscore(values: np.ndarray) -> np.ndarray:
    """Per-match z-score over the frames where the value is known (label-free), so a side that
    always sits deep or wide isn't off-scale. NaN stays NaN."""
    known = ~np.isnan(values)
    if known.sum() < 2:
        return np.full_like(values, np.nan)
    mean, sd = values[known].mean(), values[known].std()
    return (values - mean) / sd if sd > 0 else values - mean
