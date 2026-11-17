"""Cue `turn` (wave 5, backlog B38): switch timing from the players' forward acceleration.

Why (the owner's error review of v6, 14 Nov): switches land slightly late. At a turnover the team
that loses the ball often keeps running forward a little but slows down, while the team that wins
it accelerates forward; the losing team's turn in direction comes later still. v6's switch cue
(B36, src/models/switch_cue.py) reads changes of `work`, `vx` and `recovery` over 1-2 s windows;
this cue adds faster and more specific movement, per team, in its attack direction:

  acc_front W   forward acceleration of the FRONT_N most advanced outfielders: mean vx over
                (t, t+W] minus over [t-W, t), W in ACC_WINDOWS_S (0.5 s, 1 s)
  acc_all W     the same for all outfielders (every player but the deepest, the goalkeeper)
  brake W       braking while still going forward: mean vx before minus after, where the after
                mean is still > 0 (0 where it isn't or where vx rose); W in ACC_WINDOWS_S
  turn L        the delayed turn: acc_all over TURN_WINDOW_S, read LAGS_S later (0.5 s, 1 s), so
                the cue at a switch sees the turn in direction that follows it

Each is home minus away (src.shape.differences' symmetric form) times a sign so that u > 0 means
"the away team is gaining the ball" (H->A): the team losing it decelerates (acc -), brakes (+) and
turns back later (turn -). u_toward_home = -u_toward_away. Velocities are per player over
VEL_SPAN_S, players moving faster than shape.MAX_SPEED_MPS dropped (tracking glitches), sampled at
CUE_RATE_HZ and held to every frame as switch_cue does. NaN where a team has fewer than
shape.MIN_OUTFIELD outfielders seen or its direction is unknown. Players only: ball-free and
label-free, so it means the same on BAR-ATH with the ball hidden.

Two feature sets, two fits in one file (TURN_FILE, written by scripts/fit_turn_cue.py):
  "new"   the 8 columns above: the cue ON TOP of v6's (switch_cue_gain=0.5, cues=turn:g)
  "full"  switch_cue's 6 columns (v1's features, switch_cue.cue_values) then the 8 new ones:
          the cue INSTEAD of v6's (switch_cue_gain=0, cues=turn:g)
Option `set` (cues=turn:g/set=new|full): "auto" (default) is "full" when the decoder's own switch
cue is off and "new" when it is on, so `cues=turn:g` does the right thing in both variants.

The hazard and the multiplier are switch_cue's: a binary logit over a hat basis per column, y = a
true switch toward the other team within +/- switch_cue.CUE_TOL_S, and a multiplier on both H<->A
costs of clip(exp(-gain * (z - base_logit)), *switch_cue.COST_RANGE), 1 where u is unknown. Folds
as switch_cue.fold_fit (all / lomo / loso by the decoder's `fold`). Gain 0 is no effect.

Runtime: numpy only; deterministic (float64, no randomness; the decoder puts costs on its grid).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from src.models import switch_cue
from src.models.cues import Effect

CUE_RATE_HZ = 10.0
VEL_SPAN_S = 0.4  # per-player velocity: displacement over [t - span/2, t + span/2]
FRONT_N = 4  # the most advanced outfielders (backlog: 3-5)
ACC_WINDOWS_S = (0.5, 1.0)
TURN_WINDOW_S = 1.0
LAGS_S = (0.5, 1.0)
NEW_COLUMNS = (("acc_front", 0.5), ("acc_front", 1.0), ("acc_all", 0.5), ("acc_all", 1.0),
               ("brake", 0.5), ("brake", 1.0), ("turn", 0.5), ("turn", 1.0))
SIGNS = {"acc_front": -1.0, "acc_all": -1.0, "brake": 1.0, "turn": -1.0}  # u > 0: away gaining
# hat knots per feature (m/s), symmetric, spanning the pooled 2-98% range of both windows on the
# five sources' training rows (scripts/fit_turn_cue.py --knots, 14 Nov)
KNOTS = {"acc_front": (-3.0, -1.6, -0.6, 0.0, 0.6, 1.6, 3.0),
         "acc_all": (-2.6, -1.4, -0.6, 0.0, 0.6, 1.4, 2.6),
         "brake": (-1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0),
         "turn": (-3.0, -1.7, -0.65, 0.0, 0.65, 1.7, 3.0)}
SETS = ("new", "full")
TURN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turn_coefs.json")
SPEC = {"rate_hz": CUE_RATE_HZ, "vel_span_s": VEL_SPAN_S, "front_n": FRONT_N,
        "acc_windows_s": list(ACC_WINDOWS_S), "turn_window_s": TURN_WINDOW_S, "lags_s": list(LAGS_S),
        "columns": [list(c) for c in NEW_COLUMNS], "signs": SIGNS,
        "knots": {k: list(v) for k, v in KNOTS.items()}, "tol_s": switch_cue.CUE_TOL_S,
        "v1_spec_hash": switch_cue.SPEC_HASH, "version": 1}


def spec_hash(spec=None) -> str:
    spec = SPEC if spec is None else spec
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


SPEC_HASH = spec_hash()


def columns(which: str) -> list[tuple[str, float]]:
    """The columns of u for feature set `which`: (feature, window) pairs; "full" starts with
    switch_cue.COLUMNS (their features are prefixed "v1_")."""
    if which not in SETS:
        raise ValueError("turn: set must be one of {}, not {!r}".format(SETS, which))
    new = list(NEW_COLUMNS)
    return new if which == "new" else [("v1_" + f, w) for f, w in switch_cue.COLUMNS] + new


def knots_of(feature: str):
    if feature.startswith("v1_"):
        return switch_cue.CUE_KNOTS[feature[3:]]
    return KNOTS[feature]


# -- features --------------------------------------------------------------------------------------

def team_motion(match) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(idx (m,), v_all (m, 2), v_front (m, 2), rate): every step-th frame, each team's (column
    HOME, AWAY) mean forward velocity of its outfielders and of its FRONT_N most advanced ones, in
    its own attack direction (m/s). NaN outside play, where the direction is unknown, or with fewer
    than shape.MIN_OUTFIELD outfielders seen."""
    from src import shape
    from src.schema import AWAY, HOME

    step = max(1, int(round(match.fps / CUE_RATE_HZ)))
    idx = np.arange(0, match.n_frames, step)
    v_all = np.full((len(idx), 2), np.nan)
    v_front = np.full((len(idx), 2), np.nan)
    if not len(idx):
        return idx, v_all, v_front, match.fps / step
    sign = shape.attack_sign(match)
    h = max(1, int(round(VEL_SPAN_S * match.fps / 2)))
    lo, hi, same = shape._shifted(match, idx, h)
    dt = 2 * h / match.fps
    for t in (HOME, AWAY):
        cols = match.slot_team == t
        if not np.any(cols):
            continue
        s = sign[idx, t].astype(np.float64)
        for start in range(0, len(idx), shape.CHUNK):
            sl = slice(start, start + shape.CHUNK)
            p_lo, p_hi = match.players[lo[sl]][:, cols], match.players[hi[sl]][:, cols]
            disp = p_hi - p_lo
            with np.errstate(invalid="ignore"):
                speed = np.hypot(disp[..., 0], disp[..., 1]) / dt
                vx = np.where(speed <= shape.MAX_SPEED_MPS, disp[..., 0] / dt, np.nan) * s[sl, None]
            x = match.players[idx[sl]][:, cols, 0] * s[sl, None]  # own-frame x' up to a constant
            valid = ~np.isnan(vx) & ~np.isnan(x)
            count = valid.sum(axis=1)
            order = np.argsort(np.where(valid, x, np.inf), axis=1, kind="stable")
            vs = np.take_along_axis(np.where(valid, vx, 0.0), order, axis=1)
            pos = np.arange(vs.shape[1])[None, :]
            outfield = (pos >= 1) & (pos < count[:, None])  # every seen player but the deepest
            front = outfield & (pos >= count[:, None] - FRONT_N)
            ok = same[sl] & (count - 1 >= shape.MIN_OUTFIELD) & (s[sl] != 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                a = np.sum(vs * outfield, axis=1) / np.maximum(outfield.sum(axis=1), 1)
                f = np.sum(vs * front, axis=1) / np.maximum(front.sum(axis=1), 1)
            v_all[sl, t] = np.where(ok, a, np.nan)
            v_front[sl, t] = np.where(ok, f, np.nan)
    return idx, v_all, v_front, match.fps / step


def _shift(values: np.ndarray, k: int) -> np.ndarray:
    """values[i + k] (NaN past the end)."""
    out = np.full_like(values, np.nan)
    if k < len(values):
        out[:len(values) - k] = values[k:]
    return out


def new_values(match) -> np.ndarray:
    """(n, 8) u toward the away team for NEW_COLUMNS, held to every frame."""
    from src.models.learned import _offset_mean

    idx, v_all, v_front, rate = team_motion(match)
    u = np.full((match.n_frames, len(NEW_COLUMNS)), np.nan)
    if not len(idx):
        return u
    step = max(1, int(round(match.fps / CUE_RATE_HZ)))  # as team_motion
    nearest = np.minimum((np.arange(match.n_frames) + step // 2) // step, len(idx) - 1)
    k_of = lambda s: max(1, int(round(s * rate)))  # noqa: E731
    per_team = {}
    for name in ("acc_front", "acc_all", "brake", "turn"):
        for window in (LAGS_S if name == "turn" else ACC_WINDOWS_S):
            vals = np.full((len(idx), 2), np.nan)
            for t in range(2):
                v = v_front[:, t] if name == "acc_front" else v_all[:, t]
                if name == "brake":
                    k = k_of(window)
                    with np.errstate(invalid="ignore"):
                        before = _offset_mean(v, -k, -1)
                        after = _offset_mean(v, 1, k)
                        b = np.where(after > 0, np.maximum(before - after, 0.0), 0.0)
                    vals[:, t] = np.where(np.isnan(before) | np.isnan(after) | np.isnan(v), np.nan, b)
                elif name == "turn":
                    vals[:, t] = _shift(switch_cue._change(v, k_of(TURN_WINDOW_S)), k_of(window))
                else:
                    vals[:, t] = switch_cue._change(v, k_of(window))
            per_team[(name, window)] = vals
    for j, (name, window) in enumerate(NEW_COLUMNS):
        vals = per_team[(name, window)]
        u[:, j] = (SIGNS[name] * (vals[:, 0] - vals[:, 1]))[nearest]
    return u


_LAST: list = [None, None, None]  # (match, set, (u_away, u_home)): one match is decoded many ways


def cue_values(match, which: str = "new") -> tuple[np.ndarray, np.ndarray]:
    """(u_toward_away, u_toward_home), (n, len(columns(which))) each; u_toward_home = -u_toward_away.
    The last match's values are kept."""
    columns(which)  # validates
    if _LAST[0] is not match or _LAST[1] != which:
        u = new_values(match)
        if which == "full":
            u = np.hstack([switch_cue.cue_values(match)[0], u])
        _LAST[:] = [match, which, (u, -u)]
    return _LAST[2]


# -- hazard ----------------------------------------------------------------------------------------

def design(u: np.ndarray, which: str) -> np.ndarray:
    """(m, 1 + sum of knot counts): a bias column, then a hat basis per column (NaN -> zero row)."""
    from src.models.learned import hat

    cols = [np.ones((len(u), 1))]
    for j, (name, _) in enumerate(columns(which)):
        cols.append(hat(u[:, j], knots_of(name)))
    return np.hstack(cols)


def logits_of(u: np.ndarray, beta, which: str) -> np.ndarray:
    """design(u) @ beta; NaN on rows where every column of u is NaN."""
    z = design(u, which) @ np.asarray(beta, dtype=float)
    return np.where(np.isnan(u).all(axis=1), np.nan, z)


_FILES: dict = {}


def load(path: str = TURN_FILE) -> dict:
    """The fit file written by scripts/fit_turn_cue.py; refuses one fit for another spec."""
    if path not in _FILES:
        with open(path) as handle:
            cue = json.load(handle)
        if cue.get("spec_hash") != SPEC_HASH:
            raise ValueError("{} was fit for another turn-cue spec ({} != {}); rerun "
                             "scripts/fit_turn_cue.py".format(path, cue.get("spec_hash"), SPEC_HASH))
        _FILES[path] = cue
    return _FILES[path]


def pick_set(which: str, decoder) -> str:
    if which == "auto":
        return "new" if getattr(decoder, "switch_cue_gain", 0.0) else "full"
    columns(which)
    return which


def effect(match, original, decoder, gain, set="auto", file=TURN_FILE):  # noqa: A002
    """Multipliers on both H<->A costs from the turn hazard (see the module docstring)."""
    del original  # the cue reads the decoder's (repaired) match, as switch_cue does
    if not gain:
        return None
    which = pick_set(str(set), decoder)
    fit = switch_cue.fold_fit(load(str(file))["sets"][which], decoder.fold, match.match_id)
    u_away, u_home = cue_values(match, which)
    base = float(fit["base_logit"])
    mult = [switch_cue.multiplier(logits_of(u, fit["beta"], which), base, float(gain))
            for u in (u_away, u_home)]
    return Effect(ha=mult[0], ah=mult[1])
