"""Fast team-shape changes as a switch-timing cue for the learned decoder (local L2, findings F200+).

Why: `work`, `vx` and `recovery` flip within ~1 s of a turnover (F89), while the spread features lag
~3.5 s. As emissions they vote for a team; here they make an H<->A switch cheaper (or dearer) at the
frame they point to, which is what missed short spells need (F200: a flat cheaper switch won't do,
switch precision is already low).

Cue, per frame (ball-free, label-free; players only, from src/shape.py at CUE_RATE_HZ):
  d_f = home minus away of team feature f (src.shape.differences of team_features)
  change(d, W) = mean of d over (t, t + W] minus its mean over [t - W, t), for W in CUE_WINDOWS_S
  direction-normalised so u > 0 means "the away team is gaining the ball" (an H->A switch):
    u_work = +change(d_work)       (the team that lost the ball runs more, F88)
    u_vx   = -change(d_vx)         (the team with the ball moves forward)
    u_rec  = +change(d_recovery)   (the team that lost it sprints back)
  u_toward_home = -u_toward_away, so swapping the teams swaps the two directions.

Hazard: a binary logit on design(u) (a bias, then a hat basis per column), fit by
scripts/fit_switch_cue.py on external matches: y = 1 if a true switch in that direction is within
+/- CUE_TOL_S. The decoder's per-frame switch cost is
  switch_cost * clip(exp(-gain * (z - base_logit)), *COST_RANGE)
with base_logit the fit's weighted positive rate as a logit: above-average hazard -> cheaper switch.
1 (the flat cost) where the cue is unknown. LearnedDecoder.switch_cue_gain = 0 (default) is off.

Coverage gate (B36): the cue is a team-shape signal, so it only means something where both teams'
players are actually tracked (F203: it failed on SkillCorner's TV tracking, where ~43% of positions
are estimates, F93). gate(match, min_seen) is True on frames where both teams have at least
min_seen outfield players *seen* (src/shape_coverage.py, C10) on every frame within the cue's widest
window either side; elsewhere the multiplier is 1 (the flat cost). "Seen" = a position that isn't
NaN, except on SkillCorner 2024/25, whose loader fills off-camera players with SkillCorner's
estimates: there its is_detected flags count (shape_coverage.skillcorner_extras). BAR-ATH's tactical
tracking leaves NaN where a tracklet is absent, like IDSSE, Metrica, SkillCorner 2019/20 and
SoccerTrack. LearnedDecoder.switch_cue_min_seen = 0 (default) is no gate.

Runtime: numpy only. Deterministic: float64, no randomness, costs on decoder.to_grid like the scores.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from src.models.decoder import to_grid

CUE_RATE_HZ = 5.0
CUE_FEATURES = ("work", "vx", "recovery")
CUE_WINDOWS_S = (1.0, 2.0)  # change = mean over (t, t+W] minus mean over [t-W, t)
CUE_SIGNS = {"work": 1.0, "vx": -1.0, "recovery": 1.0}  # u > 0: the away team is gaining the ball
CUE_KNOTS = {"work": (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0),
             "vx": (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0),
             "recovery": (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)}
CUE_TOL_S = 0.5  # a training row is positive if a true switch in that direction is within +/- this
COST_RANGE = (0.25, 2.0)  # the per-frame multiplier on switch_cost is clipped to this
COEF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "switch_cue_coefs.json")
SPEC = {"features": list(CUE_FEATURES), "windows_s": list(CUE_WINDOWS_S),
        "knots": {k: list(v) for k, v in CUE_KNOTS.items()}, "tol_s": CUE_TOL_S,
        "rate_hz": CUE_RATE_HZ, "version": 1}
# columns of u, in this order: (feature, window) for feature in CUE_FEATURES for window in CUE_WINDOWS_S
COLUMNS = tuple((f, w) for f in CUE_FEATURES for w in CUE_WINDOWS_S)


def spec_hash(spec=None) -> str:
    spec = SPEC if spec is None else spec
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


SPEC_HASH = spec_hash()


def _change(values: np.ndarray, k: int) -> np.ndarray:
    """mean over samples (i, i + k] minus mean over [i - k, i), ignoring NaN; NaN where either side
    has no known sample or values[i] itself is unknown."""
    from src.models.learned import _offset_mean  # lazy: learned imports this module

    with np.errstate(invalid="ignore"):
        out = _offset_mean(values, 1, k) - _offset_mean(values, -k, -1)
    return np.where(np.isnan(values), np.nan, out)


_LAST: list = [None, None]  # (match, cue_values(match)): decoding several settings reads one match


def cue_values(match) -> tuple[np.ndarray, np.ndarray]:
    """(u_toward_away, u_toward_home): (n, 6) direction-normalised changes, held to every frame
    (as learned.shape_values does: sample at CUE_RATE_HZ, nearest sample). u_toward_home =
    -u_toward_away. Rows are NaN where shape is unknown (< MIN_OUTFIELD on either team).
    The last match's values are kept (the same object decoded again costs nothing)."""
    if _LAST[0] is not match:
        _LAST[:] = [match, _cue_values(match)]
    return _LAST[1]


def _cue_values(match) -> tuple[np.ndarray, np.ndarray]:
    from src import shape

    step = max(1, int(round(match.fps / CUE_RATE_HZ)))
    idx, feats = shape.team_features(match, step=step, features=CUE_FEATURES)
    diff = shape.differences(feats)
    rate = match.fps / step
    nearest = np.minimum((np.arange(match.n_frames) + step // 2) // step, max(len(idx) - 1, 0))
    u = np.full((match.n_frames, len(COLUMNS)), np.nan)
    if len(idx):
        for j, (name, window) in enumerate(COLUMNS):
            k = max(1, int(round(window * rate)))
            u[:, j] = CUE_SIGNS[name] * _change(diff[name], k)[nearest]
    return u, -u


def design(u: np.ndarray) -> np.ndarray:
    """(n, 1 + sum of knot counts): a bias column, then hat(u[:, j], knots) per column (NaN -> zero row)."""
    from src.models.learned import hat  # lazy: learned imports this module

    cols = [np.ones((len(u), 1))]
    for j, (name, _) in enumerate(COLUMNS):
        cols.append(hat(u[:, j], CUE_KNOTS[name]))
    return np.hstack(cols)


def fit_binary(x: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 30) -> np.ndarray:
    """w maximising sum w_i log p(y_i) - L2/2 |w|^2, p(1) = sigmoid(x @ w). Newton, deterministic.
    Copy of scripts/check_shape_model.py's fit_binary (scripts are not a package)."""
    from src.models.learned import L2

    x = x.astype(np.float64)
    beta = np.zeros(x.shape[1])

    def objective(b):
        z = x @ b
        return float(np.sum(w * (y * z - np.logaddexp(0.0, z)))) - 0.5 * L2 * b @ b

    value = objective(beta)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        grad = x.T @ (w * (y - p)) - L2 * beta
        hess = -(x.T @ (x * (w * p * (1 - p))[:, None])) - L2 * np.eye(len(beta))
        step = np.linalg.solve(hess, -grad)
        scale = 1.0
        while scale > 1e-4:
            new = beta + scale * step
            new_value = objective(new)
            if new_value >= value:
                break
            scale /= 2
        else:
            break
        beta, gain, value = new, new_value - value, new_value
        if gain < 1e-10:
            break
    return beta


def logits_of(u: np.ndarray, beta) -> np.ndarray:
    """design(u) @ beta; NaN on rows where every column of u is NaN."""
    z = design(u) @ np.asarray(beta, dtype=float)
    return np.where(np.isnan(u).all(axis=1), np.nan, z)


def hazard_logits(match, beta) -> tuple[np.ndarray, np.ndarray]:
    """(z_HA, z_AH) per frame: design(u) @ beta for each direction; NaN where u is NaN."""
    u_away, u_home = cue_values(match)
    return logits_of(u_away, beta), logits_of(u_home, beta)


def multiplier(z: np.ndarray, base_logit: float, gain: float) -> np.ndarray:
    """clip(exp(-gain * (z - base_logit)), *COST_RANGE); 1 where z is NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        mult = np.clip(np.exp(-gain * (z - base_logit)), *COST_RANGE)
    return np.where(np.isnan(z), 1.0, mult)


def costs(match, coefs: dict, switch_cost: float, gain: float, where: str,
          gate_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame H->A and A->H switch costs. mult = clip(exp(-gain * (z - coefs['base_logit'])),
    *COST_RANGE); mult = 1 where z is NaN, where the ball is seen if where == "missing", and where
    gate_mask (n,) bool is False when one is given. cost = switch_cost * mult."""
    if where not in ("all", "missing"):
        raise ValueError("where must be all or missing, not {!r}".format(where))
    z_ha, z_ah = hazard_logits(match, coefs["beta"])
    out = []
    for z in (z_ha, z_ah):
        mult = multiplier(z, float(coefs["base_logit"]), gain)
        if where == "missing":
            mult = np.where(np.isnan(match.ball[:, 0]), mult, 1.0)
        if gate_mask is not None:
            mult = np.where(gate_mask, mult, 1.0)
        out.append(switch_cost * mult)
    return out[0], out[1]


# -- coverage gate (B36) -----------------------------------------------------------------------------

_SEEN: dict = {}  # SkillCorner match id -> detected (n, k): read once per worker


def seen_mask(match) -> np.ndarray | None:
    """(n, k) bool of positions that were observed, for shape_coverage.coverage(seen=...): the
    is_detected flags on SkillCorner 2024/25 (its loader fills off-camera players with estimates);
    None elsewhere (a position that isn't NaN was tracked: BAR-ATH and the other sources)."""
    source, _, raw_id = match.match_id.partition(":")
    if source != "skillcorner":
        return None
    if raw_id not in _SEEN:
        from src.shape_coverage import skillcorner_extras

        _SEEN.clear()  # one match at a time is enough (workers score match by match)
        _SEEN[raw_id] = skillcorner_extras(raw_id)["detected"]
    detected = _SEEN[raw_id]
    if detected.shape != match.players.shape[:2]:
        raise ValueError("{}: detection flags {} don't match the players {}".format(
            match.match_id, detected.shape, match.players.shape[:2]))
    return detected


def gate(match, min_seen: int) -> np.ndarray:
    """(n,) bool: in play, and both teams have >= min_seen seen outfield players on every frame
    within max(CUE_WINDOWS_S) either side (the frames the cue reads). All True for min_seen <= 0."""
    if min_seen <= 0:
        return np.ones(match.n_frames, dtype=bool)
    from src.shape_coverage import coverage

    cov = coverage(match, seen=seen_mask(match))
    ok = cov.in_period & (cov.outfield.min(axis=1) >= min_seen)
    half = int(round(max(CUE_WINDOWS_S) * match.fps))
    if half and len(ok):
        bad = np.concatenate([[0], np.cumsum(~ok)])
        lo = np.maximum(np.arange(len(ok)) - half, 0)
        hi = np.minimum(np.arange(len(ok)) + half + 1, len(ok))
        ok = (bad[hi] - bad[lo]) == 0
    return ok


def viterbi_var(scores: np.ndarray, cost_ha, cost_ah, dead_cost: float) -> np.ndarray:
    """decoder.viterbi with per-frame switch costs: cost_ha[t] is the cost of H at t-1 -> A at t,
    cost_ah[t] of A at t-1 -> H at t (index 0 unused). The same recursion, tie rule (a change happens
    as late as possible) and backtracking as decoder.viterbi; scores and costs on the grid first."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    scores = to_grid(scores)
    c_ha = to_grid(np.broadcast_to(np.asarray(cost_ha, dtype=np.float64), (n,))).tolist()
    c_ah = to_grid(np.broadcast_to(np.asarray(cost_ah, dtype=np.float64), (n,))).tolist()
    dead_cost = float(to_grid(dead_cost))
    sh, sa, sd = (scores[:, k].tolist() for k in range(3))
    back = bytearray(3 * n)
    vh, va, vd = sh[0], sa[0], sd[0]
    for t in range(1, n):
        # H
        best, arg = vh, 0
        x = va - c_ah[t]
        if x >= best:
            best, arg = x, 1
        x = vd - dead_cost
        if x >= best:
            best, arg = x, 2
        nh, bh = best + sh[t], arg
        # A
        best, arg = va, 1
        x = vh - c_ha[t]
        if x >= best:
            best, arg = x, 0
        x = vd - dead_cost
        if x >= best:
            best, arg = x, 2
        na, ba = best + sa[t], arg
        # D
        best, arg = vd, 2
        x = vh - dead_cost
        if x >= best:
            best, arg = x, 0
        x = va - dead_cost
        if x >= best:
            best, arg = x, 1
        nd, bd = best + sd[t], arg
        vh, va, vd = nh, na, nd
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
    state = int(np.argmax([vh, va, vd]))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


_CUES: dict = {}


def load_cue(path: str = COEF_FILE) -> dict:
    """The fit file written by scripts/fit_switch_cue.py; refuses one fit for another spec."""
    if path not in _CUES:
        with open(path) as handle:
            cue = json.load(handle)
        if cue.get("spec_hash") != SPEC_HASH:
            raise ValueError("{} was fit for another cue spec ({} != {}); rerun "
                             "scripts/fit_switch_cue.py".format(path, cue.get("spec_hash"), SPEC_HASH))
        _CUES[path] = cue
    return _CUES[path]


def fold_fit(cue: dict, fold: str, match_id: str) -> dict:
    """The fit to use for this match: "all", or the lomo / loso fit without it (the same key rule as
    LearnedDecoder.coefs)."""
    if fold == "all":
        return cue["all"]
    key = match_id if fold == "lomo" else match_id.split(":", 1)[0]
    if key not in cue[fold]:
        raise KeyError("no {} switch-cue fit without {!r}".format(fold, key))
    return cue[fold][key]


def fit_entry(beta: np.ndarray, base_logit: float) -> dict:
    """One fit as stored in the file (floats rounded to 6 decimals)."""
    return {"beta": [round(float(v), 6) for v in beta], "base_logit": round(float(base_logit), 6)}
