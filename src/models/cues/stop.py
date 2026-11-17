"""Stop and go: the players say when the ball goes dead and when play restarts (wave 5, B37, F300+).

Why: the owner's error review of v6 (14 Nov): stoppages are missed although the players slow down
reliably (the player at the ball first, e.g. offside, then the defenders), and players often kick the
ball on after the whistle, which reads as play. Tried before only as hard rules (F82's braking
"whistle" snap tied on v3) and as emission inputs (`deadm`, lost IDSSE held out by source, F172).
Here it is a soft per-frame cost, the form that turned B36's switch cue into a win (F179).

Two hazards, each a binary logit over a hat basis (as src/models/switch_cue.py), fit by
scripts/fit_stop_cue.py on external matches (lomo / loso / all fits in src/models/cues/stop_coefs.json):

  stop  y = 1 if a true live -> D onset is within +/- TOL_S (rows: live TOL_S earlier). Columns:
        each team's mean-speed change over 0.5 / 1 / 2 s (as the min and the max over the two teams,
        so swapping the teams changes nothing), the share of players braking (F82: 1 m/s slower over
        the next 1 s than the last) and under walking pace (2 m/s over the next 1 s), the speed and
        the 1 s deceleration of the player nearest the ball and the mean speed of the 3 nearest
        (where the ball is seen).
  go    y = 1 if a true D -> live is within +/- TOL_S (rows: dead TOL_S earlier). Columns: the
        collective speed change over 0.5 / 1 / 2 s, the share of players speeding up (1 m/s faster
        over the next 1 s), and the ball's speed jump from near-still (its speed over the next 0.4 s
        where it was slower than BALL_STILL_MPS over the last 1 s, else 0) where the ball is seen.
  Both also get a bias and a ball-seen column (so a hidden ball doesn't bias the logit).

Effect (spec "stop:<gain>/go=<go gain>"):
  team -> D (H->D, A->D)  x clip(exp(-gain    * (z_stop - base_stop)), *STOP_RANGE)
  D -> team (D->H, D->A)  x clip(exp(-go_gain * (z_go   - base_go)),   *GO_RANGE)
base = the fit's weighted positive rate as a logit: an above-average hazard makes the change cheaper.
GO_RANGE reaches 4: without go evidence a restart gets dearer, which is what stops a kick after the
whistle from restarting play. Multiplier 1 where a logit is unknown (nobody tracked). gain 0 = no
effect (the whole cue, by the cue contract); go=0 (default) leaves the restarts alone.

Players only (plus the ball where it is seen), label-free, NaN-safe. Features at RATE_HZ, held to
every frame. Runtime: numpy, ~1-2 s per 90-minute match, cached per match object (the grid decodes
several settings of one match). Deterministic: float64, no randomness; the cue framework puts the
costs on decoder.to_grid.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from src.models.cues import Effect
from src.schema import AWAY, HOME

RATE_HZ = 5.0
WINDOWS_S = (0.5, 1.0, 2.0)  # speed change = mean over (t, t+W] minus mean over [t-W, t)
SPAN_S = 1.0  # braking / speeding-up span (F82)
BRAKE_MPS = 1.0  # F82: 1 m/s slower (or faster) over the next SPAN_S than over the last
WALK_MPS = 2.0  # walking pace
BALL_STILL_MPS, BALL_BEFORE_S, BALL_AFTER_S = 1.5, 1.0, 0.4
MAX_SPEED_MPS, MAX_BALL_MPS = 12.0, 40.0  # tracking glitches
MIN_TEAM = 3  # a team's mean speed needs this many seen players
TOL_S = 0.5  # a row is positive if a true onset (restart) is within +/- this
STOP_RANGE = (0.25, 2.0)
GO_RANGE = (0.25, 4.0)

_SPEED_KNOTS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_SHARE_KNOTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_LEVEL_KNOTS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
_DECEL_KNOTS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0)
_JUMP_KNOTS = (0.0, 2.0, 5.0, 10.0, 20.0)

STOP_COLUMNS = (tuple(("team_min", w) for w in WINDOWS_S) + tuple(("team_max", w) for w in WINDOWS_S)
                + (("brake_share", SPAN_S), ("walk_share", SPAN_S), ("near_speed", 0.0),
                   ("near_decel", SPAN_S), ("near3_speed", 0.0)))
GO_COLUMNS = (tuple(("all", w) for w in WINDOWS_S) + (("go_share", SPAN_S), ("ball_jump", BALL_AFTER_S)))
KNOTS = {"team_min": _SPEED_KNOTS, "team_max": _SPEED_KNOTS, "all": _SPEED_KNOTS,
         "brake_share": _SHARE_KNOTS, "walk_share": _SHARE_KNOTS, "go_share": _SHARE_KNOTS,
         "near_speed": _LEVEL_KNOTS, "near3_speed": _LEVEL_KNOTS, "near_decel": _DECEL_KNOTS,
         "ball_jump": _JUMP_KNOTS}
COEF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_coefs.json")
SPEC = {"rate_hz": RATE_HZ, "windows_s": list(WINDOWS_S), "span_s": SPAN_S, "brake_mps": BRAKE_MPS,
        "walk_mps": WALK_MPS, "ball": [BALL_STILL_MPS, BALL_BEFORE_S, BALL_AFTER_S],
        "max": [MAX_SPEED_MPS, MAX_BALL_MPS], "min_team": MIN_TEAM, "tol_s": TOL_S,
        "stop": [list(c) for c in STOP_COLUMNS], "go": [list(c) for c in GO_COLUMNS],
        "knots": {k: list(v) for k, v in sorted(KNOTS.items())}, "version": 1}
SPEC_HASH = hashlib.sha1(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:10]


# -- features ----------------------------------------------------------------------------------------

def _mean(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    from src.models.learned import _offset_mean  # lazy: learned imports the cue framework

    return _offset_mean(values, lo, hi)


def _change(values: np.ndarray, k: int) -> np.ndarray:
    """mean over samples (i, i + k] minus mean over [i - k, i), ignoring NaN; NaN where either side
    has nothing or values[i] itself is unknown."""
    with np.errstate(invalid="ignore"):
        out = _mean(values, 1, k) - _mean(values, -k, -1)
    return np.where(np.isnan(values), np.nan, out)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[i + k] at row i (NaN past the ends), along axis 0."""
    out = np.full_like(a, np.nan)
    if k > 0 and k < len(a):
        out[:-k] = a[k:]
    elif k < 0 and -k < len(a):
        out[-k:] = a[:k]
    elif k == 0:
        out[:] = a
    return out


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1])


def _nanmean_rows(values: np.ndarray, mask: np.ndarray | None = None, min_count: int = 1) -> np.ndarray:
    seen = ~np.isnan(values)
    if mask is not None:
        seen &= mask
    count = seen.sum(axis=1)
    total = np.where(seen, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_count, total / np.maximum(count, 1), np.nan)


def _share(flag: np.ndarray, seen: np.ndarray) -> np.ndarray:
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (flag & seen).sum(axis=1) / np.maximum(count, 1), np.nan)


def sample_features(match) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(index (m,), stop (m, len(STOP_COLUMNS)), go (m, len(GO_COLUMNS)), ball_seen (m,)) at the
    RATE_HZ samples index = 0, step, 2 step, ... NaN where a feature is unknown."""
    step = max(1, int(round(match.fps / RATE_HZ)))
    rate = match.fps / step
    index = np.arange(0, match.n_frames, step)
    p = match.players[index]  # (m, k, 2)
    b = match.ball[index]  # (m, 2)
    m = len(index)
    # per-player speed over +/- one sample (m/s)
    spd = np.minimum(_dist(_shift(p, 1), _shift(p, -1)) * rate / 2.0, MAX_SPEED_MPS)
    # per-player speed over the next / the last SPAN_S (displacement: noise cancels, C5)
    ks = max(1, int(round(SPAN_S * rate)))
    fwd = np.minimum(_dist(_shift(p, ks), p) / SPAN_S, MAX_SPEED_MPS)
    back = np.minimum(_dist(p, _shift(p, -ks)) / SPAN_S, MAX_SPEED_MPS)
    with np.errstate(invalid="ignore"):
        both = ~np.isnan(fwd) & ~np.isnan(back)
        brake = (back - fwd) >= BRAKE_MPS
        speedup = (fwd - back) >= BRAKE_MPS
        walk = fwd < WALK_MPS
    stop = np.full((m, len(STOP_COLUMNS)), np.nan)
    go = np.full((m, len(GO_COLUMNS)), np.nan)
    team_speed = [_nanmean_rows(spd, (match.slot_team == t)[None, :], MIN_TEAM) for t in (HOME, AWAY)]
    all_speed = _nanmean_rows(spd, None, 2 * MIN_TEAM)
    for j, w in enumerate(WINDOWS_S):
        k = max(1, int(round(w * rate)))
        changes = np.stack([_change(v, k) for v in team_speed])
        with np.errstate(invalid="ignore"):
            stop[:, j] = np.min(changes, axis=0)
            stop[:, len(WINDOWS_S) + j] = np.max(changes, axis=0)
        go[:, j] = _change(all_speed, k)
    c = 2 * len(WINDOWS_S)
    stop[:, c] = _share(brake, both)
    stop[:, c + 1] = _share(walk, ~np.isnan(fwd))
    go[:, len(WINDOWS_S)] = _share(speedup, both)
    # the players nearest the ball, where it is seen
    ball_seen = ~np.isnan(b[:, 0])
    dist = _dist(p, b[:, None, :])
    dist = np.where(np.isnan(dist) | np.isnan(spd), np.inf, dist)
    order = np.argsort(dist, axis=1, kind="stable")[:, :3]
    rows = np.arange(m)[:, None]
    near_ok = np.isfinite(dist[rows, order])
    near_spd = np.where(near_ok, spd[rows, order], np.nan)
    near_decel = np.where(near_ok[:, 0], (fwd - back)[np.arange(m), order[:, 0]], np.nan)
    stop[:, c + 2] = np.where(ball_seen & near_ok[:, 0], near_spd[:, 0], np.nan)
    stop[:, c + 3] = np.where(ball_seen, near_decel, np.nan)
    with np.errstate(invalid="ignore"):
        near3 = near_spd.mean(axis=1) if near_spd.shape[1] == 3 else np.full(m, np.nan)
    stop[:, c + 4] = np.where(ball_seen & near_ok.all(axis=1), near3, np.nan)
    # the ball's jump from near-still
    bspd = np.minimum(_dist(_shift(b, 1), _shift(b, -1)) * rate / 2.0, MAX_BALL_MPS)
    before = _mean(bspd, -max(1, int(round(BALL_BEFORE_S * rate))), -1)
    after = _mean(bspd, 1, max(1, int(round(BALL_AFTER_S * rate))))
    with np.errstate(invalid="ignore"):
        jump = np.where(before < BALL_STILL_MPS, after, 0.0)
    go[:, len(WINDOWS_S) + 1] = np.where(np.isnan(before) | np.isnan(after) | ~ball_seen, np.nan, jump)
    return index, stop, go, ball_seen.astype(np.float64)


def design(features: np.ndarray, ball_seen: np.ndarray, columns) -> np.ndarray:
    """(m, 2 + sum of knot counts): bias, ball seen, then a hat basis per column (NaN -> zeros)."""
    from src.models.learned import hat  # lazy: learned imports the cue framework

    cols = [np.ones((len(features), 1)), ball_seen.reshape(-1, 1)]
    for j, (name, _) in enumerate(columns):
        cols.append(hat(features[:, j], KNOTS[name]))
    return np.hstack(cols)


def logits(features: np.ndarray, ball_seen: np.ndarray, columns, beta) -> np.ndarray:
    """design @ beta; NaN on rows where every feature is NaN."""
    z = design(features, ball_seen, columns) @ np.asarray(beta, dtype=float)
    return np.where(np.isnan(features).all(axis=1), np.nan, z)


_LAST: list = [None, None]  # (match, sample_features(match))


def _features(match):
    if _LAST[0] is not match:
        _LAST[:] = [match, sample_features(match)]
    return _LAST[1]


def multiplier(z: np.ndarray, base_logit: float, gain: float, bounds) -> np.ndarray:
    """clip(exp(-gain * (z - base_logit)), *bounds); 1 where z is NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        mult = np.clip(np.exp(-gain * (z - base_logit)), *bounds)
    return np.where(np.isnan(z), 1.0, mult)


# -- coefficients ------------------------------------------------------------------------------------

_FILES: dict = {}


def load(path: str = COEF_FILE) -> dict:
    """The fit file written by scripts/fit_stop_cue.py; refuses one fit for another spec."""
    if path not in _FILES:
        with open(path) as handle:
            fits = json.load(handle)
        if fits.get("spec_hash") != SPEC_HASH:
            raise ValueError("{} was fit for another stop-cue spec ({} != {}); rerun "
                             "scripts/fit_stop_cue.py".format(path, fits.get("spec_hash"), SPEC_HASH))
        _FILES[path] = fits
    return _FILES[path]


def fold_fit(fits: dict, fold: str, match_id: str) -> dict:
    """{"stop": {...}, "go": {...}} for this match: "all", or the lomo / loso fit without it."""
    if fold == "all":
        return fits["all"]
    key = match_id if fold == "lomo" else match_id.split(":", 1)[0]
    if key not in fits[fold]:
        raise KeyError("no {} stop-cue fit without {!r}".format(fold, key))
    return fits[fold][key]


def frame_logits(match, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """(z_stop, z_go) per frame (each sample's value held to the frames nearest it)."""
    index, stop, go, seen = _features(match)
    step = max(1, int(round(match.fps / RATE_HZ)))
    nearest = np.minimum((np.arange(match.n_frames) + step // 2) // step, max(len(index) - 1, 0))
    if not len(index):
        empty = np.full(match.n_frames, np.nan)
        return empty, empty.copy()
    z_stop = logits(stop, seen, STOP_COLUMNS, fit["stop"]["beta"])
    z_go = logits(go, seen, GO_COLUMNS, fit["go"]["beta"])
    return z_stop[nearest], z_go[nearest]


def effect(match, original, decoder, gain, go=0.0, file=COEF_FILE) -> Effect | None:
    """The cue on one match: `gain` on team -> D, `go` on D -> team (see the module docstring)."""
    gain, go = float(gain), float(go)
    if gain == 0:
        return None
    # `file` names a coefficient file next to this module (the spec syntax has no "/")
    fit = fold_fit(load(os.path.join(os.path.dirname(COEF_FILE), os.path.basename(str(file)))),
                   decoder.fold, match.match_id)
    z_stop, z_go = frame_logits(match, fit)
    mult = multiplier(z_stop, float(fit["stop"]["base_logit"]), gain, STOP_RANGE)
    live = multiplier(z_go, float(fit["go"]["base_logit"]), go, GO_RANGE) if go else None
    return Effect.dead(mult=mult, live=live)


def fit_entry(beta, base_logit: float) -> dict:
    """One hazard as stored in the file (floats rounded to 6 decimals)."""
    return {"beta": [round(float(v), 6) for v in beta], "base_logit": round(float(base_logit), 6)}
