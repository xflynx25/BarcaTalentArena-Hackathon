"""Cue `flight` (wave 5, B39): a ball that only passes a player is no touch.

Why (owner's error review, 14 Nov): very brief overlaps of the ball with a player, with no clear
change in the ball's path (smooth arcs included), read as contact. The labels keep the team while
the ball travels (findings F2), and in-flight frames are the largest wrong-team block (F200). So:
from the *seen* ball only, measure how much the ball's path bends at each frame. In smooth flight an
H<->A switch gets dearer, even with a player within reach; at a deflection next to a player of the
other team it gets cheaper.

Features, per frame t, toward team X (the team a switch would go to; ball from `original`, the
match as delivered, so a filled or imputed ball never counts as seen; players from the prepared
match):
  res    log10(0.01 + RMS residual, m) of a constant-acceleration (quadratic) fit to the seen ball
         over [t - W, t + W], both axes
  dv     |v_after - v_before| (m/s): constant-velocity fits over [t - W, t] and [t, t + W] (a change
         of direction and / or speed)
  speed  (|v_before| + |v_after|) / 2 (m/s)
  dX     distance (m) from the ball to the nearest player of team X, its minimum over +/- NEAR_S
  dvX    dv where dX <= REACH_M, else 0 (a deflection next to a player of team X)
W = WINDOW_S. A fit needs at least MIN_SEEN of its window's frames seen (and 3 per side); else the
row is NaN, and so is it when the ball isn't seen at t. NaN rows get multiplier 1: no ball, no cue
(the ball is missing on ~41% of BAR-ATH's graded frames, official DATA.md).

Hazard (B36's way, src/models/switch_cue.py): a binary logit on a bias plus a hat basis per column,
y = 1 if a true switch to team X is within +/- TOL_S (rows every 1 / RATE_HZ s on frames labelled
H or A, the current team the labelled one LOOKBACK_S earlier). Fits: all, lomo (one per match,
without it), loso (one per source, without it), in COEF_FILE. Per frame

    mult_{H->A} = clip(exp(-gain * (z_toward_away - base_logit)), *COST_RANGE)   (1 where z is NaN)

and the same toward home for A->H. Gain 0 = no effect. Options (`cues=flight:0.5/jitter=0.045`):
  jitter  (m, default 0 = off) a label-free gate on the ball's own noise (B9's ball jitter, read
          while the ball moves: ball_measures()): the cue is off (no effect) on a match whose moving
          jitter is above this (SkillCorner's smoothed broadcast ball, F68).
  lin     (default 0 = off) the cue is off on a match whose straight share (ball_measures()) is
          above this: a ball drawn as constant-speed straight lines between annotated actions
          rather than measured (SoccerTrack, F151).
  Both gates read the ball as delivered, label-free, one number per match. B39 picked them on
  external data after the ungated grid lost exactly these sources (findings F309-F310).
  fit     "" (default: every column, COEF_FILE) or "nodx": the hazard fit without the plain
          proximity dX (flight only: the emission models already read who is near the ball), file
          flight_coefs_nodx.json. `cues=flight:0.5/fit=nodx`.

Fit and readout (external matches only, labels needed; seconds to minutes):

    .venv/bin/python -m src.models.cues.flight fit --jobs 3                   # writes COEF_FILE
    .venv/bin/python -m src.models.cues.flight fit --jobs 3 --variant nodx    # the flight-only fit
    .venv/bin/python -m src.models.cues.flight check --jobs 3 [--variant nodx]  # AUC / lift, held out

Runtime: numpy, a loop over the 2W+1 window offsets (no Python loop over frames). Deterministic:
float64, no randomness; the decoder puts costs on its grid (src/models/cues/__init__.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from src.models.cues import Effect

RATE_HZ = 5.0  # training rows per second
WINDOW_S = 0.3  # half-width of the ball-path fits (the task's 0.2-0.4 s)
NEAR_S = 0.2  # dX is the minimum distance over +/- this
REACH_M = 2.0  # "next to a player": dvX counts the deflection only within this
MOVE_STEP_M = 0.05  # ball_measures() reads frames where the ball moves more than this
MIN_SEEN = 0.6  # share of the window's frames the ball must be seen on
TOL_S = 0.5  # a row is positive if a true switch to team X is within +/- this
LOOKBACK_S = 1.0  # the current team is the labelled team this long before the row
COST_RANGE = (0.25, 4.0)  # the per-frame multiplier on the H<->A cost is clipped to this
COLUMNS = ("res", "dv", "speed", "dX", "dvX")
KNOTS = {"res": (-2.0, -1.5, -1.0, -0.5, 0.0),
         "dv": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0),
         "speed": (0.0, 2.0, 5.0, 10.0, 20.0),
         "dX": (0.0, 1.0, 2.0, 3.0, 5.0, 10.0),
         "dvX": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)}
COEF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flight_coefs.json")
SPEC = {"columns": list(COLUMNS), "knots": {k: list(v) for k, v in KNOTS.items()},
        "window_s": WINDOW_S, "near_s": NEAR_S, "reach_m": REACH_M, "min_seen": MIN_SEEN,
        "tol_s": TOL_S, "lookback_s": LOOKBACK_S, "rate_hz": RATE_HZ, "version": 1}
VARIANTS = {"": (), "nodx": ("dX",)}  # fit variants: columns left out ("nodx": flight only, no plain
# proximity, which the emission models already read)


def spec_for(variant: str = "") -> dict:
    return SPEC if not variant else {**SPEC, "drop": list(VARIANTS[variant])}


def spec_hash(variant: str = "") -> str:
    return hashlib.sha1(json.dumps(spec_for(variant), sort_keys=True).encode()).hexdigest()[:10]


def coef_file(variant: str = "") -> str:
    return COEF_FILE if not variant else COEF_FILE.replace(".json", "_{}.json".format(variant))


SPEC_HASH = spec_hash()


# -- ball path ------------------------------------------------------------------------------------

def _shifted(a: np.ndarray, j: int) -> np.ndarray:
    """a[t + j] for every t (NaN past the ends)."""
    out = np.full_like(a, np.nan)
    n = len(a)
    if j >= 0:
        out[:n - j] = a[j:]
    else:
        out[-j:] = a[:n + j]
    return out


def _linear_velocity(ball: np.ndarray, offsets, fps: float) -> np.ndarray:
    """(n, 2) least-squares slope (m/s) of the seen ball over t + offsets; NaN with < 3 seen."""
    s0 = s1 = s2 = 0.0
    sx = np.zeros_like(ball)
    sjx = np.zeros_like(ball)
    for j in offsets:
        b = _shifted(ball, j)
        w = (~np.isnan(b[:, 0])).astype(np.float64)
        b = np.nan_to_num(b)
        s0 = s0 + w
        s1 = s1 + w * j
        s2 = s2 + w * j * j
        sx += w[:, None] * b
        sjx += (w * j)[:, None] * b
    det = s0 * s2 - s1 * s1
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (s0[:, None] * sjx - s1[:, None] * sx) / det[:, None]
    slope[(s0 < 3) | (det <= 0)] = np.nan
    return slope * fps


def ball_path(ball: np.ndarray, fps: float) -> dict:
    """{"res", "dv", "speed"}: (n,) per-frame ball-path features (module docstring); NaN where the
    ball isn't seen at t or a window has too few seen frames."""
    n = len(ball)
    k = max(2, int(round(WINDOW_S * fps)))
    offsets = range(-k, k + 1)
    # quadratic fit over [t - k, t + k] in u = j / k (well conditioned), both axes
    s = np.zeros((n, 5))
    bx = np.zeros((n, 3, 2))
    sxx = np.zeros(n)
    for j in offsets:
        b = _shifted(ball, j)
        w = (~np.isnan(b[:, 0])).astype(np.float64)
        b = np.nan_to_num(b)
        u = j / k
        for p in range(5):
            s[:, p] += w * u ** p
        for p in range(3):
            bx[:, p, :] += (w * u ** p)[:, None] * b
        sxx += w * (b * b).sum(axis=1)
    count = s[:, 0]
    seen_t = ~np.isnan(ball[:, 0])
    ok = seen_t & (count >= max(5.0, MIN_SEEN * (2 * k + 1)))
    gram = np.stack([s[:, 0:3], s[:, 1:4], s[:, 2:5]], axis=1)  # (n, 3, 3)
    gram[~ok] = np.eye(3)
    beta = np.linalg.solve(gram, bx)  # (n, 3, 2)
    resid = sxx - np.einsum("npa,npa->n", beta, bx)
    with np.errstate(invalid="ignore"):
        rms = np.sqrt(np.maximum(resid, 0.0) / np.maximum(count, 1.0))
    res = np.where(ok, np.log10(0.01 + rms), np.nan)
    before = _linear_velocity(ball, range(-k, 1), fps)
    after = _linear_velocity(ball, range(0, k + 1), fps)
    dv = np.linalg.norm(after - before, axis=1)
    speed = 0.5 * (np.linalg.norm(before, axis=1) + np.linalg.norm(after, axis=1))
    known = ok & ~np.isnan(dv)
    return {"res": np.where(known, res, np.nan), "dv": np.where(known, dv, np.nan),
            "speed": np.where(known, speed, np.nan)}


def nearest(match, ball: np.ndarray, team: int) -> np.ndarray:
    """(n,) distance (m) from the ball to the nearest player of `team`, its minimum over +/- NEAR_S
    (NaN where the ball or every such player is unseen)."""
    from src.models.learned import window_min

    cols = np.flatnonzero(match.slot_team == team)
    if len(cols) == 0:
        return np.full(len(ball), np.nan)
    d = np.linalg.norm(match.players[:, cols, :] - ball[:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        best = np.where(np.isnan(d).all(axis=1), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
    best = np.where(np.isfinite(best), best, np.nan)
    h = int(round(NEAR_S * match.fps))
    return window_min(best, h, h) if h else best


_LAST: list = [None, None, None]  # (ball, players id, values): several settings decode one match


def cue_values(match, original) -> tuple[np.ndarray, np.ndarray]:
    """(v_toward_away, v_toward_home): (n, 5) columns COLUMNS, NaN rows where the path is unknown."""
    key = id(match.players)
    if _LAST[0] is not original.ball or _LAST[1] != key:
        _LAST[:] = [original.ball, key, _cue_values(match, original)]
    return _LAST[2]


def _cue_values(match, original):
    from src.official import A, H

    ball = np.asarray(original.ball, dtype=np.float64)
    path = ball_path(ball, match.fps)
    out = []
    for team in (A, H):  # toward away, toward home
        dx = nearest(match, ball, team)
        dvx = np.where(dx <= REACH_M, path["dv"], 0.0)
        v = np.column_stack([path["res"], path["dv"], path["speed"], dx, dvx])
        v[np.isnan(path["dv"])] = np.nan
        v[:, 4] = np.where(np.isnan(dx) & ~np.isnan(path["dv"]), 0.0, v[:, 4])
        out.append(v)
    return out[0], out[1]


def design(v: np.ndarray, drop=()) -> np.ndarray:
    """(n, 1 + sum of knot counts): a bias column, then a hat basis per column (NaN -> zero row; a
    column in `drop` is all zero)."""
    from src.models.learned import hat

    cols = [np.ones((len(v), 1))]
    for j, name in enumerate(COLUMNS):
        cols.append(hat(np.full(len(v), np.nan) if name in drop else v[:, j], KNOTS[name]))
    return np.hstack(cols)


def logits_of(v: np.ndarray, beta, drop=()) -> np.ndarray:
    """design(v, drop) @ beta; NaN where the ball path is unknown (column res or dv NaN)."""
    z = design(v, drop) @ np.asarray(beta, dtype=np.float64)
    return np.where(np.isnan(v[:, 0]) | np.isnan(v[:, 1]), np.nan, z)


def jitter(ball: np.ndarray) -> float:
    """B9's label-free ball noise: the median |second difference| (m) of the seen ball, over frames
    where three consecutive frames are seen. NaN with none."""
    d2 = ball[2:] - 2 * ball[1:-1] + ball[:-2]
    m = np.linalg.norm(d2, axis=1)
    m = m[~np.isnan(m)]
    return float(np.median(m)) if len(m) else float("nan")


def ball_measures(ball: np.ndarray) -> tuple[float, float]:
    """(moving jitter m, straight share): over consecutive seen triples where the ball moves (the
    larger step > MOVE_STEP_M), the median |second difference| and the share of second differences
    below 1e-6 m (exactly on a straight constant-speed line). Read only while the ball moves, so
    they hardly depend on which frames are hidden (B39, F310: at every dropout-grid point, moving
    jitter IDSSE <= 0.032, Metrica <= 0.0013, SkillCorner >= 0.061, SC 2019/20 >= 0.105 m; straight
    share SoccerTrack >= 0.49, every other source <= 0.19). (NaN, NaN) with no such triple."""
    d1 = np.linalg.norm(ball[1:] - ball[:-1], axis=1)
    d2 = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=1)
    big = np.maximum(d1[1:], d1[:-1])
    ok = ~np.isnan(d2) & (big > MOVE_STEP_M)
    if not ok.any():
        return float("nan"), float("nan")
    return float(np.median(d2[ok])), float(np.mean(d2[ok] < 1e-6))


# -- coefficients -----------------------------------------------------------------------------------

_COEFS: dict = {}


def load(variant: str = "", path: str = "") -> dict:
    """The fit file of `variant` (VARIANTS; or `path`); refuses a fit for another spec."""
    if variant not in VARIANTS:
        raise ValueError("flight: unknown fit variant {!r} (one of {})".format(variant, sorted(VARIANTS)))
    path = path or coef_file(variant)
    if path not in _COEFS:
        with open(path) as handle:
            coefs = json.load(handle)
        if coefs.get("spec_hash") != spec_hash(variant):
            raise ValueError("{} was fit for another flight spec ({} != {}); rerun "
                             "python -m src.models.cues.flight fit".format(
                                 path, coefs.get("spec_hash"), spec_hash(variant)))
        _COEFS[path] = coefs
    return _COEFS[path]


def fold_fit(coefs: dict, fold: str, match_id: str) -> dict:
    """The fit for this match: "all" (BAR-ATH, and any match the fit never saw), or the lomo / loso
    fit without it (the key rule of switch_cue.fold_fit)."""
    if fold not in ("lomo", "loso"):
        return coefs["all"]
    key = match_id if fold == "lomo" else match_id.split(":", 1)[0]
    return coefs[fold].get(key, coefs["all"])  # a match the fit never saw is held out already


# -- the cue --------------------------------------------------------------------------------------

def effect(match, original, decoder, gain, jitter: float = 0.0, lin: float = 0.0, fit: str = "",
           **opts):
    """Per-frame multipliers on H->A and A->H (module docstring)."""
    if opts:
        raise ValueError("flight: unknown options {}".format(sorted(opts)))
    if not gain:
        return None
    if jitter or lin:
        jit, straight = _measures(original)
        if jitter and not (jit <= float(jitter)):
            return None  # the ball is too noisy to read its path (or never seen moving)
        if lin and not (straight <= float(lin)):
            return None  # the ball is drawn as straight lines, not measured
    variant = "" if fit in ("", 0) else str(fit)
    drop = VARIANTS.get(variant, ())
    entry = fold_fit(load(variant), getattr(decoder, "fold", "all"), match.match_id)
    base = float(entry["base_logit"])
    out = []
    for v in cue_values(match, original):
        z = logits_of(v, entry["beta"], drop)
        with np.errstate(over="ignore", invalid="ignore"):
            mult = np.clip(np.exp(-gain * (z - base)), *COST_RANGE)
        out.append(np.where(np.isnan(z), 1.0, mult))
    return Effect(ha=out[0], ah=out[1])


_MEAS: list = [None, None]


def _measures(original) -> tuple[float, float]:
    """ball_measures() of the ball as delivered; the last match's are kept."""
    if _MEAS[0] is not original.ball:
        _MEAS[:] = [original.ball, ball_measures(np.asarray(original.ball, dtype=np.float64))]
    return _MEAS[1]


# -- fit (external matches only: reads the labels) -----------------------------------------------

def rows(match) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(index (m,), v (m, 5), y (m,)) for one labelled match: rows every 1 / RATE_HZ s on frames in
    a half labelled H or A, v toward the team other than the labelled one LOOKBACK_S earlier, y = 1
    if a true switch to that team is within +/- TOL_S. Rows with an unknown ball path are dropped.
    switch_cue_rows.cue_rows with this cue's values."""
    from src.error_analysis import switch_times, team_ffill
    from src.models.postprocess import play_runs
    from src.official import A, H

    fps = match.fps
    step = max(1, int(round(fps / RATE_HZ)))
    runs = play_runs(match.period)
    team = team_ffill(match.labels, runs)
    run_start = np.zeros(match.n_frames, dtype=np.int64)
    for start, stop in runs:
        run_start[start:stop] = start
    index = np.arange(0, match.n_frames, step)
    index = index[(match.period[index] > 0) & np.isin(match.labels[index], (H, A))]
    current = team[np.maximum(index - int(round(LOOKBACK_S * fps)), run_start[index])]
    keep = np.isin(current, (H, A))
    index, current = index[keep], current[keep]
    v_away, v_home = cue_values(match, match)
    v = np.where((current == H)[:, None], v_away[index], v_home[index])
    other = np.where(current == H, A, H)
    at, new = switch_times(match.labels, runs)
    tol = int(round(TOL_S * fps))
    y = np.zeros(len(index), dtype=np.int64)
    for code in (H, A):
        frames = np.sort(at[new == code])
        near = (np.searchsorted(frames, index + tol, side="right")
                - np.searchsorted(frames, index - tol, side="left")) > 0
        y |= (near & (other == code)).astype(np.int64)
    known = ~(np.isnan(v[:, 0]) | np.isnan(v[:, 1]))
    return index[known], v[known], y[known]


ROWS: dict = {}  # match key -> (v, y); set before the fit pool forks


def _rows_job(job):
    from src import evaluate

    key, opts = job
    match = evaluate.load_match(key, opts)
    _, v, y = rows(match)
    return key, (v, y, jitter(np.asarray(match.ball, dtype=np.float64)))


def _weights(keys: list[str]) -> np.ndarray:
    """Each source its registry weight, each match the same within its source (fit_switch_cue's)."""
    from src import sources as registry

    srcs = sorted({k.split(":", 1)[0] for k in keys})
    share = registry.weights(srcs)
    per = {s: sum(k.startswith(s + ":") for k in keys) for s in srcs}
    return np.concatenate([np.full(len(ROWS[k][1]), share[k.split(":", 1)[0]]
                                   / (per[k.split(":", 1)[0]] * max(len(ROWS[k][1]), 1)))
                           for k in keys])


def _fit_job(job):
    from src.models.switch_cue import fit_binary

    name, keys, drop = job
    keys = sorted(keys)
    x = design(np.concatenate([ROWS[k][0] for k in keys]), drop)
    y = np.concatenate([ROWS[k][1] for k in keys]).astype(np.float64)
    w = _weights(keys)
    beta = fit_binary(x, y, w)
    rate = float(np.sum(w * y) / np.sum(w))
    return name, {"beta": [round(float(b), 6) for b in beta],
                  "base_logit": round(float(np.log(rate / (1.0 - rate))), 6)}


def _auc(z: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted half."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    values, inverse, counts = np.unique(z, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    r = (upper - (counts - 1) / 2.0)[inverse]  # average ranks, 1-based
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _load_rows(sources: list[str], jobs: int) -> dict:
    import multiprocessing

    from src import evaluate
    from src import sources as registry

    registry.check_trainable(sources, "fitting or tuning")  # A19: never on a test-only source
    work = [(key, dict(registry.load_opts(s, train=True))) for s in sources
            for key in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(jobs) as pool:
        return dict(pool.map(_rows_job, work))


def main(argv=None):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import multiprocessing

    from src import sources as registry
    from src.models import learned

    parser = argparse.ArgumentParser(description="fit / check the flight cue (B39)")
    parser.add_argument("what", choices=("fit", "check"))
    parser.add_argument("--sources", default=",".join(registry.names(everything=True)))
    parser.add_argument("--variant", default="", choices=sorted(VARIANTS),
                        help="fit variant: \"\" all columns, nodx without the plain proximity")
    parser.add_argument("--coef", default="", help="file to write (fit) or read (check); default "
                        "the variant's file next to this module")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--json", default="", help="check: also write the readout here")
    args = parser.parse_args(argv)
    started = time.time()
    sources = args.sources.split(",")
    rows_by_key = _load_rows(sources, args.jobs)
    ROWS.update({k: (v[0], v[1]) for k, v in rows_by_key.items()})
    for s in sources:
        keys = [k for k in ROWS if k.startswith(s + ":")]
        y = np.concatenate([ROWS[k][1] for k in keys])
        jit = np.median([rows_by_key[k][2] for k in keys])
        print("rows   {:<16} {:>2} matches {:>8} rows  positive {:.4f}  ball jitter {:.3f} m".format(
            s, len(keys), len(y), y.mean(), jit))
    keys = sorted(ROWS)
    drop = VARIANTS[args.variant]
    path = args.coef or coef_file(args.variant)
    if args.what == "fit":
        work = [("all", keys, drop)] + [("lomo:" + k, [j for j in keys if j != k], drop) for k in keys]
        if len(sources) > 1:
            work += [("loso:" + s, [k for k in keys if not k.startswith(s + ":")], drop)
                     for s in sources]
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            fits = dict(pool.map(_fit_job, work))
        out = {"spec": spec_for(args.variant), "spec_hash": spec_hash(args.variant),
               "fit": {"sources": sources, "weights": registry.weights(sources), "l2": learned.L2,
                       "load": {s: registry.load_opts(s, train=True) for s in sources}},
               "all": fits["all"],
               "lomo": {k[5:]: v for k, v in fits.items() if k.startswith("lomo:")},
               "loso": {k[5:]: v for k, v in fits.items() if k.startswith("loso:")}}
        with open(path, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print("fits   {} ({} lomo, {} loso), base_logit {:.3f}, {:.0f} s -> {}".format(
            len(fits), len(out["lomo"]), len(out["loso"]), out["all"]["base_logit"],
            time.time() - started, os.path.basename(path)))
        return
    coefs = load(args.variant, path)
    report = {}
    for fold in ("lomo", "loso"):
        for s in sources:
            zs, ys = [], []
            for k in (k for k in keys if k.startswith(s + ":")):
                fit = fold_fit(coefs, fold, k)
                zs.append(logits_of(ROWS[k][0], fit["beta"], drop))
                ys.append(ROWS[k][1])
            z, y = np.concatenate(zs), np.concatenate(ys)
            top = z >= np.quantile(z, 0.9)
            low = z <= np.quantile(z, 0.5)
            r = {"auc": round(_auc(z, y), 4), "lift_top10": round(float(y[top].mean() / y.mean()), 3),
                 "rate_low50": round(float(y[low].mean() / y.mean()), 3)}
            report["{}:{}".format(fold, s)] = r
            print("check  {:<4} {:<16} AUC {:.3f}  top-10% lift {:.2f}  bottom-50% rate {:.2f}x".format(
                fold, s, r["auc"], r["lift_top10"], r["rate_low50"]))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
