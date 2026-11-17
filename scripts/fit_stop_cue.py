"""Fit the stop / go cue's two hazards (B37, src/models/cues/stop.py) on the external matches.

    .venv/bin/python scripts/fit_stop_cue.py --jobs 3          # every registered source, minutes
    .venv/bin/python scripts/fit_stop_cue.py --out /tmp/s.json --sources idsse,metrica
    .venv/bin/python scripts/fit_stop_cue.py --timing --jobs 1   # D timing per source (no fit)

Loads every external match with the registry's training load options (src/sources.py; the cue reads
the players and, where seen, the clean ball). Rows at stop.RATE_HZ on frames inside a half:
  stop rows  the label TOL_S earlier (not before the run start) is H or A;
             y = 1 if a true live -> D onset (H/A then D) is within +/- TOL_S
  go rows    the label TOL_S earlier is D; y = 1 if a true D -> live (D then H/A) is within +/- TOL_S
Rows whose features are all NaN are dropped. Weights: each source its registry weight, each match the
same within its source (as scripts/fit_switch_cue.py). Fits: all, lomo (one per match, without it),
loso (one per source, without it), each with both hazards; base_logit = logit of the fit's weighted
positive rate. Newton with learned.L2 (switch_cue.fit_binary).

--timing (no fit): v6 and the cue settings in TIMING, held out by match (lomo) at F30's hidden ball,
one emission per match shared by every setting (as check_dropout_grid.py). Per source: for every true
D run of at least 3 s (not touching X), the nearest predicted D start (end) within 6 s: the share
found and the median |predicted - true| in s (F52's measure), plus spellF1 and macroF1.

External data only. Deterministic: no randomness, floats rounded to 6 decimals. Prints a short
summary. Writes src/models/cues/stop_coefs.json by default.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from src import evaluate  # noqa: E402
from src import sources as registry  # noqa: E402
from src.models import learned, switch_cue  # noqa: E402
from src.models.cues import stop  # noqa: E402
from src.models.postprocess import play_runs  # noqa: E402
from src.official import A, D, H  # noqa: E402

SOURCES = {s: registry.load_opts(s, train=True) for s in registry.names(everything=True)}


def changes(labels: np.ndarray, runs) -> tuple[np.ndarray, np.ndarray]:
    """(onsets, restarts): frames t inside a play run where labels go live -> D (onsets) or
    D -> live (restarts) between t - 1 and t."""
    onsets, restarts = [], []
    live = np.isin(labels, (H, A))
    dead = labels == D
    for start, stop_ in runs:
        lv, dd = live[start:stop_], dead[start:stop_]
        onsets.append(start + 1 + np.flatnonzero(lv[:-1] & dd[1:]))
        restarts.append(start + 1 + np.flatnonzero(dd[:-1] & lv[1:]))
    cat = (lambda parts: np.sort(np.concatenate(parts)).astype(np.int64) if parts
           else np.zeros(0, dtype=np.int64))
    return cat(onsets), cat(restarts)


def _near(index: np.ndarray, frames: np.ndarray, tol: int) -> np.ndarray:
    return (np.searchsorted(frames, index + tol, side="right")
            - np.searchsorted(frames, index - tol, side="left")) > 0


def rows(match) -> dict:
    """{"stop": (x, y), "go": (x, y)} design rows for one labelled match (external only)."""
    index, f_stop, f_go, seen = stop.sample_features(match)
    runs = play_runs(match.period)
    run_start = np.full(match.n_frames, -1, dtype=np.int64)
    for start, stop_ in runs:
        run_start[start:stop_] = start
    tol = int(round(stop.TOL_S * match.fps))
    inside = run_start[index] >= 0
    index, f_stop, f_go, seen = index[inside], f_stop[inside], f_go[inside], seen[inside]
    before = match.labels[np.maximum(index - tol, run_start[index])]
    onsets, restarts = changes(match.labels, runs)
    out = {}
    for name, feats, cols, keep, frames in (
            ("stop", f_stop, stop.STOP_COLUMNS, np.isin(before, (H, A)), onsets),
            ("go", f_go, stop.GO_COLUMNS, before == D, restarts)):
        keep = keep & ~np.isnan(feats).all(axis=1)
        x = stop.design(feats[keep], seen[keep], cols).astype(np.float32)
        y = _near(index[keep], frames, tol).astype(np.int8)
        out[name] = (x, y)
    return out


def _rows(job):
    key, opts = job
    return key, rows(evaluate.load_match(key, opts))


ROWS = {}  # set before the fit pool forks


def weights(keys: list[str], hazard: str) -> np.ndarray:
    sources = sorted({k.split(":", 1)[0] for k in keys})
    share = registry.weights(sources)
    per = {s: sum(k.startswith(s + ":") for k in keys) for s in sources}
    return np.concatenate([np.full(len(ROWS[k][hazard][1]), share[k.split(":", 1)[0]]
                                   / (per[k.split(":", 1)[0]] * max(1, len(ROWS[k][hazard][1]))))
                           for k in keys])


def _fit(job):
    name, keys = job
    keys = sorted(keys)
    out = {}
    for hazard in ("stop", "go"):
        x = np.concatenate([ROWS[k][hazard][0] for k in keys]).astype(np.float64)
        y = np.concatenate([ROWS[k][hazard][1] for k in keys]).astype(np.float64)
        w = weights(keys, hazard)
        beta = switch_cue.fit_binary(x, y, w)
        rate = float(np.sum(w * y) / np.sum(w))
        out[hazard] = stop.fit_entry(beta, np.log(rate / (1.0 - rate)))
    return name, out


# --timing settings: name -> decoder params on top of TIMING_BASE (v6, lomo)
TIMING_BASE = {"fold": "lomo", "track_repair": "1", "switch_cue_gain": 0.5, "restart_rule_gain": 2.0}
TIMING = {"v6": {}, "s05g0": {"cues": "stop:0.5/go=0"}, "s05g1": {"cues": "stop:0.5/go=1"},
          "s1g0": {"cues": "stop:1/go=0"}, "s1g1": {"cues": "stop:1/go=1"},
          "r0": {"restart_rule_gain": 0.0}, "r0s05g1": {"restart_rule_gain": 0.0, "cues": "stop:0.5/go=1"}}
MIN_RUN_S, SEARCH_S = 3.0, 6.0


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def boundary_errors(match, pred: np.ndarray) -> dict:
    """{"onset": [...], "end": [...]}: per long true D run, |nearest predicted boundary - true| in s,
    or NaN when none is within SEARCH_S (as scripts/check_dead_timing.py, unsigned)."""
    from src.official import X

    live = match.period > 0
    t_start, t_stop = _runs((match.labels == D) & live)
    p_start, p_stop = _runs((pred == D) & live)
    out = {"onset": [], "end": []}
    for a, b in zip(t_start, t_stop):
        if b - a < MIN_RUN_S * match.fps:
            continue
        if (a > 0 and match.labels[a - 1] == X) or (b < match.n_frames and match.labels[b] == X):
            continue
        for t, cand, kind in ((a, p_start, "onset"), (b, p_stop, "end")):
            d = np.abs(cand - t)
            out[kind].append(float(d.min()) / match.fps if len(d) and d.min() <= SEARCH_S * match.fps
                             else np.nan)
    return out


def _timing(job):
    from src.error_analysis import HIDDEN
    from src.models import build

    key, opts = job
    match = evaluate.load_match(key, dict(opts, **HIDDEN))
    base = build("bigger", **TIMING_BASE)
    emission = base.emission(match)
    out = {}
    for name, extra in TIMING.items():
        model = build("bigger", **dict(TIMING_BASE, **extra))
        pred = model.decode(match, model.combine(model.read(match), emission))
        out[name] = dict(evaluate.score(match.labels, pred), **boundary_errors(match, pred))
    return key, out


def timing(sources: list[str], jobs: int) -> None:
    work = [(k, registry.load_opts(s)) for s in sources for k in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(jobs) as pool:
        res = dict(pool.map(_timing, work))
    print("D timing, lomo, F30 hidden ball: found share / median |error| s (onset; end), spellF1, macroF1")
    for s in sources:
        keys = [k for k in res if k.startswith(s + ":")]
        for name in TIMING:
            cells = []
            for kind in ("onset", "end"):
                v = np.array([e for k in keys for e in res[k][name][kind]])
                found = v[~np.isnan(v)]
                cells.append("{:.2f} / {:.2f}".format(len(found) / max(1, len(v)),
                                                      np.median(found) if len(found) else np.nan))
            print("{:<15} {:<8} onset {}  end {}  spell {:.4f}  macro {:.4f}".format(
                s, name, cells[0], cells[1], np.mean([res[k][name]["spellF1"] for k in keys]),
                np.mean([res[k][name]["macroF1"] for k in keys])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(SOURCES), help="registered sources")
    parser.add_argument("--out", default=stop.COEF_FILE)
    parser.add_argument("--jobs", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument("--timing", action="store_true", help="report D timing (no fit)")
    args = parser.parse_args(argv)
    if args.timing:
        timing(args.sources.split(","), args.jobs)
        return

    started = time.time()
    sources = args.sources.split(",")
    registry.check_trainable(sources, "fitting or tuning")
    jobs = [(key, dict(SOURCES[s])) for s in sources for key in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        found = dict(pool.map(_rows, jobs))
    for s in sources:
        keys = [k for k in found if k.startswith(s + ":")]
        parts = ["{} {:>7} rows pos {:.4f}".format(h, sum(len(found[k][h][1]) for k in keys),
                                                     np.concatenate([found[k][h][1] for k in keys]).mean())
                 for h in ("stop", "go")]
        print("rows       {:<15} {:>2} matches  {}".format(s, len(keys), "  ".join(parts)))
    print("rows       {} matches, {:.0f} s".format(len(found), time.time() - started))

    ROWS.update(found)
    keys = list(found)
    work = [("all", keys)] + [("lomo:" + k, [j for j in keys if j != k]) for k in keys]
    if len(sources) > 1:
        work += [("loso:" + s, [k for k in keys if not k.startswith(s + ":")]) for s in sources]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        fits = dict(pool.map(_fit, work))
    out = {"spec": stop.SPEC, "spec_hash": stop.SPEC_HASH,
           "fit": {"sources": sources, "load": {s: SOURCES[s] for s in sources},
                   "weights": registry.weights(sources), "l2": learned.L2},
           "all": fits["all"],
           "lomo": {k[5:]: v for k, v in fits.items() if k.startswith("lomo:")},
           "loso": {k[5:]: v for k, v in fits.items() if k.startswith("loso:")}}
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("fits       {} ({} lomo, {} loso), base_logit stop {:.3f} go {:.3f}, {:.0f} s".format(
        len(fits), len(out["lomo"]), len(out["loso"]), out["all"]["stop"]["base_logit"],
        out["all"]["go"]["base_logit"], time.time() - started))
    print("wrote      {}".format(os.path.relpath(args.out, evaluate.ROOT)))


if __name__ == "__main__":
    main()
