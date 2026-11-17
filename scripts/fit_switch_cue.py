"""Fit the switch-timing cue's hazard (local L2, src/models/switch_cue.py) on the external matches.

    .venv/bin/python scripts/fit_switch_cue.py --out /tmp/cue.json --jobs 3     # every registered
                                                                               # source (B36), minutes
    .venv/bin/python scripts/check_switch_cue.py --coef /tmp/cue.json --jobs 4  # the go / no-go

Loads every external match clean (the cue is ball-free, so ball dropout doesn't matter), with the
registry's training load options (src/sources.py), every registered source by default (B36; F203
fit idsse, skillcorner, metrica: --sources idsse,skillcorner,metrica). Rows every 1 / CUE_RATE_HZ s on frames inside
a half labelled H or A:
  c   the current team: error_analysis.team_ffill(labels) 1 s earlier (not before the run start)
  u   the cue toward the other team (switch_cue.cue_values: u_toward_away if c is H, else u_toward_home)
  y   1 if error_analysis.switch_times has a switch to the other team within +/- CUE_TOL_S
Rows whose u is all NaN are dropped (the rows are switch_cue_rows.cue_rows). Weights: each source its
registry weight (src/sources.py), each match the same within its source (as learned.fit). Fits: all, lomo (one per match, without it), loso (one per source, without
it); base_logit = logit of the fit's weighted positive rate.

External data only. Deterministic: no randomness, floats rounded to 6 decimals. Prints a short
summary. The default --out is src/models/switch_cue_coefs.json (not committed while the cue is off).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time

# one BLAS thread per worker: the fits run in parallel, and threads would only contend
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from src import evaluate  # noqa: E402
from src import sources as registry  # noqa: E402
from src.models import learned, switch_cue, switch_cue_rows  # noqa: E402

# the registry's training load options (src/sources.py; as scripts/fit_emission.py); the cue is
# ball-free, so only the label options (SkillCorner's onsets) matter
SOURCES = {s: registry.load_opts(s, train=True) for s in registry.names(everything=True)}
LOOKBACK_S = switch_cue_rows.LOOKBACK_S


def cue_rows(match) -> tuple[np.ndarray, np.ndarray]:
    """(u (m, 6), y (m,)) for one labelled match (switch_cue_rows.cue_rows without the index)."""
    _, u, y = switch_cue_rows.cue_rows(match)
    return u, y


def _rows(job):
    key, opts = job
    return key, cue_rows(evaluate.load_match(key, opts))


ROWS = {}  # set before the fit pool forks, so workers share it instead of pickling it


def weights(keys: list[str]) -> np.ndarray:
    """Each source its registry weight (renormalised over the sources in keys; idsse, skillcorner
    and metrica are 1 each, so the three-source fit is F203's), each match the same within its
    source (as learned.fit)."""
    sources = sorted({k.split(":", 1)[0] for k in keys})
    share = registry.weights(sources)
    per = {s: sum(k.startswith(s + ":") for k in keys) for s in sources}
    return np.concatenate([np.full(len(ROWS[k][1]), share[k.split(":", 1)[0]]
                                   / (per[k.split(":", 1)[0]] * len(ROWS[k][1]))) for k in keys])


def _fit(job):
    name, keys = job
    keys = sorted(keys)
    x = switch_cue.design(np.concatenate([ROWS[k][0] for k in keys]))
    y = np.concatenate([ROWS[k][1] for k in keys]).astype(float)
    w = weights(keys)
    beta = switch_cue.fit_binary(x, y, w)
    rate = float(np.sum(w * y) / np.sum(w))
    return name, switch_cue.fit_entry(beta, np.log(rate / (1.0 - rate)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(SOURCES), help="registered sources")
    parser.add_argument("--out", default=switch_cue.COEF_FILE)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    started = time.time()
    sources = args.sources.split(",")
    registry.check_trainable(sources, "fitting or tuning")  # A19: never on a test-only source
    jobs = [(key, dict(SOURCES[s])) for s in sources
            for key in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        rows = dict(pool.map(_rows, jobs))
    for s in sources:
        keys = [k for k in rows if k.startswith(s + ":")]
        y = np.concatenate([rows[k][1] for k in keys])
        print("rows       {:<12} {:>2} matches, {:>7} rows, positive {:.4f}".format(
            s, len(keys), len(y), y.mean()))
    print("rows       {} matches, {} rows, {:.0f} s".format(
        len(rows), sum(len(r[1]) for r in rows.values()), time.time() - started))

    ROWS.update(rows)
    keys = list(rows)
    work = [("all", keys)] + [("lomo:" + k, [j for j in keys if j != k]) for k in keys]
    if len(sources) > 1:
        work += [("loso:" + s, [k for k in keys if not k.startswith(s + ":")]) for s in sources]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        fits = dict(pool.map(_fit, work))
    out = {"spec": switch_cue.SPEC, "spec_hash": switch_cue.SPEC_HASH,
           "fit": {"sources": sources, "load": {s: SOURCES[s] for s in sources},
                   "weights": registry.weights(sources),
                   "lookback_s": LOOKBACK_S, "l2": learned.L2},
           "all": fits["all"],
           "lomo": {k[5:]: v for k, v in fits.items() if k.startswith("lomo:")},
           "loso": {k[5:]: v for k, v in fits.items() if k.startswith("loso:")}}
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("fits       {} ({} lomo, {} loso), {} weights, base_logit {:.3f}, {:.0f} s".format(
        len(fits), len(out["lomo"]), len(out["loso"]), len(out["all"]["beta"]),
        out["all"]["base_logit"], time.time() - started))
    print("wrote      {}".format(os.path.relpath(args.out, evaluate.ROOT)))


if __name__ == "__main__":
    main()
