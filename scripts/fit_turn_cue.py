"""Fit the `turn` cue's hazards (wave 5, B38, src/models/cues/turn.py) on the external matches, and
check them.

    .venv/bin/python scripts/fit_turn_cue.py --knots --jobs 3       # pooled quantiles of the new
                                                                    # columns (how KNOTS were set)
    .venv/bin/python scripts/fit_turn_cue.py --jobs 3               # fit both sets, write the file,
                                                                    # print held-out AUC / lift
    .venv/bin/python scripts/fit_turn_cue.py --timing --cand o05=cues=turn:0.5 \
        --cand i05=switch_cue_gain=0,cues=turn:0.5                  # F61's switch readout, decoded

Fit (as scripts/fit_switch_cue.py, B36): every registered source loaded clean with the registry's
training options (the cue is ball-free); rows every 1 / switch_cue.CUE_RATE_HZ s on frames labelled
H or A; the current team is the labelled team switch_cue_rows.LOOKBACK_S earlier; u is the cue
toward the other team; y = 1 if a true switch to the other team is within +/- CUE_TOL_S. Two sets
(turn.SETS): "new" (the 8 new columns) and "full" (switch_cue's 6 v1 columns, then the 8). Rows with
the set's u all NaN are dropped. Weights: each source its registry weight, each match the same
within its source. Fits: all, lomo (without each match), loso (without each source).

Check: per source, pooled over its matches, the lomo hazard's AUC and top-5% lift for each set and
for v6's own cue (src/models/switch_cue_coefs.json, lomo) on the same rows (where all are known).

--timing: decode each external match (the configs must differ only in decode-time params) BAR-ATH-like (the hidden ball of `scripts/models.py compare`,
evaluate.HIDDEN) with `bigger` + v6's params (fold=lomo) + the --param overrides, and print per
source F61's switch precision / recall at +/- 1 s, predicted switches per live minute, and the
median signed switch-time error (s, prediction minus truth; a true switch's nearest predicted switch
to the same team within +/- 3 s, D carried over as in F61) and its median |error|, for v6 and each
--cand (v6's params plus the candidate's).

External data only. Deterministic. Prints short tables.
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
from src.models import learned, switch_cue, switch_cue_rows  # noqa: E402
from src.models.cues import turn  # noqa: E402

SOURCES = {s: registry.load_opts(s, train=True) for s in registry.names(everything=True)}
TOP = 0.05
N_V1 = len(switch_cue.COLUMNS)
V6 = {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5, "fold": "lomo"}
NEAR_S = 3.0  # --timing: a true switch's nearest predicted switch within this counts for the error


def cue_rows(match):
    """(index (m,), u_full (m, 14), y (m,)): switch_cue_rows.cue_rows' rows (the same current team,
    tolerance and labels), u from turn.cue_values(match, "full"); rows where all of u is NaN dropped."""
    from src.error_analysis import switch_times, team_ffill
    from src.models.postprocess import play_runs
    from src.official import A, H

    fps = match.fps
    step = max(1, int(round(fps / switch_cue.CUE_RATE_HZ)))
    runs = play_runs(match.period)
    team = team_ffill(match.labels, runs)
    run_start = np.zeros(match.n_frames, dtype=np.int64)
    for start, stop in runs:
        run_start[start:stop] = start
    index = np.arange(0, match.n_frames, step)
    index = index[(match.period[index] > 0) & np.isin(match.labels[index], (H, A))]
    current = team[np.maximum(index - int(round(switch_cue_rows.LOOKBACK_S * fps)), run_start[index])]
    keep = np.isin(current, (H, A))
    index, current = index[keep], current[keep]
    u_away, u_home = turn.cue_values(match, "full")
    u = np.where((current == H)[:, None], u_away[index], u_home[index])
    other = np.where(current == H, A, H)
    at, new = switch_times(match.labels, runs)
    tol = int(round(switch_cue.CUE_TOL_S * fps))
    y = np.zeros(len(index), dtype=np.int64)
    for code in (H, A):
        frames = np.sort(at[new == code])
        near = (np.searchsorted(frames, index + tol, side="right")
                - np.searchsorted(frames, index - tol, side="left")) > 0
        y |= (near & (other == code)).astype(np.int64)
    known = ~np.isnan(u).all(axis=1)
    return index[known], u[known], y[known]


def set_u(u_full: np.ndarray, which: str) -> np.ndarray:
    return u_full if which == "full" else u_full[:, N_V1:]


def _rows(job):
    key, opts = job
    _, u, y = cue_rows(evaluate.load_match(key, opts))
    return key, (u, y)


ROWS = {}  # set before the fit pool forks


def weights(keys, masks) -> np.ndarray:
    sources = sorted({k.split(":", 1)[0] for k in keys})
    share = registry.weights(sources)
    per = {s: sum(k.startswith(s + ":") for k in keys) for s in sources}
    return np.concatenate([np.full(int(masks[k].sum()), share[k.split(":", 1)[0]]
                                   / (per[k.split(":", 1)[0]] * max(1, int(masks[k].sum()))))
                           for k in keys])


def known(which: str) -> dict:
    return {k: ~np.isnan(set_u(u, which)).all(axis=1) for k, (u, _) in ROWS.items()}


def _fit(job):
    which, name, keys = job
    keys = sorted(keys)
    masks = known(which)
    x = turn.design(np.concatenate([set_u(ROWS[k][0], which)[masks[k]] for k in keys]), which)
    y = np.concatenate([ROWS[k][1][masks[k]] for k in keys]).astype(float)
    w = weights(keys, masks)
    beta = switch_cue.fit_binary(x, y, w)
    rate = float(np.sum(w * y) / np.sum(w))
    return which, name, switch_cue.fit_entry(beta, np.log(rate / (1.0 - rate)))


def auc(score, y):
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def lift(score, y, top=TOP):
    k = max(1, int(round(top * len(y))))
    order = np.argsort(-score, kind="stable")[:k]
    return float(y[order].mean() / y.mean()) if y.mean() > 0 else float("nan")


def check(out: dict, sources: list[str]) -> dict:
    """Per source: lomo AUC / top-5% lift of v6's cue, "new" and "full", on rows all three know."""
    v1 = switch_cue.load_cue()
    table = {}
    for s in sources:
        zs = {"v6cue": [], "new": [], "full": []}
        ys = []
        for k in sorted(ROWS):
            if not k.startswith(s + ":"):
                continue
            u, y = ROWS[k]
            ok = ~np.isnan(u[:, :N_V1]).all(axis=1) & ~np.isnan(u[:, N_V1:]).all(axis=1)
            zs["v6cue"].append(switch_cue.logits_of(u[ok, :N_V1], switch_cue.fold_fit(v1, "lomo", k)["beta"]))
            for which in turn.SETS:
                fit = switch_cue.fold_fit(out["sets"][which], "lomo", k)
                zs[which].append(turn.logits_of(set_u(u[ok], which), fit["beta"], which))
            ys.append(y[ok])
        y = np.concatenate(ys)
        table[s] = {name: {"auc": round(auc(np.concatenate(z), y), 4),
                           "lift": round(lift(np.concatenate(z), y), 3)} for name, z in zs.items()}
        table[s]["rows"] = int(len(y))
        print("check      {:<15} rows {:>6}  AUC / lift5%: v6 cue {:.3f} / {:.2f}   new {:.3f} / {:.2f}"
              "   full {:.3f} / {:.2f}".format(s, len(y), *(table[s][n][m] for n in ("v6cue", "new", "full")
                                                          for m in ("auc", "lift"))))
    return table


def print_knots():
    u = np.concatenate([ROWS[k][0][:, N_V1:] for k in sorted(ROWS)])
    for j, (name, window) in enumerate(turn.NEW_COLUMNS):
        q = np.nanpercentile(u[:, j], (2, 10, 30, 50, 70, 90, 98))
        print("knots      {:<10} {:>4}  {}".format(name, window, " ".join("{:+.2f}".format(v) for v in q)))


# -- timing readout --------------------------------------------------------------------------------

def timing(gt, pred, fps, runs) -> dict:
    """F61's switch precision / recall (+/- 1 s) plus every true switch's signed offset to its
    nearest predicted switch to the same team within +/- NEAR_S (prediction minus truth, s)."""
    from src.error_analysis import match_switches, switch_times

    gt_at, gt_team = switch_times(gt, runs)
    pr_at, pr_team = switch_times(pred, runs)
    hits = match_switches(gt_at, gt_team, pr_at, pr_team, round(1.0 * fps))
    offsets = []
    near = round(NEAR_S * fps)
    for t, team in zip(gt_at, gt_team):
        cand = pr_at[(pr_team == team) & (np.abs(pr_at - t) <= near)]
        if len(cand):
            offsets.append((cand[np.argmin(np.abs(cand - t))] - t) / fps)
    return {"hits": int(hits), "pred": int(len(pr_at)), "true": int(len(gt_at)), "offsets": offsets}


def _timing_job(job):
    key, opts, params_list = job
    from src.error_analysis import HIDDEN
    from src.models import build
    from src.official import pm

    match = evaluate.load_match(key, dict(opts, **HIDDEN))
    out, scores = [], None
    for params in params_list:  # the configs differ only at decode time: one set of scores
        model = build("bigger", **params)
        scores = model.scores(match) if scores is None else scores
        pred = model.decode(match, scores)
        out.append(timing(match.labels, pred, match.fps, pm.valid_runs(match.labels)))
    return key, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(registry.names()), help="registered sources")
    parser.add_argument("--out", default=turn.TURN_FILE)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--knots", action="store_true", help="only print the new columns' quantiles")
    parser.add_argument("--timing", action="store_true", help="the decoded switch readout")
    parser.add_argument("--cand", action="append", default=[], metavar="NAME=K=V,K=V",
                        help="--timing: a candidate, params on top of v6's (fold=lomo)")
    parser.add_argument("--json", help="--timing / fit: also write the summary here")
    args = parser.parse_args(argv)
    started = time.time()
    sources = args.sources.split(",")
    registry.check_trainable(sources, "fitting or tuning")
    if args.timing:
        return run_timing(args, sources)

    jobs = [(key, dict(SOURCES[s])) for s in sources for key in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        rows = dict(pool.map(_rows, jobs))
    ROWS.update(rows)
    for s in sources:
        keys = [k for k in rows if k.startswith(s + ":")]
        y = np.concatenate([rows[k][1] for k in keys])
        print("rows       {:<15} {:>2} matches, {:>7} rows, positive {:.4f}".format(
            s, len(keys), len(y), y.mean()))
    print("rows       {:.0f} s".format(time.time() - started))
    if args.knots:
        print_knots()
        return

    keys = list(rows)
    work = []
    for which in turn.SETS:
        work += [(which, "all", keys)] + [(which, "lomo:" + k, [j for j in keys if j != k]) for k in keys]
        work += [(which, "loso:" + s, [k for k in keys if not k.startswith(s + ":")]) for s in sources]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        fits = pool.map(_fit, work)
    out = {"spec": turn.SPEC, "spec_hash": turn.SPEC_HASH,
           "fit": {"sources": sources, "load": {s: SOURCES[s] for s in sources},
                   "weights": registry.weights(sources), "lookback_s": switch_cue_rows.LOOKBACK_S,
                   "l2": learned.L2},
           "sets": {}}
    for which in turn.SETS:
        mine = {name: fit for w, name, fit in fits if w == which}
        out["sets"][which] = {"all": mine["all"],
                              "lomo": {k[5:]: v for k, v in mine.items() if k.startswith("lomo:")},
                              "loso": {k[5:]: v for k, v in mine.items() if k.startswith("loso:")}}
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("fits       {} per set, base_logit new {:.3f} full {:.3f}, {:.0f} s".format(
        len(fits) // 2, out["sets"]["new"]["all"]["base_logit"], out["sets"]["full"]["all"]["base_logit"],
        time.time() - started))
    print("wrote      {}".format(os.path.relpath(args.out, evaluate.ROOT)))
    table = check(out, sources)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"check": table, "fold": "lomo", "top": TOP}, handle, indent=1, sort_keys=True)


def run_timing(args, sources):
    from src.run import parse_params

    names, configs = ["v6"], [dict(V6)]
    for spec in args.cand:
        name, _, rest = spec.partition("=")
        names.append(name)
        configs.append(dict(V6, **parse_params(rest.split(","))))
    jobs = [(key, dict(registry.load_opts(s)), configs) for s in sources
            for key in evaluate.parse_matches(s + ":all")]
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        res = dict(pool.map(_timing_job, jobs, chunksize=1))
    summary = {}
    for s in sources:
        keys = sorted(k for k in res if k.startswith(s + ":"))
        summary[s] = {}
        for i, name in enumerate(names):
            parts = [res[k][i] for k in keys]
            prec = np.nanmean([p["hits"] / p["pred"] if p["pred"] else np.nan for p in parts])
            rec = np.nanmean([p["hits"] / p["true"] if p["true"] else np.nan for p in parts])
            off = np.concatenate([p["offsets"] for p in parts]) if parts else np.zeros(0)
            row = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                   "pred": int(sum(p["pred"] for p in parts)), "true": int(sum(p["true"] for p in parts)),
                   "median_offset_s": round(float(np.median(off)), 3) if len(off) else None,
                   "median_abs_s": round(float(np.median(np.abs(off))), 3) if len(off) else None,
                   "paired": int(len(off))}
            summary[s][name] = row
            print("timing     {:<15} {:<5} P {:.3f} R {:.3f}  pred/true {:>5}/{:<5} median offset {:+.2f} s"
                  "  median |err| {:.2f} s".format(s, name, row["precision"], row["recall"], row["pred"],
                                                  row["true"], row["median_offset_s"] or 0.0,
                                                  row["median_abs_s"] or 0.0))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"timing": summary, "params": {n: c for n, c in zip(names, configs)},
                       "hidden": True, "near_s": NEAR_S}, handle, indent=1, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
