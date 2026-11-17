"""B29: fit the hidden-ball imputer (src/ball_impute.py) on external matches and score it held out.

    .venv/bin/python scripts/fit_ball_impute.py                      # fit + write + held-out tables
    .venv/bin/python scripts/fit_ball_impute.py --out /tmp/x.json    # a scratch copy
    .venv/bin/python scripts/fit_ball_impute.py --no-write           # tables only

Training rows: on every external match, under dropout grid points (src/dropout.py; --mix K draws per
match, seeded by its key as in fit_learned.py; 0 = all 15), the in-play frames whose ball the grid
hides but the source tracked, every --stride-s. Target: the tracked ball. Weighted so each source
counts the same and each match the same within its source. Fits: "all", one per held-out match
("lomo") and one per held-out source ("loso"), written to src/ball_impute_coefs.json.

Scored held out at 5 Hz on the frames each of the 15 grid points hides (mean over points, then over
matches within a source):
  error (m) of the estimate vs the tracked ball (frames where the source has one): the last seen
      ball, the straight line across the gap ("line"), the imputer (lomo, loso);
  nearest team: share of frames where the team nearest the estimate is the team nearest the tracked
      ball; for "last seen" the team nearest the ball when it was last seen (as F96's baseline);
  possession: share of live frames (labels H/A) where that team is the team with the ball (unknown
      calls count half), comparable with F96's shape-only classifier and last-seen ball.
Labels are used for scoring only. External data only; prints tables. Deterministic.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import ball_impute as bi  # noqa: E402
from src import dropout, evaluate  # noqa: E402
from src import sources as sources_registry  # noqa: E402
from src.official import A, H  # noqa: E402

SOURCES = {"idsse": {}, "skillcorner": {"cuts": "ungraded"}, "metrica": {}}
EVAL_HZ = 5.0
METHODS = ("last", "line", "lomo", "loso", "true")  # true: the tracked ball itself (a ceiling)
GAP_BINS = ((0.0, 1.0), (1.0, 4.0), (4.0, np.inf))  # gap length, seconds


def _points(key: str, mix: int) -> list[int]:
    return list(range(len(dropout.GRID))) if not mix else dropout.draws(key, mix)


def _train_job(task):
    key, opts, mix, stride_s = task
    clean = evaluate.load_match(key, opts)
    seen = ~np.isnan(clean.ball[:, 0])
    step = max(1, int(round(stride_s * clean.fps)))
    grid_rows = np.zeros(clean.n_frames, bool)
    grid_rows[::step] = True
    points = _points(key, mix)
    parts, n_rows = [], 0
    for i in points:
        deg = evaluate.load_match(key, dict(opts, **dropout.GRID[i]))
        rows = np.flatnonzero(grid_rows & (deg.period > 0) & np.isnan(deg.ball[:, 0]) & seen)
        n_rows += len(rows)
        ctx = bi.context(deg)
        for s in range(0, len(rows), bi.CHUNK):
            part = rows[s:s + bi.CHUNK]
            start, columns = bi.design(deg, part, ctx)
            parts.append((start, columns, clean.ball[part]))
    # each draw's rows weigh the same in total; the match weighs 1 (scaled later per source)
    pieces = []
    for start, columns, target in parts:
        pieces.append(bi.gram(start, columns, target, np.full(len(start), 1.0 / max(1, n_rows))))
    G = sum(p[0] for p in pieces) if pieces else np.zeros((bi.N_WEIGHTS, bi.N_WEIGHTS))
    g = sum(p[1] for p in pieces) if pieces else np.zeros(bi.N_WEIGHTS)
    return key, (G, g, 1.0 if n_rows else 0.0), n_rows


def _eval_job(task):
    key, opts, betas = task
    clean = evaluate.load_match(key, opts)
    step = max(1, int(round(clean.fps / EVAL_HZ)))
    ev = np.zeros(clean.n_frames, bool)
    ev[::step] = True
    seen = ~np.isnan(clean.ball[:, 0])
    live = np.isin(clean.labels, (H, A)) & (clean.period > 0)
    out = []
    for point in dropout.GRID:
        deg = evaluate.load_match(key, dict(opts, **point))
        rows = np.flatnonzero(ev & (deg.period > 0) & np.isnan(deg.ball[:, 0]))
        ctx = bi.context(deg)
        last, nxt = ctx[:2]
        start, columns = bi.design(deg, rows, ctx)
        est = {"line": start, "true": clean.ball[rows]}
        for name in ("lomo", "loso"):
            if betas.get(name) is not None:
                est[name] = bi.estimate(start, columns, betas[name], deg.pitch)
        lo = last[rows]
        est["last"] = np.where((lo >= 0)[:, None], deg.ball[np.maximum(lo, 0)], np.nan)
        p = deg.players[rows]
        calls = {m: bi.nearest_team_call(p, deg.slot_team, e) for m, e in est.items()}
        # last seen, as F96: the team nearest the ball on the frame it was last seen
        calls["last"] = np.where(lo >= 0, bi.nearest_team_call(
            deg.players[np.maximum(lo, 0)], deg.slot_team, deg.ball[np.maximum(lo, 0)]), np.nan)
        truth_team = bi.nearest_team_call(p, deg.slot_team, clean.ball[rows])
        has_truth = seen[rows]
        lo_i, hi_i = np.where(lo >= 0, lo, rows), np.where(nxt[rows] >= 0, nxt[rows], rows)
        gap_s = (hi_i - lo_i) / clean.fps
        res = {"n": len(rows), "gap_s": gap_s[has_truth].astype(np.float32),
               "live": live[rows], "y": (clean.labels[rows] == H)}
        for m in est:
            e = np.hypot(*(est[m][has_truth] - clean.ball[rows][has_truth]).T)
            res["err_" + m] = e.astype(np.float32)
            t = truth_team[has_truth]
            c = calls[m][has_truth]
            ok = ~np.isnan(t)
            res["team_" + m] = np.where(np.isnan(c[ok]), 0.5, c[ok] == t[ok]).astype(np.float32)
            c = calls[m][res["live"]]
            yv = res["y"][res["live"]]
            res["poss_" + m] = np.where(np.isnan(c), 0.5, c == yv).astype(np.float32)
        out.append(res)
    return key, out


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--mix", type=int, default=0,
                        help="grid points per training match (seeded draws); 0 = all 15")
    parser.add_argument("--stride-s", type=float, default=0.2)
    parser.add_argument("--load", action="append", default=[], metavar="SOURCE:KEY=VALUE")
    parser.add_argument("--out", default=bi.COEF_FILE)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    started = time.time()
    sources = args.sources.split(",")
    sources_registry.check_trainable(sources, "fitting or tuning")  # A19: never on a test-only source
    load = {s: dict(SOURCES[s]) for s in sources}
    for spec in args.load:
        source, pair = spec.split(":", 1)
        load[source].update(evaluate.parse_params([pair]))
    keys = [k for s in sources for k in evaluate.parse_matches(s + ":all")]
    tasks = [(k, load[k.split(":", 1)[0]], args.mix, args.stride_s) for k in keys]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        trained = pool.map(_train_job, tasks)
    pieces = {k: p for k, p, _ in trained}
    n_rows = {k: n for k, _, n in trained}

    def fit(subset):
        subset = [k for k in subset if n_rows[k]]
        w = bi.weights(subset)
        return bi.solve([(w[k] * pieces[k][0], w[k] * pieces[k][1], w[k]) for k in subset])

    betas = {"all": fit(keys), "lomo": {k: fit([j for j in keys if j != k]) for k in keys}}
    betas["loso"] = ({s: fit([j for j in keys if not j.startswith(s + ":")]) for s in sources}
                     if len(sources) > 1 else {})
    print("rows       {} matches, {} rows, {} weights, {:.0f} s".format(
        len(keys), sum(n_rows.values()), bi.N_WEIGHTS, time.time() - started))
    if not args.no_write:
        rnd = lambda b: [round(float(v), 6) for v in b]  # noqa: E731
        out = {"spec_hash": bi.SPEC_HASH, "spec": bi.SPEC,
               "fit": {"sources": sources, "load": load, "stride_s": args.stride_s,
                       "points": ("all" if not args.mix else "mix {}".format(args.mix)),
                       "grid": list(dropout.GRID), "rows": sum(n_rows.values())},
               "all": rnd(betas["all"]),
               "lomo": {k: rnd(v) for k, v in betas["lomo"].items()},
               "loso": {k: rnd(v) for k, v in betas["loso"].items()}}
        with open(args.out, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print("wrote      {}".format(os.path.relpath(args.out, evaluate.ROOT)))
    if args.no_eval:
        return
    # the evaluation uses the rounded coefficients, as the written file does
    rounded = lambda b: np.round(np.asarray(b, float), 6)  # noqa: E731
    tasks = [(k, load[k.split(":", 1)[0]],
              {"lomo": rounded(betas["lomo"][k]),
               "loso": rounded(betas["loso"][k.split(":", 1)[0]]) if betas["loso"] else None})
             for k in keys]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = dict(pool.map(_eval_job, tasks))
    report(results, sources, keys)
    print("total      {:.0f} s".format(time.time() - started))


def _per_source(results, keys, sources, fn):
    """{source: mean over matches of the mean over grid points of fn(point result)}."""
    out = {}
    for s in sources:
        per_match = []
        for k in keys:
            if k.startswith(s + ":"):
                per_match.append(_mean([fn(r) for r in results[k]]))
        out[s] = _mean(per_match)
    return out


def report(results, sources, keys):
    methods = [m for m in METHODS if "err_" + m in results[keys[0]][0]]
    print("\nheld out, frames the grid hides at {:g} Hz (mean over the 15 points, then matches); "
          "per source {}".format(EVAL_HZ, " / ".join(sources)))
    rows = [
        ("mean error (m)", "err_", lambda a: float(a.mean()) if len(a) else None),
        ("median error (m)", "err_", lambda a: float(np.median(a)) if len(a) else None),
        ("share within 5 m", "err_", lambda a: float((a <= 5).mean()) if len(a) else None),
        ("nearest team = true", "team_", lambda a: float(a.mean()) if len(a) else None),
        ("possession acc (live)", "poss_", lambda a: float(a.mean()) if len(a) else None),
    ]
    print("{:<24}".format("") + "".join("{:>24}".format(m) for m in methods))
    for title, prefix, fn in rows:
        cells = []
        for m in methods:
            per = _per_source(results, keys, sources, lambda r: fn(r[prefix + m]))
            cells.append(" / ".join("{:.3f}".format(per[s]) if per[s] < 10 else "{:.1f}".format(
                per[s]) for s in sources))
        print("{:<24}".format(title) + "".join("{:>24}".format(c) for c in cells))
    print("\nby gap length (lomo vs line vs last seen), mean error (m) / nearest team; "
          "mean over sources")
    for lo, hi in GAP_BINS:
        cells = []
        for m in ("last", "line", "lomo"):
            if m not in methods:
                continue

            def err(r, m=m):
                sel = (r["gap_s"] >= lo) & (r["gap_s"] < hi)
                return float(r["err_" + m][sel].mean()) if sel.any() else None

            def team(r, m=m):
                sel = (r["gap_s"] >= lo) & (r["gap_s"] < hi)
                return float(r["team_" + m][sel].mean()) if sel.any() else None

            e = _mean(list(_per_source(results, keys, sources, err).values()))
            t = _mean(list(_per_source(results, keys, sources, team).values()))
            cells.append("{} {:5.1f} m {:.3f}".format(m, e, t))
        share = _mean([_mean([float(((r["gap_s"] >= lo) & (r["gap_s"] < hi)).mean())
                              for r in results[k] if len(r["gap_s"])]) for k in keys])
        print("  gap {:>4g}-{:<4g}s ({:4.0%})  ".format(lo, hi, share) + "   ".join(cells))
    print("\nper grid point, mean over sources: error (m) last / line / lomo, possession acc "
          "last / lomo")
    for i, point in enumerate(dropout.GRID):
        e = {m: _mean([_mean([float(results[k][i]["err_" + m].mean())
                              for k in keys if k.startswith(s + ":")
                              and len(results[k][i]["err_" + m])]) for s in sources])
             for m in ("last", "line", "lomo")}
        pa = {m: _mean([_mean([float(results[k][i]["poss_" + m].mean())
                               for k in keys if k.startswith(s + ":")
                               and len(results[k][i]["poss_" + m])]) for s in sources])
              for m in ("last", "lomo")}
        print("  {:<16} {:5.1f} / {:5.1f} / {:5.1f}   {:.3f} / {:.3f}".format(
            dropout.name(point), e["last"], e["line"], e["lomo"], pa["last"], pa["lomo"]))


if __name__ == "__main__":
    main()
