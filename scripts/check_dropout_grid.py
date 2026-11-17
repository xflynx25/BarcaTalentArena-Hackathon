"""C6 + C12: score candidates held out across the ball-dropout grid and choose between them with
uncertainty; or measure each source's real dropout.

Why (backlog C6): one dropout setting was retired because it had been chosen to match published
full-match scores. Every model is scored on every point of src/dropout.py's grid (ball missing
0.30 / 0.41 / 0.50 x quiet share 0 / 0.5 / 1 x gap 1 / 4 s, 15 points) and ranked by the mean over grid
points and sources; the worst grid point is reported too (a model that wins at one point only is a
red flag). 0.41 is the missing-ball share in the official data notes. Nothing here is fit to a
possession score.

How we choose (C12, findings F114-F115, decisions.md):
  - the headline is **held out by source** (`learned` with fold=loso; a source the fit never saw is
    scored with the fit on everything, which is held out by source too). fold=lomo follows as a check.
  - every candidate is compared with --ref by a **seeded paired bootstrap over matches**: per-match
    differences (mean over the grid) are resampled within each source, and the source means are
    weighted. It prints the weighted difference with its 95% interval, the same per source, and
    "wins k of n sources". Verdict: `better` if the weighted interval is above 0 and no source's
    interval is below 0; `worse` if the weighted interval is below 0; else `tie`. This replaces the
    fixed 0.005 tie margin (F55).
  - sources and weights come from the registry (src/sources.py: the selection set, weights from the
    written principles). --weights c6 gives C6's 0.25 / 0.5 / 0.25 (only for its three sources) to
    reproduce older runs; the equal-weight table is printed as a check.

    .venv/bin/python scripts/check_dropout_grid.py --measure          # real dropout per source, ~1 min
    .venv/bin/python scripts/check_dropout_grid.py --coef v3=src/models/learned_coefs_v3.json --ref v3
    .venv/bin/python scripts/check_dropout_grid.py --coef f30=/tmp/f30.json:min_spell_s=1 --json /tmp/grid.json

Candidates: CANDIDATES below; `--coef name=path` adds `learned` with that coefficient file,
`--coef name=path:key=value,...` with extra params, `--extra name=model:k=v,...` any model.
`learned` is always held out, once per fold in --folds (unless its params fix `fold`). --sources picks
a subset of the registry. Prints tables only (external data). Deterministic (fixed seeds).

Lockbox (A19): lockbox and test-only matches are left out unless --lockbox <tranche> names
their tranche; a test-only source in --sources needs it. `--lockbox explore` is not recorded;
named tranches are opened once each, recorded in experiments/lockbox.json, for a candidate
that has passed C12 (and C14), never to choose.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import dropout, evaluate  # noqa: E402
from src import sources as registry  # noqa: E402
from src.models import build  # noqa: E402
from src.models.learned import LearnedDecoder  # noqa: E402
from src.official import A, D, H, X  # noqa: E402

# scoring load options and selection weights, from the registry; main() narrows them with --sources
SOURCES = {s: registry.load_opts(s) for s in registry.names()}
WEIGHTS = registry.weights()
EQUAL = {s: 1 / len(SOURCES) for s in SOURCES}
SHORT = {"idsse": "IDSSE", "skillcorner": "SC", "metrica": "Metrica", "skillcorner2020": "SC20",
         "soccertrack": "ST"}  # table headers
FOLDS = ("loso", "lomo")  # held out by source (the headline, C12), then by match
BOOT_N, BOOT_SEED, BOOT_LEVEL = 10000, 0, 0.95
CANDIDATES = {
    "shipped": ("learned", {}),  # src.run's default (v4 since 10 Nov, F99); fold set per --folds
    "viterbi": ("viterbi", {}),
    "deadball+delay1": ("deadball", {"switch_delay_s": 1.0}),
}
METRICS = ("spellF1", "macroF1", "switch_precision", "switch_recall")
# Settings applied after an emission model's emission(): the restart rule in combine() (B35) and the
# switch cue and wave 5's `cues` in decode() (B36, LearnedDecoder.decode reads them, scores() doesn't).
# Candidates that differ only in these share one emission() (or, for `learned`, one scores() for
# CUE_ONLY) per task. `repairs` changes the match the emissions read, so it is not in either.
RESTART_ONLY = ("restart_rule_gain", "restart_rule_back_s", "restart_rule_kick_mps",
                "restart_rule_search_s", "restart_rule_noise")
CUE_ONLY = ("switch_cue_gain", "switch_cue_where", "switch_cue_file", "switch_cue_min_seen", "cues")
DECODE_ONLY = RESTART_ONLY + CUE_ONLY


# -- real dropout ------------------------------------------------------------------------------------

def dropout_stats(match) -> dict:
    """Label-based description of where the ball is missing (external matches only): share of graded
    frames, of live (H/A) and of dead (D) frames without a ball, the share of missing graded frames
    that are dead, and the median gap length (s) over graded frames."""
    graded = match.labels != X
    live = (match.labels == H) | (match.labels == A)
    dead = match.labels == D
    missing = np.isnan(match.ball[:, 0])
    gaps = []
    edges = np.diff(np.concatenate(([0], (missing & graded).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        gaps.append((stop - start) / match.fps)
    return {"missing": float(missing[graded].mean()), "live": float(missing[live].mean()),
            "dead": float(missing[dead].mean()),
            "dead share": float(dead[missing & graded].mean()) if (missing & graded).any() else 0.0,
            "median gap": float(np.median(gaps)) if gaps else 0.0}


def _measure_job(task):
    key, point = task
    source = key.split(":", 1)[0]
    opts = dict(SOURCES[source], **(dropout.GRID[point] if point is not None else {}))
    return key, point, dropout_stats(evaluate.load_match(key, opts))


def measure(keys: dict, jobs: int) -> None:
    """Real dropout per source, and the same numbers for every grid point (applied to IDSSE, whose
    ball is never missing, so only the simulation shows), to place each source on the grid."""
    tasks = [(k, None) for s in SOURCES for k in keys[s]]
    tasks += [(k, i) for k in keys["idsse"] for i in range(len(dropout.GRID))]
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        res = pool.map(_measure_job, tasks)
    cols = ("missing", "live", "dead", "dead share", "median gap")
    print("{:<22}".format("ball missing on") + "".join("{:>12}".format(c) for c in cols)
          + "    nearest grid point")
    grid_rows = {}
    for i in range(len(dropout.GRID)):
        rows = [r for k, p, r in res if p == i]
        grid_rows[i] = {c: float(np.mean([r[c] for r in rows])) for c in cols}

    def nearest(row):  # scaled distance over missing share, dead share and log gap length
        def dist(g):
            return (abs(row["missing"] - g["missing"]) / 0.1 + abs(row["dead share"] - g["dead share"])
                    / 0.2 + abs(np.log1p(row["median gap"]) - np.log1p(g["median gap"])))
        return dropout.name(dropout.GRID[min(grid_rows, key=lambda i: dist(grid_rows[i]))])

    for s in SOURCES:
        rows = [r for k, p, r in res if p is None and k.startswith(s + ":")]
        row = {c: float(np.mean([r[c] for r in rows])) for c in cols}
        print("{:<22}".format("{} x{} (real)".format(s, len(rows)))
              + "".join("{:>12.2f}".format(row[c]) for c in cols)
              + ("    " + nearest(row) if row["missing"] > 0.01 else "    (never missing)"))
    for i, row in grid_rows.items():
        print("{:<22}".format("idsse " + dropout.name(dropout.GRID[i]))
              + "".join("{:>12.2f}".format(row[c]) for c in cols))


# -- scoring -----------------------------------------------------------------------------------------

def predict_shared(name: str, params: dict, match, cache: dict) -> np.ndarray:
    """build(name, **params).predict(match), sharing the expensive part among candidates that differ
    only in decode-time settings. An emission model (bigger, boosted, temporal): one emission() per
    cache for any DECODE_ONLY difference (B35: restart-rule gains, applied in combine(); B36: switch-
    cue settings, applied in decode()). Another LearnedDecoder: one scores() per cache for CUE_ONLY
    differences (the restart rule is inside its scores()). The same codes as predict(), since
    predict() is decode(scores()) and an emission model's scores() is combine(read(m), emission(m))
    (tests/test_grid_shared.py)."""
    model = build(name, **params)

    def key(kind, skip):
        return (kind, name, tuple(sorted((k, repr(v)) for k, v in params.items() if k not in skip)))

    if hasattr(model, "emission") and hasattr(model, "combine"):
        k = key("emission", DECODE_ONLY)
        if k not in cache:
            cache[k] = model.emission(match)
        return model.decode(match, model.combine(model.read(match), cache[k]))
    if isinstance(model, LearnedDecoder):
        k = key("scores", CUE_ONLY)
        if k not in cache:
            cache[k] = model.scores(match)
        return model.decode(match, cache[k])
    return model.predict(match)


def _score_job(task):
    key, point, candidates = task
    source = key.split(":", 1)[0]
    match = evaluate.load_match(key, dict(SOURCES[source], **dropout.GRID[point]))
    out, cache = {}, {}
    for name, (model, params) in candidates.items():
        res = evaluate.score(match.labels, predict_shared(model, params, match, cache), match.fps)
        out[name] = {m: (np.nan if res[m] is None else res[m]) for m in METRICS}
    return key, point, out


def run_tasks(pool, fn, tasks: list, spec, checkpoint: str | None = None, chunksize: int = 2) -> dict:
    """{(match, grid point): scores} for every task. With a checkpoint file (B30: cloud containers
    restart without warning), each finished task is appended to it and flushed, and a rerun with the
    same spec (candidates, grid, load options) only scores the tasks it doesn't hold yet. JSON floats
    round-trip exactly, so a resumed run gives the same numbers as an uninterrupted one."""
    tag = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]
    done = {}
    if checkpoint and os.path.isfile(checkpoint):
        with open(checkpoint) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:  # a line cut off by a restart
                    continue
                if row.get("spec") == tag:
                    done[(row["k"], row["i"])] = row["out"]
    todo = [t for t in tasks if (t[0], t[1]) not in done]
    if done:
        print("checkpoint: {} of {} tasks already scored".format(len(tasks) - len(todo), len(tasks)),
              flush=True)
    fh = open(checkpoint, "a") if checkpoint else None
    if fh and fh.tell():  # start on a fresh line after a line a restart cut off
        with open(checkpoint, "rb") as tail:
            tail.seek(-1, os.SEEK_END)
            if tail.read(1) != b"\n":
                fh.write("\n")
    try:
        for k, i, out in pool.imap_unordered(fn, todo, chunksize=chunksize):
            done[(k, i)] = out
            if fh:
                fh.write(json.dumps({"spec": tag, "k": k, "i": i, "out": out}) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    return {(t[0], t[1]): done[(t[0], t[1])] for t in tasks}


def expand(candidates: dict, folds: list[str]) -> tuple[dict, dict]:
    """The jobs to run ({job: (model, params)}) and, per fold, which job each candidate name reads.
    `learned` without a fixed fold runs once per fold; every other model once for all folds."""
    jobs, view = {}, {f: {} for f in folds}
    for name, (model, params) in candidates.items():
        if model == "learned" and "fold" not in params:
            for f in folds:
                jobs[name + "@" + f] = (model, dict(params, fold=f))
                view[f][name] = name + "@" + f
        else:
            jobs[name] = (model, params)
            for f in folds:
                view[f][name] = name
    return jobs, view


def fold_view(res: dict, view: dict) -> dict:
    """res with each job renamed to its candidate name, for one fold."""
    return {ki: {name: out[job] for name, job in view.items()} for ki, out in res.items()}


def match_score(res: dict, key: str, name: str) -> float:
    """One match's spellF1 for a candidate: the mean over the grid points."""
    return float(np.mean([res[(key, i)][name]["spellF1"] for i in range(len(dropout.GRID))]))


def paired_bootstrap(res: dict, keys: dict, a: str, b: str, weights: dict, n_boot: int = BOOT_N,
                     seed: int = BOOT_SEED, level: float = BOOT_LEVEL) -> dict:
    """a minus b, paired by match: per-match differences (mean over the grid) resampled with
    replacement within each source (seeded, so the same draws every run), source means weighted.
    Returns the weighted difference and interval, per-source ones, and the verdict (C12)."""
    rng = np.random.default_rng(seed)
    tails = [(1 - level) / 2, (1 + level) / 2]
    total, point, per = np.zeros(n_boot), 0.0, {}
    for s in weights:  # registry order: the draws depend only on the seed and the match counts
        d = np.array([match_score(res, k, a) - match_score(res, k, b) for k in keys[s]])
        draws = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        lo, hi = np.quantile(draws, tails)
        per[s] = {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
                  "better": int((d > 0).sum()), "n": int(len(d))}
        total += weights[s] * draws
        point += weights[s] * float(d.mean())
    lo, hi = np.quantile(total, tails)
    losses = [s for s, r in per.items() if r["hi"] < 0]
    verdict = ("worse" if hi < 0 else "tie" if lo <= 0 else
               "better" if not losses else "better overall, worse on " + ", ".join(losses))
    return {"a": a, "b": b, "diff": point, "lo": float(lo), "hi": float(hi),
            "p_better": float((total > 0).mean()), "per source": per,
            "sources won": sum(r["diff"] > 0 for r in per.values()), "n sources": len(per),
            "losses": losses, "verdict": verdict}


def print_bootstrap(title: str, rows: list[dict]) -> None:
    print("\n== {} ==  paired bootstrap over matches ({} draws, seed {}), spellF1 difference (mean "
          "over the grid), {:.0%} interval; per source: difference [interval] matches better"
          .format(title, BOOT_N, BOOT_SEED, BOOT_LEVEL))
    for r in rows:
        print("{} vs {}: {:+.4f} [{:+.4f}, {:+.4f}]  wins {} of {} sources  P(better) {:.2f}  -> {}"
              .format(r["a"], r["b"], r["diff"], r["lo"], r["hi"], r["sources won"],
                      r["n sources"], r["p_better"], r["verdict"].upper()))
        for s, p in r["per source"].items():
            print("    {:<9}{:+.4f} [{:+.4f}, {:+.4f}]  {}/{}{}".format(
                SHORT.get(s, s[:8]), p["diff"], p["lo"], p["hi"], p["better"], p["n"],
                "  (few matches: interval too narrow)" if p["n"] < 5 else ""))


def summarise(res: dict, keys: dict, names: list[str], weights: dict) -> dict:
    """{name: {"grid": [weighted spellF1 per point], "mean", "worst", "worst point",
    "per source": {s: mean over grid}, metric means}}."""
    out = {}
    n = len(dropout.GRID)
    for name in names:
        def source_mean(s, i, metric="spellF1"):
            return float(np.nanmean([res[(k, i)][name][metric] for k in keys[s]]))
        grid = [sum(weights[s] * source_mean(s, i) for s in SOURCES) for i in range(n)]
        worst = int(np.argmin(grid))
        row = {"grid": grid, "mean": float(np.mean(grid)), "worst": grid[worst],
               "worst point": dropout.name(dropout.GRID[worst]),
               "per source": {s: float(np.mean([source_mean(s, i) for i in range(n)]))
                              for s in SOURCES}}
        for metric in METRICS[1:]:
            row[metric] = float(np.mean([sum(weights[s] * source_mean(s, i, metric) for s in SOURCES)
                                         for i in range(n)]))
        out[name] = row
    return out


def print_summary(title: str, summary: dict, res: dict, keys: dict, ref: str) -> None:
    names = sorted(summary, key=lambda n: -summary[n]["mean"])
    print("\n== {} ==  held-out spellF1, mean over the {} grid points; worst point; per source "
          "(mean over grid); switch P/R; grid points won vs {}".format(title, len(dropout.GRID), ref))
    print("{:<20}{:>8}{:>8}  {:<16}".format("", "mean", "worst", "worst point")
          + "".join("{:>8}".format(SHORT.get(s, s[:7])) for s in SOURCES)
          + "{:>8}{:>12}{:>7}".format("macro", "sw P / R", "wins"))
    for name in names:
        r = summary[name]
        wins = sum(a > b for a, b in zip(r["grid"], summary[ref]["grid"]))
        print("{:<20}{:>8.4f}{:>8.4f}  {:<16}".format(name, r["mean"], r["worst"], r["worst point"])
              + "".join("{:>8.3f}".format(r["per source"][s]) for s in SOURCES)
              + "{:>8.3f}{:>12}{:>7}".format(r["macroF1"], "{:.2f} / {:.2f}".format(
                  r["switch_precision"], r["switch_recall"]), "{}/{}".format(wins, len(r["grid"]))))


def print_grid(summary: dict) -> None:
    names = sorted(summary, key=lambda n: -summary[n]["mean"])
    print("\nweighted spellF1 per grid point (rows) and candidate (columns); * = best at that point")
    print("{:<16}".format("") + "".join("{:>18}".format(n[:17]) for n in names))
    for i, point in enumerate(dropout.GRID):
        vals = [summary[n]["grid"][i] for n in names]
        best = int(np.argmax(vals))
        print("{:<16}".format(dropout.name(point)) + "".join(
            "{:>18}".format("{:.4f}{}".format(v, "*" if j == best else " ")) for j, v in enumerate(vals)))


def lomo_selection(res: dict, keys: dict, names: list[str], weights: dict) -> dict:
    """Leave-one-match-out over the pick (best weighted mean over the grid): for each match, pick on
    the other matches, score it. How much choosing among the candidates overfits (cf. F63)."""
    n = len(dropout.GRID)
    score = {(k, name): float(np.mean([res[(k, i)][name]["spellF1"] for i in range(n)]))
             for s in SOURCES for k in keys[s] for name in names}

    def wmean(kept, name):
        return sum(w * np.mean([score[(k, name)] for k in kept[s]]) for s, w in weights.items())

    held, picked = {s: [] for s in SOURCES}, {}
    for s in SOURCES:
        for left in keys[s]:
            kept = {t: [k for k in keys[t] if k != left] for t in SOURCES}
            best = max(names, key=lambda name: wmean(kept, name))
            picked[best] = picked.get(best, 0) + 1
            held[s].append(score[(left, best)])
    return {"weighted": sum(w * float(np.mean(held[s])) for s, w in weights.items()),
            "picked": picked}


def parse_coef(spec: str) -> tuple[str, tuple]:
    name, _, rest = spec.partition("=")
    path, _, extra = rest.partition(":")
    params = {"coef_file": os.path.abspath(path)}
    if extra:
        from src.run import parse_params
        params.update(parse_params(extra.split(",")))
    return name, ("learned", params)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", action="store_true", help="only measure real dropout")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--coef", action="append", default=[], metavar="NAME=PATH[:K=V,...]",
                        help="add learned (held out per --folds) with this coefficient file")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=MODEL:K=V,...",
                        help="add any model with these params")
    parser.add_argument("--ref", help="compare every candidate with this one (default: shipped, "
                                      "else the first)")
    parser.add_argument("--folds", default=",".join(FOLDS),
                        help="held-out folds for learned, the first is the headline (C12: loso)")
    parser.add_argument("--weights", default="registry", choices=("registry", "c6", "equal"),
                        help="source weights: the registry's (C12), C6's old 1/4 1/2 1/4, or equal")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--json", help="also write the per-match scores and summaries here")
    parser.add_argument("--checkpoint", help="B30: append each scored (match, point) here and "
                        "resume from it (same candidates and grid only)")
    parser.add_argument("--sources", default=",".join(registry.names()),
                        help="registered sources to score (src/sources.py); weights renormalised")
    parser.add_argument("--lockbox", help="A19: score this tranche's lockbox matches (explore, T1, "
                        "T2); T1/T2 open once, recorded in experiments/lockbox.json")
    args = parser.parse_args(argv)
    chosen = args.sources.split(",")
    test_only = [s for s in chosen if registry.get(s).test_only]
    if test_only and not args.lockbox:
        parser.error("{} is test-only (A19): give --lockbox explore|T1|T2".format(",".join(test_only)))
    if args.weights == "c6":
        if set(chosen) - set(registry.C6_WEIGHTS):
            parser.error("--weights c6 only covers " + ",".join(registry.C6_WEIGHTS))
        total = sum(registry.C6_WEIGHTS[s] for s in chosen)
        weights = {s: registry.C6_WEIGHTS[s] / total for s in chosen}
    else:
        weights = registry.weights(chosen) if args.weights == "registry" else None
    # narrow the module-level tables in place: the forked workers and the summaries read them
    for table, new in ((SOURCES, {s: registry.load_opts(s) for s in chosen}),
                       (EQUAL, {s: 1 / len(chosen) for s in chosen})):
        table.clear()
        table.update(new)
    WEIGHTS.clear()
    WEIGHTS.update(weights or EQUAL)

    keys = {s: evaluate.parse_matches(s + ":all", lockbox=args.lockbox) for s in SOURCES}
    if args.measure:
        measure(keys, args.jobs)
        return
    candidates = {n: CANDIDATES[n] for n in args.candidates.split(",") if n}
    candidates.update(parse_coef(c) for c in args.coef)
    for spec in args.extra:
        from src.run import parse_params
        name, _, rest = spec.partition("=")
        model, _, extra = rest.partition(":")
        candidates[name] = (model, parse_params(extra.split(",")) if extra else {})
    names = list(candidates)
    folds = args.folds.split(",")
    if args.lockbox:
        registry.open_tranche(args.lockbox, {"by": "check_dropout_grid", "sources": chosen,
                                             "candidates": {n: list(c) for n, c in candidates.items()},
                                             "folds": folds})
    jobs, view = expand(candidates, folds)
    tasks = [(k, i, jobs) for s in SOURCES for k in keys[s] for i in range(len(dropout.GRID))]
    print("scoring {} candidates ({} runs) on {} at {} grid points; held out by {}".format(
        len(names), len(jobs), ", ".join("{} x{}".format(s, len(keys[s])) for s in SOURCES),
        len(dropout.GRID), " then ".join(folds)), flush=True)
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        res = run_tasks(pool, _score_job, tasks, [jobs, list(dropout.GRID), SOURCES], args.checkpoint)

    ref = args.ref or ("shipped" if "shipped" in names else names[0])
    label = "weighted " + ", ".join("{} {:g}".format(SHORT.get(s, s), round(w, 3))
                                    for s, w in WEIGHTS.items())
    out = {"grid": list(dropout.GRID), "candidates": candidates, "folds": folds,
           "weights": dict(WEIGHTS), "ref": ref, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
                                                                "level": BOOT_LEVEL},
           "per_match": {"{}|{}".format(k, i): v for (k, i), v in res.items()}}
    for j, fold in enumerate(folds):
        fres = fold_view(res, view[fold])
        held = "held out by " + {"loso": "source", "lomo": "match"}.get(fold, fold)
        weighted = summarise(fres, keys, names, WEIGHTS)
        print_summary("{}, {}".format(held, label), weighted, fres, keys, ref)
        out[fold] = {"weighted": weighted}
        if j == 0:  # the headline fold gets the full set of tables
            equal = summarise(fres, keys, names, EQUAL)
            print_summary(held + ", equal weights", equal, fres, keys, ref)
            print_grid(weighted)
            sel = lomo_selection(fres, keys, names, WEIGHTS)
            print("\nleave-one-match-out over the pick (weighted mean over grid): {:.4f}; picked {}"
                  .format(sel["weighted"], sel["picked"]))
            out[fold].update(equal=equal, lomo_selection=sel)
        boot = [paired_bootstrap(fres, keys, n, ref, WEIGHTS) for n in names if n != ref]
        print_bootstrap("{}, vs {}".format(held, ref), boot)
        out[fold]["bootstrap"] = boot
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    return out


if __name__ == "__main__":
    main()
