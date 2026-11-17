"""Score a model on labelled external matches with the official scorer. Writes a run file and the leaderboard.

    .venv/bin/python -m src.evaluate --model nearest --param window_s=2 --matches skillcorner:all
    .venv/bin/python -m src.evaluate --model nearest --param window_s=1,2,4 \
        --matches skillcorner:1874553,1886347 --no-record          # grid, quick look, no file
    .venv/bin/python -m src.evaluate --model nearest --matches synthetic:all --no-record

--matches is a comma list of <source>:<id> or <source>:all. --sources idsse,skillcorner (`all`: the selection set,
C12; `every`: every registered source) is the registry's way (src/sources.py, A14): every match of those sources, each loaded with its source's
default scoring options (SkillCorner cuts=ungraded); --load still applies on top, to every source. A bare id reuses the previous source,
so skillcorner:1874553,1886347 is two SkillCorner matches. A source is any module in src/ingest/ that
has list_matches() and load(id) returning a Match with labels (skillcorner, synthetic, and idsse once
A2 lands). BAR-ATH is refused: model selection never uses it (CLAUDE.md).

--param key=v1,v2 runs a grid. Each config gets its own run file in experiments/runs/ and
experiments/LEADERBOARD.md is regenerated. --load key=value passes loader options (e.g. SkillCorner's
cuts=ungraded) and is part of the run's config and of the cache key. ball_missing / ball_gap_s /
ball_seed work on every source: they hide the ball like BAR-ATH loses it (src/ingest/degrade.py);
camera / camera_fill / player_drop / id_swap degrade the players and clean=1 drops them all (C11).
fps=<n> resamples any source to n fps, truth included (src/resample.py, backlog C1), e.g.
SkillCorner at BAR-ATH's 25 fps: --load fps=25.

Scoring: possession_metrics.evaluate() from official/evaluation, per match, the same function
run_evaluation.py uses. The headline is the unweighted mean over matches (findings F6), with
per-match numbers in the run file.

Models with a fit(matches) method are trained leave-one-match-out: for each evaluated match a fresh
model is fit on the --train matches (default: the evaluated ones) minus that match. Models without
fit are just run.

Loaded matches are cached as .npz under data/cache/matches/ (gitignored). The key hashes the load
options and the code that source depends on: its loader plus every src/ module it imports,
transitively. Editing one loader invalidates only the sources that use it. Ball options are applied
after loading and not cached (~1 s per match). --no-cache skips the cache.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
import re
import sys
import time

import numpy as np

from src import resample, sources
from src.ingest import degrade
from src.models import MODELS, build
from src.official import ROOT, pm
from src.presets import PRESETS, model_and_params
from src.run import git_commit, parse_params, set_params
from src.schema import Match

RUNS_DIR = os.path.join(ROOT, "experiments", "runs")
CACHE_DIR = os.path.join(ROOT, "data", "cache", "matches")
REFUSED_SOURCES = {"bar_ath": "BAR-ATH may not be used for model selection (CLAUDE.md)",
                   "epts": "epts is a file parser, not a source",
                   "degrade": "degrade is a transform; use --load ball_missing=..."}


# -- which matches ---------------------------------------------------------------------------------

def loader_name(source: str) -> str:
    """The src/ingest/ module behind a source: the registry's `loader` (src/sources.py), else its name."""
    return sources.SOURCES[source].loader if source in sources.SOURCES else source


def source_module(source: str):
    if source in REFUSED_SOURCES:
        raise ValueError("source {!r} refused: {}".format(source, REFUSED_SOURCES[source]))
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", source):
        raise ValueError("bad source name {!r}".format(source))
    try:
        module = importlib.import_module("src.ingest." + loader_name(source))
    except ModuleNotFoundError:
        raise ValueError("no loader src/ingest/{}.py".format(loader_name(source))) from None
    if not (hasattr(module, "list_matches") and hasattr(module, "load")):
        raise ValueError("src/ingest/{}.py has no list_matches()/load()".format(source))
    return module


def parse_matches(spec: str, lockbox: str | None = None) -> list[str]:
    """'skillcorner:all,idsse:X' -> ['skillcorner:<id>', ..., 'idsse:X'], in order, no repeats.
    Lockbox and test-only matches (A19, src/sources.py) are left out unless they are in tranche
    `lockbox`; naming one explicitly without it is refused, and so is a source left with none."""
    keys, explicit, source = [], set(), None
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        if ":" in item:
            source, match_id = item.split(":", 1)
        elif source is None:
            raise ValueError("{!r} has no source; write <source>:<id>".format(item))
        else:
            match_id = item
        module = source_module(source)
        if match_id == "all":
            ids = module.list_matches()
            if not ids:
                raise ValueError("{}: no matches found locally (fetch them first: "
                                 "scripts/fetch_external_data.sh {})".format(source, source))
        else:
            ids = [match_id]
            explicit.add("{}:{}".format(source, match_id))
        wanted = ["{}:{}".format(source, i) for i in ids]
        kept = sources.filter_lockbox(wanted, lockbox, explicit)
        if wanted and not kept:
            raise sources.LockboxError(
                "{}: every match is a lockbox match (A19: {}); score it only through a tranche "
                "with --lockbox {}".format(source, "test-only source" if source in sources.SOURCES
                                           and sources.SOURCES[source].test_only else "lockbox",
                                           "|".join(sources.tranche_names())))
        keys.extend(kept)
    return list(dict.fromkeys(keys))


def describe_matches(keys: list[str]) -> str:
    """Short label for a match list, e.g. 'skillcorner x20 + idsse x7' or 'skillcorner:1874553'."""
    if len(keys) <= 3:
        return ", ".join(keys)
    counts = {}
    for key in keys:
        source = key.split(":", 1)[0]
        counts[source] = counts.get(source, 0) + 1
    return " + ".join("{} x{}".format(source, n) for source, n in counts.items())


# -- loading, with a cache -------------------------------------------------------------------------

def src_imports(path: str) -> set[str]:
    """Paths of the src/ modules a file imports directly (from src.x import y, from src import x,
    import src.x). Stdlib, third-party and official/ imports don't count."""
    with open(path) as handle:
        tree = ast.parse(handle.read(), path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "src":
                names.update("src." + alias.name for alias in node.names)
            elif node.module.startswith("src."):
                names.add(node.module)
                names.update(node.module + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("src."))
    found = set()
    for name in names:
        base = os.path.join(ROOT, *name.split("."))
        if os.path.isfile(base + ".py"):
            found.add(base + ".py")
    return found


def loader_files(source: str) -> list[str]:
    """The code a cached match depends on: the source's loader and every src/ module it imports,
    transitively. Editing another source's loader leaves this source's cache alone (findings F25).
    degrade.py isn't in it: degraded matches aren't cached (see load_match)."""
    todo = [os.path.join(ROOT, "src", "ingest", loader_name(source) + ".py")]
    seen = set()
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo.extend(src_imports(path) - seen)
    return sorted(seen)


def loader_hash(source: str, load_opts: dict) -> str:
    digest = hashlib.sha1(json.dumps(load_opts, sort_keys=True).encode())
    for path in loader_files(source):
        digest.update(os.path.relpath(path, ROOT).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


FIELDS = ("period", "ball", "players", "slot_team", "labels")


def load_match(key: str, load_opts: dict, cache: bool = True) -> Match:
    """A labelled match. Ball and player options (degrade.OPTIONS) are applied to the clean match, which is
    what gets cached: hiding the ball takes ~1 s, re-reading a source tens of seconds.
    fps=<n> resamples last (src.resample.to_fps, backlog C1), after any ball hiding, so the same
    frames lose the ball at every rate and only the frame rate changes. Truth is resampled too."""
    if "fps" in load_opts:
        rest = {k: v for k, v in load_opts.items() if k != "fps"}
        return resample.to_fps(load_match(key, rest, cache), int(load_opts["fps"]))
    degrade_opts = {k: load_opts[k] for k in degrade.OPTIONS if k in load_opts}
    if degrade_opts:  # ball and player options (C11), applied to the clean match
        match = load_match(key, {k: v for k, v in load_opts.items() if k not in degrade_opts}, cache)
        return degrade.apply(match, degrade_opts)

    source, match_id = key.split(":", 1)
    path = os.path.join(CACHE_DIR, source,
                        "{}.{}.npz".format(match_id, loader_hash(source, load_opts)))
    if cache and os.path.isfile(path):
        with np.load(path) as saved:
            return Match(match_id=key, fps=int(saved["fps"]), pitch=tuple(saved["pitch"].tolist()),
                         **{name: saved[name] for name in FIELDS}).validate()

    match = source_module(source).load(match_id, **load_opts)
    if match.labels is None:
        raise ValueError("{}: the loader gave no labels, so it can't be scored".format(key))
    match.match_id = key
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = "{}.{}.partial.npz".format(path, os.getpid())  # unique: the session-start
        # cache warm-up may be writing the same match from another process
        np.savez(partial, fps=match.fps, pitch=np.array(match.pitch),
                 **{name: getattr(match, name) for name in FIELDS})
        os.replace(partial, path)
    return match


# -- scoring ----------------------------------------------------------------------------------------

SWITCH_TOL_S = 1.0  # switch precision/recall tolerance (findings F53, F61)
SWITCH_KEYS = ("switch_precision", "switch_recall")


def score(gt: np.ndarray, pred: np.ndarray, fps: int | None = None) -> dict:
    """The official metrics for one match, rounded. Same call as official run_evaluation.py.
    With fps, also H<->A switch precision / recall at +/- 1 s (B16, findings F61-F62), None when
    there is nothing to divide by. They're diagnostics next to spellF1, not a target (F62)."""
    pred = np.asarray(pred)
    if pred.shape != gt.shape:
        raise ValueError("prediction shape {} != {}".format(pred.shape, gt.shape))
    if not np.isin(pred, np.arange(len(pm.LABELS))).all():
        raise ValueError("prediction has codes outside 0..{}".format(len(pm.LABELS) - 1))
    result = pm.evaluate(gt, pred.astype(np.int8))
    out = {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "n_poss_pred": result["n_pred_spells"],
        "n_poss_true": result["n_gt_spells"],
    }
    if fps is not None:
        from src.error_analysis import switch_breakdown  # it imports this module
        sw = switch_breakdown(gt, pred, fps, pm.valid_runs(gt), SWITCH_TOL_S)
        for key, name in zip(SWITCH_KEYS, ("precision", "recall")):
            out[key] = None if np.isnan(sw[name]) else round(float(sw[name]), 4)
    return out


def _mean_known(values) -> float | None:
    """Mean of the values that aren't None, rounded; None if there are none."""
    known = [v for v in values if v is not None]
    return round(float(np.mean(known)), 4) if known else None


def summarise(per_match: dict) -> dict:
    """Unweighted means over matches, overall and per source when there are several."""
    rows = list(per_match.values())
    out = {
        "spellF1": round(float(np.mean([r["spellF1"] for r in rows])), 4),
        "macroF1": round(float(np.mean([r["macroF1"] for r in rows])), 4),
        "spellF1_min": min(r["spellF1"] for r in rows),
        "spellF1_max": max(r["spellF1"] for r in rows),
        "n_poss_pred_mean": round(float(np.mean([r["n_poss_pred"] for r in rows])), 1),
        "n_poss_true_mean": round(float(np.mean([r["n_poss_true"] for r in rows])), 1),
        "n_matches": len(rows),
    }
    for key in SWITCH_KEYS:  # skip matches without a value (no switches to divide by)
        if any(key in r for r in rows):
            out[key] = _mean_known(r.get(key) for r in rows)
    sources = sorted({key.split(":", 1)[0] for key in per_match})
    if len(sources) > 1:
        out["per_source_spellF1"] = {
            s: round(float(np.mean([r["spellF1"] for k, r in per_match.items()
                                    if k.startswith(s + ":")])), 4) for s in sources}
        if any(SWITCH_KEYS[0] in r for r in rows):
            out["per_source_switch_precision"] = {
                s: _mean_known(r.get(SWITCH_KEYS[0]) for k, r in per_match.items()
                               if k.startswith(s + ":")) for s in sources}
    out["per_match"] = per_match
    return out


def _predict_all(model_name: str, configs: list[dict], match: Match,
                 train: list[Match] | None) -> list[dict]:
    """Scores of every config on one match. train is None for models without fit."""
    scores = []
    for params in configs:
        model = build(model_name, **params)
        if train is not None:
            model.fit([m for m in train if m.match_id != match.match_id])
        scores.append(score(match.labels, model.predict(match), match.fps))
    return scores


def _worker(job: tuple) -> tuple[str, list[dict]]:
    key, load_opts, cache, model_name, configs = job
    match = load_match(key, load_opts, cache)
    return key, _predict_all(model_name, configs, match, None)


def _load_worker(job: tuple) -> Match:
    return load_match(*job)


def evaluate(model_name: str, configs: list[dict], eval_keys: list[str],
             train_keys: list[str] | None = None, load_opts: dict | None = None,
             cache: bool = True, jobs: int = 1, log=print,
             source_opts: dict | None = None) -> list[dict]:
    """[{params, metrics, mode}] per config, in the order given. source_opts {source: options}
    are loaded under load_opts for that source's matches only (src/sources.py defaults)."""
    load_opts = load_opts or {}
    source_opts = source_opts or {}

    def opts(key):
        return dict(source_opts.get(key.split(":", 1)[0], {}), **load_opts)

    fits = hasattr(build(model_name, **configs[0]), "fit")
    by_match = {}
    if not fits:
        work = [(key, opts(key), cache, model_name, configs) for key in eval_keys]
        if jobs > 1 and len(work) > 1:
            with multiprocessing.get_context().Pool(min(jobs, len(work))) as pool:
                results = pool.imap(_worker, work)
                by_match = dict(results)
        else:
            by_match = dict(map(_worker, work))
    else:
        keys = list(dict.fromkeys(eval_keys + (train_keys or [])))
        work = [(key, opts(key), cache) for key in keys]
        if jobs > 1 and len(work) > 1:
            with multiprocessing.get_context().Pool(min(jobs, len(work))) as pool:
                loaded = dict(zip(keys, pool.map(_load_worker, work)))
        else:
            loaded = dict(zip(keys, map(_load_worker, work)))
        # A19: a lockbox or test-only match is never trained on (refused, not silently dropped)
        train_from = train_keys or eval_keys
        if sources.drop_lockbox(train_from) != list(train_from):
            raise sources.LockboxError("lockbox matches in the training set (A19)")
        train = [loaded[k] for k in train_from]
        for key in eval_keys:
            by_match[key] = _predict_all(model_name, configs, loaded[key], train)
            log("  fit+scored {}".format(key))

    return [{"params": params,
             "mode": "leave-one-match-out" if fits else "no training",
             "metrics": summarise({key: by_match[key][i] for key in eval_keys})}
            for i, params in enumerate(configs)]


# -- run files ----------------------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def write_run(result: dict, model_name: str, given: dict, args, eval_keys: list[str],
              train_keys: list[str], runs_dir: str) -> str:
    date = time.strftime("%Y-%m-%d")
    if args.id:  # a grid still needs one file per config, so add the grid keys
        grid = [k for k, v in given.items() if isinstance(v, list)]
        base = slug("-".join([args.id] + ["{}{}".format(k, result["params"][k]) for k in grid]))
    else:
        parts = [model_name] + ["{}{}".format(k, result["params"][k]) for k in given]
        parts.append(describe_matches(eval_keys).replace(" x", ""))
        base = slug("-".join(parts))
    base = date + "-" + base
    os.makedirs(runs_dir, exist_ok=True)
    path, n = os.path.join(runs_dir, base + ".json"), 1
    while os.path.exists(path):  # never overwrite another run
        n += 1
        path = os.path.join(runs_dir, "{}-{}.json".format(base, n))

    model = build(model_name, **result["params"])
    run = {
        "id": os.path.basename(path)[:-5],
        "commit": git_commit(),
        "author": args.author,
        "harness": "src.evaluate",
        "train": train_keys if result["mode"] != "no training" else [],
        "eval": eval_keys,
        "eval_label": describe_matches(eval_keys),
        "config": {"model": model_name,
                   "params": set_params(model) if dataclasses.is_dataclass(model)
                   else result["params"],
                   "load": args.load_opts, "mode": result["mode"],
                   **({"source_load": args.source_opts} if args.source_opts else {})},
        "metrics": result["metrics"],
        "seconds": args.seconds,
        "note": args.note,
    }
    with open(path, "w") as handle:
        json.dump(run, handle, indent=2)
        handle.write("\n")
    return path


# -- command line --------------------------------------------------------------------------------------

def expand_grid(pairs: list[str]) -> tuple[list[dict], dict]:
    """--param key=v1,v2 ... -> (list of param dicts, {key: value or [values]} as given)."""
    given = {}
    for pair in pairs:
        key, _, values = pair.partition("=")
        parsed = [parse_params(["x=" + v])["x"] for v in values.split(",")]
        given[key] = parsed if len(parsed) > 1 else parsed[0]
    keys = list(given)
    grids = [v if isinstance(v, list) else [v] for v in given.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)], given


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--preset", choices=list(PRESETS),
                        help="named candidate (src/presets.py); learned presets are held out (lomo)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=V[,V...]",
                        help="model parameter, repeatable; a comma list runs a grid")
    parser.add_argument("--matches", help="e.g. skillcorner:all,idsse:<id>")
    parser.add_argument("--sources", help="registered sources (src/sources.py) or 'all' (the "
                        "selection set, C12; 'every': every registered one): their matches, each with its "
                        "source's default load options")
    parser.add_argument("--train", help="training matches for models with fit() (default: --matches)")
    parser.add_argument("--load", action="append", default=[], metavar="KEY=VALUE",
                        help="loader option, repeatable (passed to <source>.load)")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="print only; no run file")
    parser.add_argument("--id", help="run id after the date (default: model, params, matches)")
    parser.add_argument("--note", default="", help="one line for the run file")
    parser.add_argument("--author", default="cloud-agent" if os.environ.get("CLAUDE_CODE_REMOTE")
                        else "owner")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--quiet", action="store_true", help="no per-match lines")
    parser.add_argument("--lockbox", help="A19: also score the lockbox matches of this tranche "
                        "(explore, T1, T2...). Opening T1/T2 is recorded in experiments/lockbox.json "
                        "and allowed once; never to tune or choose between candidates")
    args = parser.parse_args(argv)
    if args.preset:
        if args.model:
            parser.error("give --preset or --model, not both")
        args.model, preset_params = model_and_params(args.preset, for_eval=True)
        args.param = ["{}={}".format(k, v) for k, v in preset_params.items()] + args.param
        args.load = ["{}={}".format(k, v) for k, v in PRESETS[args.preset].get("load", {}).items()
                     ] + args.load
    elif not args.model:
        parser.error("--model or --preset is required")
    if not args.no_record and not args.note:
        parser.error("--note is required when recording a run (or pass --no-record)")

    if bool(args.matches) == bool(args.sources):
        parser.error("give --matches or --sources (one of them)")
    configs, given = expand_grid(args.param)
    args.load_opts = parse_params(args.load)
    args.source_opts = {}
    if args.sources:
        chosen = (sources.names() if args.sources == "all" else
                  sources.names(everything=True) if args.sources == "every" else args.sources.split(","))
        args.source_opts = {s: sources.load_opts(s) for s in chosen if sources.load_opts(s)}
        args.matches = ",".join(s + ":all" for s in chosen)
    eval_keys = parse_matches(args.matches, lockbox=args.lockbox)
    # a lockbox match is never trained on (A19): explicit --train refuses one, the default drops them
    train_keys = (parse_matches(args.train) if args.train else
                  [k for k in eval_keys if not sources.is_lockbox(*k.split(":", 1))])
    if args.lockbox:
        sources.open_tranche(args.lockbox, {"by": "src.evaluate", "model": args.model,
                                            "params": args.param, "load": args.load,
                                            "matches": len(eval_keys), "note": args.note})
    print("evaluate   {} on {} ({} configs)".format(args.model, describe_matches(eval_keys),
                                                    len(configs)))
    started = time.time()
    results = evaluate(args.model, configs, eval_keys, train_keys, args.load_opts,
                       cache=not args.no_cache, jobs=args.jobs, source_opts=args.source_opts)
    args.seconds = round(time.time() - started, 1)

    for result in results:
        metrics = result["metrics"]
        if not args.quiet and len(results) == 1:
            for key, row in metrics["per_match"].items():
                print("  {:<28} spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  "
                      "spells {n_poss_pred}/{n_poss_true}".format(key, **row))
        print("{:<40} spellF1 {:.4f} [{:.3f}-{:.3f}]  macroF1 {:.4f}  spells {}/{}".format(
            json.dumps(result["params"]) if result["params"] else "(defaults)",
            metrics["spellF1"], metrics["spellF1_min"], metrics["spellF1_max"],
            metrics["macroF1"], metrics["n_poss_pred_mean"], metrics["n_poss_true_mean"]))
        if "per_source_spellF1" in metrics:
            print("  per source " + "  ".join("{} {:.4f}".format(s, v) for s, v in
                                              metrics["per_source_spellF1"].items()))
    print("time       {} s".format(args.seconds))

    if not args.no_record:
        for result in results:
            path = write_run(result, args.model, given, args, eval_keys, train_keys, args.runs_dir)
            result["run_file"] = path
            print("recorded   {}".format(os.path.relpath(path, ROOT)))
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import make_leaderboard
        board = make_leaderboard.main(["--runs-dir", args.runs_dir])
        print("leaderboard {}".format(os.path.relpath(board, ROOT)))
    return results


if __name__ == "__main__":
    main()
