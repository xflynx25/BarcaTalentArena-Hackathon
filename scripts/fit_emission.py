"""Fit a bigger emission model (B31) on the external matches and write its coefficient file.

    .venv/bin/python scripts/fit_emission.py --kind boosted               # src/models/boosted_coefs.json (+ _heldout.json.gz)
    .venv/bin/python scripts/fit_emission.py --kind temporal              # src/models/temporal_coefs.json (+ _heldout.json.gz)
    .venv/bin/python scripts/fit_emission.py --kind temporal --spec taps=0 --out /tmp/mlp.json  # no context

Same data as scripts/fit_learned.py: every training source of the registry (src/sources.py, with its
train_load options; --load adds options, e.g. B30's <source>:track_repair=1), each match under --mix K settings of the ball-dropout grid (src/dropout.py, drawn
per match from its key), the same features (learned.features with v4's groups, --groups to change).
Fits, like fit_learned.py --folds K:
  all      every match: what src.run would use
  fold:i   without fold i's matches (grouped by match, stratified by source, fit_learned's
           assignment); fold=lomo scores each match with the fit of its fold
  loso:s   without source s (fold=loso)
Inside each fit, INNER_SHARE of its matches (by match, per source) is kept aside to stop the boosting
rounds / training epochs (src/emission_train.py); the held-out fold is never seen.

Training rows are cached in data/cache/emission_rows/ (gitignored), keyed by the match, its load
options, the feature groups and spec, and a hash of the code that makes them. Deterministic: fixed
seeds (a hash of each fit's name), one BLAS thread per worker, coefficients on fixed grids, the
gzip header fixed: a refit on the same machine writes the same bytes. Prints a short summary only.
Then score held out with scripts/check_dropout_grid.py --extra NAME=boosted:fold=lomo,coef_file=PATH
(and fold=loso), or scripts/check_emission.py for decoder settings.
"""

from __future__ import annotations

import os

# one BLAS thread per worker, set before numpy is imported anywhere (findings F100)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import dropout, evaluate  # noqa: E402
from src import emission_train as train  # noqa: E402
from src import sources as registry  # noqa: E402
from src.models import emission, learned  # noqa: E402

SOURCES = {s: registry.load_opts(s, train=True) for s in registry.names(everything=True)}
ROW_CACHE = os.path.join(evaluate.ROOT, "data", "cache", "emission_rows")
ROW_CODE = [os.path.join(evaluate.ROOT, "src", *p) for p in
            (("models", "emission.py"), ("emission_train.py",), ("models", "learned.py"),
             ("ingest", "degrade.py"), ("evaluate.py",))]
DEFAULT_OUT = {"boosted": emission.BOOSTED_FILE, "temporal": emission.TEMPORAL_FILE}


def code_digest() -> str:
    todo, seen = list(ROW_CODE), set()
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo.extend(evaluate.src_imports(path) - seen)
    digest = hashlib.sha1()
    for path in sorted(seen):
        digest.update(os.path.relpath(path, evaluate.ROOT).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def row_path(key: str, opts: dict, groups: tuple, feat_spec: dict, code: str) -> str:
    spec = {"key": key, "opts": opts, "groups": list(groups), "spec": feat_spec,
            "feature_hash": learned.feature_hash(groups), "stride_s": emission.STRIDE_S,
            "loader": evaluate.loader_hash(key.split(":", 1)[0], opts), "code": code}
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    source, rest = key.split(":", 1)
    return os.path.join(ROW_CACHE, source, "{}.{}.npz".format(rest.replace("|", "_"), digest))


def _rows(job):
    key, opts, groups, feat_spec, path = job
    if path and os.path.isfile(path):
        with np.load(path) as saved:
            return key, {k: saved[k] for k in saved.files}, True
    match = evaluate.load_match(key.split("|", 1)[0], opts)
    rows = train.grid_rows(match, groups, feat_spec)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = "{}.{}.partial.npz".format(path, os.getpid())
        np.savez(partial, **rows)
        os.replace(partial, path)
    return key, rows, False


ROWS: dict = {}  # filled before the fit pool forks, so workers share it
SETTINGS: dict = {}


def _fit(job):
    name, keys = job
    started = time.time()
    fit_keys, valid_keys = train.inner_split(sorted(keys))
    data, valid = train.stack(ROWS, fit_keys), train.stack(ROWS, valid_keys)
    kind, spec = SETTINGS["kind"], SETTINGS["spec"]
    log = print if SETTINGS["verbose"] else None
    out = train.FITTERS[kind](data, valid, spec, zlib.crc32(("b31|" + name).encode()), log)
    out["seconds"] = time.time() - started
    out["rows"] = int((data["y"] >= 0).sum())
    return name, out


def parse_spec(pairs: list[str], base: dict) -> dict:
    spec = dict(base)
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in spec:
            raise SystemExit("unknown spec key {!r}; have {}".format(key, sorted(spec)))
        old = spec[key]
        if isinstance(old, list):
            spec[key] = [int(v) for v in value.split("/") if v != ""]
        else:
            spec[key] = type(old)(value)
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=emission.KINDS)
    parser.add_argument("--sources", default=",".join(registry.names("train")))
    parser.add_argument("--load", action="append", default=[], metavar="SOURCE:KEY=VALUE",
                        help="extra load option for one source (repeatable), as fit_learned.py "
                             "(B30: <source>:track_repair=1)")
    parser.add_argument("--mix", type=int, default=1, help="dropout grid draws per match (fit_learned's --mix)")
    parser.add_argument("--groups", default=",".join(emission.GROUPS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--spec", action="append", default=[], metavar="KEY=VALUE",
                        help="model setting (emission_train.BOOSTED_SPEC / TEMPORAL_SPEC); "
                             "lists as a/b/c, e.g. taps=-4/-2/0/2/4")
    parser.add_argument("--only", default="", help="comma list of fits to run (all, fold:0, loso:idsse)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--verbose", action="store_true", help="print each fit's progress")
    parser.add_argument("--no-row-cache", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    sources = args.sources.split(",")
    registry.check_trainable(sources)  # A19: never fit on a test-only source
    load = {s: dict(SOURCES[s]) if s in SOURCES else {} for s in sources}
    for item in args.load:
        source, pair = item.split(":", 1)
        load[source].update(evaluate.parse_params([pair]))
    groups = tuple(sorted(g for g in args.groups.split(",") if g))
    spec = parse_spec(args.spec, train.SPECS[args.kind])
    feat_spec = {"speed_span_s": learned.SPEED_SPAN_S}
    code = None if args.no_row_cache else code_digest()
    jobs = []
    for source in sources:
        for key in evaluate.parse_matches(source + ":all"):
            for i in dropout.draws(key, args.mix):
                opts = dict(load[source], **dropout.GRID[i])
                k = "{}|g{}".format(key, i)
                jobs.append((k, opts, groups, feat_spec,
                             row_path(k, opts, groups, feat_spec, code) if code else None))
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        loaded = pool.map(_rows, jobs)
    ROWS.update({k: r for k, r, _ in loaded})
    print("rows       {} row sets, {} grid points ({} targets), {} cached, {:.0f} s".format(
        len(ROWS), sum(len(r["y"]) for r in ROWS.values()),
        sum(int((r["y"] >= 0).sum()) for r in ROWS.values()), sum(h for *_, h in loaded),
        time.time() - started))

    p_team, p_common = (int(v) for v in next(iter(ROWS.values()))["p"])
    spec_out = dict(spec, n_in=2 * p_team + p_common, p_team=p_team, p_common=p_common,
                    stride_s=emission.STRIDE_S, **feat_spec)
    SETTINGS.update(kind=args.kind, spec=spec, verbose=args.verbose)
    keys = list(ROWS)
    matches = list(dict.fromkeys(k.split("|", 1)[0] for k in keys))
    fold_of = {}
    if args.folds:
        i = 0
        for source in dict.fromkeys(m.split(":", 1)[0] for m in matches):  # fit_learned.fold_groups
            own = sorted((m for m in matches if m.startswith(source + ":")),
                         key=lambda m: (zlib.crc32(("a14|" + m).encode()), m))
            for m in own:
                fold_of[m] = i % args.folds
                i += 1
    work = [("all", keys)]
    work += [("fold:{}".format(i), [k for k in keys if fold_of[k.split("|", 1)[0]] != i])
             for i in range(args.folds)]
    if len(sources) > 1:
        work += [("loso:" + s, [k for k in keys if not k.startswith(s + ":")]) for s in sources]
    if args.only:
        wanted = set(args.only.split(","))
        work = [w for w in work if w[0] in wanted]
    fit_started = time.time()
    with multiprocessing.get_context("fork").Pool(min(args.jobs, len(work))) as pool:
        fits = dict(pool.imap_unordered(_fit, work))

    def entry(fit):
        return {"model": emission.encode_model(args.kind, fit["model"]),
                "log_prior": emission.to_int(fit["log_prior"], emission.LEAF_BITS)}

    payload = {"kind": args.kind, "spec": spec_out, "groups": list(groups),
               "feature_hash": learned.feature_hash(groups),
               "fit": {"sources": sources, "mix": args.mix, "grid": list(dropout.GRID),
                       "load": load, "folds": args.folds,
                       "inner_share": train.INNER_SHARE,
                       "best": {n: f["best"] for n, f in sorted(fits.items())},
                       "valid_loss": {n: round(f["valid_loss"], 6) for n, f in sorted(fits.items())}},
               "fold_of": {m: i for m, i in sorted(fold_of.items())}}
    if "all" in fits:
        payload["all"] = entry(fits["all"])
    payload["folds"] = {n.split(":")[1]: entry(f) for n, f in sorted(fits.items()) if n.startswith("fold:")}
    payload["loso"] = {n.split(":")[1]: entry(f) for n, f in sorted(fits.items()) if n.startswith("loso:")}
    for name, fit in fits.items():  # refuse to write a file inference can't evaluate exactly
        emission.check_bounds(args.kind, fit["model"], spec_out["n_in"])
    out = args.out or DEFAULT_OUT[args.kind]
    emission.write_coefs(out, payload)
    for name, fit in sorted(fits.items()):
        print("fit        {:<16} {} {:>4}, inner valid loss {:.4f}, {} rows, {:.0f} s".format(
            name, "rounds" if args.kind == "boosted" else "epochs", fit["best"], fit["valid_loss"],
            fit["rows"], fit["seconds"]))
    print("fits       {} {} fits in {:.0f} s (total {:.0f} s)".format(
        len(fits), args.kind, time.time() - fit_started, time.time() - started))
    held = emission.heldout_path(out)
    print("wrote      {} ({} KB) and {} ({} KB)".format(
        os.path.relpath(out, evaluate.ROOT), os.path.getsize(out) // 1024,
        os.path.relpath(held, evaluate.ROOT), os.path.getsize(held) // 1024 if os.path.isfile(held) else 0))


if __name__ == "__main__":
    main()
