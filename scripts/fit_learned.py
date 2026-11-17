"""Fit the `learned` model's emissions on the external matches and write src/models/learned_coefs.json (B10).

    .venv/bin/python scripts/fit_learned.py                  # the committed file (v4), ~72 min on 4 cores (F100)
    .venv/bin/python scripts/fit_learned.py --folds 5 --out /tmp/f5.json           # 5 grouped folds, ~18 min
    .venv/bin/python scripts/fit_learned.py --groups touch --out /tmp/touch.json   # an ablation (B4)
    .venv/bin/python scripts/fit_learned.py --mix 0 --out /tmp/f30.json            # F30's one setting

--mix K (C6, default 5: the committed file, findings F66) trains each match on K settings from the dropout grid (src/dropout.py; drawn per match,
seeded by its key) instead of the one BAR-ATH-like setting; each draw weighs 1/K of the match. The
lomo / loso folds leave out every draw of the held-out match / source.

Loads every source in the registry (src/sources.py, A14: IDSSE, SkillCorner with cuts=ungraded and
onsets=out, Metrica; each with its `load` and `train_load` options) with the ball hidden (see --mix),
takes training rows from each (src/models/learned.py: training_rows), and fits:
  all    every match: what src.run uses on BAR-ATH
  lomo   one fit per match, without that match (leave-one-match-out across sources)
  loso   one fit per source, without that source (leave-one-source-out, findings F40)
Then score held out with the decoder costs of your choice, e.g.
    .venv/bin/python scripts/tune_decoder.py --model learned --param fold=lomo --param switch_cost_s=0.25,0.5,1
--load SOURCE:KEY=VALUE adds or overrides a load option for one source. Since v4 (B22) SkillCorner is
loaded with onsets=out by default (its train_load in src/sources.py) (B20, F84: its D onsets moved to the ball leaving the pitch;
src/ingest/skillcorner.py); --load skillcorner:onsets=label gives its own labels back.
--player-noise 0,0.25,0.5 (C7) adds position noise of --noise-kind (src/noise.py, tau --noise-tau)
to the players of each training row set, with sd drawn from the list per row set, seeded by its key.
--groups picks the optional feature groups (learned.GROUP_FEATURES; default learned.GROUPS, what the
committed file uses); the file records them, so `--param coef_file=<out>` scores any ablation.

--train-matches N (B30, the learning curve) caps every fit's training set at N matches, chosen within
each source in proportion to its share of that fit's matches (at least one each), in a fixed order (a
hash of the key); the loso fits are skipped. Score it with fold=lomo against the same --folds file
without the cap: every match is then scored by a fit that never saw it, trained on N matches vs all.

--folds K (A14) replaces the per-match fits with K fits, each without one group of matches (grouped
by match, stratified by source, assigned by a hash of the key): every match is still scored by a fit
that never saw it (`fold=lomo` reads it the same way), but a refit on ~100 matches needs K + 1 + the
loso fits instead of ~100. The fit step dominates the time (F100), so this is what makes big refits
fit in a session. Default 0: one fit per match, as before.

Training rows are cached on disk (A14) in data/cache/rows/ (gitignored), one file per row set, keyed
by the match, its load options (dropout setting included), the feature groups and their hash, the
stride, the player noise, and a hash of the code that makes the rows (the loader and learned.py with
every src/ module they import, degrade.py, noise.py, src/evaluate.py). A refit with warm rows only
fits. --no-row-cache skips it.

Per-source changes to the training rows (A18, findings F150-F152), all off by default, applied to the
rows after loading (the row cache is unaffected) and recorded in the file's "fit" block:
  --source-weight SOURCE=W   the source weighs W relative to the others' 1 (default: every source the
                             same, each match the same within its source; weights renormalised)
  --source-rows SOURCE=missing   keep only that source's rows with the ball missing
  --source-zero SOURCE=GROUP[+GROUP...]   zero that source's columns of fitted optional groups
                             (touch since A18; any group since B30, e.g. soccertrack=touch+impute;
                             repeatable): its rows then teach those groups' weights nothing (a missing
                             value, as a hidden ball's distance hats are zero)
Row sets with no graded frames (SoccerTrack labels=status on 2 matches) are skipped.

Smart Tagging (B32's recipe, moved here from scripts/check_smart_tags.py in B30): with the "tags"
group, --tag-mix gives every row set --draws tag conditions drawn from the mix (learned.TAG_CONDITIONS:
stand-ins at noise n0-n4, the unthinned oracle, SkillCorner's own phases "sc" on SkillCorner only,
"none" = whole-tag dropout), interleaved over its rows; --flip is the share of conditions whose
defending codes name the other team before --tag-convention (auto: smart_tags.canonical) reads them.
B32's pick (findings F132): --tag-mix n1:1,n2:1,n3:1,sc:1,none:1 --draws 4 --flip 0.5. The tags
columns may sit anywhere among the groups (learned.group_layout), so "impute" and "tags" combine.

Deterministic: no randomness, fixed row sampling, coefficients rounded to 6 decimals. The JSON is
committed, so src.run never refits. Prints a short summary only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import zlib

# one BLAS thread per worker: the fits run in parallel, and threads would only contend. Set before
# numpy is imported (anywhere), or BLAS starts one thread per core in every worker.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import evaluate  # noqa: E402
from src import dropout, noise  # noqa: E402
from src import sources as registry  # noqa: E402
from src.error_analysis import HIDDEN  # noqa: E402
from src.models import learned  # noqa: E402

# Fit load options per source, from the registry (SkillCorner: cuts=ungraded, onsets=out since v4, F84)
SOURCES = {s: registry.load_opts(s, train=True) for s in registry.names(everything=True)}
ROW_CACHE = os.path.join(evaluate.ROOT, "data", "cache", "rows")
# code that makes a row set, beyond the source's loader (which evaluate.loader_hash covers)
ROW_CODE = [os.path.join(evaluate.ROOT, "src", *p) for p in
            (("models", "learned.py"), ("ingest", "degrade.py"), ("noise.py",), ("evaluate.py",))]


def noise_sd(key: str, sds: list[float]) -> float:
    """The player-noise sd for one row set (C7), deterministic: seeded by its key alone."""
    return sds[zlib.crc32(("c7|" + key).encode()) % len(sds)] if sds else 0.0


def code_files() -> list[str]:
    """ROW_CODE and every src/ module they import, transitively."""
    todo, seen = list(ROW_CODE), set()
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo.extend(evaluate.src_imports(path) - seen)
    return sorted(seen)


def row_hash(key: str, opts: dict, groups: tuple, noise_spec: tuple, code: str, tagspec=None) -> str:
    """Cache key of one row set: everything its rows depend on (see the module docstring)."""
    match_key = key.split("|", 1)[0]
    spec = {"key": key, "opts": opts, "groups": list(groups), "noise": list(noise_spec),
            "feature_hash": learned.feature_hash(groups), "stride_s": learned.FIT_STRIDE_S,
            "loader": evaluate.loader_hash(match_key.split(":", 1)[0], opts), "code": code}
    if tagspec:  # B30: the tag conditions (none: the hash stays what it was before them)
        spec["tags"] = [[list(c) for c in tagspec[0]], tagspec[1]]
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def row_path(key: str, digest: str) -> str:
    source, rest = key.split(":", 1)
    return os.path.join(ROW_CACHE, source, "{}.{}.npz".format(rest.replace("|", "_"), digest))


def code_digest() -> str:
    digest = hashlib.sha1()
    for path in code_files():
        digest.update(os.path.relpath(path, evaluate.ROOT).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def _rows(job):
    key, opts, groups, (kind, sd, tau), tagspec, cache_path = job
    if cache_path and os.path.isfile(cache_path):
        with np.load(cache_path) as saved:
            return key, (tuple(saved["b{}".format(i)] for i in range(3)), saved["y"]), True
    match = evaluate.load_match(key.split("|", 1)[0], opts)
    if sd:
        match = noise.add_noise(match, kind, sd, tau, zlib.crc32(("c7n|" + key).encode()))
    if tagspec:  # B32's tag mix: one condition per interleaved slice of the rows
        rows = learned.mixed_rows(match, groups, [tuple(c) for c in tagspec[0]], tagspec[1])
    else:
        rows = learned.training_rows(match, groups=groups)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        partial = "{}.{}.partial.npz".format(cache_path, os.getpid())
        np.savez(partial, y=rows[1], **{"b{}".format(i): b for i, b in enumerate(rows[0])})
        os.replace(partial, cache_path)
    return key, rows, False


ROWS = {}  # set before the fit pool forks, so workers share it instead of pickling it


def fold_groups(matches: list[str], k: int) -> dict[str, int]:
    """{match key: fold index}: within each source, matches ordered by a hash of their key and dealt
    round-robin, continuing across sources, so every fold mixes sources. Deterministic."""
    out, i = {}, 0
    for source in dict.fromkeys(m.split(":", 1)[0] for m in matches):
        own = sorted((m for m in matches if m.startswith(source + ":")),
                     key=lambda m: (zlib.crc32(("a14|" + m).encode()), m))
        for m in own:
            out[m] = i % k
            i += 1
    return out


SOURCE_WEIGHT: dict[str, float] = {}  # --source-weight (A18); empty = learned.fit as is


def touch_columns() -> slice:
    """The "touch" group's columns in a team block: right after the base team block (learned.features)."""
    base = 4 * len(learned.DIST_KNOTS) + len(learned.MARGIN_KNOTS) + len(learned.CONTROL_KNOTS) + len(
        learned.COUNT_KNOTS)
    return slice(base, base + 4 + len(learned.SINCE_KNOTS) + len(learned.LEAD_KNOTS))


def parse_zero(specs: list[str]) -> dict:
    """["soccertrack=touch+impute", "x=duel"] -> {"soccertrack": "impute+touch", "x": "duel"}
    (repeatable per source; groups sorted, so the fit block is the same however they were given)."""
    out: dict = {}
    for spec in specs:
        source, groups = spec.split("=", 1)
        out.setdefault(source, set()).update(g for g in groups.split("+") if g)
    return {s: "+".join(sorted(g)) for s, g in out.items()}


def adjust_rows(rows: dict, keep: dict, zero: dict, groups: tuple) -> dict:
    """--source-rows / --source-zero applied to {row set key: (blocks, y)}; other sources untouched."""
    layout = learned.group_layout(groups) if zero else {}
    for source, spec in zero.items():
        for g in spec.split("+"):
            if g not in groups:
                raise ValueError("--source-zero {}={}: {!r} is not a fitted group".format(source, spec, g))
    out = {}
    for key, (blocks, y) in rows.items():
        source = key.split(":", 1)[0]
        if keep.get(source) == "missing":  # common column 1: the ball missing now
            sel = blocks[2][:, 1] == 1
            blocks, y = tuple(b[sel] for b in blocks), y[sel]
        if source in zero:
            blocks = tuple(b.copy() for b in blocks)
            for g in zero[source].split("+"):
                team, common = layout[g]
                blocks[0][:, team] = 0
                blocks[1][:, team] = 0
                blocks[2][:, common] = 0
        out[key] = (blocks, y)
    return out


def weighted_fit(rows: dict, source_weight: dict) -> dict:
    """learned.fit with source s weighing source_weight.get(s, 1) relative to the others."""
    keys = sorted(rows)
    sources = sorted({k.split(":", 1)[0] for k in keys})
    share = {s: source_weight.get(s, 1.0) for s in sources}
    total = sum(share.values())
    per_source = {s: sum(k.startswith(s + ":") for k in keys) for s in sources}
    w = np.concatenate([np.full(len(rows[k][1]), share[k.split(":", 1)[0]] / total
                                / (per_source[k.split(":", 1)[0]] * len(rows[k][1]))) for k in keys])
    blocks = tuple(np.concatenate([rows[k][0][i] for k in keys]) for i in range(3))
    y = np.concatenate([rows[k][1] for k in keys])
    theta = learned.fit_logit(blocks, y, w)
    prior = np.array([w[y == k].sum() for k in range(3)])
    return {"theta": [round(float(v), 6) for v in theta],
            "log_prior": [round(float(v), 6) for v in np.log(prior / prior.sum())],
            "train": keys}


def cap_matches(matches: list[str], n: int) -> list[str]:
    """At most n of these matches (B30's learning curve): per source in proportion to its share,
    at least one each, the first in a fixed hash order. Deterministic."""
    if n <= 0 or n >= len(matches):
        return list(matches)
    by_source: dict = {}
    for m in matches:
        by_source.setdefault(m.split(":", 1)[0], []).append(m)
    keep = []
    for source, own in by_source.items():
        take = max(1, int(round(n * len(own) / len(matches))))
        keep += sorted(own, key=lambda m: (zlib.crc32(("b30lc|" + m).encode()), m))[:take]
    return sorted(keep)


def _fit(job):
    name, keys = job
    part = {k: ROWS[k] for k in keys}
    return name, weighted_fit(part, SOURCE_WEIGHT) if SOURCE_WEIGHT else learned.fit(part)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(registry.names("train")))
    parser.add_argument("--clean", action="store_true", help="fit with the ball kept instead")
    parser.add_argument("--mix", type=int, default=5,
                        help="train each match on K dropout grid points (C6); 0 = F30's one setting")
    parser.add_argument("--groups", default=",".join(learned.GROUPS),
                        help="optional feature groups, comma-separated ('' = none)")
    parser.add_argument("--load", action="append", default=[], metavar="SOURCE:KEY=VALUE",
                        help="extra load option for one source (repeatable)")
    parser.add_argument("--player-noise", default="",
                        help="comma list of player-noise sd (m), one drawn per row set (C7)")
    parser.add_argument("--noise-kind", default="players", choices=noise.KINDS)
    parser.add_argument("--noise-tau", type=float, default=5.0)
    parser.add_argument("--out", default=learned.COEF_FILE)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-row-cache", action="store_true", help="recompute every training row")
    parser.add_argument("--folds", type=int, default=0,
                        help="K grouped folds instead of one fit per held-out match (0 = per match)")
    parser.add_argument("--source-weight", action="append", default=[], metavar="SOURCE=W",
                        help="relative weight of one source (A18; default 1 each)")
    parser.add_argument("--source-rows", action="append", default=[], metavar="SOURCE=missing",
                        help="keep only that source's rows with the ball missing (A18)")
    parser.add_argument("--source-zero", action="append", default=[], metavar="SOURCE=GROUP[+GROUP]",
                        help="zero that source's columns of fitted groups, e.g. soccertrack=touch+impute")
    parser.add_argument("--train-matches", type=int, default=0,
                        help="B30 learning curve: at most N training matches per fit (no loso fits)")
    parser.add_argument("--tag-mix", default="",
                        help="B32: tag conditions and weights, e.g. " + learned.TAG_MIX_B)
    parser.add_argument("--draws", type=int, default=1, help="tag conditions per row set (interleaved)")
    parser.add_argument("--flip", type=float, default=0.0,
                        help="share of tag conditions with the defending codes' Team flipped")
    parser.add_argument("--tag-convention", default="auto", choices=learned.TAG_CONVENTIONS)
    args = parser.parse_args(argv)

    started = time.time()
    sources = args.sources.split(",")
    registry.check_trainable(sources)  # A19: never fit on a test-only source
    groups = tuple(sorted(g for g in args.groups.split(",") if g))
    load = {s: dict(SOURCES[s]) if s in SOURCES else {} for s in sources}
    for spec in args.load:
        source, pair = spec.split(":", 1)
        load[source].update(evaluate.parse_params([pair]))
    sds = [float(v) for v in args.player_noise.split(",") if v]
    jobs = []
    for source in sources:
        for key in evaluate.parse_matches(source + ":all"):
            if args.mix:  # one row set per draw, keyed "source:id|g<i>" (the source prefix stays)
                jobs += [("{}|g{}".format(key, i), dict(load[source], **dropout.GRID[i]), groups)
                         for i in dropout.draws(key, args.mix)]
            else:
                jobs.append((key, dict(load[source], **({} if args.clean else HIDDEN)), groups))
    jobs = [j + ((args.noise_kind, noise_sd(j[0], sds), args.noise_tau),) for j in jobs]
    mix = learned.parse_tag_mix(args.tag_mix) if args.tag_mix else {}
    if mix and "tags" not in groups:
        raise ValueError("--tag-mix needs the tags group in --groups")
    jobs = [j + ((learned.draw_tag_conditions(j[0], mix, args.draws, args.flip), args.tag_convention)
                 if mix else None,) for j in jobs]
    code = None if args.no_row_cache else code_digest()
    jobs = [j + (row_path(j[0], row_hash(*j[:4], code, j[4])) if code else None,) for j in jobs]
    with multiprocessing.get_context().Pool(args.jobs) as pool:
        loaded = pool.map(_rows, jobs)
    rows = {key: r for key, r, _ in loaded}
    keep = dict(spec.split("=", 1) for spec in args.source_rows)
    zero = parse_zero(args.source_zero)
    if keep or zero:
        rows = adjust_rows(rows, keep, zero, groups)
    empty = [k for k, r in rows.items() if not len(r[1])]
    rows = {k: r for k, r in rows.items() if len(r[1])}
    SOURCE_WEIGHT.clear()
    SOURCE_WEIGHT.update({s: float(w) for s, w in (spec.split("=", 1) for spec in args.source_weight)})
    for s in list(keep) + list(zero) + list(SOURCE_WEIGHT):
        if s not in sources:
            raise ValueError("{} is not among --sources".format(s))
    print("rows       {} row sets, {} rows, {} from the row cache, {:.0f} s".format(
        len(rows), sum(len(r[1]) for r in rows.values()), sum(hit for *_, hit in loaded),
        time.time() - started) + (", {} empty skipped".format(len(empty)) if empty else ""))
    fit_started = time.time()

    ROWS.update(rows)
    matches = list(dict.fromkeys(k.split("|", 1)[0] for k in rows))
    work = [("all", list(rows))]
    groups_of = fold_groups(matches, args.folds) if args.folds else {}
    if args.folds:
        work += [("fold:{}".format(i), [k for k in rows if groups_of[k.split("|", 1)[0]] != i])
                 for i in range(args.folds)]
    else:
        work += [("lomo:" + key, [k for k in rows if k.split("|", 1)[0] != key]) for key in matches]
    if len(sources) > 1 and not args.train_matches:
        work += [("loso:" + s, [k for k in rows if not k.startswith(s + ":")]) for s in sources]
    if args.train_matches:  # the learning curve: each fit on at most N of its matches
        capped = []
        for name, keys in work:
            kept = set(cap_matches(list(dict.fromkeys(k.split("|", 1)[0] for k in keys)),
                                   args.train_matches))
            capped.append((name, [k for k in keys if k.split("|", 1)[0] in kept]))
        work = capped
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        fits = dict(pool.map(_fit, work))
    for key, i in groups_of.items():  # each match gets its fold's fit, under "lomo" as before
        fits["lomo:" + key] = dict(fits["fold:{}".format(i)])
    out = {"feature_hash": learned.feature_hash(groups), "features": learned.FEATURES,
           "fit": {"sources": sources,
                   "setting": ({"mix": args.mix, "grid": list(dropout.GRID)} if args.mix
                               else "clean" if args.clean else HIDDEN),
                   "load": load,
                   "stride_s": learned.FIT_STRIDE_S, "l2": learned.L2,
                   **({"folds": args.folds, "fold_of": groups_of} if args.folds else {}),
                   **({"train_matches": args.train_matches,
                       "trained_on": {n: sorted(dict.fromkeys(k.split("|", 1)[0] for k in keys))
                                      for n, keys in work}} if args.train_matches else {}),
                   **({"source_weight": SOURCE_WEIGHT} if SOURCE_WEIGHT else {}),
                   **({"source_rows": keep} if keep else {}),
                   **({"source_zero": zero} if zero else {}),
                   **({"tags": {"mix": mix, "draws": args.draws, "flip": args.flip,
                                "convention": args.tag_convention, "stand_in": "box_v2"}}
                      if mix else {}),
                   **({"player_noise": {"kind": args.noise_kind, "sd": sds, "tau_s": args.noise_tau}}
                      if sds else {})},
           "all": fits["all"],
           "lomo": {k[5:]: v for k, v in fits.items() if k.startswith("lomo:")},
           "loso": {k[5:]: v for k, v in fits.items() if k.startswith("loso:")}}
    if groups:  # B10's file has no groups key; keep it byte-identical when none are used
        out["groups"] = list(groups)
        out["group_features"] = {g: learned.GROUP_FEATURES[g] for g in groups}
    for fold in ("lomo", "loso"):
        for value in out[fold].values():
            del value["train"]  # implied by the key; keeps the file small
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    prior = [round(float(v), 3) for v in out["all"]["log_prior"]]
    print("fits       {} ({} lomo, {} loso), groups {}, {} weights, log prior H/A/D {}, {:.0f} s "
          "(total {:.0f} s)".format(len(work), len(out["lomo"]), len(out["loso"]),
                                    ",".join(groups) or "none", len(out["all"]["theta"]), prior,
                                    time.time() - fit_started, time.time() - started))
    print("wrote      {}".format(os.path.relpath(args.out, evaluate.ROOT)))


if __name__ == "__main__":
    main()
