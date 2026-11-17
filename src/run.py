"""Produce a submission: load BAR-ATH, run a model, write, validate and sanity-check the CSV.

On the real match (owner's machine, in your own terminal, not through an AI assistant):

    export MTAH_DATA_DIR=$PWD/data/mtah2027
    .venv/bin/python -m src.run --out submissions/<team>.csv          # default: preset v5 (B30)
    .venv/bin/python -m src.run --preset v3 --out submissions/v3.csv  # any preset in src/presets.py
    .venv/bin/python -m src.run --model nearest --param window=50 --compare-official baseline_smooth50

On the synthetic bundle (anywhere, including cloud agents):

    .venv/bin/python scripts/make_synthetic_bar_ath.py
    .venv/bin/python -m src.run --data-dir data/synthetic/mtah2027 \
        --gt data/synthetic/mtah2027/synthetic_gt.csv --out outputs/synthetic.csv

It prints aggregates only (counts and scores), never rows. The public-window score is a check that
the pipeline is wired correctly. The rules forbid using it to choose between models or settings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

from src import submission
from src.ingest import bar_ath
from src.models import MODELS, build
from src.presets import DEFAULT, PRESETS, model_and_params
from src.official import PUBLIC_GT, RESULTS_DIR, ROOT


def parse_params(pairs: list[str]) -> dict:
    """["window=50", "ball_quiet=0.5", "extrapolated=false"] -> {"window": 50, "ball_quiet": 0.5,
    "extrapolated": False}. Values become int, float or bool (true/false in any case) when they
    parse as one, otherwise stay strings. Booleans matter: the string "False" is truthy (F50)."""
    params = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
            continue
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        params[key] = value
    return params


def set_params(model) -> dict:
    """The model's dataclass fields, without unset (None) ones, e.g. {"window": 50} rather than
    {"window_s": None, "window": 50}. For printing and run files."""
    return {k: v for k, v in dataclasses.asdict(model).items() if v is not None}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve_model(args, for_eval: bool = False) -> tuple[str, dict]:
    """(model name, preset params) from --model / --preset. --param is applied on top by the caller."""
    if args.preset:
        if args.model:
            raise SystemExit("give --preset or --model, not both")
        return model_and_params(args.preset, for_eval)
    if args.model:
        return args.model, {}
    return model_and_params(DEFAULT, for_eval)  # no flags: the default preset (v5 since B30)


def restart_rule_note(model, match) -> str | None:
    """B35: the restart rule (restart_rule_gain) reads BAR-ATH's windows from Smart Tagging's SET
    PIECES in Match.tags. Without the file (or if it doesn't read) Match.tags is None and the rule
    adds nothing: the same codes as gain 0, every time. Prints which case applies; None when the
    model doesn't use the rule."""
    if not getattr(model, "restart_rule_gain", 0):
        return None
    if match.tags is None:
        note = "off: no Smart Tagging file read, so no restart windows (same as gain 0)"
    else:
        note = "on: windows from Smart Tagging SET PIECES"
    print("restart    rule {}".format(note), file=sys.stderr)
    return note


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="bundle root (default: MTAH_DATA_DIR or official/data)")
    parser.add_argument("--model", choices=sorted(MODELS),
                        help="model block (default: the default preset, src/presets.py DEFAULT)")
    parser.add_argument("--preset", choices=list(PRESETS),
                        help="named candidate from src/presets.py (see scripts/models.py list)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="model parameter, repeatable, e.g. --param window=50")
    parser.add_argument("--out", default=os.path.join("submissions", "submission.csv"))
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="frame,label truth to sanity-check against (default: public window)")
    parser.add_argument("--compare-official", metavar="NAME",
                        help="count frames that differ from official/evaluation/results/NAME.csv")
    parser.add_argument("--record", metavar="PATH",
                        help="also write the printed summary as JSON (e.g. experiments/local/x.json)")
    args = parser.parse_args(argv)

    args.model, preset_params = resolve_model(args)
    model = build(args.model, **dict(preset_params, **parse_params(args.param)))
    params = set_params(model) if dataclasses.is_dataclass(model) else {}
    started = time.time()

    match = bar_ath.load(args.data_dir)  # Smart Tagging in Match.tags when the file reads (B28)
    rule = restart_rule_note(model, match)
    from src.models.cues import whistle  # V3b: the tactical video's sound, only if the cue is on
    heard = whistle.attach_run(model, match, args.data_dir)
    reads_ball2 = getattr(model, "reads_ball2", lambda: False)()
    if reads_ball2:  # V4: the broadcast recording, aligned with the tactical timeline
        from src import broadcast
        match = broadcast.attach(match, args.data_dir)
    codes = model.predict(match)
    path = submission.write(args.out, codes)

    summary = {
        "model": args.model,
        "params": params,
        "commit": git_commit(),
        "out": os.path.relpath(path, ROOT) if os.path.isabs(path) else path,
        "n_frames": match.n_frames,
        "validation": submission.validate(path, match.n_frames) or "OK",
        "prediction": submission.describe(codes, match.fps),
        "seconds": None,
    }
    summary["broadcast"] = "aligned" if reads_ball2 else "not used"
    if rule:
        summary["restart_rule"] = rule
    if heard:
        summary["whistle"] = heard
    print("model      {} {}".format(args.model, params))
    if reads_ball2:
        print("broadcast  aligned with the tactical timeline, its ball attached (src/align.py)")
    print("wrote      {}  ({} rows)".format(summary["out"], match.n_frames))
    print("validation {}".format(summary["validation"]))
    pred = summary["prediction"]
    print("labels     " + "  ".join("{} {}".format(k, v) for k, v in pred["counts"].items()))
    print("predicted  {} possessions over the whole file (ungraded frames too), mean {} s".format(
        pred["n_possessions"], pred["mean_possession_s"]))

    if args.gt:
        result = submission.score(codes, args.gt)
        summary["sanity_check"] = dict(result, gt=os.path.basename(args.gt))
        note = ("public 5-min window: a wiring check only, NOT for tuning or model choice"
                if os.path.abspath(args.gt) == os.path.abspath(PUBLIC_GT) else args.gt)
        print("sanity     spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  graded possessions "
              "{n_poss_pred} predicted vs {n_poss_true} true".format(**result))
        print("           ({})".format(note))

    if args.compare_official:
        reference = os.path.join(RESULTS_DIR, args.compare_official + ".csv")
        differing = submission.compare(codes, reference)
        summary["compare_official"] = {"name": args.compare_official, "frames_differing": differing}
        verdict = "IDENTICAL" if differing == 0 else "DIFFERENT"
        print("official   {}: {} ({} of {} frames differ)".format(
            args.compare_official, verdict, differing, match.n_frames))

    summary["seconds"] = round(time.time() - started, 1)
    print("time       {} s".format(summary["seconds"]))
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        with open(args.record, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print("recorded   {}".format(args.record))
    return summary


if __name__ == "__main__":
    main()
