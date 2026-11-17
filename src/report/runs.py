"""Read experiments/runs/*.json for the leaderboard and the dashboard.

Run files come in two shapes: src.evaluate's (config.params, config.load, metrics.per_match) and the
older hand-written ones (params inline in config, eval as a free-text label, per_match_spellF1).
Everything here reads both.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(ROOT, "experiments", "runs")

# Ball settings that make an external match look like BAR-ATH (findings F27, F30).
HIDDEN_KEYS = ("ball_missing", "ball_quiet")


def eval_label(run: dict) -> str:
    label = run.get("eval_label") or ", ".join(run.get("eval", [])) or "?"
    label = re.sub(r"(\w+):all \((\d+)\)", r"\1 x\2", label)  # older files: "skillcorner:all (20)"
    load = (run.get("config") or {}).get("load") or {}
    per_source = (run.get("config") or {}).get("source_load") or {}  # src.evaluate --sources (A14)
    items = ["{}={}".format(k, v) for k, v in sorted(load.items())]
    items += ["{}:{}={}".format(s, k, v) for s, opts in sorted(per_source.items())
              for k, v in sorted(opts.items())]
    if items:
        label += " [" + ", ".join(items) + "]"
    return label


def params_text(config: dict) -> str:
    params = config.get("params")
    if params is None:  # older run files: params inline next to "model"
        params = {k: v for k, v in config.items() if k not in ("model", "load", "mode")}
    return " ".join("{}={}".format(k, v) for k, v in params.items()) or "-"


def load_runs(runs_dir: str = RUNS_DIR) -> list[dict]:
    """Every run file with a spellF1, in file order, with its file name under "_file"."""
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as handle:
            run = json.load(handle)
        metrics = run.get("metrics") or {}
        if "spellF1" not in metrics:
            continue
        run["_file"] = os.path.basename(path)
        runs.append(run)
    return runs


def per_match(run: dict) -> dict[str, float]:
    """{match key: spellF1}, from either run file shape."""
    metrics = run.get("metrics") or {}
    if isinstance(metrics.get("per_match"), dict):
        return {k: float(v["spellF1"]) for k, v in metrics["per_match"].items() if "spellF1" in v}
    source = (run.get("eval") or ["?"])[0].split(":", 1)[0]
    return {"{}:{}".format(source, k): float(v)
            for k, v in (metrics.get("per_match_spellF1") or {}).items()}


def setting_tag(run: dict) -> str:
    """How the matches were shown to the model: 'BAR-ATH-like' (F30), 'ball 41% random' (F27),
    'cuts ungraded' or 'clean'."""
    load = (run.get("config") or {}).get("load") or {}
    if all(k in load for k in HIDDEN_KEYS):
        return "BAR-ATH-like"
    if "ball_missing" in load:
        return "ball 41% random"
    per_source = (run.get("config") or {}).get("source_load") or {}
    if load.get("cuts") == "ungraded" or any(o.get("cuts") == "ungraded" for o in per_source.values()):
        return "cuts ungraded"
    return "clean"


def sources(run: dict) -> list[str]:
    keys = run.get("eval") or []
    found = [k.split(":", 1)[0] for k in keys if ":" in k]
    if not found:  # very old files: a free-text label only
        found = re.findall(r"([a-z_]+)[:\s]", run.get("eval_label") or "")
    return sorted(set(found))


def commit_times(commits: list[str], cwd: str = ROOT) -> dict[str, str]:
    """{commit: ISO commit time} for the commits git knows; others are left out."""
    out = {}
    for commit in sorted(set(c for c in commits if c and re.fullmatch(r"[0-9a-f]{4,40}", c))):
        try:
            text = subprocess.run(["git", "show", "-s", "--format=%cI", commit], cwd=cwd,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if text:
            out[commit] = text
    return out


def run_time(run: dict, times: dict[str, str]) -> str:
    """When the run was made, as UTC 'YYYY-MM-DDTHH:MM:SSZ': its commit's time, else noon on the
    date in its file name."""
    stamp = times.get(run.get("commit", ""))
    if stamp:
        moment = datetime.datetime.fromisoformat(stamp).astimezone(datetime.timezone.utc)
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    match = re.match(r"(\d{4}-\d{2}-\d{2})", run.get("_file") or run.get("id") or "")
    return (match.group(1) if match else "1970-01-01") + "T12:00:00Z"
