"""Synthetic labelled matches for the evaluation harness, so it can be tested without external data.

Each id is a seed. The bundle is the short (4,000-frame) fake BAR-ATH from src/synthetic.py, written
once to data/synthetic/eval/<seed>/ and read back through the real BAR-ATH loader, with the
simulation's labels attached. Scores on it say the code runs, nothing about football.
"""

from __future__ import annotations

import os

from src import submission, synthetic
from src.ingest import bar_ath
from src.official import ROOT as REPO_ROOT
from src.schema import Match

ROOT = os.path.join(REPO_ROOT, "data", "synthetic", "eval")
SEEDS = ("0", "1", "2")


def list_matches(root: str | None = None) -> list[str]:
    return list(SEEDS)


def load(match_id: str | int, root: str | None = None) -> Match:
    spec = synthetic.Spec.small(seed=int(match_id))
    folder = os.path.join(root or ROOT, str(match_id))
    gt_path = os.path.join(folder, "synthetic_gt.csv")
    if not os.path.isfile(gt_path):
        synthetic.write_bundle(folder, spec, broadcast=False)  # tactical only
    match = bar_ath.load(folder, n_frames=spec.n_frames, halves=list(spec.halves))
    match.match_id = "synthetic:{}".format(match_id)
    match.labels = submission.read_codes(gt_path)
    return match.validate()
