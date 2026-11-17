"""Write, validate, describe and score submission files, using the official scorer's own functions.

Everything returned here is an aggregate (counts, scores), so it is safe to print from a run on the
real match data.
"""

from __future__ import annotations

import os

import numpy as np

from src.official import A, H, pm

LETTERS = np.array(pm.LABELS)


def write(path: str, codes: np.ndarray) -> str:
    """frame,label CSV, written by hand so the bytes are identical on every OS (no \\r\\n)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(LETTERS[codes]))
    return path


def read_codes(path: str) -> np.ndarray:
    return pm.label_codes(pm.read_label_csv(path))


def validate(path: str, n_frames: int) -> str | None:
    """None if the file passes the official checks, else the official one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_frames)


def describe(codes: np.ndarray, fps: int = 25) -> dict:
    """Label counts, possession count and mean possession length, like validate_submission.py."""
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    alive = int(np.sum((codes == H) | (codes == A)))
    return {
        "counts": {name: int(np.sum(codes == k)) for k, name in enumerate(pm.LABELS)},
        "n_possessions": len(spells),
        "mean_possession_s": round(alive / float(fps) / len(spells), 2) if len(spells) else 0.0,
    }


def score(codes: np.ndarray, gt_path: str) -> dict:
    """spellF1 (the ranking metric), macroF1 and possession counts against a frame,label truth."""
    gt = read_codes(gt_path)
    result = pm.evaluate(gt, codes)
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "n_poss_pred": result["n_pred_spells"],
        "n_poss_true": result["n_gt_spells"],
        "n_graded": result["n_graded"],
    }


def compare(codes: np.ndarray, reference_path: str) -> int:
    """How many frames differ from another submission file."""
    reference = read_codes(reference_path)
    if len(reference) != len(codes):
        raise ValueError("length {} vs reference {}".format(len(codes), len(reference)))
    return int(np.sum(reference != codes))
