"""Full-length synthetic match through every stage, exactly as the owner will run the real one."""

import hashlib
import os
import subprocess
import sys

import pytest

from src import check_data, run
from src.official import EVALUATION, games

pytestmark = pytest.mark.slow


def test_check_data_passes_on_synthetic(full):
    results = check_data.run_checks(full["dir"])
    assert [name for name, ok in results if not ok] == []


def test_official_loaders_read_synthetic(full):
    from kloppy import sportscode
    from starter import load_homography, load_smart_tagging

    matrices, error, valid = load_homography.load(data_dir=full["dir"])
    assert matrices.shape == (games.GAMES["BAR-ATH"]["n_video_frames"], 3, 3) and valid.all()
    assert len(load_smart_tagging.load(data_dir=full["dir"])) == 638
    events = sportscode.load(full["paths"]["smart_tagging"]).to_df()
    assert len(events) == 638 and "Team" in events.columns


def test_run_end_to_end_and_deterministic(full, tmp_path):
    outs = []
    for name in ("a", "b"):
        out = str(tmp_path / (name + ".csv"))
        summary = run.main(["--data-dir", full["dir"], "--gt", full["gt"], "--out", out,
                            "--model", "nearest", "--param", "window=50", "--record", str(tmp_path / (name + ".json"))])
        assert summary["validation"] == "OK"
        assert summary["sanity_check"]["spellF1"] > 0.1
        outs.append(hashlib.sha256(open(out, "rb").read()).hexdigest())
    assert outs[0] == outs[1]

    result = subprocess.run([sys.executable, os.path.join(EVALUATION, "validate_submission.py"),
                             out], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
