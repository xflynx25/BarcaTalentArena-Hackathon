import os
import subprocess
import sys

import numpy as np

from src import submission
from src.official import D, EVALUATION, H, PUBLIC_GT, RESULTS_DIR, X

N = 199319


def test_write_passes_official_validator(tmp_path):
    codes = np.full(N, X, dtype=np.int8)
    codes[11185:90652] = H
    codes[110808:190198] = D
    path = submission.write(str(tmp_path / "x.csv"), codes)
    assert submission.validate(path, N) is None
    assert np.array_equal(submission.read_codes(path), codes)
    result = subprocess.run([sys.executable, os.path.join(EVALUATION, "validate_submission.py"),
                             path], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_write_is_byte_stable(tmp_path):
    codes = np.random.default_rng(0).integers(0, 4, N).astype(np.int8)
    a = open(submission.write(str(tmp_path / "a.csv"), codes), "rb").read()
    b = open(submission.write(str(tmp_path / "b.csv"), codes), "rb").read()
    assert a == b and b"\r" not in a


def test_score_reproduces_published_public_numbers():
    """EVALUATION.md: baseline_smooth50 scores spellF1 0.3256 on the public window."""
    codes = submission.read_codes(os.path.join(RESULTS_DIR, "baseline_smooth50.csv"))
    assert submission.score(codes, PUBLIC_GT)["spellF1"] == 0.3256
    assert submission.compare(codes, os.path.join(RESULTS_DIR, "baseline_smooth50.csv")) == 0
