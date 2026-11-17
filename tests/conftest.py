import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import synthetic  # noqa: E402
from src.ingest import bar_ath  # noqa: E402


@pytest.fixture(scope="session")
def small(tmp_path_factory):
    """4,000-frame synthetic bundle: {dir, spec, gt, match}."""
    out = str(tmp_path_factory.mktemp("synthetic_small"))
    spec = synthetic.Spec.small(seed=0)
    info = synthetic.write_bundle(out, spec)
    match = bar_ath.load(out, n_frames=spec.n_frames, halves=spec.halves)
    return {"dir": out, "spec": spec, "gt": info["gt"], "match": match, "paths": info["paths"]}


@pytest.fixture(scope="session")
def full(tmp_path_factory):
    """Full-length (199,319-frame) synthetic bundle: {dir, gt}."""
    out = str(tmp_path_factory.mktemp("synthetic_full"))
    info = synthetic.write_bundle(out, synthetic.Spec(seed=0))
    return {"dir": out, "gt": info["gt"], "paths": info["paths"]}
