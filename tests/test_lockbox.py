"""A19: a test-only source never reaches training, tuning or selection by default, and a tranche
opens once. The stand-in source exists only inside these tests. The analysis scripts' side
(grid, tuner, fit_emission) is in test_lockbox_scripts.py (not exported)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import fit_learned  # noqa: E402

from src import evaluate, sources  # noqa: E402


def _install(monkeypatch):
    entry = sources.Source(
        "holdout", "holdout", kind="broadcast", labels="events", fps=25, weight=1.0,
        select=False, train=False, test_only=True, lockbox=("1", "2", "3"))
    monkeypatch.setitem(sources.SOURCES, "holdout", entry)
    monkeypatch.setitem(sources.TRANCHES, "holdout", {
        "explore": ("1",), "T1": ("2",), "T2": ("3",)})

    class _Loader:
        @staticmethod
        def list_matches(root=None):
            return ["1", "2", "3"]

        @staticmethod
        def load(*a, **k):
            raise AssertionError("loaded a match")

    monkeypatch.setattr(evaluate, "source_module", lambda source: _Loader)


def test_tranche_split_partitions_and_is_fixed():
    ids = [str(i) for i in range(10)]
    sizes = (("explore", 2), ("T1", 4), ("T2", 4))
    first = sources.tranche_split("holdout", ids, sizes)
    assert sources.tranche_split("holdout", list(reversed(ids)), sizes) == first
    flat = [m for t in first.values() for m in t]
    assert sorted(flat) == sorted(ids)
    assert [len(first[t]) for t in ("explore", "T1", "T2")] == [2, 4, 4]


def test_a_test_only_source_is_out_of_every_default_set(monkeypatch):
    _install(monkeypatch)
    entry = sources.get("holdout")
    assert entry.test_only and not entry.select and not entry.train
    for names in (sources.names(), sources.names("train"), sources.names(everything=True)):
        assert "holdout" not in names
    assert "holdout" in sources.names(everything=True, test_only=True)
    assert sources.is_lockbox("holdout", "2") and not sources.is_lockbox("idsse", "x")
    assert sources.tranche_of("holdout", "2") == "T1"


def test_parse_matches_leaves_lockbox_matches_out(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(sources.LockboxError):
        evaluate.parse_matches("holdout:all")
    with pytest.raises(sources.LockboxError):
        evaluate.parse_matches("holdout:3")  # named explicitly without its tranche
    with pytest.raises(sources.LockboxError):
        evaluate.parse_matches("holdout:3", lockbox="explore")  # 3 is in T2
    assert evaluate.parse_matches("holdout:all", lockbox="explore") == ["holdout:1"]
    assert evaluate.parse_matches("holdout:all", lockbox="T1") == ["holdout:2"]
    assert evaluate.parse_matches("holdout:all", lockbox=sources.LABELS_ONLY) == [
        "holdout:1", "holdout:2", "holdout:3"]
    with pytest.raises(ValueError):
        evaluate.parse_matches("holdout:all", lockbox="T9")


def test_fit_learned_refuses_a_test_only_source(monkeypatch):
    _install(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("loaded a match")
    monkeypatch.setattr(evaluate, "load_match", boom)
    assert "holdout" not in fit_learned.SOURCES
    with pytest.raises(sources.LockboxError):
        fit_learned.main(["--sources", "holdout"])
    with pytest.raises(sources.LockboxError):
        fit_learned.main(["--sources", "idsse,holdout"])


def test_training_on_a_lockbox_match_is_refused(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(sources.LockboxError):
        sources.drop_lockbox(["holdout:2"])
    assert sources.drop_lockbox(["idsse:a"]) == ["idsse:a"]
    keys = ["idsse:a", "holdout:2"]
    assert [k for k in keys if not sources.is_lockbox(*k.split(":", 1))] == ["idsse:a"]


def test_a_tranche_opens_once(monkeypatch, tmp_path):
    _install(monkeypatch)
    ledger = str(tmp_path / "lockbox.json")
    sources.open_tranche("explore", {"x": 1}, ledger)  # not a tranche: never recorded
    assert not os.path.exists(ledger)
    sources.open_tranche("T1", {"model": "v6"}, ledger)
    with pytest.raises(sources.LockboxError):
        sources.open_tranche("T1", {"model": "v7"}, ledger)
    sources.open_tranche("T2", {"model": "v6"}, ledger)
    with pytest.raises(ValueError):
        sources.open_tranche("T3", {}, ledger)
