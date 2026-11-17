import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pytest

from src import evaluate
from src.ingest import synthetic as synthetic_source
from src.models import MODELS
from src.models.nearest import NearestPlayer
from src.official import pm

KEYS = ["synthetic:0", "synthetic:1", "synthetic:2"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Synthetic bundles and the match cache go to a temp dir, never into data/."""
    monkeypatch.setattr(synthetic_source, "ROOT", str(tmp_path / "bundles"))
    monkeypatch.setattr(evaluate, "CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


@dataclass
class Memorise:
    """A model with fit(): predicts the training matches' most common live label everywhere,
    and records which matches it was trained on."""
    calls = []

    def fit(self, matches):
        Memorise.calls.append(sorted(m.match_id for m in matches))
        labels = np.concatenate([m.labels for m in matches])
        self.code = int(np.bincount(labels[labels < 2]).argmax())

    def predict(self, match):
        return np.full(match.n_frames, self.code, dtype=np.int8)


def test_parse_matches():
    assert evaluate.parse_matches("synthetic:all") == KEYS
    assert evaluate.parse_matches("synthetic:0,2, synthetic:0") == ["synthetic:0", "synthetic:2"]
    for bad in ("bar_ath:all", "0", "nosuchsource:1", "epts:x"):
        with pytest.raises(ValueError):
            evaluate.parse_matches(bad)


def test_describe_matches():
    assert evaluate.describe_matches(KEYS[:2]) == "synthetic:0, synthetic:1"
    assert evaluate.describe_matches(["a:1", "a:2", "a:3", "b:1"]) == "a x3 + b x1"


def test_scores_match_official_and_cache_is_exact(isolated):
    cold = evaluate.evaluate("nearest", [{"window": 25}], KEYS, cache=True, log=lambda *_: None)
    assert len(glob.glob(str(isolated / "cache" / "synthetic" / "*.npz"))) == 3
    warm = evaluate.evaluate("nearest", [{"window": 25}], KEYS, cache=True)
    fresh = evaluate.evaluate("nearest", [{"window": 25}], KEYS, cache=False)
    assert cold == warm == fresh

    match = synthetic_source.load("1")
    pred = NearestPlayer(window=25).predict(match)
    official = pm.evaluate(match.labels, pred)
    row = cold[0]["metrics"]["per_match"]["synthetic:1"]
    assert row["spellF1"] == round(official["spell_f1_mean"], 4)
    assert row["macroF1"] == round(official["macro_f1"], 4)
    mean = np.mean([r["spellF1"] for r in cold[0]["metrics"]["per_match"].values()])
    assert cold[0]["metrics"]["spellF1"] == round(mean, 4)


def test_grid_and_parallel_agree():
    configs = [{"window": 0}, {"window": 25}]
    serial = evaluate.evaluate("nearest", configs, KEYS, jobs=1)
    parallel = evaluate.evaluate("nearest", configs, KEYS, jobs=2)
    assert serial == parallel
    assert [r["params"] for r in serial] == configs
    assert serial[0]["metrics"]["spellF1"] != serial[1]["metrics"]["spellF1"]


def test_leave_one_match_out(monkeypatch):
    monkeypatch.setitem(MODELS, "memorise", Memorise)
    Memorise.calls = []
    results = evaluate.evaluate("memorise", [{}], KEYS)
    assert results[0]["mode"] == "leave-one-match-out"
    assert Memorise.calls == [[k for k in KEYS if k != held] for held in KEYS]

    Memorise.calls = []
    evaluate.evaluate("memorise", [{}], ["synthetic:2"], train_keys=KEYS[:2])
    assert Memorise.calls == [KEYS[:2]]


def test_bad_predictions_rejected():
    gt = np.zeros(10, dtype=np.int8)
    with pytest.raises(ValueError):
        evaluate.score(gt, np.zeros(9, dtype=np.int8))
    with pytest.raises(ValueError):
        evaluate.score(gt, np.full(10, 7, dtype=np.int8))


def test_main_writes_runs_and_leaderboard(isolated, capsys):
    runs = isolated / "experiments" / "runs"
    argv = ["--model", "nearest", "--param", "window=10,25", "--matches", "synthetic:all",
            "--runs-dir", str(runs), "--note", "test", "--jobs", "1"]
    results = evaluate.main(argv)
    evaluate.main(argv)  # same ids again: must not overwrite
    files = sorted(os.listdir(runs))
    assert len(files) == 4 and len(set(files)) == 4
    assert any(f.endswith("-nearest-window10-synthetic-0-synthetic-1-synthetic-2.json")
               for f in files)

    run = json.loads(open(results[1]["run_file"]).read())
    assert run["config"] == {"model": "nearest", "params": {"window": 25}, "load": {},
                             "mode": "no training"}
    assert run["eval"] == KEYS and run["train"] == [] and run["note"] == "test"
    assert set(run["metrics"]["per_match"]) == set(KEYS)

    board = (isolated / "experiments" / "LEADERBOARD.md").read_text()
    assert board.count("| nearest |") == 4
    first = board.index("window=25")
    assert first < board.index("window=10")  # window 25 scores higher on synthetic


def test_note_required_when_recording():
    with pytest.raises(SystemExit):
        evaluate.main(["--model", "nearest", "--matches", "synthetic:0"])


def test_leaderboard_reads_repo_runs(tmp_path):
    """The committed run files (old hand-written format included) all render."""
    import sys
    sys.path.insert(0, os.path.join(evaluate.ROOT, "scripts"))
    import make_leaderboard
    if not os.path.isdir(evaluate.RUNS_DIR):
        pytest.skip("no experiments/runs/ (the public export leaves run logs out)")
    runs = make_leaderboard.load_runs(evaluate.RUNS_DIR)
    text = make_leaderboard.render(runs)
    rows = sum(text.count("| {} |".format(name)) for name in MODELS)
    assert rows >= len(runs) > 0


def _rel(paths):
    return {os.path.relpath(p, evaluate.ROOT) for p in paths}


def test_cache_key_covers_only_the_sources_own_code():
    sc = _rel(evaluate.loader_files("skillcorner"))
    assert sc == {"src/ingest/skillcorner.py", "src/official.py", "src/schema.py"}
    metrica = _rel(evaluate.loader_files("metrica"))
    assert {"src/ingest/metrica.py", "src/ingest/epts.py", "src/schema.py"} <= metrica
    assert "src/ingest/skillcorner.py" not in metrica
    # synthetic goes through bar_ath, epts, submission and src/synthetic.py (transitively)
    synth = _rel(evaluate.loader_files("synthetic"))
    assert {"src/synthetic.py", "src/ingest/bar_ath.py", "src/ingest/epts.py",
            "src/submission.py"} <= synth
    assert "src/ingest/degrade.py" not in sc | metrica | synth
    assert evaluate.loader_hash("skillcorner", {}) != evaluate.loader_hash("idsse", {})
    assert evaluate.loader_hash("skillcorner", {}) != evaluate.loader_hash(
        "skillcorner", {"cuts": "ungraded"})


def test_ball_options_reuse_the_clean_cache(isolated):
    evaluate.load_match("synthetic:0", {"ball_missing": 0.6})
    assert len(glob.glob(str(isolated / "cache" / "synthetic" / "*.npz"))) == 1  # only the clean one


def test_fps_option_resamples_after_hiding_the_ball(isolated):
    native = evaluate.load_match("synthetic:0", {"ball_missing": 0.6})
    resampled = evaluate.load_match("synthetic:0", {"ball_missing": 0.6, "fps": 10})
    assert native.fps == 25 and resampled.fps == 10
    assert abs(resampled.n_frames - native.n_frames * 10 / 25) < 1
    hidden = np.isnan(native.ball[:, 0])
    nearest = np.minimum(np.floor(np.arange(resampled.n_frames) * 2.5 + 0.5).astype(int),
                         native.n_frames - 1)
    assert np.isnan(resampled.ball[hidden[nearest], 0]).all()  # hidden frames stay hidden
    assert np.array_equal(resampled.labels, native.labels[nearest])
    assert len(glob.glob(str(isolated / "cache" / "synthetic" / "*.npz"))) == 1


def test_score_switch_precision_recall():
    """B16: with fps, score() adds switch precision/recall at +/-1 s; summarise() averages them."""
    H, A = pm.LABELS.index("H"), pm.LABELS.index("A")
    gt = np.array([H] * 50 + [A] * 50 + [H] * 50, dtype=np.int8)
    assert "switch_precision" not in evaluate.score(gt, gt)
    perfect = evaluate.score(gt, gt, fps=25)
    assert perfect["switch_precision"] == 1.0 and perfect["switch_recall"] == 1.0
    late = gt.copy()
    late[50:100] = H  # misses the H->A and A->H switches entirely
    late[120:130] = A  # one switch pair, far from the true ones
    s = evaluate.score(gt, late, fps=25)
    assert s["switch_precision"] == 0.0 and s["switch_recall"] == 0.0
    flat = evaluate.score(gt, np.full_like(gt, H), fps=25)
    assert flat["switch_precision"] is None and flat["switch_recall"] == 0.0
    summary = evaluate.summarise({"a:1": perfect, "a:2": flat, "b:1": s})
    assert summary["switch_precision"] == 0.5  # mean of 1.0 and 0.0, None skipped
    assert summary["per_source_switch_precision"] == {"a": 1.0, "b": 0.0}
    json.dumps(summary)  # None, not NaN: run files stay valid JSON
