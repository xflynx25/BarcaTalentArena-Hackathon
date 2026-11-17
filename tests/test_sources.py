"""The source registry (A14): the fit reads it, results don't move, the row cache is exact.
The analysis scripts' side (grid, tuner, check_labels) is in test_check_labels.py (not exported)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import fit_learned  # noqa: E402

from src import evaluate, sources  # noqa: E402
from src.ingest import synthetic as synthetic_source  # noqa: E402

# what the scripts hard-coded before A14: the registry must give exactly these
OLD_FIT = {"idsse": {}, "skillcorner": {"cuts": "ungraded", "onsets": "out"}, "metrica": {}}
OLD_SCORE = {"idsse": {}, "skillcorner": {"cuts": "ungraded"}, "metrica": {}}
OLD_WEIGHTS = {"idsse": 0.25, "skillcorner": 0.5, "metrica": 0.25}  # C6's, now sources.C6_WEIGHTS


def test_registry_is_valid_and_matches_the_old_tables():
    """v4's refit must not move (C12 split the sets): its training set and load options are A14's."""
    sources.validate()
    assert sources.names("train") == ["idsse", "skillcorner", "metrica"]
    assert {s: fit_learned.SOURCES[s] for s in sources.names("train")} == OLD_FIT
    assert list(fit_learned.SOURCES)[:3] == list(OLD_FIT)  # order decides the fit's row order
    assert {s: sources.load_opts(s) for s in OLD_SCORE} == OLD_SCORE
    assert sources.C6_WEIGHTS == OLD_WEIGHTS
    assert sources.weights(["idsse", "metrica"]) == {"idsse": 0.5, "metrica": 0.5}
    assert {s.kind for s in sources.SOURCES.values()} <= set(sources.KINDS)
    with pytest.raises(ValueError):
        sources.get("bar_ath")


def test_selection_scores_all_five_and_training_keeps_v4s_three():
    """C12: the wave-1 sources (A15, A17) are selected on, not trained on (B30 / A18 decide)."""
    every = sources.names(everything=True)
    assert sources.names() == sources.names("select") == every
    assert set(every) - set(sources.names("train")) == {"skillcorner2020", "soccertrack"}
    assert fit_learned.SOURCES["skillcorner2020"] == {"cuts": "ungraded", "onsets": "out"}
    # A18 (F151-F152): scored with its grid-stepped ball smoothed, trained on the shipped ball
    assert sources.load_opts("soccertrack") == {"ball": "smooth"}
    assert sources.load_opts("soccertrack", train=True) == {"ball": "shipped"}
    with pytest.raises(ValueError):
        sources.names("default")


def test_weights_follow_the_written_principles():
    """decisions.md (C12): every source counts the same; one whose input is rebuilt from the
    annotations its labels come from (SoccerTrack's ball, F108) counts half."""
    raw = {n: s.weight for n, s in sources.SOURCES.items()}
    assert raw == {"idsse": 1.0, "skillcorner": 1.0, "metrica": 1.0, "skillcorner2020": 1.0,
                   "soccertrack": 0.5}
    w = sources.weights()
    assert sum(w.values()) == pytest.approx(1)
    assert w["idsse"] == w["skillcorner"] == w["metrica"] == w["skillcorner2020"] == pytest.approx(2 / 9)
    assert sources.weights(sources.names("train")) == pytest.approx(
        {"idsse": 1 / 3, "skillcorner": 1 / 3, "metrica": 1 / 3})


def test_load_opts_are_fresh_copies():
    opts = sources.load_opts("skillcorner")
    opts["cuts"] = "keep"
    assert sources.load_opts("skillcorner") == {"cuts": "ungraded"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic_source, "ROOT", str(tmp_path / "bundles"))
    monkeypatch.setattr(evaluate, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fit_learned, "ROW_CACHE", str(tmp_path / "rows"))
    return tmp_path


def test_evaluate_sources_applies_each_sources_defaults(isolated, monkeypatch):
    monkeypatch.setitem(sources.SOURCES, "synthetic", sources.Source(
        "synthetic", "synthetic", kind="optical", labels="frame", fps=25, weight=1.0,
        load={"ball_missing": 0.9}))
    by_sources = evaluate.main(["--model", "nearest", "--sources", "synthetic", "--no-record",
                                "--jobs", "1", "--quiet"])
    by_matches = evaluate.main(["--model", "nearest", "--matches", "synthetic:all", "--no-record",
                                "--load", "ball_missing=0.9", "--jobs", "1", "--quiet"])
    plain = evaluate.main(["--model", "nearest", "--matches", "synthetic:all", "--no-record",
                           "--jobs", "1", "--quiet"])
    assert by_sources[0]["metrics"] == by_matches[0]["metrics"]
    assert by_sources[0]["metrics"] != plain[0]["metrics"]
    with pytest.raises(SystemExit):
        evaluate.main(["--model", "nearest", "--sources", "synthetic", "--matches", "synthetic:0",
                       "--no-record"])


def test_row_cache_gives_the_same_fit(isolated, capsys):
    args = ["--sources", "synthetic", "--mix", "1", "--jobs", "2"]
    first, second, fresh = (str(isolated / n) for n in ("a.json", "b.json", "c.json"))
    fit_learned.main(args + ["--out", first])
    assert "0 from the row cache" in capsys.readouterr().out
    fit_learned.main(args + ["--out", second])
    assert "3 from the row cache" in capsys.readouterr().out
    fit_learned.main(args + ["--out", fresh, "--no-row-cache"])
    with open(first, "rb") as a, open(second, "rb") as b, open(fresh, "rb") as c:
        assert a.read() == b.read() == c.read()
    assert len(os.listdir(isolated / "rows" / "synthetic")) == 3


def test_row_hash_changes_with_what_the_rows_depend_on():
    base = ("synthetic:0|g1", {"ball_missing": 0.3}, ("touch",), ("players", 0.0, 5.0), "code")
    digest = fit_learned.row_hash(*base)
    assert digest == fit_learned.row_hash(*base)
    for i, other in ((0, "synthetic:0|g2"), (1, {"ball_missing": 0.41}), (2, ()),
                     (3, ("players", 0.25, 5.0)), (4, "other code")):
        changed = list(base)
        changed[i] = other
        assert fit_learned.row_hash(*changed) != digest
    files = {os.path.relpath(p, evaluate.ROOT) for p in fit_learned.code_files()}
    assert {"src/models/learned.py", "src/ingest/degrade.py", "src/shape.py"} <= files


def test_grouped_folds_hold_each_match_out(isolated):
    out = str(isolated / "folds.json")
    fit_learned.main(["--sources", "synthetic", "--mix", "1", "--jobs", "2", "--folds", "2",
                      "--out", out])
    with open(out) as handle:
        coefs = json.load(handle)
    fold_of = coefs["fit"]["fold_of"]
    assert sorted(fold_of) == sorted(coefs["lomo"]) and set(fold_of.values()) == {0, 1}
    by_fold = {}
    for key, i in fold_of.items():
        by_fold.setdefault(i, []).append(coefs["lomo"][key]["theta"])
    assert all(t == ts[0] for ts in by_fold.values() for t in ts)
    assert by_fold[0][0] != by_fold[1][0] != coefs["all"]["theta"]


def test_fold_groups_are_stratified_and_deterministic():
    keys = ["a:{}".format(i) for i in range(5)] + ["b:{}".format(i) for i in range(4)]
    groups = fit_learned.fold_groups(keys, 3)
    assert {groups[k] for k in keys if k.startswith("b:")} == {0, 1, 2}  # each source spread out
    assert groups == fit_learned.fold_groups(keys, 3)
    assert sorted(list(groups.values()).count(i) for i in range(3)) == [3, 3, 3]


def test_held_out_folds_use_the_all_fit_on_a_source_the_fit_never_saw(tmp_path):
    """C12: lomo / loso on a source outside the fit (e.g. SoccerTrack for v4) is the all fit, which
    is held out by source; a trained source without its fold entry still fails loudly."""
    from types import SimpleNamespace

    from src.models.learned import LearnedDecoder, feature_hash
    path = tmp_path / "coefs.json"
    path.write_text(json.dumps({"all": "ALL", "lomo": {"a:0": "A0"}, "loso": {"a": "A"},
                                "fit": {"sources": ["a"]}, "groups": [],
                                "feature_hash": feature_hash(())}))
    for fold, want in (("lomo", "A0"), ("loso", "A"), ("all", "ALL")):
        model = LearnedDecoder(fold=fold, coef_file=str(path))
        assert model.coefs(SimpleNamespace(match_id="a:0")) == want
        assert model.coefs(SimpleNamespace(match_id="new:3")) == "ALL"
    with pytest.raises(KeyError):
        LearnedDecoder(fold="lomo", coef_file=str(path)).coefs(SimpleNamespace(match_id="a:1"))
