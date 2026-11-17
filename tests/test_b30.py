"""B30: the integration refit's plumbing (synthetic data only): where each feature group's columns
sit (learned.group_layout), --source-zero for any group, the tag mix with groups after "tags", and
the "a source the fit never saw gets the fit on everything" fallbacks of the imputer and `bigger`."""

import importlib.util
import os

import numpy as np
import pytest

from src import ball_impute
from src.models import emission, learned
from test_smart_tags import toy_match

_spec = importlib.util.spec_from_file_location(
    "fit_learned", os.path.join(os.path.dirname(__file__), "..", "scripts", "fit_learned.py"))
fit_learned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_learned)


def _gappy(match):
    match.ball[1000:1300] = np.nan  # a 30 s gap, so "impute" and "shapem" speak
    return match


@pytest.mark.parametrize("groups", [("shapem", "touch"), ("impute", "shapem", "touch"),
                                    ("impute", "shapem", "tags", "touch"), ("gap", "duel", "touch")])
def test_group_layout_is_where_features_put_each_group(groups):
    match = _gappy(toy_match(seconds=200))
    full = learned.features(match, groups, tags=[])
    layout = learned.group_layout(groups)
    assert set(layout) == set(groups)
    for g, (team, common) in layout.items():
        rest = tuple(x for x in groups if x != g)
        less = learned.features(match, rest, tags=[])
        keep_t = np.ones(full[0].shape[1], bool)
        keep_t[team] = False
        keep_c = np.ones(full[2].shape[1], bool)
        keep_c[common] = False
        for i, keep in ((0, keep_t), (1, keep_t), (2, keep_c)):
            np.testing.assert_array_equal(full[i][:, keep], less[i])
    assert learned.group_layout(("touch",))["touch"][0] == fit_learned.touch_columns()


def test_mixed_rows_put_the_tags_before_impute():
    match = _gappy(toy_match(seconds=200))
    groups = ("impute", "shapem", "tags", "touch")
    blocks, y = learned.mixed_rows(match, groups, [("n1", 7, True)])
    match.tags = learned.tags_for(match, "standin", 1, 7, True)
    ref, ref_y = learned.training_rows(match, groups=groups)
    assert np.array_equal(y, ref_y)
    for a, b in zip(blocks, ref):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    team, common = learned.group_layout(groups)["impute"]
    assert ref[0][:, team].any() and common.stop == ref[2].shape[1]  # impute comes after the tags


def test_source_zero_takes_any_fitted_group_and_only_that_source():
    match = _gappy(toy_match(seconds=200))
    groups = ("impute", "shapem", "touch")
    rows = learned.training_rows(match, groups=groups)
    data = {"st:1|g0": rows, "idsse:1|g0": rows}
    zero = fit_learned.parse_zero(["st=touch", "st=impute"])
    assert zero == {"st": "impute+touch"} == fit_learned.parse_zero(["st=touch+impute"])
    out = fit_learned.adjust_rows(data, {}, zero, groups)
    layout = learned.group_layout(groups)
    for g in ("touch", "impute"):
        team, common = layout[g]
        assert not out["st:1|g0"][0][0][:, team].any() and not out["st:1|g0"][0][1][:, team].any()
        assert not out["st:1|g0"][0][2][:, common].any()
        assert out["idsse:1|g0"][0][0][:, team].any()
    team, _ = layout["shapem"]
    np.testing.assert_array_equal(out["st:1|g0"][0][0][:, team], rows[0][0][:, team])
    with pytest.raises(ValueError):
        fit_learned.adjust_rows(data, {}, {"st": "tags"}, groups)


def test_the_imputer_gives_an_unseen_source_its_fit_on_everything():
    every = ball_impute.beta_for("idsse:x", "all")
    for fold in ("auto", "lomo", "loso"):
        np.testing.assert_array_equal(ball_impute.beta_for("soccertrack:1", fold), every)
    seen = ball_impute.load_coefs()["fit"]["sources"][0]
    assert not np.array_equal(ball_impute.beta_for(seen + ":x", "loso"), every)
    with pytest.raises(KeyError):
        ball_impute.beta_for(seen + ":not-a-match", "lomo")


def test_bigger_gives_an_unseen_source_its_fit_on_everything(monkeypatch):
    coefs = {"all": "ALL", "loso": {"idsse": "NO-IDSSE"}, "folds": {}, "fold_of": {},
             "fit": {"sources": ["idsse", "skillcorner"]}}
    monkeypatch.setattr(emission, "load_coefs", lambda path: coefs)
    dec = emission.EmissionDecoder(fold="loso", coef_file="x.json")

    class M:
        match_id = "soccertrack:1"
    assert dec.coefs(M()) == "ALL"
    M.match_id = "idsse:1"
    assert dec.coefs(M()) == "NO-IDSSE"
    M.match_id = "skillcorner:1"  # trained on, but no loso fit: still an error
    with pytest.raises(KeyError):
        dec.coefs(M())
    lomo = emission.EmissionDecoder(fold="lomo", coef_file="x.json")
    M.match_id = "soccertrack:1"
    assert lomo.coefs(M()) == "ALL"  # never seen: no fold holds it out, "all" does
    M.match_id = "idsse:1"
    with pytest.raises(KeyError):
        lomo.coefs(M())


def test_learning_curve_cap_keeps_every_source_and_is_fixed():
    ms = (["idsse:%d" % i for i in range(6)] + ["skillcorner:%d" % i for i in range(16)]
          + ["metrica:%d" % i for i in range(2)])
    kept = fit_learned.cap_matches(ms, 12)
    assert len(kept) == 12 and kept == fit_learned.cap_matches(list(reversed(ms)), 12)
    assert {m.split(":")[0] for m in kept} == {"idsse", "skillcorner", "metrica"}
    assert fit_learned.cap_matches(ms, 0) == ms and fit_learned.cap_matches(ms, 99) == ms

