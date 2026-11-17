import json

import numpy as np
import pytest

from src.ingest import skillcorner2020
from src.official import A, D, H, X
from src.schema import AWAY, HOME

MATCH_ID = "999"
HOME_TEAM, AWAY_TEAM, BALL, REF = 10, 20, 55, 77


def _frame(frame, period, group=None, objects=()):
    return {"frame": frame, "period": period, "time": None,
            "possession": {"trackable_object": None, "group": group}, "data": list(objects)}


def _obj(obj_id, x, y, track=0, group_name=None):
    out = {"track_id": track, "x": x, "y": y}
    if obj_id is None:
        out["group_name"] = group_name
    else:
        out["trackable_object"] = obj_id
    return out


@pytest.fixture
def tiny(tmp_path):
    """Seven frames: before kick-off, four in period 1 (one a cut), half-time, one in period 2."""
    folder = tmp_path / "data" / "matches" / MATCH_ID
    folder.mkdir(parents=True)
    meta = {
        "pitch_length": 100, "pitch_width": 60, "ball": {"trackable_object": BALL},
        "home_team": {"id": HOME_TEAM, "name": "Home"}, "away_team": {"id": AWAY_TEAM, "name": "Away"},
        "players": [{"trackable_object": 7, "team_id": AWAY_TEAM},
                    {"trackable_object": 3, "team_id": HOME_TEAM},
                    {"trackable_object": 5, "team_id": HOME_TEAM}],
        "referees": [{"trackable_object": REF}],
    }
    (folder / "match_data.json").write_text(json.dumps(meta))
    frames = [
        _frame(0, None),
        _frame(1, 1, "home team", [_obj(BALL, 0.0, 0.0), _obj(3, -1.0, 2.0), _obj(REF, 0, 0),
                                   _obj(None, 5.0, 5.0, track=900, group_name="away team")]),
        _frame(2, 1, "home team", [_obj(5, 4.0, 1.0), _obj(None, 1, 1, group_name="referee"),
                                   _obj(None, 6.0, 6.0, track=900, group_name="away team"),
                                   _obj(None, 7.0, 7.0, track=901, group_name="away team")]),
        _frame(3, 1, "away team", [_obj(None, -50.0, -30.0, group_name="balls"), _obj(7, -49, -29),
                                   _obj(None, 8.0, 8.0, track=902, group_name="away team")]),
        _frame(4, 1, None),
        _frame(5, None, "home team"),
        _frame(6, 2, "away team", [_obj(BALL, 1.0, 1.0), _obj(3, 0.0, 0.0)]),
    ]
    (folder / "structured_data.json").write_text(json.dumps(frames))
    (tmp_path / "data" / "matches" / "123").mkdir()  # no tracking: skipped by list_matches
    return tmp_path


def test_labels_periods_and_ball(tiny):
    match = skillcorner2020.load(MATCH_ID, root=str(tiny))
    assert match.match_id == "skillcorner2020:999" and match.fps == 10
    assert match.labels.tolist() == [X, H, H, A, D, X, A]
    assert match.period.tolist() == [0, 1, 1, 1, 1, 0, 2]
    # corner origin; the unidentified "balls" counts as the ball; frames without one are NaN
    assert np.allclose(match.ball[1], (50.0, 30.0)) and np.allclose(match.ball[3], (0.0, 0.0))
    assert np.isnan(match.ball[2]).all()


def test_slots_identified_then_packed_anonymous(tiny):
    match = skillcorner2020.load(MATCH_ID, root=str(tiny))
    # home: ids 3, 5; away: id 7, then two anonymous slots (tracks 900+901 overlap; 902 reuses one)
    assert match.slot_team.tolist() == [HOME, HOME, AWAY, AWAY, AWAY]
    assert np.allclose(match.players[1, 0], (49.0, 32.0))
    assert np.allclose(match.players[1, 3], (55.0, 35.0)) and np.allclose(match.players[2, 3], (56, 36))
    assert np.allclose(match.players[2, 4], (57.0, 37.0))
    assert np.allclose(match.players[3, 3], (58.0, 38.0))  # 902 takes the lowest free slot
    assert np.isnan(match.players[3, 4]).all()
    plain = skillcorner2020.load(MATCH_ID, root=str(tiny), anonymous=False)
    assert plain.slot_team.tolist() == [HOME, HOME, AWAY]


def test_cuts_ungraded(tiny):
    match = skillcorner2020.load(MATCH_ID, root=str(tiny), cuts="ungraded", min_cut_s=0.1)
    assert match.labels.tolist() == [X, H, H, A, X, X, A]


def test_unknown_object_raises(tiny):
    path = tiny / "data" / "matches" / MATCH_ID / "structured_data.json"
    frames = json.loads(path.read_text())
    frames[1]["data"].append(_obj(12345, 0, 0))
    path.write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="unknown object"):
        skillcorner2020.load(MATCH_ID, root=str(tiny))


def test_list_matches(tiny):
    assert skillcorner2020.list_matches(str(tiny)) == [MATCH_ID]


@pytest.mark.skipif(not skillcorner2020.list_matches(), reason="SkillCorner 2019/20 not fetched")
def test_real_match_loads():
    match = skillcorner2020.load(skillcorner2020.list_matches()[0], cuts="ungraded")
    graded = match.labels != X
    assert 0.5 < graded[match.period > 0].mean() < 0.9
    assert {H, A, D} <= set(np.unique(match.labels[graded]).tolist())
