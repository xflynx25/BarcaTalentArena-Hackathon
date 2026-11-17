import json

import numpy as np
import pytest

from src import resample
from src.ingest import skillcorner
from src.official import A, D, H, X
from src.schema import AWAY, HOME

MATCH_ID = "999"
HOME_TEAM, AWAY_TEAM = 10, 20


def _frame(frame, period, group=None, ball=None, players=(), player_id=None):
    return {
        "frame": frame, "timestamp": None, "period": period,
        "ball_data": {"x": ball[0] if ball else None, "y": ball[1] if ball else None, "z": 0.0,
                      "is_detected": ball[2] if ball else None},
        "possession": {"player_id": player_id, "group": group},
        "image_corners_projection": {},
        "player_data": [{"player_id": i, "x": x, "y": y, "is_detected": det}
                        for i, x, y, det in players],
    }


@pytest.fixture
def tiny(tmp_path):
    """Six frames: before kick-off, three in period 1, a broadcast cut, one in period 2."""
    folder = tmp_path / "data" / "matches" / MATCH_ID
    folder.mkdir(parents=True)
    meta = {
        "pitch_length": 100, "pitch_width": 60,
        "home_team": {"id": HOME_TEAM, "name": "Home"}, "away_team": {"id": AWAY_TEAM, "name": "Away"},
        "players": [{"id": 7, "team_id": AWAY_TEAM}, {"id": 3, "team_id": HOME_TEAM},
                    {"id": 5, "team_id": HOME_TEAM}, {"id": 99, "team_id": AWAY_TEAM}],
    }
    (folder / "{}_match.json".format(MATCH_ID)).write_text(json.dumps(meta))
    frames = [
        _frame(0, None),
        _frame(1, 1, "home team", (0.0, 0.0, True), [(3, -1.0, 2.0, True), (7, 10.0, -5.0, False)], 3),
        _frame(2, 1, "home team", (5.0, 1.0, False), [(5, 4.0, 1.0, True)]),
        _frame(3, 1, "away team", (-50.0, -30.0, True), [(7, -49.0, -29.0, True)], 7),
        _frame(4, 1, None),
        _frame(5, 2, "away team", (1.0, 1.0, True), [(3, 0.0, 0.0, True)]),
    ]
    raw = folder / "{}_tracking_extrapolated.jsonl".format(MATCH_ID)
    raw.write_text("\n".join(json.dumps(f) for f in frames) + "\n")
    # a match without tracking must be skipped by list_matches
    (tmp_path / "data" / "matches" / "123").mkdir()
    return tmp_path


def test_labels_and_periods(tiny):
    match = skillcorner.load(MATCH_ID, root=str(tiny))
    assert match.fps == 10 and match.pitch == (100.0, 60.0)
    assert match.labels.tolist() == [X, H, H, A, D, A]
    assert match.period.tolist() == [0, 1, 1, 1, 1, 2]


def test_slots_and_coordinates(tiny):
    match = skillcorner.load(MATCH_ID, root=str(tiny))
    # slots ordered (team, id): home 3, home 5, away 7; player 99 never tracked -> no slot
    assert match.slot_team.tolist() == [HOME, HOME, AWAY]
    np.testing.assert_allclose(match.players[1, 0], [49.0, 32.0])
    np.testing.assert_allclose(match.players[1, 2], [60.0, 25.0])
    np.testing.assert_allclose(match.ball[3], [0.0, 0.0])  # corner of the pitch
    assert np.isnan(match.ball[0]).all() and np.isnan(match.players[4]).all()


def test_drop_extrapolated(tiny):
    match = skillcorner.load(MATCH_ID, root=str(tiny), extrapolated=False)
    assert np.isnan(match.players[1, 2]).all() and np.isnan(match.ball[2]).all()
    assert np.isfinite(match.ball[1]).all()


def test_cuts_ungraded(tiny):
    # frame 4 is a one-frame cut: only masked once min_cut_s allows a single 0.1 s frame
    assert skillcorner.load(MATCH_ID, root=str(tiny), cuts="ungraded").labels.tolist() == \
        [X, H, H, A, D, A]
    match = skillcorner.load(MATCH_ID, root=str(tiny), cuts="ungraded", min_cut_s=0.1)
    assert match.labels.tolist() == [X, H, H, A, X, A]
    with pytest.raises(ValueError):
        skillcorner.load(MATCH_ID, root=str(tiny), cuts="drop")


def test_cut_mask():
    tracked = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    period = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    assert skillcorner.cut_mask(tracked, period, 3).tolist() == \
        [False, False, False, False, True, True, True, False, False, False]
    assert skillcorner.cut_mask(tracked, period, 2).sum() == 5


def test_list_matches(tiny):
    assert skillcorner.list_matches(str(tiny)) == [MATCH_ID]
    assert skillcorner.list_matches(str(tiny / "nowhere")) == []


def test_unknown_group_rejected(tiny):
    raw = tiny / "data" / "matches" / MATCH_ID / "{}_tracking_extrapolated.jsonl".format(MATCH_ID)
    raw.write_text(raw.read_text().replace('"away team"', '"referee"', 1))
    with pytest.raises(ValueError, match="possession group"):
        skillcorner.load(MATCH_ID, root=str(tiny))


def test_resample_10_to_25(tiny):
    match = skillcorner.load(MATCH_ID, root=str(tiny))
    up = resample.to_fps(match, 25)
    assert up.fps == 25 and up.n_frames == 15  # 6 frames = 0.6 s = 15 frames at 25 fps
    # output frame 3 = source 1.2: same period, both observed -> interpolated
    np.testing.assert_allclose(up.ball[3], match.ball[1] * 0.8 + match.ball[2] * 0.2)
    # output frame 12 = source 4.8 -> nearest (frame 5), period 2, no interpolation across halves
    assert up.period[12] == 2 and up.labels[12] == A
    np.testing.assert_allclose(up.ball[12], match.ball[5])
    assert resample.labels_to_fps(up.labels, 25, 10).tolist() == match.labels.tolist()


@pytest.mark.skipif(not skillcorner.list_matches(), reason="SkillCorner tracking not downloaded")
def test_real_match():
    """On real data: shapes, label mix, and agreement with SkillCorner's own phases of play."""
    import pandas as pd

    match_id = skillcorner.list_matches()[0]
    match = skillcorner.load(match_id)
    in_play = match.period > 0
    assert in_play.mean() > 0.95 and set(np.unique(match.period)) == {0, 1, 2}
    assert (match.labels[~in_play] == X).all() and (match.labels[in_play] != X).all()
    share = np.bincount(match.labels[in_play], minlength=3) / in_play.sum()
    assert all(0.15 < s < 0.6 for s in share[:3])
    assert 18 <= len(match.slot_team) <= 40 and set(match.slot_team.tolist()) == {HOME, AWAY}
    ball = match.ball[np.isfinite(match.ball[:, 0])]
    assert np.median(ball[:, 0]) == pytest.approx(match.pitch[0] / 2, abs=15)

    # Frames inside a phase of play should mostly be live (H/A); frames outside, mostly D.
    phases = pd.read_csv(skillcorner._paths(match_id, skillcorner.ROOT)[0].replace(
        "_match.json", "_phases_of_play.csv"))
    in_phase = np.zeros(match.n_frames, dtype=bool)
    for start, end in zip(phases["frame_start"], phases["frame_end"]):
        in_phase[start:end + 1] = True
    live = np.isin(match.labels, [H, A])
    assert (live == in_phase)[in_play].mean() > 0.9
