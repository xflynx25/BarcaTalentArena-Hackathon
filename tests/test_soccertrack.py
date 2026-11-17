"""src/ingest/soccertrack.py on tiny synthetic files: label rules, layout, coordinates, the GSR parser.
An optional test on the real compact files skips when they are absent."""

import json
import os

import numpy as np
import pytest

from src.ingest import soccertrack as st
from src.official import A, D, H, X
from src.schema import AWAY, HOME


def ev(half, frame, label, team="left", player=None):
    e = {"gameTime": "{} - 0:00".format(half), "label": label, "frame": frame, "team": team,
         "position": str((frame - 1) * 40)}
    if player is not None:
        e["player_id"] = str(player)
    return e


# -- label rules ---------------------------------------------------------------------------------------

def labels_for(events, n=100):
    frames, teams, kinds = st.half_events(events, 1, {"left": HOME, "right": AWAY})
    return st.action_labels(frames, teams, kinds, n)


def test_actions_possession_and_dead_ball():
    out = labels_for([ev(1, 11, "PASS"), ev(1, 21, "DRIVE", "right"), ev(1, 31, "OUT", "right"),
                      ev(1, 41, "THROW IN", "left"), ev(1, 51, "PASS", "left")])
    assert (out[:10] == D).all()  # before the first action
    assert (out[10:20] == H).all() and (out[20:30] == A).all()
    assert (out[30:40] == D).all()  # Out -> frame before the restart
    assert (out[40:] == H).all()  # the throw-in restarts with its team, to the end of the half


def test_touches_do_not_switch_but_a_tackle_does():
    out = labels_for([ev(1, 1, "PASS"), ev(1, 11, "HEADER", "right"), ev(1, 21, "BALL PLAYER BLOCK", "right"),
                      ev(1, 31, "PLAYER SUCCESSFUL TACKLE", "right"), ev(1, 41, "BALL PLAYER BLOCK", "left")])
    assert (out[:30] == H).all()
    assert (out[30:] == A).all()


def test_goal_on_the_shot_frame_is_dead_until_kickoff():
    out = labels_for([ev(1, 1, "PASS"), ev(1, 21, "GOAL"), ev(1, 21, "SHOT"), ev(1, 61, "PASS", "right")])
    assert (out[:20] == H).all()
    assert (out[20:60] == D).all()  # the Shot on the Goal's own frame is not the restart
    assert (out[60:] == A).all()


def test_last_action_out_is_dead_to_the_end():
    out = labels_for([ev(1, 1, "PASS"), ev(1, 51, "OUT")])
    assert (out[50:] == D).all() and (out[:50] == H).all()


def test_unknown_class_fails_loudly():
    with pytest.raises(ValueError):
        labels_for([ev(1, 1, "CORNER")])


def test_status_labels():
    status = np.array([0, 1, 1, 3, 3, 2, 3, 0, 3, -1, 2])
    assert st.status_labels(status).tolist() == [D, H, H, H, H, A, A, D, A, X, A]
    assert st.status_labels(np.array([3, 3, 1])).tolist() == [D, D, H]  # no team yet in the half
    assert st.status_labels(np.array([4, 4, -1])).tolist() == [X, X, X]  # no vendor status
    with pytest.raises(ValueError):
        st.status_labels(np.array([5]))


# -- a whole synthetic match ---------------------------------------------------------------------------

def write_match(root, match_id="900001", n=(60, 80)):
    c = st.compact_dir(str(root))
    os.makedirs(c, exist_ok=True)
    players = {"11": "home", "12": "home", "21": "away", "22": "away", "23": "away"}
    with open(os.path.join(c, "{}_meta.json".format(match_id)), "w") as h:
        json.dump({"format": st.FORMAT, "teams": {"home": "1", "away": "2"},
                   "players": {p: {"team": t, "shirt": "1", "position": "MF"} for p, t in players.items()}}, h)
    events = [ev(1, 5, "PASS", "left", 11), ev(1, 30, "DRIVE", "right", 21),
              ev(2, 3, "PASS", "left", 22), ev(2, 50, "OUT", "left", 22), ev(2, 70, "FREE KICK", "right", 12)]
    with open(os.path.join(c, "{}_events.json".format(match_id)), "w") as h:
        json.dump({"match_id": match_id, "fps": 25.0, "clock": "ms", "actions": events}, h)
    # half 1: home (11, 12) on the left; half 2: sides swap and 23 replaces 21
    layout = {"1st": ([11, 12, 21, 22], [0, 0, 1, 1]), "2nd": ([11, 12, 22, 23], [1, 1, 0, 0])}
    for (half, (ids, sides)), size in zip(layout.items(), n):
        xy = np.zeros((size, len(ids), 2), dtype=np.float32)
        xy[:, :, 0] = np.arange(len(ids)) * 10 - 15
        xy[:, :, 1] = 5.0
        xy[0, 0] = np.nan
        ball = np.tile(np.array([[1.0, -2.0]], dtype=np.float32), (size, 1))
        labeled = np.ones(size, dtype=bool)
        if half == "1st":
            labeled[:2] = False
        np.savez_compressed(os.path.join(c, "{}_{}.npz".format(match_id, half)), xy=xy, ball=ball,
                            status=np.full(size, 1, dtype=np.int8), labeled=labeled,
                            player_id=np.array(ids, dtype=np.int64), side=np.array(sides, dtype=np.int8),
                            gk=np.zeros(len(ids), bool), jersey=np.arange(len(ids), dtype=np.int16))
    return match_id


def test_load_layout_coordinates_and_labels(tmp_path):
    mid = write_match(tmp_path)
    assert st.list_matches(str(tmp_path)) == []  # list_matches only knows the 10 real ids
    m = st.load(mid, root=str(tmp_path))
    gap = st.GAP_S * st.FPS
    assert m.n_frames == 60 + gap + 80 and m.fps == 25 and m.pitch == (105.0, 68.0)
    assert m.slot_team.tolist() == [HOME, HOME, AWAY, AWAY, AWAY]  # 11, 12 | 21, 22, 23
    assert (m.period[:2] == 0).all() and (m.period[2:60] == 1).all()
    assert (m.period[60:60 + gap] == 0).all() and (m.period[60 + gap:] == 2).all()
    assert (m.labels[m.period == 0] == X).all()
    # corner origin, y flipped so y = 0 is the near touchline: (1, -2) -> (53.5, 36)
    assert np.allclose(m.ball[5], [53.5, 36.0])
    assert np.isnan(m.players[0, 0]).all() and np.isnan(m.players[5, 4]).all()  # 23 absent in half 1
    hidden = st.load(mid, root=str(tmp_path), ball="none")  # A18: same players and labels, no ball
    assert np.isnan(hidden.ball).all() and np.array_equal(hidden.labels, m.labels)
    assert np.array_equal(hidden.players, m.players, equal_nan=True)
    with pytest.raises(ValueError):
        st.load(mid, root=str(tmp_path), ball="maybe")
    assert np.allclose(m.players[5, 0], [52.5 - 15, 29.0])
    lab1 = m.labels[:60]
    assert (lab1[2:4] == D).all() and (lab1[4:29] == H).all() and (lab1[29:] == A).all()
    lab2 = m.labels[60 + gap:]
    # half 2: home is on the right, so 'left' events are the away team's
    assert (lab2[:2] == D).all() and (lab2[2:49] == A).all()
    assert (lab2[49:69] == D).all() and (lab2[69:] == H).all()
    s = st.load(mid, root=str(tmp_path), labels="status")
    assert (s.labels[m.period > 0] == H).all()


def test_mixed_side_refused(tmp_path):
    mid = write_match(tmp_path)
    path = os.path.join(st.compact_dir(str(tmp_path)), "{}_1st.npz".format(mid))
    half = dict(np.load(path))
    half["side"] = np.array([0, 1, 1, 1], dtype=np.int8)  # a home player on the away side
    np.savez_compressed(path, **half)
    with pytest.raises(ValueError):
        st.load(mid, root=str(tmp_path))


# -- the streaming GSR parser --------------------------------------------------------------------------

def test_parse_gsr(tmp_path):
    def obj(frame, pid, x, y, team, role="player", jersey="7"):
        return {"id": "3901{:06d}01".format(frame), "image_id": "3901{:06d}".format(frame),
                "supercategory": "object", "attributes": {"role": role, "jersey": jersey, "team": team,
                                                          "player_id": pid},
                "bbox_pitch": {"x_bottom_left": x, "y_bottom_left": y, "x_bottom_middle": x,
                               "y_bottom_middle": y, "x_bottom_right": x, "y_bottom_right": y}}

    doc = {"info": {"seq_length": 3},
           "images": [{"is_labeled": f != 1, "image_id": "3901{:06d}".format(f)} for f in (1, 2, 3)],
           "annotations": [{"id": "3901000002P", "image_id": "3901000002", "supercategory": "pitch",
                            "lines": {"a": [{"x": 0.1, "y": 0.2}]}},
                           obj(2, 5, 1.0, 2.0, "left", "goalkeeper", "1"), obj(2, 5, 9.0, 9.0, "left"),
                           obj(3, 5, 1.5, 2.5, "left", "goalkeeper", "1"), obj(3, 8, -3.0, 4.0, "right")]}
    path = tmp_path / "half.json"
    path.write_text(json.dumps(doc))
    out = st.parse_gsr(str(path), 3)
    assert out["labeled"].tolist() == [False, True, True]
    assert out["player_id"].tolist() == [5, 8] and out["side"].tolist() == [0, 1]
    assert out["gk"].tolist() == [True, False] and out["jersey"].tolist() == [1, 7]
    assert out["duplicates"] == 1
    assert np.allclose(out["xy"][1, 0], [1.0, 2.0])  # the first record of a duplicate wins
    assert np.isnan(out["xy"][0]).all() and np.isnan(out["xy"][1, 1]).all()
    assert out["xy"].dtype == np.float32


@pytest.mark.skipif(not st.list_matches(), reason="SoccerTrack compact files not fetched")
def test_real_match_loads():
    m = st.load(st.list_matches()[0])
    graded = m.labels != X
    assert m.fps == 25 and graded.mean() > 0.9
    assert set(np.unique(m.labels[graded]).tolist()) == {H, A, D}
    assert np.isfinite(m.ball[m.period > 0, 0]).mean() > 0.99


def test_smooth_ball_turns_a_staircase_into_a_line():
    # a ball moving at 3 m/s drawn on a 1.05 m grid: still on most frames, then one-cell jumps
    fps, n = 25, 200
    true = np.stack([np.arange(n) * 3.0 / fps, np.full(n, 10.0)], axis=1)
    stair = np.stack([np.round(true[:, 0] / 1.05) * 1.05, true[:, 1]], axis=1)
    period = np.ones(n, np.int8)
    period[:5] = 0
    out = st.smooth_ball(stair, period, fps, 0.3)
    inner = slice(20, n - 20)
    assert np.abs(out[inner] - true[inner]).max() < np.abs(stair[inner] - true[inner]).max() / 2
    assert np.array_equal(out[:5], stair[:5]) and np.allclose(out[:, 1], 10.0)
    hidden = stair.copy()
    hidden[50:60] = np.nan
    assert np.isnan(st.smooth_ball(hidden, period, fps, 0.3)[50:60]).all()
