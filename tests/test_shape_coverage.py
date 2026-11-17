import json

import numpy as np

from src import shape_coverage as sc
from src.ingest import skillcorner
from src.schema import AWAY, HOME, Match

L, W = 100.0, 60.0


def two_halves(n_half=300, fps=10, seed=0):
    """11 v 11, slot 0 home keeper, slot 11 away keeper. Home defends x=0 in period 1, swaps in 2."""
    rng = np.random.default_rng(seed)
    n = 2 * n_half + 10
    period = np.zeros(n, dtype=np.int8)
    period[5:5 + n_half] = 1
    period[5 + n_half:5 + 2 * n_half] = 2
    players = np.full((n, 22, 2), np.nan)
    for half, rows in ((1, period == 1), (2, period == 2)):
        m = rows.sum()
        home_x = rng.uniform(25, 50, (m, 10))
        away_x = rng.uniform(50, 75, (m, 10))
        home_gk, away_gk = rng.uniform(1, 6, m), rng.uniform(94, 99, m)
        if half == 2:  # swap ends
            home_x, away_x = L - home_x, L - away_x
            home_gk, away_gk = L - home_gk, L - away_gk
        players[rows, 0, 0], players[rows, 1:11, 0] = home_gk, home_x
        players[rows, 11, 0], players[rows, 12:, 0] = away_gk, away_x
        players[rows, :, 1] = rng.uniform(5, 55, (m, 22))
        players[rows, 0, 1] = players[rows, 11, 1] = W / 2
    return Match(match_id="t", fps=fps, pitch=(L, W), period=period, ball=np.full((n, 2), np.nan),
                 players=players, slot_team=np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8))


def test_full_teams():
    m = two_halves()
    cov = sc.coverage(m)
    assert cov.keeper_slots[HOME].tolist() == [0] and cov.keeper_slots[AWAY].tolist() == [11]
    live = m.period > 0
    assert (cov.present[live] == 11).all() and (cov.outfield[live] == 10).all()
    assert (cov.present[~live] == 0).all()
    assert cov.mask().sum() == live.sum() and np.allclose(cov.weight()[live], 1.0)
    p1, p2 = m.period == 1, m.period == 2
    assert (cov.own_goal_x[p1] == [0.0, L]).all() and (cov.own_goal_x[p2] == [L, 0.0]).all()
    assert (cov.attack_sign[p1] == [1, -1]).all() and (cov.attack_sign[p2] == [-1, 1]).all()
    assert (cov.attack_sign[~live] == 0).all()
    s = sc.summary(cov, m.fps)
    assert s["both>=10"] == 1.0 and s["outfield>10"] == 0.0 and s["keeper slots"] == 1.0


def test_missing_players_and_seen_mask():
    m = two_halves()
    rows = np.flatnonzero(m.period == 1)[:100]
    m.players[rows, 1:3] = np.nan          # home loses two outfield players
    m.players[rows, 11] = np.nan           # away loses its keeper
    cov = sc.coverage(m)
    assert (cov.outfield[rows, HOME] == 8).all() and (cov.present[rows, AWAY] == 10).all()
    # away keeper slot gone; deepest away outfield player is >20 m from the goal: no keeper
    assert not cov.keeper[rows, AWAY].any() and (cov.outfield[rows, AWAY] == 10).all()
    assert not cov.mask()[rows].any() and np.allclose(cov.weight()[rows], 0.8)
    seen = np.ones(m.players.shape[:2], dtype=bool)
    seen[:, 5] = False
    assert (sc.coverage(m, seen=seen).present[m.period == 2, HOME] == 10).all()


def test_tracklet_keeper_and_fallback():
    """The keeper's slot is cut in two (as BAR-ATH's tracklets do): both halves are keeper slots."""
    m = two_halves()
    extra = np.full((m.n_frames, 1, 2), np.nan)
    rows = np.flatnonzero(m.period == 2)
    extra[rows, 0] = m.players[rows, 0]
    m.players[rows, 0] = np.nan
    m = Match(match_id="t", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
              players=np.concatenate([m.players, extra], axis=1),
              slot_team=np.append(m.slot_team, HOME).astype(np.int8))
    cov = sc.coverage(m)
    assert cov.keeper_slots[HOME].tolist() == [0, 22]
    assert (cov.keeper_slot[m.period == 1, HOME] == 0).all()
    assert (cov.keeper_slot[rows, HOME] == 22).all()
    assert (cov.outfield[m.period > 0] == 10).all()


def test_box_fallback_for_short_keeper_tracklets():
    m = two_halves()
    m.players[:, 0] = np.nan
    rows = np.flatnonzero(m.period == 1)[:30]   # 3 s: too short to be a keeper slot
    m.players[rows, 0] = (3.0, W / 2)
    cov = sc.coverage(m)
    assert cov.keeper_slots[HOME].tolist() == []
    assert cov.keeper[rows, HOME].all() and (cov.keeper_slot[rows, HOME] == 0).all()
    assert (cov.outfield[rows, HOME] == 10).all()


def test_view_share_and_quads():
    full = np.array([[[-1, -1], [-1, W + 1], [L + 1, W + 1], [L + 1, -1]]], dtype=float)
    half = np.array([[[0, 0], [L / 2, 0], [L / 2, W], [0, W]]], dtype=float)  # other winding
    none = np.full((1, 4, 2), np.nan)
    share = sc.view_share(np.concatenate([full, half, none]), (L, W), step=2.0)
    assert np.allclose(share, [1.0, 0.5, 0.0])


def test_skillcorner_extras_layout(tmp_path):
    folder = tmp_path / "data" / "matches" / "5"
    folder.mkdir(parents=True)
    meta = {"pitch_length": 100, "pitch_width": 60, "home_team_side": ["left_to_right", "right_to_left"],
            "home_team": {"id": 1, "name": "H"}, "away_team": {"id": 2, "name": "A"},
            "players": [{"id": 9, "team_id": 2, "player_role": {"acronym": "GK"}},
                        {"id": 4, "team_id": 1, "player_role": {"acronym": "CB"}},
                        {"id": 6, "team_id": 1, "player_role": {"acronym": "GK"}}]}
    (folder / "5_match.json").write_text(json.dumps(meta))

    def frame(i, players, cam=None):
        return {"frame": i, "timestamp": None, "period": 1, "possession": {"group": None},
                "ball_data": {"x": None, "y": None}, "image_corners_projection": cam or {},
                "player_data": [{"player_id": p, "x": x, "y": 0.0, "is_detected": d}
                                for p, x, d in players]}
    cam = {"x_top_left": -60, "y_top_left": 40, "x_bottom_left": -60, "y_bottom_left": -40,
           "x_bottom_right": 60, "y_bottom_right": -40, "x_top_right": 60, "y_top_right": 40}
    frames = [frame(0, [(9, 1.0, True), (6, -40.0, False)], cam), frame(1, [(4, 0.0, True)]),
              frame(2, [])]
    (folder / "5_tracking_extrapolated.jsonl").write_text("\n".join(json.dumps(f) for f in frames))

    match = skillcorner.load("5", root=str(tmp_path))
    extras = sc.skillcorner_extras("5", root=str(tmp_path))
    assert extras["detected"].shape == match.players.shape[:2]
    assert extras["slot_ids"] == [4, 6, 9]  # home first, then by id, as the loader
    assert extras["detected"].tolist() == [[False, False, True], [True, False, False],
                                           [False, False, False]]
    assert extras["keeper_ids"] == {6, 9}
    assert extras["home_side"] == ["left_to_right", "right_to_left"]
    assert np.allclose(extras["view"][0, 0], [-10.0, 70.0]) and np.isnan(extras["view"][1]).all()
    assert np.allclose(sc.view_share(extras["view"], match.pitch), [1.0, 0.0, 0.0])

