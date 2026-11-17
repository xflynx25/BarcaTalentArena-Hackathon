"""Team-shape features (src/shape.py, backlog B23): geometry, symmetry and direction, synthetic only."""

import numpy as np

from src import shape
from src.schema import AWAY, HOME, Match


def _match(n=60, seed=0, missing=0):
    """Two 11-player teams, home defending x = 0 in period 1 and x = 105 in period 2."""
    rng = np.random.default_rng(seed)
    period = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)].astype(np.int8)
    base_home = np.c_[np.r_[3, rng.uniform(10, 50, 10)], rng.uniform(5, 63, 11)]
    base_away = np.c_[np.r_[102, rng.uniform(55, 95, 10)], rng.uniform(5, 63, 11)]
    walk = np.cumsum(rng.normal(0, 0.1, (n, 22, 2)), axis=0)
    pos = np.concatenate([base_home, base_away])[None] + walk
    second = period == 2
    pos[second, :, 0] = 105 - pos[second, :, 0]
    pos[second, :, 1] = 68 - pos[second, :, 1]
    if missing:
        pos[:, 1:1 + missing] = np.nan
    return Match(match_id="t", fps=10, pitch=(105.0, 68.0), period=period,
                 ball=np.full((n, 2), np.nan), players=pos,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_hull_area_square_with_inside_and_duplicate_points():
    pts = np.array([[[0, 0], [4, 0], [4, 3], [0, 3], [2, 1], [2, 1], [np.nan, np.nan]]], float)
    assert np.allclose(shape.hull_area(pts), 12.0, atol=1e-4)
    assert np.isnan(shape.hull_area(np.array([[[0, 0], [1, 1], [np.nan, np.nan]]], float)))[0]


def test_attack_sign_follows_halves():
    sign = shape.attack_sign(_match())
    assert (sign[:30, HOME] == 1).all() and (sign[30:, HOME] == -1).all()
    assert (sign[:, AWAY] == -sign[:, HOME]).all()


def test_swapping_teams_negates_differences():
    m = _match()
    swapped = Match(match_id="s", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
                    players=m.players, slot_team=1 - m.slot_team)
    a = shape.differences(shape.team_features(m)[1])
    b = shape.differences(shape.team_features(swapped)[1])
    for name in shape.FEATURES:
        known = ~np.isnan(a[name])
        assert known.sum() > 20, name
        assert np.allclose(a[name][known], -b[name][known], atol=1e-6), name


def test_features_are_the_same_in_both_halves():
    """The second half is the first rotated by 180 degrees, so own-frame features match."""
    m = _match()
    first = Match(match_id="f", fps=m.fps, pitch=m.pitch, period=np.ones(30, np.int8),
                  ball=m.ball[:30], players=m.players[:30], slot_team=m.slot_team)
    second_players = m.players[:30].copy()
    second_players[..., 0] = 105 - second_players[..., 0]
    second_players[..., 1] = 68 - second_players[..., 1]
    second = Match(match_id="g", fps=m.fps, pitch=m.pitch, period=np.full(30, 2, np.int8),
                   ball=m.ball[:30], players=second_players, slot_team=m.slot_team)
    a, b = shape.team_features(first)[1], shape.team_features(second)[1]
    for name in shape.FEATURES:
        assert np.allclose(a[name], b[name], atol=1e-6, equal_nan=True), name


def test_values_make_sense_and_few_players_give_nan():
    idx, f = shape.team_features(_match(), step=2)
    assert len(idx) == 30
    assert (f["gk"][:, HOME] < 6).all() and (f["gk"][:, AWAY] < 6).all()  # deepest, own frame
    assert np.nanmax(f["opp_half"][:, HOME]) == 0  # home outfield sits in x' < 52.5
    assert np.nanmax(f["last_line"][:, HOME]) <= 10
    ok = ~np.isnan(f["control"])
    assert ((f["control"][ok] >= 0) & (f["control"][ok] <= 1)).all()
    _, few = shape.team_features(_match(missing=4))
    assert np.isnan(few["width"][:, HOME]).all() and not np.isnan(few["width"][:, AWAY]).all()


def test_zscore():
    z = shape.zscore(np.array([1.0, 2.0, 3.0, np.nan]))
    assert np.allclose(z[:3], [-1.2247449, 0, 1.2247449]) and np.isnan(z[3])

