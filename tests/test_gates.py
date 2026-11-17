"""Local L1/L3: the free-ball / in-flight gates and the gap evidence (synthetic data only)."""

import dataclasses

import numpy as np

from src.models import gates, learned
from src.models.decoder import team_distances
from src.schema import AWAY, HOME, Match


def _two_player_match(ball, fps=10):
    """A home player at (50, 34) and an away player at (60, 34), standing still."""
    n = len(ball)
    players = np.stack([np.tile([50.0, 34.0], (n, 1)), np.tile([60.0, 34.0], (n, 1))], axis=1)
    return Match(match_id="test:gates", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=np.asarray(ball, float), players=players,
                 slot_team=np.array([HOME, AWAY], np.int8)).validate()


def _gates(match):
    d_home, d_away = team_distances(match)
    return gates.free_ball(match, d_home, d_away), gates.flight(match, d_home, d_away)


def test_free_ball_needs_a_seen_ball_far_from_everyone():
    ball = np.repeat([[np.nan, np.nan], [50.0, 46.0], [51.0, 34.0], [55.0, 50.0]], 5, axis=0)
    free, _ = _gates(_two_player_match(ball))
    assert free.dtype == bool and free[::5].tolist() == [False, True, False, True]


def test_free_ball_counts_a_team_without_players_as_far():
    match = _two_player_match(np.tile([60.0, 34.5], (5, 1)))  # on the away player
    alone = dataclasses.replace(match, players=match.players[:, :1], slot_team=match.slot_team[:1])
    d_home, d_away = team_distances(alone)
    assert np.isnan(d_away).all()
    assert gates.free_ball(alone, d_home, d_away).all()  # home player is 10 m away
    d_home, d_away = team_distances(match)
    assert not gates.free_ball(match, d_home, d_away).any()


def test_flight_needs_a_fast_free_ball():
    n, fps = 40, 10
    rest = np.tile([20.0, 10.0], (n, 1))
    _, flying = _gates(_two_player_match(rest, fps))
    assert not flying.any()
    moving = np.stack([5.0 + 20.0 * np.arange(n) / fps, np.full(n, 5.0)], axis=1)  # 20 m/s
    free, flying = _gates(_two_player_match(moving, fps))
    assert free.all() and flying[5:-5].all()
    with_gap = moving.copy()
    with_gap[10:20] = np.nan
    _, flying = _gates(_two_player_match(with_gap, fps))
    assert not flying[10:20].any()


def _shape_ball_match(n=120, fps=10, wide_home=True):
    """Two 11-player teams (as tests/test_learned.py's _shape_match); the ball is missing for the
    first third, on a home player for the second (controlled) and flies at 20 m/s far from
    everyone for the last third."""
    rng = np.random.default_rng(1)
    home_y = np.linspace(4, 64, 10) if wide_home else np.linspace(24, 44, 10)
    away_y = np.linspace(24, 44, 10) if wide_home else np.linspace(4, 64, 10)
    home = np.c_[np.r_[3, rng.uniform(20, 50, 10)], np.r_[34, home_y]]
    away = np.c_[np.r_[102, rng.uniform(55, 85, 10)], np.r_[34, away_y]]
    walk = np.cumsum(rng.normal(0, 0.05, (n, 22, 2)), axis=0)
    players = np.concatenate([home, away])[None] + walk
    third = n // 3
    ball = np.full((n, 2), np.nan)
    ball[third:2 * third] = players[third:2 * third, 1]
    t = np.arange(n - 2 * third) / fps
    ball[2 * third:] = np.c_[10.0 + 20.0 * t, np.full(len(t), 90.0)]  # 20+ m beyond the touchline
    return Match(match_id="test:shapeball", fps=fps, pitch=(105.0, 68.0),
                 period=np.ones(n, np.int8), ball=ball, players=players,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_shapef_adds_gated_team_columns_and_keeps_v4_columns():
    match = _shape_ball_match()
    width = learned.shape_blocks(match)[0].shape[1]
    base = learned.features(match, ("touch",))
    v4 = learned.features(match, ("shapem", "touch"))
    both = learned.features(match, ("shapem", "shapef", "touch"))
    assert both[0].shape[1] == base[0].shape[1] + 2 * width and both[2].shape == base[2].shape
    assert np.array_equal(both[0][:, :v4[0].shape[1]], v4[0])  # v4's columns first, unchanged
    assert np.array_equal(both[1][:, :v4[1].shape[1]], v4[1])
    f_home, f_away = both[0][:, -width:], both[1][:, -width:]
    third = match.n_frames // 3
    assert not f_home[:2 * third].any() and not f_away[:2 * third].any()  # missing, controlled
    assert f_home[2 * third + 5:].any(axis=1).all()  # flying: shape evidence
    swapped = dataclasses.replace(match, slot_team=1 - match.slot_team)
    s = learned.features(swapped, ("shapem", "shapef", "touch"))
    assert np.allclose(both[0], s[1]) and np.allclose(both[1], s[0]) and np.allclose(both[2], s[2])


def test_shapeo_is_the_free_ball_gate():
    match = _shape_ball_match()
    width = learned.shape_blocks(match)[0].shape[1]
    t_home = learned.features(match, ("shapeo",))[0][:, -width:]
    d_home, d_away = team_distances(match)
    free = gates.free_ball(match, d_home, d_away)
    assert not t_home[~free].any() and t_home[free].any(axis=1).all()


def test_v4_feature_hash_is_unchanged():
    assert learned.feature_hash(("shapem", "touch")) == learned.load_coefs()["feature_hash"]
    assert learned.FEATURE_HASH == learned.load_coefs()["feature_hash"]
    hashes = {learned.feature_hash(("shapem", "touch", g)) for g in ("shapef", "shapeo", "deadm")}
    assert len(hashes) == 3 and learned.FEATURE_HASH not in hashes


def test_dead_gap_raw_shapes_and_dtypes():
    match = _shape_ball_match()
    s2, s05, masks = gates.dead_gap_raw(match)
    n = match.n_frames
    assert s2.shape == (n,) and s05.shape == (n,) and masks.shape == (n, len(gates.DEAD_SHAPES))
    assert s2.dtype == np.float64 and s05.dtype == np.float64 and masks.dtype == np.float64
    assert set(np.unique(masks)) <= {0.0, 1.0}
    assert "goalkick" not in gates.DEAD_SHAPES
    again = gates.dead_gap_raw(match)  # cached masks: the same answer
    assert np.array_equal(again[2], masks)


def test_deadm_adds_common_columns_only_on_missing_frames():
    match = _shape_ball_match()
    base = learned.features(match, ("shapem", "touch"))
    dead = learned.features(match, ("shapem", "touch", "deadm"))
    extra = 2 * len(learned.SPEED_KNOTS) + len(gates.DEAD_SHAPES)
    assert dead[0].shape == base[0].shape and dead[1].shape == base[1].shape
    assert np.array_equal(dead[0], base[0]) and np.array_equal(dead[1], base[1])
    assert dead[2].shape[1] == base[2].shape[1] + extra
    assert np.array_equal(dead[2][:, :base[2].shape[1]], base[2])
    cols = dead[2][:, -extra:]
    seen = ~np.isnan(match.ball[:, 0])
    assert not cols[seen].any() and cols[~seen].any(axis=1).all()  # speeds give every gap frame evidence
