"""C16 (2): the holder stand-in repair (src/repairs/holder.py)."""

import numpy as np

from src import repairs
from src.ingest import synthetic as synthetic_source
from src.schema import AWAY, HOME, Match


def _match(n=100, fps=25):
    t = np.arange(n) / fps
    ball = np.stack([20 + 3 * t, 30 + 0 * t], 1)  # 3 m/s: carried
    players = np.stack([ball + [1.0, 0.0], np.stack([40 + 0 * t, 50 + 0 * t], 1),
                        np.stack([60 + 0 * t, 10 + 0 * t], 1)], axis=1)
    return Match(match_id="test:holder", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=ball, players=players,
                 slot_team=np.array([HOME, HOME, AWAY], np.int8)).validate()


def test_holder_moves_with_the_ball_then_stops():
    m = _match()
    m.players[40:, 0] = np.nan
    out = repairs.apply(m, "holder/max_s=1")
    seen = ~np.isnan(out.players).any(axis=2)
    assert seen[40:65, 0].all() and not seen[65:, 0].any()
    assert np.allclose(out.players[50, 0], m.ball[50] + [1.0, 0.0])


def test_holder_stops_when_someone_else_arrives_or_the_ball_goes():
    m = _match()
    m.players[40:, 0] = np.nan
    m.players[50:, 2] = m.ball[50:] + [0.5, 0.5]  # an opponent reaches the ball at 50
    out = repairs.apply(m, "holder")
    seen = ~np.isnan(out.players).any(axis=2)
    assert seen[40:50, 0].all() and not seen[50:, 0].any()
    m2 = _match()
    m2.players[40:, 0] = np.nan
    m2.ball[45:] = np.nan
    seen2 = ~np.isnan(repairs.apply(m2, "holder").players).any(axis=2)
    assert seen2[40:45, 0].all() and not seen2[45:, 0].any()


def test_holder_ignores_a_player_away_from_the_ball_and_returns_the_same_object():
    m = _match()
    m.players[40:, 1] = np.nan  # not at the ball
    assert repairs.apply(m, "holder") is m
    clean = synthetic_source.load("0")
    out = repairs.apply(clean, "holder")
    assert out is clean or np.isnan(clean.players).sum() > np.isnan(out.players).sum()


def test_holder_does_not_follow_a_passed_ball():
    m = _match()
    m.players[40:, 0] = np.nan
    m.ball[40:] = m.ball[39] + np.arange(1, 61)[:, None] * [0.6, 0.0]  # 15 m/s from frame 40
    assert repairs.apply(m, "holder") is m
