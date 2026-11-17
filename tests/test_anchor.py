"""Cue `anchor` (src/models/cues/anchor.py): synthetic only."""

import dataclasses

import numpy as np

from src.models import cues
from src.models.cues import anchor
from src.schema import AWAY, HOME, Match

FPS = 25


def _match(n=100):
    """Home player at (30, 30), away at (60, 30). Ball on the home player on frames 0-19 and 60-79
    (two home anchors 1.6 s apart), unseen in between and after."""
    players = np.zeros((n, 2, 2))
    players[:, 0] = (30.0, 30.0)
    players[:, 1] = (60.0, 30.0)
    ball = np.full((n, 2), np.nan)
    ball[:20] = (30.0, 30.0)
    ball[60:80] = (30.0, 30.0)
    return Match(match_id="syn", fps=FPS, pitch=(105.0, 68.0), period=np.ones(n, dtype=np.int8),
                 ball=ball, players=players, slot_team=np.array([HOME, AWAY], dtype=np.int8))


def _add(match, gain=1.0, gap=4.0):
    anchor._LAST[:] = [None, None, None]
    eff = anchor.effect(match, match, None, gain, gap=gap)
    return None if eff is None else eff.add


def test_fills_between_same_team_anchors():
    add = _add(_match())
    assert np.all(add[21:59, 0] > 0) and not np.any(add[:, 1:])
    assert not np.any(add[:20]) and not np.any(add[60:])  # anchors themselves and after the last
    # decays with the distance to the nearer anchor
    assert add[21, 0] > add[40, 0] and add[58, 0] > add[40, 0]
    assert _add(_match(), gap=1.5) is None  # the anchors are 1.6 s apart: over the cap


def test_gain_zero_is_no_effect():
    m = _match()
    assert _add(m, gain=0.0) is None
    base = {k: 1.0 for k in cues.KINDS}
    c0, a0 = cues.combine("", m, m, None, base, m.n_frames)
    c1, a1 = cues.combine("anchor:0/gap=4", m, m, None, base, m.n_frames)
    assert a0 is None and a1 is None
    assert all(np.array_equal(c0[k], c1[k]) for k in cues.KINDS)


def test_swapping_teams_swaps_the_evidence():
    m = _match()
    swapped = dataclasses.replace(m, slot_team=m.slot_team[::-1].copy())
    a, b = _add(m), _add(swapped)
    assert np.any(a) and np.array_equal(a[:, [1, 0, 2]], b)


def test_ball_off_the_pitch_blocks_the_fill():
    m = _match()
    ball = m.ball.copy()
    ball[40] = (-5.0, 30.0)  # seen off the pitch between the two home anchors
    assert _add(dataclasses.replace(m, ball=ball)) is None


def test_period_change_and_other_team_block_the_fill():
    m = _match()
    period = m.period.copy()
    period[50:] = 2
    assert _add(dataclasses.replace(m, period=period)) is None
    ball = m.ball.copy()
    ball[60:80] = (60.0, 30.0)  # the second anchor is the away team's
    assert _add(dataclasses.replace(m, ball=ball)) is None
