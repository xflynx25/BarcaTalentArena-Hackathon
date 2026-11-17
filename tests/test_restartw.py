"""Cue `restartw` (B35b, src/models/cues/restartw.py): restart windows from the tracking; synthetic only."""

import types

import numpy as np

from src.models import cues
from src.models.cues import restartw
from src.schema import Match
from src.smart_tags import NA, SET_PIECES, Tag

FPS = 25


def _match(ball_rest, rest=(500, 600), n=1000, tags=None):
    """60 s-ish: the ball rolls across midfield, rests at ball_rest over `rest` (frames), then is
    kicked at 15 m/s towards the centre. HOME (x 10-45) defends x = 0, AWAY (x 60-95) the far end;
    everyone stands still, a HOME player 1 m from the ball's rest spot."""
    t = np.arange(n) / FPS
    ball = np.stack([50 + 5 * np.sin(t), 34 + 10 * np.cos(t / 2)], axis=1)
    a, b = rest
    ball[a:b] = ball_rest
    kick = np.array(ball_rest, dtype=float)
    direction = np.array([52.5, 34.0]) - kick
    direction /= np.linalg.norm(direction)
    for i, f in enumerate(range(b, n)):
        ball[f] = kick + direction * 15.0 * (i + 1) / FPS
    home = np.stack([np.linspace(10, 45, 11), np.linspace(5, 63, 11)], axis=1)
    away = np.stack([np.linspace(60, 95, 11), np.linspace(5, 63, 11)], axis=1)
    home[0] = np.array(ball_rest) + [1.0, 0.0]
    players = np.repeat(np.concatenate([home, away])[None], n, axis=0)
    return Match("synthetic:restartw", FPS, (105.0, 68.0), np.ones(n, dtype=np.int8), ball, players,
                 np.array([0] * 11 + [1] * 11, dtype=np.int8), tags=tags).validate()


def test_goal_kick_is_found_and_marks_before_the_kick():
    m = _match((4.0, 34.0))
    wins = restartw.detect(m)
    assert len(wins) == 1
    w = wins[0]
    assert w.kind == "goalkick" and w.route == "ball"
    assert 600 <= w.kick <= 600 + 10  # the kick test sees the ball moving within 0.4 s
    assert 500 <= w.rest <= 500 + 6 and w.start == w.rest  # speed is measured over 0.2 s


def test_goal_kick_with_an_attacker_in_the_box_is_not_one():
    m = _match((4.0, 34.0))
    m.players[:, 11] = [10.0, 30.0]  # an AWAY player inside HOME's penalty area: a keeper's hold
    assert all(w.kind != "goalkick" for w in restartw.detect(m))


def test_corner_and_throw_spots():
    assert [w.kind for w in restartw.detect(_match((104.5, 67.5)))] == ["corner"]
    assert [w.kind for w in restartw.detect(_match((60.0, 0.3)))] == ["throw"]


def test_a_still_ball_in_midfield_is_no_restart():
    assert restartw.detect(_match((40.0, 30.0))) == []


def test_effect_adds_dead_evidence_and_gain_zero_is_nothing():
    m = _match((4.0, 34.0))
    dec = types.SimpleNamespace(restart_rule_gain=0.0)
    assert restartw.effect(m, m, dec, 0.0) is None
    eff = restartw.effect(m, m, dec, 2.0)
    w = restartw.detect(m)[0]
    assert np.allclose(eff.add[w.start:w.kick, 2], 2.0 / FPS)
    assert not eff.add[:w.start].any() and not eff.add[w.kick:].any() and not eff.add[:, :2].any()
    assert eff.ha is None and eff.dh is None


def test_union_with_the_restart_rule_counts_no_frame_twice():
    tags = [Tag(560, 640, SET_PIECES, NA)]
    m = _match((4.0, 34.0), tags=tags)
    rule = types.SimpleNamespace(restart_rule_gain=2.0, restart_rule_back_s=2.0, restart_rule_kick_mps=4.0,
                                 restart_rule_search_s=8.0, restart_rule_noise=0)
    eff = restartw.effect(m, m, rule, 1.0)
    w = restartw.detect(m)[0]
    # the rule marks the 2 s before the kick; the cue keeps only the frames before that
    assert eff.add[w.start:w.kick - 2 * FPS, 2].all()
    assert not eff.add[w.kick - 2 * FPS:w.kick, 2].any()


def test_registered_as_a_cue_and_options_checked():
    cues.check("restartw:2/back_s=6/out_w=1")
    assert cues.lookup("restartw") is restartw.effect
    try:
        restartw.detect(_match((4.0, 34.0)), nosuch=1)
    except ValueError as exc:
        assert "unknown options" in str(exc)
    else:
        raise AssertionError("an unknown option must raise")


def test_deterministic():
    a = [restartw.detect(_match(p)) for p in ((4.0, 34.0), (104.5, 67.5))]
    b = [restartw.detect(_match(p)) for p in ((4.0, 34.0), (104.5, 67.5))]
    assert a == b
