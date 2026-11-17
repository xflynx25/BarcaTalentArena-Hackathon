"""Restart rule (src/restarts.py): windows from Match.tags, the dead stretch before the kick."""

import numpy as np

from src import restarts
from src.schema import Match
from src.smart_tags import Tag


def _match(match_id="BAR-ATH:tactical", tags=None, fps=10):
    n = 120
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[60:, 0] = 50.0 + np.arange(n - 60) * 1.0  # 10 m/s from frame 60 at 10 fps
    return Match(match_id=match_id, fps=fps, pitch=(100.0, 60.0), period=np.ones(n, dtype=np.int8),
                 ball=ball, players=np.full((n, 2, 2), np.nan), slot_team=np.array([0, 1], dtype=np.int8),
                 tags=tags).validate()


def test_dead_before_the_kick():
    m = _match(tags=[Tag(55, 90, "SET PIECES", -1), Tag(0, 120, "BUILD UP", 0)])
    dead = restarts.dead_stretch(m, back_s=2.0, kick_mps=4.0, search_s=8.0)
    kick = 61  # the first frame whose 0.2 s speed is over 4 m/s
    assert dead[kick - 20:kick].all() and dead.sum() == 20


def test_no_kick_in_the_search_window():
    m = _match(tags=[Tag(0, 30, "SET PIECES", -1)])
    assert not restarts.dead_stretch(m).any()


def test_no_windows_without_a_source():
    assert len(restarts.windows(_match(tags=None))) == 0
    assert len(restarts.windows(_match("synthetic:x"))) == 0
    assert len(restarts.windows(_match("skillcorner:1", fps=25))) == 0  # only at the source's own rate


def test_off_by_default_and_added_to_dead_only():
    from src.models.learned import LearnedDecoder

    m = _match(tags=[Tag(55, 90, "SET PIECES", -1)])
    out = np.zeros((m.n_frames, 3))
    assert LearnedDecoder().add_restart_rule(m, out) is out  # gain 0: untouched
    on = LearnedDecoder(restart_rule_gain=4.0).add_restart_rule(m, out)
    assert not out.any() and not on[:, :2].any()
    np.testing.assert_allclose(on[:, 2], 4.0 * restarts.dead_stretch(m) / m.fps)


def test_emission_models_add_the_rule():
    # v5 (bigger) builds its scores in EmissionDecoder.combine, not LearnedDecoder.scores
    from src.models.emission import BiggerDecoder

    m = _match(tags=[Tag(55, 90, "SET PIECES", -1)])
    lp = np.zeros((m.n_frames, 3))
    off = BiggerDecoder(hand_team=0.0, hand_dead=0.0).combine(m, lp)
    on = BiggerDecoder(hand_team=0.0, hand_dead=0.0, restart_rule_gain=4.0).combine(m, lp)
    np.testing.assert_array_equal(on[:, :2], off[:, :2])
    np.testing.assert_allclose(on[:, 2] - off[:, 2], 4.0 * restarts.dead_stretch(m) / m.fps)
    assert on[:, 2].max() > off[:, 2].max()


def test_noisy_windows_are_seeded_and_off_at_zero():
    m = _match(tags=[Tag(10 * i, 10 * i + 8, "SET PIECES", -1) for i in range(11)])
    wins = restarts.windows(m)
    assert restarts.noisy(m, wins, 0) is wins
    a, b = restarts.noisy(m, wins, 2), restarts.noisy(m, wins, 2)
    np.testing.assert_array_equal(a, b)
    assert a.shape[1] == 2 and not np.array_equal(a, wins)
    assert restarts.dead_stretch(m, noise=0).sum() == restarts.dead_stretch(m).sum()

