"""B39's cue `flight` (src/models/cues/flight.py); synthetic only."""

import dataclasses

import numpy as np
import pytest

from src.models.cues import flight
from src.models.learned import LearnedDecoder


def _kinked(fps=25, n=200, at=100):
    t = np.arange(n) / fps
    ball = np.column_stack([10 * t, 2 * t])
    ball[at:] = ball[at] + np.column_stack([-8 * (t[at:] - t[at]), 5 * (t[at:] - t[at])])
    return ball


def test_ball_path_smooth_and_kink():
    ball = _kinked()
    path = flight.ball_path(ball, 25)
    assert path["dv"][50] < 1e-6 and path["res"][50] == pytest.approx(-2.0)  # straight: no bend
    assert path["dv"][100] > 15 and path["res"][100] > -1.0  # the kink
    assert path["speed"][50] == pytest.approx(np.hypot(10, 2))
    ball[150:160] = np.nan  # too few seen frames in the window -> unknown
    path = flight.ball_path(ball, 25)
    assert np.isnan(path["dv"][149]) and np.isnan(path["res"][155])


def test_a_smooth_arc_is_smooth():
    t = np.arange(100) / 25
    ball = np.column_stack([12 * t, 8 * t - 4.9 * t ** 2])  # constant acceleration
    path = flight.ball_path(ball, 25)
    assert np.nanmax(path["res"][10:90]) == pytest.approx(-2.0, abs=1e-3)
    assert np.nanmax(path["dv"][10:90]) < 3.5  # 9.8 m/s2 over the 0.32 s between the two fits


def test_no_ball_no_cue_and_gain_zero(small):
    match = small["match"]
    dec = LearnedDecoder()
    assert flight.effect(match, match, dec, 0.0) is None
    gone = dataclasses.replace(match, ball=np.full_like(match.ball, np.nan))
    eff = flight.effect(gone, gone, dec, 1.0)
    assert np.all(eff.ha == 1.0) and np.all(eff.ah == 1.0)
    for fit in ("", "nodx"):
        eff = flight.effect(match, match, dec, 1.0, fit=fit)
        assert np.all((eff.ha >= flight.COST_RANGE[0]) & (eff.ha <= flight.COST_RANGE[1]))
        assert not np.all(eff.ha == 1.0)
    with pytest.raises(ValueError):
        flight.effect(match, match, dec, 1.0, fit="nosuch")
    with pytest.raises(ValueError, match="unknown options"):
        flight.effect(match, match, dec, 1.0, win=2)


def test_decoder_runs_and_is_deterministic(small):
    match = small["match"]
    a = LearnedDecoder(cues="flight:0.5").predict(match)
    b = LearnedDecoder(cues="flight:0.5").predict(match)
    assert np.array_equal(a, b) and len(a) == match.n_frames
    assert np.array_equal(LearnedDecoder(cues="flight:0").predict(match),
                          LearnedDecoder().predict(match))


def test_spec_hash_is_checked(tmp_path):
    bad = tmp_path / "coefs.json"
    bad.write_text('{"spec_hash": "nope"}')
    with pytest.raises(ValueError, match="another flight spec"):
        flight.load("", str(bad))


def test_ball_artefact_gates(small):
    ball = _kinked()
    jit, straight = flight.ball_measures(ball)
    assert jit < 1e-9 and straight > 0.9  # drawn as two straight lines
    noisy = ball + np.random.default_rng(0).normal(0, 0.02, ball.shape)
    jit, straight = flight.ball_measures(noisy)
    assert jit > 0.01 and straight == 0.0
    match = small["match"]
    dec = LearnedDecoder()
    jit, straight = flight.ball_measures(match.ball)
    assert flight.effect(match, match, dec, 1.0, jitter=jit / 2) is None  # too noisy: off
    assert flight.effect(match, match, dec, 1.0, jitter=jit * 2) is not None
    if straight > 0:
        assert flight.effect(match, match, dec, 1.0, lin=straight / 2) is None  # drawn: off
    assert flight.effect(match, match, dec, 1.0, lin=min(1.0, straight * 2 + 1e-3)) is not None
