import numpy as np
import pytest

from src.models import build
from src.models.deadball import DeadBall, drop_runs, extend, fill_gaps
from src.models.postprocess import delay_switches, play_runs
from src.official import A, D, H, X
from src.schema import AWAY, HOME, Match

FPS = 10


def _match(n=600, seed=0) -> Match:
    """Players jog around their own half, the ball sits at the home player 0's feet."""
    rng = np.random.default_rng(seed)
    slot_team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    base = np.stack([np.where(slot_team == HOME, 30.0, 75.0), np.linspace(5, 63, 22)], axis=1)
    phase = rng.uniform(0, 2 * np.pi, 22)[None]
    angle = phase + np.arange(n)[:, None] / FPS  # 1 rad/s on a 3 m circle: 3 m/s, lively
    players = base[None] + 3.0 * np.stack([np.cos(angle), np.sin(angle)], axis=2)
    ball = players[:, 0, :].copy() + 0.5
    period = np.ones(n, dtype=np.int8)
    return Match("t", FPS, (105.0, 68.0), period, ball, players, slot_team).validate()


def _calm(match: Match, start: int, stop: int) -> Match:
    """Freeze the players on [start, stop): they stand, so the calm cue fires."""
    match.players[start:stop] = match.players[start]
    match.ball[start:stop] = match.ball[start]
    return match


def test_mask_helpers():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert drop_runs(mask, 3).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert fill_gaps(mask, 2).tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    assert extend(mask, 1).tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]


def test_live_match_has_no_dead_time():
    match = _match()
    model = DeadBall()
    assert not model.dead(match).any()
    assert set(np.unique(model.predict(match))) <= {H, A}


def test_calm_stretch_becomes_dead():
    match = _calm(_match(), 200, 400)
    codes = DeadBall().predict(match)
    assert (codes[230:370] == D).all()
    assert (codes[:150] != D).all() and (codes[450:] != D).all()


def test_short_calm_is_ignored():
    match = _calm(_match(), 200, 225)  # 2.5 s, under min_dead_s
    assert not DeadBall(min_dead_s=4).dead(match).any()


def test_ball_out_is_dead_for_out_s():
    match = _match()
    match.ball[300:330] = (50.0, -3.0)  # over the touchline for 3 s
    dead = DeadBall(speed_mps=0, out_s=2.0, min_dead_s=0).dead(match)
    assert dead[300:350].all() and not dead[:300].any() and not dead[351:].any()
    assert not DeadBall(speed_mps=0, out_s=-1).dead(match).any()


def test_missing_team_fills_unseen_frames():
    match = _match()
    match.ball[200:300] = np.nan  # ball unseen for 10 s, players lively
    keep = DeadBall().predict(match)
    team = DeadBall(missing="team").predict(match)
    assert (keep[230:270] == D).all()
    assert (team[200:300] == H).all()


def test_frames_outside_periods_stay_untouched():
    match = _calm(_match(), 0, 600)
    match.period[:100] = 0
    dead = DeadBall().dead(match)
    assert not dead[:100].any() and dead[150:].all()


def test_registered_deterministic_and_valid_on_synthetic(small):
    match = small["match"]
    model = build("deadball", min_spell_s=2.0)
    first, second = model.predict(match), model.predict(match)
    assert first.dtype == np.int8 and len(first) == match.n_frames
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {H, A, D}


def test_bad_missing_rejected():
    with pytest.raises(ValueError):
        DeadBall(missing="maybe")


def test_switch_delay_s_runs_after_the_filter(small):
    match = small["match"]
    base = build("deadball").predict(match)
    delayed = build("deadball", switch_delay_s=1.0).predict(match)
    assert np.array_equal(delayed, delay_switches(base, play_runs(match.period), 25))
