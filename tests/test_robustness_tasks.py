"""B18 / B19 / B20 / B21 / C7 / C8 (5 Nov wave): controlled touches, the dropout-robust ball-noise
measure, SkillCorner's onset option, the whistle cue, and the position-noise kinds."""

import numpy as np
import pytest

from src import noise
from src.ingest.skillcorner import out_onsets
from src.models import build
from src.models.decoder import ball_jitter, ball_noise, noise_level, player_jitter
from src.models.learned import controlled, features
from src.models.onset import braking, cue_starts, whistle_events
from src.official import A, D, H
from src.schema import Match


def _line(n=200, fps=10, speed=3.0):
    """A ball rolling along x at `speed` m/s, nobody else; one half."""
    t = np.arange(n) / fps
    ball = np.stack([10 + speed * t, np.full(n, 30.0)], axis=1)
    players = np.full((n, 2, 2), np.nan)
    players[:, 0] = ball + [1.0, 0.0]
    players[:, 1] = ball + [5.0, 0.0]
    return Match(match_id="t:1", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=ball, players=players, slot_team=np.array([0, 1], np.int8),
                 labels=np.full(n, H, np.int8)).validate()


def test_ball_noise_ignores_still_frames_and_sees_noise():
    m = _line()
    rng = np.random.default_rng(0)
    noisy = Match(**{**m.__dict__, "ball": m.ball + rng.normal(0, 0.05, m.ball.shape)})
    assert ball_noise(m) < 1e-9 < ball_noise(noisy)
    # a still ball for half the match pulls ball_jitter down but leaves ball_noise alone
    still = noisy.ball.copy()
    still[:100] = still[100]
    half = Match(**{**m.__dict__, "ball": still})
    assert ball_jitter(half) < ball_jitter(noisy)
    assert np.isclose(ball_noise(half), ball_noise(Match(**{**m.__dict__, "ball": noisy.ball})),
                      rtol=0.35)
    assert noise_level(noisy, "noise") == ball_noise(noisy)
    with pytest.raises(ValueError):
        noise_level(noisy, "nope")
    with pytest.raises(ValueError):
        build("learned", noise_measure="nope")


def test_controlled_keeps_only_touches_the_team_keeps():
    t = np.zeros(10)
    t[[1, 5]] = 1.0
    own = np.array([1, 1, 1, 1, 1, 1, 9, 9, 9, 9], float)
    other = np.full(10, 3.0)
    out = controlled(t, own, other, fps=1, control_s=2, keep=0.7)
    assert out.tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_ctouch_group_adds_the_touch_columns(small):
    match = small["match"]
    base = features(match, ("touch",))
    both = features(match, ("ctouch", "touch"))
    assert both[0].shape[1] == base[0].shape[1] + 15 and both[2].shape == base[2].shape


def test_out_onsets_moves_dead_to_the_ball_leaving():
    n = 12
    labels = np.array([H] * 4 + [D] * 8, np.int8)
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[6:, 0] = 106.0  # off the pitch from frame 6
    out = out_onsets(labels, ball, (105.0, 68.0), np.ones(n, np.int8), within=3)
    assert out.tolist() == [H] * 6 + [D] * 6
    # too far after the onset: unchanged
    assert out_onsets(labels, ball, (105.0, 68.0), np.ones(n, np.int8), within=1).tolist() == \
        labels.tolist()


def test_whistle_fires_when_both_teams_brake():
    n, fps = 100, 10
    players = np.zeros((n, 4, 2))
    speed = np.where(np.arange(n) < 50, 4.0, 0.0)
    x = np.cumsum(speed) / fps
    for j in range(4):
        players[:, j, 0] = x + j
        players[:, j, 1] = 10.0 * j
    m = Match(match_id="t:1", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
              ball=np.full((n, 2), 50.0), players=players,
              slot_team=np.array([0, 0, 1, 1], np.int8), labels=np.full(n, H, np.int8)).validate()
    home, away, mean = braking(m)
    assert home[50] == 1 and away[50] == 1 and mean[50] > 3
    events = np.flatnonzero(whistle_events(m))
    assert len(events) == 1 and abs(events[0] - 50) <= 2
    assert "whistle" in cue_starts(m)


@pytest.mark.parametrize("kind", noise.KINDS)
def test_noise_kinds_are_seeded_and_sized(small, kind):
    match = small["match"]
    a = noise.add_noise(match, kind, 0.5, 5.0, 1)
    b = noise.add_noise(match, kind, 0.5, 5.0, 1)
    assert np.array_equal(a.players, b.players, equal_nan=True)
    moved = (a.ball - match.ball) if kind == "ball" else (a.players - match.players)
    assert 0.2 < np.nanstd(moved) < 0.8
    if kind in ("smooth", "offset", "shift", "calib"):  # no frame-level steps (C8)
        assert player_jitter(a) < 3 * player_jitter(match) + 0.02
    if kind == "shift":  # ball and players move together
        move = (a.players - match.players)[:, 0]
        seen = np.isfinite(move[:, 0]) & np.isfinite(match.ball[:, 0])
        assert seen.any() and np.allclose((a.ball - match.ball)[seen], move[seen])
