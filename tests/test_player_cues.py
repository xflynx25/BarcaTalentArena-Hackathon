"""Players-only features for the ball-missing stretch (src/player_cues.py), synthetic only."""

import numpy as np

from src import player_cues
from src.schema import Match


def _match(n=200, seed=0, run=0.0):
    rng = np.random.default_rng(seed)
    period = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)].astype(np.int8)
    base_home = np.c_[np.r_[3, rng.uniform(10, 60, 10)], rng.uniform(5, 63, 11)]
    base_away = np.c_[np.r_[102, rng.uniform(45, 95, 10)], rng.uniform(5, 63, 11)]
    walk = np.cumsum(rng.normal(0, 0.3, (n, 22, 2)), axis=0)
    pos = np.concatenate([base_home, base_away])[None] + walk
    pos[:, :, 1] += run * np.sin(np.arange(n) / 15)[:, None] * np.r_[np.ones(11), -np.ones(11)]
    second = period == 2
    pos[second, :, 0] = 105 - pos[second, :, 0]
    pos[second, :, 1] = 68 - pos[second, :, 1]
    return Match(match_id="t", fps=10, pitch=(105.0, 68.0), period=period,
                 ball=np.full((n, 2), np.nan), players=pos,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_swapping_teams_negates_differences_and_keeps_two_team_values():
    m = _match(run=4.0)  # a lateral sway, so both teams have a heading
    swapped = Match(match_id="s", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
                    players=m.players, slot_team=1 - m.slot_team)
    a = player_cues.pair_values(player_cues.team_values(m)[1])
    b = player_cues.pair_values(player_cues.team_values(swapped)[1])
    symmetric = {"brake_both", "coh_min", "dhead1_max", "dhead2_max"}
    for name, value in a.items():
        known = ~np.isnan(value)
        assert known.sum() > 20, name
        sign = 1 if name in symmetric else -1
        assert np.allclose(value[known], sign * b[name][known], atol=1e-6), name


def test_team_running_straight_is_coherent_and_keeps_its_heading():
    m = _match()
    p = m.players.copy()
    p[:, :11, 0] += np.arange(m.n_frames)[:, None] * 0.5 * np.where(m.period == 1, 1, -1)[:, None]
    m = Match(match_id="r", fps=10, pitch=m.pitch, period=m.period, ball=m.ball, players=p,
              slot_team=m.slot_team)
    _, v = player_cues.team_values(m)
    coh, dh = v["coherence"][20:80, 0], v["dhead1"][20:80, 0]
    assert np.nanmedian(coh) > 0.9 and np.nanmedian(dh) < 0.2
    assert np.nanmedian(v["v_opp_spot"][20:80, 0]) > 0  # running forward: toward the box it attacks


def test_too_few_seen_outfielders_is_nan():
    m = _match()
    p = m.players.copy()
    p[:, 1:5] = np.nan  # home keeps 6 outfielders
    m = Match(match_id="n", fps=10, pitch=m.pitch, period=m.period, ball=m.ball, players=p,
              slot_team=m.slot_team)
    _, v = player_cues.team_values(m)
    for name in ("opp_box", "coherence", "brake", "back4_width"):
        assert np.isnan(v[name][:, 0]).all(), name
        assert (~np.isnan(v[name][:, 1])).sum() > 20, name
