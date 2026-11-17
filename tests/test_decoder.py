"""Switch-timing decoder (backlog B3 + B7): evidence, Viterbi and the registered model."""

import numpy as np
import pytest

from src.models import build
from src.models.deadball import player_speed
from src.models.decoder import (SwitchDecoder, ball_jitter, contact, forward_window,
                                 player_jitter, shift, team_evidence, viterbi)
from src.official import A, D, H

HS, AS, DS = 0, 1, 2  # viterbi states


def scores(team, dead=None):
    team = np.asarray(team, dtype=float)
    dead = np.zeros_like(team) - 1.0 if dead is None else np.asarray(dead, dtype=float)
    return np.stack([team, -team, dead], axis=1)


def test_team_evidence_is_zero_out_of_reach_and_when_missing():
    d_home = np.array([0.0, 1.0, 3.0, np.nan, 0.5])
    d_away = np.array([5.0, 1.0, 4.0, np.nan, np.nan])
    assert np.allclose(team_evidence(d_home, d_away, 2.0), [1.0, 0.0, 0.0, 0.0, 0.75])


def test_far_weight_gives_the_nearer_team_evidence_out_of_reach():
    d_home = np.array([3.0, 6.0, 0.5, np.nan])
    d_away = np.array([4.0, 3.0, 5.0, 3.0])
    out = team_evidence(d_home, d_away, 2.0, far_weight=0.2)
    assert np.allclose(out, [0.1, -0.2, 0.75, 0.0])


def test_gauss_contact_decays_smoothly_and_is_zero_when_missing():
    """C5: the gauss shape has no cut-off at reach, but still vanishes far away and when missing."""
    d = np.array([0.0, 1.5, 3.0, 10.0, np.nan])
    c = contact(d, 1.5, "gauss")
    assert c[0] == 1.0 and 0.5 < c[1] < 0.7 and 0 < c[2] < c[1] and c[3] < 1e-5 and c[4] == 0.0
    assert np.allclose(contact(d, 1.5), [1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        contact(d, 1.5, "nope")


def _noisy_players(match, sd, seed=0):
    rng = np.random.default_rng(seed)
    return type(match)(**{**match.__dict__,
                          "players": match.players + rng.normal(0, sd, match.players.shape)})


def test_span_speed_ignores_frame_noise_but_keeps_motion(small):
    """C5/F72: frame-to-frame position noise inflates frame-step speeds, not span speeds."""
    match = small["match"]
    noisy = _noisy_players(match, 0.05)
    step_gain = np.nanmean(player_speed(noisy, 1.0)) - np.nanmean(player_speed(match, 1.0))
    span_gain = (np.nanmean(player_speed(noisy, 1.0, span_s=1.0))
                 - np.nanmean(player_speed(match, 1.0, span_s=1.0)))
    assert step_gain > 0.5 and abs(span_gain) < step_gain / 5
    clean = np.nanmean(player_speed(match, 1.0))
    assert abs(np.nanmean(player_speed(match, 1.0, span_s=1.0)) - clean) < 0.25 * clean


def test_player_jitter_scales_reach_and_new_knobs_default_off(small):
    match = small["match"]
    noisy = _noisy_players(match, 0.2)
    assert player_jitter(noisy) > 2 * player_jitter(match)
    base = SwitchDecoder().scores(match)
    assert np.array_equal(SwitchDecoder(speed_span_s=0, reach_shape="linear",
                                        pjitter_ref_m=0).scores(match), base)
    wide = SwitchDecoder(pjitter_ref_m=player_jitter(noisy) / 4).scores(noisy)
    assert not np.array_equal(wide, SwitchDecoder().scores(noisy))


def test_ball_jitter_grows_with_noise_and_scales_the_costs(small):
    match = small["match"]
    rng = np.random.default_rng(0)
    noisy = type(match)(**{**match.__dict__, "ball": match.ball + rng.normal(0, 0.2, match.ball.shape)})
    assert ball_jitter(noisy) > ball_jitter(match)
    model = SwitchDecoder(noise_ref_m=ball_jitter(noisy) / 2)
    assert np.isclose(model.noise_scale(noisy), 2.0)
    assert model.noise_scale(match) == 1.0  # never scales down
    assert SwitchDecoder(noise_ref_m=0).noise_scale(noisy) == 1.0  # 0 turns it off


def test_switch_goes_to_the_latest_frame_of_a_flight():
    # home controls, ball in flight (no evidence), away controls: switch at the away reach
    path = viterbi(scores([1, 1, 1, 0, 0, 0, 0, -1, -1, -1]), switch_cost=0.5, dead_cost=5)
    assert path.tolist() == [HS] * 7 + [AS] * 3


def test_switch_cost_ignores_short_touches():
    team = [1] * 5 + [-1] + [1] * 5
    assert (viterbi(scores(team), switch_cost=1.5, dead_cost=5) == HS).all()
    assert viterbi(scores(team), switch_cost=0.5, dead_cost=5)[5] == AS


def test_dead_state_needs_enough_evidence():
    dead = [-1] * 4 + [1] * 6 + [-1] * 4
    team = [1] * 4 + [0] * 6 + [1] * 4
    assert viterbi(scores(team, dead), switch_cost=1, dead_cost=2).tolist() == [HS] * 4 + [DS] * 6 + [HS] * 4
    assert (viterbi(scores(team, dead), switch_cost=1, dead_cost=4) == HS).all()


def test_shift_moves_evidence_and_zero_fills():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert shift(values, 1).tolist() == [0, 1, 2, 3]
    assert shift(values, -2).tolist() == [3, 4, 0, 0]
    assert shift(values, 0) is values


def test_model_is_registered_deterministic_and_valid(small):
    match = small["match"]
    model = build("viterbi")
    assert isinstance(model, SwitchDecoder)
    first, second = model.predict(match), build("viterbi").predict(match)
    assert first.shape == (match.n_frames,) and first.dtype == np.int8
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {H, A, D}


def test_min_spell_off_and_on(small):
    match = small["match"]
    raw = build("viterbi", min_spell_s=0).predict(match)
    filtered = build("viterbi", min_spell_s=4).predict(match)
    assert np.count_nonzero(np.diff(filtered)) <= np.count_nonzero(np.diff(raw))


def test_forward_window_moves_one_edge_of_a_dead_stretch():
    dead = np.array([-1, -1, 1, 1, 1, 1, -1, -1], dtype=float)
    # min: the stretch keeps its start and ends 2 frames earlier (restart_lead_s)
    assert forward_window(dead, 2, np.min).tolist() == [-1, -1, 1, 1, -1, -1, -1, -1]
    # max: it starts 2 frames earlier and keeps its end (onset_lead_s)
    assert forward_window(dead, 2, np.max).tolist() == [1, 1, 1, 1, 1, 1, -1, -1]
    assert forward_window(dead, 0, np.min) is dead


def test_leads_move_dead_stretches_earlier(small):
    match = small["match"]
    base = build("viterbi", onset_lead_s=0, restart_lead_s=0).scores(match)[:, 2]
    led = build("viterbi", onset_lead_s=0, restart_lead_s=1).scores(match)[:, 2]
    assert np.all(led <= base + 1e-12)  # ending earlier only ever lowers the dead evidence
    led = build("viterbi", onset_lead_s=1, restart_lead_s=0).scores(match)[:, 2]
    assert np.all(led >= base - 1e-12)  # starting earlier only ever raises it
