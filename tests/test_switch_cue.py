"""Switch-timing cue (local L2, src/models/switch_cue.py): decoder, costs, symmetry, file; synthetic only."""

import json
import types

import numpy as np
import pytest

from src.models import switch_cue
from src.models.decoder import viterbi
from src.models.learned import LearnedDecoder
from src.schema import Match


def _match(n=300, seed=0):
    """Two 11-player teams that wander and speed up / slow down, home defending x = 0 then x = 105."""
    rng = np.random.default_rng(seed)
    period = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)].astype(np.int8)
    base_home = np.c_[np.r_[3, rng.uniform(10, 50, 10)], rng.uniform(5, 63, 11)]
    base_away = np.c_[np.r_[102, rng.uniform(55, 95, 10)], rng.uniform(5, 63, 11)]
    home_fast = (np.sin(np.arange(n) / 15.0) > 0)[:, None, None] * np.r_[np.ones(11), np.zeros(11)][None, :, None]
    pace = 0.05 + 0.3 * home_fast  # the home team speeds up and slows down
    walk = np.cumsum(rng.normal(0, 1.0, (n, 22, 2)) * pace, axis=0)
    pos = np.concatenate([base_home, base_away])[None] + walk
    second = period == 2
    pos[second, :, 0] = 105 - pos[second, :, 0]
    pos[second, :, 1] = 68 - pos[second, :, 1]
    return Match(match_id="t", fps=10, pitch=(105.0, 68.0), period=period,
                 ball=np.full((n, 2), np.nan), players=pos,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


@pytest.mark.parametrize("n", [0, 1, 2, 700])
def test_viterbi_var_with_constant_costs_is_viterbi(n):
    rng = np.random.default_rng(n)
    scores = rng.normal(0, 1, (n, 3))
    for switch, dead in ((0.75, 3.0), (0.3, 0.5), (2.0, 1.0)):
        flat = np.full(n, switch)
        assert np.array_equal(switch_cue.viterbi_var(scores, flat, flat, dead),
                              viterbi(scores, switch, dead))


def test_a_cheaper_frame_moves_a_marginal_switch():
    scores = np.full((20, 3), -5.0)
    scores[:8, 0], scores[13:, 1] = 1.0, 1.0  # H clearly early, A clearly late, 8..12 neutral
    scores[8:13, :2] = 0.0
    flat = np.full(20, 0.5)
    path = switch_cue.viterbi_var(scores, flat, flat, 3.0)
    assert np.flatnonzero(np.diff(path))[0] + 1 == 13  # ties: the change happens as late as possible
    cheap = flat.copy()
    cheap[9] = 0.4
    path = switch_cue.viterbi_var(scores, cheap, flat, 3.0)
    assert np.flatnonzero(np.diff(path))[0] + 1 == 9
    # the other direction's cost doesn't matter for an H->A switch
    assert np.array_equal(switch_cue.viterbi_var(scores, flat, cheap, 3.0),
                          viterbi(scores, 0.5, 3.0))


def test_costs_edge_cases_and_range(monkeypatch):
    z_ha = np.array([np.nan, -50.0, 50.0, 0.0, 1.0, np.nan])
    z_ah = np.array([0.0, np.nan, 2.0, -2.0, 50.0, -50.0])
    monkeypatch.setattr(switch_cue, "hazard_logits", lambda match, beta: (z_ha, z_ah))
    ball = np.zeros((6, 2))
    ball[[1, 2, 4], :] = np.nan  # missing on frames 1, 2, 4
    match = types.SimpleNamespace(ball=ball)
    coefs = {"beta": [0.0], "base_logit": 0.5}
    c_ha, c_ah = switch_cue.costs(match, coefs, 0.75, 1.0, "all")
    for z, c in ((z_ha, c_ha), (z_ah, c_ah)):
        assert np.all(c[np.isnan(z)] == 0.75)
        lo, hi = switch_cue.COST_RANGE
        assert np.all((c >= 0.75 * lo) & (c <= 0.75 * hi))
    assert c_ha[1] == 0.75 * switch_cue.COST_RANGE[1] and c_ha[2] == 0.75 * switch_cue.COST_RANGE[0]
    assert np.isclose(c_ha[3], 0.75 * np.exp(0.5))
    m_ha, m_ah = switch_cue.costs(match, coefs, 0.75, 1.0, "missing")
    seen = ~np.isnan(ball[:, 0])
    assert np.all(m_ha[seen] == 0.75) and np.all(m_ah[seen] == 0.75)
    assert np.array_equal(m_ha[~seen], c_ha[~seen]) and np.array_equal(m_ah[~seen], c_ah[~seen])
    g_ha, g_ah = switch_cue.costs(match, coefs, 0.75, 0.0, "all")
    assert np.all(g_ha == 0.75) and np.all(g_ah == 0.75)
    with pytest.raises(ValueError):
        switch_cue.costs(match, coefs, 0.75, 1.0, "seen")


def test_cue_values_are_antisymmetric():
    m = _match()
    swapped = Match(match_id="s", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
                    players=m.players, slot_team=1 - m.slot_team)
    u_away, u_home = switch_cue.cue_values(m)
    s_away, s_home = switch_cue.cue_values(swapped)
    assert u_away.shape == (m.n_frames, len(switch_cue.COLUMNS))
    known = ~np.isnan(u_away)
    assert known.sum() > 0.5 * known.size
    assert np.array_equal(np.isnan(u_home), ~known)
    assert np.allclose(u_home[known], -u_away[known])
    assert np.allclose(s_away[known], u_home[known], atol=1e-9)
    assert np.allclose(s_home[known], u_away[known], atol=1e-9)
    beta = np.linspace(-1, 1, switch_cue.design(u_away[:1]).shape[1])
    z_ha, z_ah = switch_cue.hazard_logits(m, beta)
    sz_ha, sz_ah = switch_cue.hazard_logits(swapped, beta)
    ok = ~np.isnan(z_ha)
    assert np.allclose(sz_ha[ok], z_ah[ok]) and np.allclose(sz_ah[ok], z_ha[ok])


def test_design_shape_and_fit_recovers_a_signal():
    rng = np.random.default_rng(1)
    u = rng.normal(0, 0.4, (4000, len(switch_cue.COLUMNS)))
    u[:5] = np.nan
    x = switch_cue.design(u)
    assert x.shape == (4000, 1 + sum(len(switch_cue.CUE_KNOTS[f]) for f, _ in switch_cue.COLUMNS))
    assert np.all(x[:5, 1:] == 0) and np.all(x[:, 0] == 1)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(3 * np.nan_to_num(u[:, 0]) - 2)))).astype(float)
    beta = switch_cue.fit_binary(x, y, np.full(4000, 1 / 4000))
    z = switch_cue.logits_of(u, beta)
    assert np.all(np.isnan(z[:5])) and np.corrcoef(z[5:], u[5:, 0])[0, 1] > 0.8


def _write_cue(path, beta=None, base=0.0, spec_hash=None, match_id="synthetic:0"):
    n = switch_cue.design(np.zeros((1, len(switch_cue.COLUMNS)))).shape[1]
    entry = switch_cue.fit_entry(np.zeros(n) if beta is None else beta, base)
    with open(path, "w") as handle:
        json.dump({"spec": switch_cue.SPEC, "spec_hash": spec_hash or switch_cue.SPEC_HASH,
                   "all": entry, "lomo": {match_id: entry}, "loso": {match_id.split(":")[0]: entry}},
                  handle)
    return str(path)


def test_load_cue_round_trip_and_bad_hash(tmp_path):
    beta = np.arange(43) / 7.0
    path = _write_cue(tmp_path / "cue.json", beta=beta, base=-2.4)
    cue = switch_cue.load_cue(path)
    assert cue["spec_hash"] == switch_cue.SPEC_HASH
    assert np.allclose(cue["all"]["beta"], np.round(beta, 6)) and cue["all"]["base_logit"] == -2.4
    assert switch_cue.fold_fit(cue, "lomo", "synthetic:0") == cue["all"]
    assert switch_cue.fold_fit(cue, "loso", "synthetic:0") == cue["all"]
    with pytest.raises(KeyError):
        switch_cue.fold_fit(cue, "lomo", "other:1")
    with pytest.raises(ValueError):
        switch_cue.load_cue(_write_cue(tmp_path / "bad.json", spec_hash="0000000000"))


def test_decoder_defaults_and_where_check():
    assert LearnedDecoder().switch_cue_gain == 0.0
    assert LearnedDecoder().switch_cue_file == switch_cue.COEF_FILE
    with pytest.raises(ValueError):
        LearnedDecoder(switch_cue_where="seen")


def test_gain_zero_keeps_the_codes(small, tmp_path, monkeypatch):
    match = small["match"]
    before = LearnedDecoder().predict(match)
    # a flat cue (beta 0, base_logit 0: multiplier 1 everywhere) through the per-frame path
    flat = _write_cue(tmp_path / "flat.json", match_id=match.match_id)
    assert np.array_equal(LearnedDecoder(switch_cue_gain=1.0, switch_cue_file=flat).predict(match),
                          before)

    def boom(*args, **kwargs):
        raise AssertionError("viterbi_var called at gain 0")

    monkeypatch.setattr(switch_cue, "viterbi_var", boom)
    assert np.array_equal(LearnedDecoder(switch_cue_gain=0).predict(match), before)


def test_gate_counts_seen_players_over_the_cue_window():
    m = _match(n=300)
    assert switch_cue.gate(m, 0).all()
    assert switch_cue.gate(m, 9).all()  # 10 outfield players each, every frame
    players = m.players.copy()
    players[100, 15:20] = np.nan  # the away team drops to 5 seen outfield players on one frame
    period = m.period.copy()
    period[290:] = 0  # out of play at the end
    gap = Match(match_id="t", fps=m.fps, pitch=m.pitch, period=period, ball=m.ball,
                players=players, slot_team=m.slot_team)
    on = switch_cue.gate(gap, 9)
    half = int(round(max(switch_cue.CUE_WINDOWS_S) * m.fps))
    off = np.zeros(m.n_frames, dtype=bool)
    off[100 - half:100 + half + 1] = True
    off[290 - half:] = True
    assert np.array_equal(on, ~off)
    assert switch_cue.gate(gap, 5).sum() == m.n_frames - 10 - half  # only the end is off
    assert switch_cue.seen_mask(gap) is None  # not SkillCorner 2024/25: NaN is "not seen"


def test_closed_gate_is_the_flat_cost(small, tmp_path, monkeypatch):
    match = small["match"]
    n = switch_cue.design(np.zeros((1, len(switch_cue.COLUMNS)))).shape[1]
    cue = _write_cue(tmp_path / "cue.json", beta=np.linspace(-2, 2, n), base=-2.0,
                     match_id=match.match_id)
    flat = LearnedDecoder().predict(match)
    assert LearnedDecoder(switch_cue_gain=1.0, switch_cue_file=cue).switch_cue_min_seen == 0
    monkeypatch.setattr(switch_cue, "gate", lambda m, k: np.zeros(m.n_frames, dtype=bool))
    closed = LearnedDecoder(switch_cue_gain=1.0, switch_cue_file=cue, switch_cue_min_seen=9)
    assert np.array_equal(closed.predict(match), flat)
    with pytest.raises(ValueError):
        LearnedDecoder(switch_cue_min_seen=-1)


def test_decode_from_shared_scores_is_predict(small, tmp_path):
    match = small["match"]
    n = switch_cue.design(np.zeros((1, len(switch_cue.COLUMNS)))).shape[1]
    cue = _write_cue(tmp_path / "cue.json", beta=np.linspace(-2, 2, n), base=-2.0,
                     match_id=match.match_id)
    scores = LearnedDecoder().scores(match)
    for gain in (0.0, 0.5, 2.0):
        m = LearnedDecoder(switch_cue_gain=gain, switch_cue_file=cue)
        assert np.array_equal(m.decode(match, scores), m.predict(match))
