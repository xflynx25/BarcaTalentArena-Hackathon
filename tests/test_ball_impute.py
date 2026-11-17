"""The hidden-ball imputer (src/ball_impute.py, backlog B29) and its `learned` groups: synthetic only."""

import dataclasses

import numpy as np

from src import ball_impute as bi
from src.models import learned
from src.schema import Match


def _match(n=200, seed=0, gaps=((40, 70), (120, 180))):
    """Two 11-player teams walking about; the ball follows player 3 (home) and then 15 (away), and
    is hidden on `gaps`. Period 1 then period 2."""
    rng = np.random.default_rng(seed)
    period = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)].astype(np.int8)
    base = np.c_[rng.uniform(10, 95, 22), rng.uniform(5, 63, 22)]
    pos = base[None] + np.cumsum(rng.normal(0, 0.2, (n, 22, 2)), axis=0)
    carrier = np.where(np.arange(n) < n // 2, 3, 15)
    ball = pos[np.arange(n), carrier] + 0.5
    for a, b in gaps:
        ball[a:b] = np.nan
    return Match(match_id="t:1", fps=10, pitch=(105.0, 68.0), period=period, ball=ball,
                 players=pos, slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_gap_context_stays_inside_the_period():
    m = _match(gaps=((95, 110),))  # a gap across the half-time boundary
    last, nxt = bi.gap_context(m.ball, m.period)
    assert last[97] == 94 and nxt[97] == -1  # period 1 ends in the gap: no next seen ball
    assert last[105] == -1 and nxt[105] == 110
    assert last[50] == 50 and nxt[50] == 50


def test_zero_weights_give_the_straight_line():
    m = _match()
    out = bi.impute(m, beta=np.zeros(bi.N_WEIGHTS))
    seen = ~np.isnan(m.ball[:, 0])
    assert np.allclose(out[seen], m.ball[seen])
    a, b = m.ball[39], m.ball[70]
    assert np.allclose(out[55], a + (55 - 39) / (70 - 39) * (b - a))
    assert not np.isnan(out[m.period > 0]).any()


def test_estimate_is_equivariant_and_team_blind():
    m = _match()
    beta = np.random.default_rng(1).normal(0, 0.1, bi.N_WEIGHTS)
    base = bi.impute(m, beta=beta)
    # a 180-degree rotation of the pitch rotates the estimate (clipping is symmetric)
    rot = dataclasses.replace(m, ball=np.array(m.pitch) - m.ball, players=np.array(m.pitch) - m.players)
    assert np.allclose(bi.impute(rot, beta=beta), np.array(m.pitch) - base)
    swapped = dataclasses.replace(m, slot_team=1 - m.slot_team)
    assert np.allclose(bi.impute(swapped, beta=beta), base)


def test_fit_learns_to_follow_the_carrier():
    """The ball moves with a player; the fit should beat the straight line on the hidden frames."""
    pieces, truth = [], []
    for seed in range(3):
        m = _match(seed=seed, gaps=())
        hidden = dataclasses.replace(m, ball=m.ball.copy())
        rows = np.r_[20:60, 120:170]
        hidden.ball[rows] = np.nan
        start, cols = bi.design(hidden, rows)
        pieces.append(bi.gram(start, cols, m.ball[rows], np.full(len(rows), 1.0 / len(rows))))
        truth.append((hidden, rows, m.ball[rows]))
    beta = bi.solve(pieces)
    for hidden, rows, target in truth:
        fit = bi.impute(hidden, beta=beta)[rows]
        line = bi.impute(hidden, beta=np.zeros(bi.N_WEIGHTS))[rows]
        assert np.hypot(*(fit - target).T).mean() < 0.5 * np.hypot(*(line - target).T).mean()


def test_weights_count_sources_equally():
    w = bi.weights(["a:1", "a:2", "b:1"])
    assert np.isclose(w["a:1"] + w["a:2"], w["b:1"]) and np.isclose(sum(w.values()), 1.0)


def test_learned_groups_are_zero_on_seen_frames_and_swap_with_the_teams(monkeypatch):
    m = _match()
    beta = np.random.default_rng(2).normal(0, 0.05, bi.N_WEIGHTS)
    orig = bi.impute
    monkeypatch.setattr(bi, "impute", lambda match, fold="auto": orig(match, beta=beta))
    seen = ~np.isnan(m.ball[:, 0])
    h, a, c = learned.impute_blocks(m, ~seen)
    assert not h[seen].any() and not a[seen].any() and h[~seen].any()
    sm = dataclasses.replace(m, slot_team=1 - m.slot_team)
    h2, a2, c2 = learned.impute_blocks(sm, ~seen)
    assert np.allclose(h, a2) and np.allclose(a, h2) and np.allclose(c, c2)
    g_home, g_away = learned.gapend_blocks(m, ~seen)
    assert not g_home[seen].any() and not g_away[seen].any()
    # gap 40-70 in period 1: home's player 3 carried the ball before and after it
    assert g_home[50].sum() > 0 and g_away[50].sum() == 0
    s_home, s_away = learned.gapend_blocks(dataclasses.replace(m, slot_team=1 - m.slot_team), ~seen)
    assert np.allclose(g_home, s_away) and np.allclose(g_away, s_home)


def test_committed_coefficients_load():
    coefs = bi.load_coefs()
    assert len(coefs["all"]) == bi.N_WEIGHTS
    assert set(coefs["loso"]) <= {"idsse", "skillcorner", "metrica"}
    assert np.array_equal(bi.beta_for("bar_ath"), np.array(coefs["all"]))
