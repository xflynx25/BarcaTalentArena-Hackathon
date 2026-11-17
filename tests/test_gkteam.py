"""C15: keeper team flips (degrade gk_flip / gk_ghost / gk_hide) and the gkteam repair."""

import numpy as np
import pytest

from src.ingest import degrade
from src.ingest import synthetic as synthetic_source
from src.models import build
from src.repairs import gkteam
from src.schema import AWAY, HOME, Match


def _keeper_match(n=2500, fps=25):
    """One period: home keeper near x = 5, away keeper near x = 100, three outfield players each.
    Home defends x = 0."""
    t = np.arange(n) / fps
    rows = []
    rows.append(np.stack([5 + np.sin(t), 34 + 2 * np.cos(t / 3)], 1))  # home keeper
    for y in (20, 34, 48):
        rows.append(np.stack([35 + 10 * np.sin(t / 7 + y), y + 0 * t], 1))
    rows.append(np.stack([100 - np.sin(t), 34 + 2 * np.sin(t / 3)], 1))  # away keeper
    for y in (15, 34, 53):
        rows.append(np.stack([60 + 10 * np.cos(t / 5 + y), y + 0 * t], 1))
    return Match(match_id="test:keepers", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=np.full((n, 2), np.nan), players=np.stack(rows, axis=1),
                 slot_team=np.array([HOME] * 4 + [AWAY] * 4, np.int8)).validate()


def test_keepers_found_label_free():
    m = _keeper_match()
    assert degrade.defending_left(m) == {1: HOME}
    assert gkteam.defending_left(m) == {1: HOME}
    assert degrade.keepers(m) == {(1, HOME): 0, (1, AWAY): 4}


def test_gk_flip_moves_keeper_to_other_team_and_is_deterministic():
    m = _keeper_match()
    out = degrade.apply(m, {"gk_flip": 20, "gk_flip_s": 3})
    again = degrade.apply(m, {"gk_flip": 20, "gk_flip_s": 3})
    assert np.array_equal(out.players, again.players, equal_nan=True)
    assert out.players.shape[1] > m.players.shape[1]
    added = np.arange(m.players.shape[1], out.players.shape[1])
    # every flipped position is a keeper's, in a slot of the other team, his own slot empty then
    for c in added:
        rows = np.flatnonzero(~np.isnan(out.players[:, c, 0]))
        team = out.slot_team[c]
        keeper = 4 if team == HOME else 0
        assert np.allclose(out.players[rows, c], m.players[rows, keeper])
        assert np.isnan(out.players[rows, keeper]).all()
    # nothing else changes
    assert np.array_equal(out.players[:, [1, 2, 3, 5, 6, 7]], m.players[:, [1, 2, 3, 5, 6, 7]])


def test_gk_ghost_and_hide_keep_counts():
    m = _keeper_match()
    out = degrade.apply(m, {"gk_flip": 20, "gk_flip_s": 3, "gk_ghost": 1, "gk_hide": 1})
    seen = ~np.isnan(out.players).any(axis=2)
    for team in (HOME, AWAY):
        assert (seen[:, out.slot_team == team].sum(axis=1) == 4).all()
    with pytest.raises(ValueError):
        degrade.apply(m, {"gk_ghost": 1})


def test_repair_undoes_flips_and_drops_ghosts():
    m = _keeper_match()
    for opts in ({"gk_flip": 20, "gk_flip_s": 3}, {"gk_flip": 20, "gk_flip_s": 3, "gk_ghost": 1}):
        damaged = degrade.apply(m, opts)
        fixed = gkteam.apply(damaged, max_share=1.0)
        assert fixed is not damaged
        k = m.players.shape[1]
        assert np.array_equal(fixed.players[:, :k], m.players, equal_nan=True)
        assert np.isnan(fixed.players[:, k:]).all()


def test_repair_leaves_clean_matches_alone():
    m = _keeper_match()
    assert gkteam.apply(m) is m
    syn = synthetic_source.load("0")
    assert gkteam.apply(syn) is syn


def test_self_guard(capsys):
    m = _keeper_match()
    damaged = degrade.apply(m, {"gk_flip": 20, "gk_flip_s": 3})
    assert gkteam.apply(damaged, max_share=0.001) is damaged
    assert "self-guard" in capsys.readouterr().out


def test_decoder_parameter_runs():
    syn = synthetic_source.load("0")
    base = build("learned").predict(syn)
    rep = build("learned", repairs="gkteam").predict(syn)
    assert np.array_equal(base, rep)
