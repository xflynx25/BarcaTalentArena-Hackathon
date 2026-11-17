"""Cue `heading` (src/models/cues/heading.py): a dearer H<->A switch while both teams keep their
heading. Synthetic only."""

import numpy as np
import pytest

from src.models.cues import heading
from src.schema import Match


def _match(n=300, run=True, players=11):
    rng = np.random.default_rng(1)
    period = np.ones(n, dtype=np.int8)
    home = np.c_[np.r_[3, rng.uniform(10, 50, 10)], rng.uniform(5, 63, 11)]
    away = np.c_[np.r_[102, rng.uniform(55, 95, 10)], rng.uniform(5, 63, 11)]
    pos = np.concatenate([home, away])[None].repeat(n, axis=0) + rng.normal(0, 0.02, (n, 22, 2))
    if run:  # both teams jog toward +x at 3 m/s (10 fps): coherent, steady heading
        pos[:, :, 0] += np.arange(n)[:, None] * 0.3
    if players < 11:
        pos[:, players:11] = np.nan
        pos[:, 11 + players:] = np.nan
    return Match(match_id="t", fps=10, pitch=(105.0, 68.0), period=period,
                 ball=np.full((n, 2), np.nan), players=pos,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_gain_zero_is_no_effect():
    assert heading.effect(_match(), None, None, 0.0) is None


def test_steady_running_makes_switches_dearer_and_only_dearer():
    eff = heading.effect(_match(), None, None, 0.5)
    assert eff.ha is eff.ah
    assert np.all((eff.ha == 1.0) | (eff.ha == 1.5))
    assert np.mean(eff.ha[30:-30] == 1.5) > 0.9
    assert eff.hd is None and eff.dh is None and eff.add is None


def test_unknown_heading_gives_exactly_one():
    for m in (_match(run=False), _match(players=7)):  # standing still; 6 outfielders
        eff = heading.effect(m, None, None, 2.0)
        assert np.all(eff.ha == 1.0)


def test_negative_gain_refused():
    with pytest.raises(ValueError):
        heading.effect(_match(), None, None, -1.0)
