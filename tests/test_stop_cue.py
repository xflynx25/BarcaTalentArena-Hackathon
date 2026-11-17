"""B37's stop / go cue (src/models/cues/stop.py); synthetic only."""

import dataclasses

import numpy as np

from src.models.cues import stop
from src.models.learned import LearnedDecoder


def test_coefficient_file_matches_the_spec():
    fits = stop.load()
    assert fits["spec_hash"] == stop.SPEC_HASH
    n_stop = 2 + sum(len(stop.KNOTS[c]) for c, _ in stop.STOP_COLUMNS)
    n_go = 2 + sum(len(stop.KNOTS[c]) for c, _ in stop.GO_COLUMNS)
    for fit in [fits["all"], *fits["lomo"].values(), *fits["loso"].values()]:
        assert len(fit["stop"]["beta"]) == n_stop and len(fit["go"]["beta"]) == n_go


def test_features_ignore_which_team_is_home(small):
    match = small["match"]
    swapped = dataclasses.replace(match, slot_team=(1 - match.slot_team).astype(match.slot_team.dtype))
    a, b = stop.sample_features(match), stop.sample_features(swapped)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)


def test_nobody_tracked_gives_no_effect(small):
    match = small["match"]
    blind = dataclasses.replace(match, players=np.full_like(match.players, np.nan),
                                ball=np.full_like(match.ball, np.nan))
    z_stop, z_go = stop.frame_logits(blind, stop.load()["all"])
    assert np.isnan(z_stop).all() and np.isnan(z_go).all()
    eff = stop.effect(blind, blind, LearnedDecoder(), 1.0, go=1.0)
    assert np.all(eff.hd == 1.0) and np.all(eff.dh == 1.0)


def test_gain_zero_keeps_the_codes_and_multipliers_stay_in_range(small):
    match = small["match"]
    before = LearnedDecoder().predict(match)
    assert stop.effect(match, match, LearnedDecoder(), 0.0, go=1.0) is None
    assert np.array_equal(LearnedDecoder(cues="stop:0/go=1").predict(match), before)
    eff = stop.effect(match, match, LearnedDecoder(), 1.0, go=1.0)
    assert eff.ha is None and eff.ah is None and eff.add is None
    assert np.all((eff.hd >= stop.STOP_RANGE[0]) & (eff.hd <= stop.STOP_RANGE[1]))
    assert np.all((eff.dh >= stop.GO_RANGE[0]) & (eff.dh <= stop.GO_RANGE[1]))
    assert stop.effect(match, match, LearnedDecoder(), 1.0).dh is None  # go=0: restarts untouched
    a = LearnedDecoder(cues="stop:1/go=1").predict(match)
    assert np.array_equal(a, LearnedDecoder(cues="stop:1/go=1").predict(match))  # deterministic
