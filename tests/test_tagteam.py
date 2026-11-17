"""B40: cue `tagteam` (src/models/cues/tagteam.py), Smart Tagging's phases as a team prior; synthetic only."""

import dataclasses

import numpy as np
import pytest

from src.models.cues import tagteam
from src.models.learned import LearnedDecoder
from src.official import A, H
from src.smart_tags import NA, SET_PIECES, Tag

V6 = dict(track_repair="1", restart_rule_gain=2.0, switch_cue_gain=0.5)


def test_weights_families_taper_and_overlap():
    tags = [Tag(10, 60, "BUILD UP", H),                    # attack: for H
            Tag(40, 90, "DEFENDING IN MIDDLE THIRD", H),   # defending: for the other team (A)
            Tag(100, 150, "LONG BALL", A),                 # long ball: for A
            Tag(160, 200, SET_PIECES, H),                  # set pieces: nothing
            Tag(200, 240, "PROGRESSION", NA)]              # no team: nothing
    w = tagteam.weights(tags, 250, fps=10, taper_s=1.0)
    assert w.shape == (250, 2)
    assert w[35, 0] == 1 and w[35, 1] == 0  # the middle of the H phase
    assert 0 < w[10, 0] < 0.1 and w[59, 0] < 0.1  # ramped down at its edges
    assert w[65, 1] == 1 and w[65, 0] == 0  # the defending tag counts for A
    assert w[125, 1] == 1
    assert not w[160:].any() and not w[:10].any()
    flat = tagteam.weights(tags, 250, fps=10, taper_s=0)
    assert flat[10, 0] == 1 and flat[59, 0] == 1
    # two tags for the same team: the larger weight, no double count
    twice = tagteam.weights([Tag(0, 50, "BUILD UP", H), Tag(0, 50, "PROGRESSION", H)], 50, 10)
    assert twice.max() == 1


def test_no_tags_or_gain_zero_is_no_effect(small):
    match = dataclasses.replace(small["match"], tags=None)
    dec = LearnedDecoder(**V6)
    assert tagteam.effect(match, match, dec, 1.0) is None
    assert tagteam.effect(small["match"], small["match"], dec, 0.0) is None
    empty = dataclasses.replace(small["match"], tags=[])
    assert tagteam.effect(empty, empty, dec, 1.0) is None
    only_sets = dataclasses.replace(small["match"], tags=[Tag(0, 100, SET_PIECES, H)])
    assert tagteam.effect(only_sets, only_sets, dec, 1.0) is None


def test_without_tags_the_codes_are_v6s(small):
    match = dataclasses.replace(small["match"], tags=None)
    v6 = LearnedDecoder(**V6).predict(match)
    assert np.array_equal(LearnedDecoder(**V6, cues="tagteam:1").predict(match), v6)


def test_tags_pull_the_codes_towards_the_tagged_team(small):
    match = small["match"]
    n = match.n_frames
    tagged = dataclasses.replace(match, tags=[Tag(0, n, "BUILD UP", A)])
    codes = LearnedDecoder(cues="tagteam:50").predict(tagged)
    base = LearnedDecoder().predict(tagged)
    assert (codes == A).sum() > (base == A).sum()
    eff = tagteam.effect(tagged, tagged, LearnedDecoder(), 2.0)
    assert eff.add.shape == (n, 3) and np.all(eff.add[:, [0, 2]] == 0)
    assert np.isclose(eff.add[n // 2, 1], 2.0 / match.fps)
    other = tagteam.effect(tagged, tagged, LearnedDecoder(), 2.0, mode="other")
    assert np.isclose(other.add[n // 2, 0], -2.0 / match.fps) and np.all(other.add[:, 1:] == 0)
    with pytest.raises(ValueError, match="mode"):
        tagteam.effect(tagged, tagged, LearnedDecoder(), 1.0, mode="nope")


def test_standin_option_refuses_bar_ath(small):
    match = small["match"]  # no labels: stand-ins can't be made
    with pytest.raises(Exception):
        tagteam.effect(match, match, LearnedDecoder(), 1.0, src="standin", noise=2)
