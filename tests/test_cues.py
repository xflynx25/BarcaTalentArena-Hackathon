"""Wave-5 hooks: decode-time cues (src/models/cues/) and run-time repairs (src/repairs/); synthetic only."""

import dataclasses
import os

import numpy as np
import pytest

from src import repairs
from src.models import cues, switch_cue
from src.models.cues import Effect, viterbi_costs
from src.models.decoder import viterbi
from src.models.learned import LearnedDecoder


@pytest.mark.parametrize("n", [0, 1, 2, 700])
def test_constant_costs_are_viterbi(n):
    rng = np.random.default_rng(n)
    scores = rng.normal(0, 1, (n, 3))
    for switch, dead in ((0.75, 3.0), (0.3, 0.5), (2.0, 1.0)):
        assert np.array_equal(viterbi_costs(scores, switch, switch, dead, dead, dead, dead),
                              viterbi(scores, switch, dead))


def test_per_frame_switch_costs_are_viterbi_var():
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 1, (900, 3))
    c_ha, c_ah = rng.uniform(0.2, 2, 900), rng.uniform(0.2, 2, 900)
    assert np.array_equal(viterbi_costs(scores, c_ha, c_ah, 3.0, 3.0, 3.0, 3.0),
                          switch_cue.viterbi_var(scores, c_ha, c_ah, 3.0))


def test_a_cheaper_dead_frame_moves_a_marginal_stoppage():
    scores = np.full((30, 3), -5.0)
    scores[:10, 0], scores[20:, 2] = 1.0, 1.0  # H early, D late, 10..19 neutral
    flat = viterbi_costs(scores, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0)
    assert flat[19] == 0 and flat[20] == 2  # ties: the change happens as late as possible
    hd = np.full(30, 2.0)
    hd[12] = 0.1  # going dead at frame 12 is nearly free
    early = viterbi_costs(scores, 1.0, 1.0, hd, 2.0, 2.0, 2.0)
    assert early[11] == 0 and early[12] == 2


def test_parse_and_errors():
    assert cues.parse("") == []
    assert cues.parse("stop:0.5/win=2/mode=x+flight") == [("stop", 0.5, {"win": 2, "mode": "x"}),
                                                           ("flight", 1.0, {})]
    for bad in ("stop:x", "stop/win", ":1", "stop+stop", "a-b:1"):
        with pytest.raises(ValueError):
            cues.parse(bad)
    with pytest.raises(ValueError, match="unknown cue"):
        cues.check("nosuchcue:1")
    with pytest.raises(ValueError, match="unknown cue"):
        LearnedDecoder(cues="nosuchcue")
    with pytest.raises(ValueError):
        LearnedDecoder(cues="x", priors="any")
    assert repairs.parse("gk/reach=3+hold") == [("gk", {"reach": 3}), ("hold", {})]
    with pytest.raises(ValueError, match="unknown repair"):
        LearnedDecoder(repairs="nosuchrepair")


@pytest.fixture
def registered(monkeypatch):
    """Test cues: `one` (every multiplier 1, no term), `none` (returns None), `free` (going dead free
    everywhere: D wins every neutral stretch), `bump` (adds evidence for D)."""
    calls = []

    def one(match, original, decoder, gain, **opts):
        calls.append(("one", gain, opts, original is not None))
        n = match.n_frames
        return Effect(*(np.ones(n) for _ in cues.KINDS), add=np.zeros((n, 3)))

    def free(match, original, decoder, gain, **opts):
        return Effect.dead(np.full(match.n_frames, 1e-3))

    def bump(match, original, decoder, gain, **opts):
        add = np.zeros((match.n_frames, 3))
        add[:, 2] = gain
        return Effect(add=add)

    def bad(match, original, decoder, gain, **opts):
        return Effect.switch(np.zeros(match.n_frames))

    for name, fn in (("one", one), ("none", lambda *a, **k: None), ("free", free), ("bump", bump),
                     ("bad", bad)):
        monkeypatch.setitem(cues.REGISTRY, name, fn)
    return calls


def test_identity_cue_keeps_the_codes(small, registered):
    match = small["match"]
    before = LearnedDecoder().predict(match)
    assert np.array_equal(LearnedDecoder(cues="one:0.5/k=1+none").predict(match), before)
    assert registered == [("one", 0.5, {"k": 1}, True)]
    assert not np.array_equal(LearnedDecoder(cues="bump:5").predict(match), before)
    with pytest.raises(ValueError, match="finite and > 0"):
        LearnedDecoder(cues="bad").predict(match)


def test_identity_cue_keeps_the_switch_cue_path(small, registered):
    match = small["match"]
    on = dict(switch_cue_gain=0.5)  # v6's cue file, fold "all"
    before = LearnedDecoder(**on).predict(match)
    assert np.array_equal(LearnedDecoder(cues="one", **on).predict(match), before)


def test_repairs_run_before_track_repair_and_identity_keeps_codes(small, monkeypatch):
    match = small["match"]
    seen = []

    def ident(m, **opts):
        seen.append(opts)
        return m

    def shift(m, **opts):
        return dataclasses.replace(m, players=m.players + 0.5).validate()

    monkeypatch.setitem(repairs.REGISTRY, "ident", ident)
    monkeypatch.setitem(repairs.REGISTRY, "shift", shift)
    before = LearnedDecoder(track_repair="1").predict(match)
    assert np.array_equal(LearnedDecoder(track_repair="1", repairs="ident/x=2").predict(match), before)
    assert seen and seen[0] == {"x": 2}
    dec = LearnedDecoder(repairs="shift")
    assert np.allclose(dec.prepare(match).players, match.players + 0.5, equal_nan=True)
    assert dec.prepare(match) is dec.prepare(match)  # cached: scores() and decode() repair once


def test_cues_is_decode_only_in_the_grid():
    """check_dropout_grid shares one emission among candidates that differ only in `cues`."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "check_dropout_grid.py")
    if not os.path.exists(path):
        pytest.skip("not in the public export")
    line = [t for t in open(path).read().splitlines() if t.startswith("CUE_ONLY = ")][0]
    assert '"cues"' in line and "repairs" not in line
