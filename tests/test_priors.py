"""B27: event-data priors and the duration-aware decoder (synthetic data only)."""

import json

import numpy as np
import pytest

from src import priors
from src.models.decoder import to_grid, viterbi
from src.models.learned import LearnedDecoder
from src.official import A, D, H

PRIO = [r for _, r in priors.SHAPE_TYPES_IDX]


def _penalty(s, a, b, g, typed, types):
    gg = g[s]
    if s == 2 and typed:
        for r in PRIO:
            if r in typed and r in types and types[r][a:b].any():
                gg = typed[r]
                break
    return gg[min(b - a, g.shape[1] - 1)]


def _best_by_segments(z, sw, dc, g, typed, types):
    """O(n^2) segment DP: the true optimum, for small n."""
    n = len(z)
    prefix = np.vstack([np.zeros(3), np.cumsum(z, 0)])
    enter = np.full((n, 3), -1e18)
    enter[0] = 0
    for tau in range(1, n):
        for s2 in range(3):
            for s in range(3):
                if s == s2:
                    continue
                c = sw if {s, s2} == {0, 1} else dc
                for a in range(tau):
                    v = (enter[a, s] + prefix[tau, s] - prefix[a, s]
                         - _penalty(s, a, tau, g, typed, types) - c)
                    enter[tau, s2] = max(enter[tau, s2], v)
    return max(enter[a, s] + prefix[n, s] - prefix[a, s] for a in range(n) for s in range(3))


def _value(path, z, sw, dc, g, typed, types):
    bounds = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
    total = z[np.arange(len(path)), path].sum()
    for i in range(len(bounds) - 2):
        s, s2 = path[bounds[i]], path[bounds[i + 1]]
        total -= (sw if {s, s2} == {0, 1} else dc) + _penalty(s, bounds[i], bounds[i + 1], g,
                                                             typed, types)
    return total


def test_aged_viterbi_with_zero_penalties_is_the_flat_viterbi():
    rng = np.random.default_rng(0)
    for _ in range(10):
        z = to_grid(rng.normal(size=(400, 3)) * 0.5)
        flat = viterbi(z, 0.4, 0.9)
        assert len(np.flatnonzero(np.diff(flat))) > 5
        assert np.array_equal(flat, priors.viterbi_aged(z, 0.4, 0.9, np.zeros((3, 60))))


@pytest.mark.parametrize("typed", [False, True])
def test_aged_viterbi_finds_the_optimum(typed):
    rng = np.random.default_rng(1 + typed)
    for _ in range(8):
        # typed: segments longer than the cap K are approximate (see viterbi_aged), so K > n
        n, k = 50, (60 if typed else int(rng.integers(4, 12)))
        z = to_grid(rng.normal(size=(n, 3)) * 0.6)
        g = to_grid(np.abs(rng.normal(size=(3, k + 1))))
        g[:, 0] = 0
        tg = {r: to_grid(np.abs(rng.normal(size=k + 1)) * 2) for r in PRIO[:3]} if typed else None
        types = {r: rng.random(n) < 0.05 for r in PRIO} if typed else None
        sw, dc = float(to_grid(0.3)), float(to_grid(0.7))
        path = priors.viterbi_aged(z, sw, dc, g, tg, types)
        assert _value(path, z, sw, dc, g, tg, types) == pytest.approx(
            _best_by_segments(z, sw, dc, g, tg, types), abs=1e-9)


def test_short_segments_pay_the_penalty():
    # a 3-frame D blip in strong H evidence survives flat costs but not a stoppage prior
    z = np.zeros((40, 3))
    z[:, 0] = 0.3
    z[18:21, 2] = 1.3
    flat = priors.viterbi_aged(z, 0.5, 0.2, np.zeros((3, 20)))
    assert (flat == 2).sum() == 3
    g = np.zeros((3, 20))
    g[2, 1:] = 5.0
    assert (priors.viterbi_aged(z, 0.5, 0.2, g) == 2).sum() == 0


def test_cdf_and_penalty_shape():
    q = list(np.linspace(1.0, 20.0, len(priors.QUANTILES)))
    lengths = np.array([0.0, 0.5, 1.0, 20.0, 30.0])
    f = priors.cdf(q, lengths)
    assert f[0] == 0 and f[2] == pytest.approx(priors.QUANTILES[0]) and f[-1] == 1.0
    assert np.all(np.diff(f) >= 0)
    g = priors.penalty(q, 25, 2.0)
    assert g[0] == 0 and g[1] == pytest.approx(-2 * np.log(priors.FLOOR), abs=1e-5)
    assert g[-1] == 0 and np.all(np.diff(g[1:]) <= 0)


def _stream(rows):
    return priors._stream([(1,) + r for r in rows])


def test_stream_segments_stoppages_and_spells():
    ON, STOP, RS, OS = priors.ON, priors.STOP, priors.RESTART, priors.ON_STOP
    throw = priors.RESTART_TYPES.index("throw")
    seg = priors.stream_segments(_stream([
        (0.0, 0, RS, priors.RESTART_TYPES.index("kickoff")),  # period start: no stoppage
        (2.0, 0, ON, -1), (5.0, 1, ON, -1), (5.0, 0, ON, -1),  # same-instant duel: no spell
        (6.0, 1, ON, -1), (9.0, 0, ON, -1), (12.0, 0, OS, -1),  # a pass out at 12
        (20.0, 1, RS, throw),  # throw-in: 8 s stoppage
        (21.0, 1, ON, -1), (30.0, 0, STOP, -1),  # a foul ...
        (31.0, 1, ON, -1),  # ... with advantage: no stoppage
        (35.0, 0, ON, -1), (40.0, 0, ON, -1),
    ]))
    assert seg["dead"] == [(8.0, throw)]
    # the zero-length away run at 5 goes, so home 0..6 (cut: period start), away 6..9,
    # home 9..12, away 20..35, home 35..40 (cut: period end)
    assert seg["team"] == [3.0, 3.0, 15.0]
    assert seg["changes"] == 3


def test_label_segments_drop_cut_segments_and_type_dead():
    labels = np.array([H] * 10 + [D] * 5 + [A] * 6 + [D] * 4 + [H] * 3, dtype=np.int8)
    period = np.ones(len(labels), dtype=np.int8)
    corner = priors.RESTART_TYPES.index("corner")
    types = {corner: np.zeros(len(labels), bool)}
    types[corner][22] = True
    seg = priors.label_segments(labels, period, 1, types)
    assert seg["dead"] == [(5.0, -1), (4.0, corner)]
    assert seg["team"] == [6.0]


def test_decoder_without_priors_is_unchanged_and_with_priors_runs(small, tmp_path):
    match = small["match"]
    base = LearnedDecoder()
    scores = base.scores(match)
    assert np.array_equal(base.predict(match), base.decode(match, scores))
    table = {"x": {"team": list(np.linspace(0.5, 30, len(priors.QUANTILES))),
                   "dead": list(np.linspace(2, 60, len(priors.QUANTILES))),
                   "by_type": {"corner": {"n": 50, "q": list(np.linspace(8, 60, 19))}}}}
    path = tmp_path / "priors.json"
    path.write_text(json.dumps(table))
    g, tg = priors.exit_costs("x", match.fps, 1.0, 1.0, True, path=str(path))
    assert g.shape == (3, int(priors.MAX_AGE_S * match.fps) + 1)
    assert list(tg) == [priors.RESTART_TYPES.index("corner")]
    off = priors.viterbi_aged(scores[:2000], 0.75, 3.0, np.zeros_like(g))
    assert np.array_equal(off, viterbi(scores[:2000], 0.75, 3.0))
    codes = priors.viterbi_aged(scores[:2000], 0.75, 3.0, g, tg,
                                {k: v[:2000] for k, v in priors.shape_types(match).items()})
    assert set(np.unique(codes)) <= {0, 1, 2}
    assert H == 0 and A == 1 and D == 2
