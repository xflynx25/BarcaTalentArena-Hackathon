"""Post-processing: drop_short + play_runs (backlog B6), delay_switches (C2), and nearest's fields."""

import numpy as np

from src.models import build
from src.models.nearest import NearestPlayer
from src.models.postprocess import delay_switches, drop_short, play_runs
from src.official import A, D, H
from src.resample import to_fps


def codes(text: str) -> np.ndarray:
    return np.array([{"H": H, "A": A, "D": D}[c] for c in text], dtype=np.int8)


def reference(values: np.ndarray, runs, min_frames: int) -> np.ndarray:
    """Plain loop version of drop_short's rule, to check the vectorised one against."""
    out = values.copy()
    for start, stop in runs:
        segs, a = [], start
        for t in range(start + 1, stop + 1):
            if t == stop or values[t] != values[t - 1]:
                segs.append((a, t, values[a]))
                a = t
        long = [s for s in segs if s[1] - s[0] >= min_frames]
        if not long:
            continue
        label = long[0][2]
        for a, b, k in segs:
            if b - a >= min_frames:
                label = k
            out[a:b] = label
    return out


def test_short_segments_take_the_previous_long_label():
    assert np.array_equal(drop_short(codes("HHHHDAHHHAAAA"), [(0, 13)], 3), codes("HHHHHHHHHAAAA"))


def test_leading_short_segments_take_the_first_long_label():
    assert np.array_equal(drop_short(codes("ADAAAHHH"), [(0, 8)], 3), codes("AAAAAHHH"))
    assert np.array_equal(drop_short(codes("DHAAAA"), [(0, 6)], 3), codes("AAAAAA"))


def test_runs_are_kept_apart_and_left_alone_without_long_segments():
    values = codes("HHHHAD" + "DAH")
    out = drop_short(values, [(0, 6), (6, 9)], 3)
    assert np.array_equal(out, codes("HHHHHH" + "DAH"))  # second run: nothing long, unchanged


def test_off_and_input_untouched():
    values = codes("HAHAD")
    assert np.array_equal(drop_short(values, [(0, 5)], 1), values)
    drop_short(values, [(0, 5)], 3)
    assert np.array_equal(values, codes("HAHAD"))


def test_matches_reference_on_random_codes():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.integers(0, 3, 400).astype(np.int8), rng.integers(1, 8, 400))
    runs = [(0, 700), (900, len(values))]
    for min_frames in (2, 3, 5, 9):
        assert np.array_equal(drop_short(values, runs, min_frames), reference(values, runs, min_frames))


def test_play_runs_split_halves():
    period = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0], dtype=np.int8)
    assert play_runs(period) == [(2, 5), (6, 8)]
    assert play_runs(np.array([1, 1, 2], dtype=np.int8)) == [(0, 2), (2, 3)]


def test_nearest_min_spell_s(small):
    match = small["match"]
    plain = NearestPlayer(window_s=1.0).predict(match)
    assert np.array_equal(NearestPlayer(window_s=1.0, min_spell_s=0).predict(match), plain)
    filtered = build("nearest", window_s=1.0, min_spell_s=2.0).predict(match)
    assert np.array_equal(filtered, drop_short(plain, play_runs(match.period), 50))
    match10 = to_fps(match, 10)
    plain10 = NearestPlayer(window_s=1.0).predict(match10)
    assert np.array_equal(NearestPlayer(window_s=1.0, min_spell_s=2.0).predict(match10),
                          drop_short(plain10, play_runs(match10.period), 20))


def test_delay_switches_moves_only_team_switches():
    values = codes("HHHAAAADDHHAA")
    assert np.array_equal(delay_switches(values, [(0, 13)], 2), codes("HHHHHAADDHHHH"))
    assert np.array_equal(delay_switches(values, [(0, 13)], 0), values)
    assert np.array_equal(values, codes("HHHAAAADDHHAA"))  # input untouched


def test_delay_switches_stays_inside_runs():
    values = codes("HHA" + "AHH")
    assert np.array_equal(delay_switches(values, [(0, 3), (3, 6)], 5), codes("HHH" + "AAA"))


def test_delay_switches_matches_reference_on_random_codes():
    rng = np.random.default_rng(1)
    values = np.repeat(rng.integers(0, 3, 300).astype(np.int8), rng.integers(1, 8, 300))
    runs = [(0, 500), (600, len(values))]
    for frames in (1, 3, 6):
        expected = values.copy()
        for start, stop in runs:
            for t in range(start + 1, stop):
                old, new = values[t - 1], values[t]
                if old != new and {old, new} == {H, A}:
                    end = min(t + frames, stop)
                    span = expected[t:end]
                    span[values[t:end] == new] = old
        assert np.array_equal(delay_switches(values, runs, frames), expected)


def test_nearest_switch_delay_s(small):
    match = small["match"]
    base = NearestPlayer(window_s=1.0, min_spell_s=2.0).predict(match)
    delayed = build("nearest", window_s=1.0, min_spell_s=2.0, switch_delay_s=0.5).predict(match)
    assert np.array_equal(delayed, delay_switches(base, play_runs(match.period), 13))
