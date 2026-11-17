"""V3: the provisional audio stack and the whistle detector, on synthetic sound only."""

import numpy as np

from src.official import A, D, H
from src.video import stack, whistle

FPS = 25


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.01, int(seconds * stack.SR))


def _tone(pcm: np.ndarray, t0: float, t1: float, hz: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(len(pcm)) / stack.SR
    on = (t >= t0) & (t < t1)
    pcm = pcm.copy()
    pcm[on] += amp * np.sin(2 * np.pi * hz * t[on])
    return pcm


def test_whistle_line_stands_out_and_voice_stack_does_not():
    pcm = _tone(_noise(10.0), 4.0, 4.6, 3700.0)
    for k in range(4, 18):  # a "voice": parallel harmonics of 250 Hz at 7-8 s
        pcm = _tone(pcm, 7.0, 7.6, 250.0 * k, 0.05)
    n = 10 * FPS
    f = stack.audio_features(pcm, n, FPS)
    assert set(f) == set(stack.FEATURES)
    assert all(len(v) == n for v in f.values())
    w = slice(int(4.1 * FPS), int(4.3 * FPS))
    assert f["line_db"][w].min() > 25
    assert np.all(np.abs(f["line_hz"][w] - 3700) < 40)
    assert f["others"][w].max() <= 1
    v = slice(int(7.1 * FPS), int(7.3 * FPS))
    assert f["others"][v].min() >= 3
    quiet = slice(int(1 * FPS), int(3 * FPS))
    assert np.median(f["line_db"][quiet]) < 15
    ev = whistle.events(f, FPS, thr_db=20.0)
    assert len(ev) == 1 and abs(ev[0] - 4.0 * FPS) <= 3
    # deterministic and rounded
    g = stack.audio_features(pcm, n, FPS)
    assert all(np.array_equal(f[k], g[k]) for k in f)
    assert np.array_equal(f["line_db"], np.round(f["line_db"], stack.ROUND))


def test_empty_features_score_zero():
    f = stack.empty_features(50)
    assert whistle.events(f, FPS, thr_db=1.0).size == 0


def test_peaks_keeps_the_highest_within_the_window():
    s = np.zeros(100)
    s[[10, 12, 40, 41]] = [5, 7, 9, 9]
    assert whistle.peaks(s, 1.0, 5).tolist() == [12, 40]


def test_snap_edges_moves_onset_and_offset_inside_their_segments():
    codes = np.array([H] * 20 + [D] * 20 + [A] * 20)
    runs = [(0, 60)]
    # whistle 3 frames before the onset: D starts there
    out = whistle.snap_edges(codes, np.array([17]), runs, "onset", before=5, after=0, delay=0)
    assert out[17] == D and out[16] == H and np.array_equal(out[40:], codes[40:])
    # whistle 4 frames after the onset with delay 1: the onset moves later, H fills the gap
    out = whistle.snap_edges(codes, np.array([24]), runs, "onset", before=0, after=5, delay=1)
    assert out[24] == H and out[25] == D
    # out of the window: nothing moves
    assert np.array_equal(whistle.snap_edges(codes, np.array([30]), runs, "onset", 5, 5, 0), codes)
    # offset: the restart moves to the whistle; a segment never vanishes
    out = whistle.snap_edges(codes, np.array([38]), runs, "offset", before=5, after=0, delay=0)
    assert out[37] == D and out[38] == A
    out = whistle.snap_edges(codes, np.array([0]), runs, "onset", before=50, after=0, delay=0)
    assert out[0] == H and out[1] == D
