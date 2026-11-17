"""The referee's whistle from per-frame audio features (src/video/stack.py), and snapping D edges to it.

Pure numpy, deterministic, label-free: it reads only the features of one match, so it could run on
BAR-ATH's video at run time (V5) with parameters chosen on external matches (V3, findings F139-F141).

    score = line_db where the frame's lasting line is alone (others <= max_others) and inside band_hz,
            else 0 (src/video/stack.py: line_db, line_hz, others)
    events = frames where score >= thr_db that are the highest within +-nms_s (first on ties)

snap_edges moves a decoded D run's start (or end) to the nearest whistle within a window around it,
plus a fixed delay; the frames between take the neighbouring segment's code.
"""

from __future__ import annotations

import numpy as np

from src.official import D

DEFAULTS = {"thr_db": 18.0, "max_others": 1, "band_hz": (2500.0, 4800.0), "nms_s": 1.0}


def score(feats: dict, max_others: int = DEFAULTS["max_others"],
          band_hz: tuple = DEFAULTS["band_hz"]) -> np.ndarray:
    ok = (feats["has_audio"] > 0) & (feats["others"] <= max_others) \
        & (feats["line_hz"] >= band_hz[0]) & (feats["line_hz"] <= band_hz[1])
    return np.where(ok, np.maximum(feats["line_db"], 0.0), 0.0)


def peaks(s: np.ndarray, thr: float, half_width: int) -> np.ndarray:
    """Frames with s >= thr that are the maximum of s within +-half_width (the first such frame on a
    plateau). Sorted."""
    cand = np.flatnonzero(s >= thr)
    if len(cand) == 0:
        return cand
    order = cand[np.lexsort((cand, -s[cand]))]  # highest first, earliest first on ties
    taken = np.zeros(len(s), dtype=bool)
    out = []
    for f in order:
        lo, hi = max(0, f - half_width), f + half_width + 1
        if not taken[lo:hi].any():
            out.append(f)
            taken[f] = True
    return np.array(sorted(out), dtype=np.int64)


def events(feats: dict, fps: int, thr_db: float = DEFAULTS["thr_db"],
           max_others: int = DEFAULTS["max_others"], band_hz: tuple = DEFAULTS["band_hz"],
           nms_s: float = DEFAULTS["nms_s"]) -> np.ndarray:
    """Whistle frames (sorted int array)."""
    return peaks(score(feats, max_others, band_hz), thr_db, int(round(nms_s * fps)))


def segments(codes: np.ndarray, runs) -> list[tuple[int, int, int]]:
    """(start, stop, code) of each constant segment inside the given (start, stop) runs."""
    out = []
    for start, stop in runs:
        w = codes[start:stop]
        if len(w) == 0:
            continue
        cuts = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        out += [(start + int(a), start + int(b), int(w[a])) for a, b in zip(cuts[:-1], cuts[1:])]
    return out


def snap_edges(codes: np.ndarray, whistles: np.ndarray, runs, edge: str, before: int, after: int,
               delay: int, dead: int = D) -> np.ndarray:
    """Copy of codes with each D segment's start (edge="onset") or end (edge="offset") moved to the
    whistle nearest to it within [edge - before, edge + after], plus `delay` frames. Only edges between
    a D segment and a non-D one inside the same run move; a moved edge stays strictly inside the two
    segments it separates, so no segment vanishes. The frames it crosses take the other side's code."""
    out = np.array(codes, copy=True)
    if len(whistles) == 0:
        return out
    segs = segments(codes, runs)
    for i in range(len(segs) - 1):
        (a0, a1, ca), (b0, b1, cb) = segs[i], segs[i + 1]
        if a1 != b0:  # different runs
            continue
        if edge == "onset" and not (cb == dead and ca != dead):
            continue
        if edge == "offset" and not (ca == dead and cb != dead):
            continue
        t = b0
        near = whistles[(whistles >= t - before) & (whistles <= t + after)]
        if len(near) == 0:
            continue
        w = int(near[np.argmin(np.abs(near - t))]) + delay
        w = min(max(w, a0 + 1), b1 - 1)
        if w < t:
            out[w:t] = cb
        elif w > t:
            out[t:w] = ca
    return out
