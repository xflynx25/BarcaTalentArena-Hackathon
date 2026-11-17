"""Change a Match's frame rate, e.g. SkillCorner's 10 fps to BAR-ATH's 25 fps.

Output frame k sits at time k / fps, which is source position k * match.fps / fps. Positions are
interpolated linearly between the two neighbouring source frames when both are observed and in the
same period, otherwise they take the nearest source frame's value (NaN stays NaN). Discrete fields
(period, labels) take the nearest source frame. Pure numpy and deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.schema import Match


def nearest_index(n_source: int, source_fps: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(position (m,), nearest (m,)): fractional source position and nearest source row of each
    output frame. The output covers the same duration (n_source / source_fps seconds), so going
    10 -> 25 -> 10 fps gives back the original length and labels."""
    n_out = int(np.ceil(n_source * fps / source_fps - 1e-9))
    position = np.arange(n_out) * (source_fps / fps)
    nearest = np.minimum(np.floor(position + 0.5).astype(np.int64), n_source - 1)
    return position, nearest


def labels_to_fps(codes: np.ndarray, source_fps: float, fps: float) -> np.ndarray:
    """Nearest-frame resampling of a label (or any discrete) array."""
    return codes[nearest_index(len(codes), source_fps, fps)[1]]


def to_fps(match: Match, fps: int) -> Match:
    if fps == match.fps:
        return match
    position, nearest = nearest_index(match.n_frames, match.fps, fps)
    low = np.minimum(np.floor(position).astype(np.int64), match.n_frames - 1)
    high = np.minimum(low + 1, match.n_frames - 1)  # past the last frame: low == high
    weight = position - low
    same_period = match.period[low] == match.period[high]

    def interpolate(values: np.ndarray) -> np.ndarray:
        shape = (-1,) + (1,) * (values.ndim - 1)
        w = weight.reshape(shape)
        blended = values[low] * (1 - w) + values[high] * w
        usable = (np.isfinite(values[low]) & np.isfinite(values[high])
                  & same_period.reshape(shape))
        return np.where(usable, blended, values[nearest])

    return dataclasses.replace(
        match,
        fps=fps,
        period=match.period[nearest],
        ball=interpolate(match.ball),
        players=interpolate(match.players),
        labels=None if match.labels is None else match.labels[nearest],
        tags=None if match.tags is None else _tags_at(match.tags, match.fps, fps, len(nearest)),
    ).validate()


def _tags_at(tags: list, old_fps: int, fps: int, n: int) -> list:
    """Smart Tagging intervals (src/smart_tags.Tag) moved to the new frame rate, clipped to n."""
    out = []
    for t in tags:
        start = min(n, int(round(t.start * fps / old_fps)))
        end = min(n, max(start + 1, int(round(t.end * fps / old_fps))))
        if end > start:
            out.append(t._replace(start=start, end=end))
    return out
