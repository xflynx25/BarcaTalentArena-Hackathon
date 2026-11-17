"""Turn durations in seconds into frame counts at a match's frame rate.

Sources come at different rates (BAR-ATH 25 fps, SkillCorner 10 fps), so model parameters that are
durations should be given in seconds and converted here, with match.fps, at predict time. Then one
config means the same thing on every source.
"""

from __future__ import annotations

import math


def to_frames(seconds: float, fps: float) -> int:
    """Nearest whole number of frames, halves rounded up (0.25 s at 10 fps -> 3). Never negative."""
    if seconds < 0:
        raise ValueError("duration must be >= 0, got {}".format(seconds))
    return int(math.floor(seconds * fps + 0.5 + 1e-9))
