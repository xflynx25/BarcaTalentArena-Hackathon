"""The official baseline, vectorised: nearest tracked player to the ball, then a centred majority vote.

Must reproduce official/starter/baseline.py exactly. tests/test_baseline_equivalence.py checks it
against the official code on synthetic data, and `python -m src.run --compare-official
baseline_smooth50` checks it against the published CSV on the real match. That makes it the
reference that proves the loader and alignment are right before anything cleverer is built on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.postprocess import delay_switches, drop_short, play_runs
from src.models.timing import to_frames
from src.official import A, D, H
from src.schema import AWAY, HOME, Match

NOBODY = -1


def nearest_team(match: Match) -> np.ndarray:
    """(n,) HOME / AWAY / NOBODY: the team of the slot closest to the ball. NOBODY when the ball or
    every player is missing. Ties go to the lowest slot, like the official loop."""
    dx = match.players[:, :, 0] - match.ball[:, None, 0]
    dy = match.players[:, :, 1] - match.ball[:, None, 1]
    distance = dx * dx + dy * dy
    distance[np.isnan(distance)] = np.inf
    best = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(len(best)), best])
    return np.where(found, match.slot_team[best], NOBODY).astype(np.int8)


def majority_smooth(teams: np.ndarray, window: int) -> np.ndarray:
    """Label codes from a centred majority vote over +/- window frames. D where nobody is seen in
    the window, ties to home. Same arithmetic as the official smooth()."""
    if window <= 0:
        return np.where(teams == HOME, H, np.where(teams == AWAY, A, D)).astype(np.int8)

    home_cum = np.concatenate(([0], np.cumsum(teams == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(teams == AWAY)))
    index = np.arange(len(teams))
    low = np.maximum(0, index - window)
    high = np.minimum(len(teams), index + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]

    codes = np.full(len(teams), D, dtype=np.int8)
    seen = (home + away) > 0
    codes[seen] = np.where(home[seen] >= away[seen], H, A)
    return codes


@dataclass(kw_only=True)
class NearestPlayer:
    """Give the smoothing half-width as window_s (seconds, the default: one config for every source)
    or as window (frames, to reproduce the official baseline file exactly). Not both.
    At 25 fps the default 2 s is 50 frames, the published floor (spellF1 0.2433).
    min_spell_s > 0 then drops predicted segments shorter than that (postprocess.drop_short,
    findings F32/F34). switch_delay_s > 0 then moves every H<->A switch that much later
    (postprocess.delay_switches, backlog C2). Both off by default, so the defaults still reproduce
    the official baseline."""

    window_s: float | None = None  # half-width in seconds, scaled by match.fps
    window: int | None = None  # half-width in frames; set this or window_s, not both
    min_spell_s: float | None = None  # drop predicted segments shorter than this; None = off
    switch_delay_s: float | None = None  # delay H<->A switches by this much; None = off

    def __post_init__(self):
        if self.window is not None and self.window_s is not None:
            raise ValueError("give window (frames) or window_s (seconds), not both")
        if self.window is None and self.window_s is None:
            self.window_s = 2.0

    def frames(self, fps: float) -> int:
        return self.window if self.window is not None else to_frames(self.window_s, fps)

    def predict(self, match: Match) -> np.ndarray:
        codes = majority_smooth(nearest_team(match), self.frames(match.fps))
        if self.min_spell_s:
            codes = drop_short(codes, play_runs(match.period), to_frames(self.min_spell_s, match.fps))
        if self.switch_delay_s:
            codes = delay_switches(codes, play_runs(match.period),
                                   to_frames(self.switch_delay_s, match.fps))
        return codes
