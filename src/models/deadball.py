"""Dead-ball (D) detector from tracking only, on top of the nearest-player team (backlog B2).

The team comes from NearestPlayer (nearest tracked player, centred majority vote). Then frames the
detector calls dead become D, and optionally (min_spell_s) short predicted segments are dropped
(postprocess.drop_short, findings F32/F34) and (switch_delay_s) H<->A switches moved later
(postprocess.delay_switches, findings F43). Cues, all label-free:
  - calm: the players' mean speed over +/- speed_s is below speed_mps. Players walk during
    stoppages on every source (findings F29). The cue that carries over.
  - out: the ball is off the pitch (by more than OUT_MARGIN_M), and the next out_s seconds after
    it leaves (the restart is being prepared).
Ball speed is left out on purpose: IDSSE freezes the ball in dead time (F29), so a detector
leaning on it would look perfect there and nowhere else. "No ball" is not a cue either: what it
means differs by source (F3, F14, F21). A kick-off shape cue (ball on the spot, everyone in their
own half) was tried and dropped: it changed nothing (findings F38).
The cue mask is cleaned like spells: dead runs shorter than min_dead_s are dropped, then live gaps
shorter than min_live_s between dead runs are filled. True stoppages last 12-24 s median (F15, F20).

Frames the nearest-player vote already calls D (nobody near the ball for the whole window) stay D
unless missing="team", which gives them the last team instead when the detector says live.

Defaults were picked on the mean spellF1 over the three BAR-ATH-like hidden-ball settings (F30),
SkillCorner first; see findings F37-F38 and `scripts/check_deadball.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.nearest import NearestPlayer
from src.models.postprocess import delay_switches, drop_short, play_runs
from src.models.timing import to_frames
from src.official import D, X, pm
from src.schema import Match

OUT_MARGIN_M = 0.5  # tracking noise around the lines
MAX_SPEED_MPS = 12.0  # tracking glitches


def _mean_over(values: np.ndarray, valid: np.ndarray, half: int) -> np.ndarray:
    """Centred mean of values over +/- half frames, counting only valid frames. NaN where none."""
    kernel = np.ones(2 * half + 1)
    total = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def player_speed(match: Match, half_s: float, span_s: float = 0.0) -> np.ndarray:
    """(n,) the players' mean speed in m/s, averaged over +/- half_s. NaN where nobody is seen.
    Each player's speed is its step from the previous frame (span_s=0) or, with span_s > 0, its
    displacement across a centred span of span_s divided by span_s (C5, findings F72): frame-to-
    frame position noise then barely adds to it, while real motion over a second hardly changes."""
    players = match.players
    if span_s > 0:
        k = max(1, to_frames(span_s / 2, match.fps))
        pad = np.full((k,) + players.shape[1:], np.nan)
        padded = np.concatenate((pad, players, pad))
        step = (padded[2 * k:] - padded[:-2 * k]) / (2 * k)
    else:
        step = np.diff(players, axis=0, prepend=players[:1])
    with np.errstate(invalid="ignore"):
        speed = np.minimum(np.hypot(step[..., 0], step[..., 1]) * match.fps, MAX_SPEED_MPS)
    seen = np.sum(~np.isnan(speed), axis=1)
    mean = np.nansum(speed, axis=1) / np.maximum(seen, 1)
    return _mean_over(mean, seen > 0, to_frames(half_s, match.fps))


def ball_out(match: Match) -> np.ndarray:
    """(n,) bool: the ball is seen off the pitch."""
    length, width = match.pitch
    x, y = match.ball[:, 0], match.ball[:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
                | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def drop_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """mask without its True runs shorter than min_frames."""
    out = mask.copy()
    for start, stop in zip(*_runs(mask)):
        if stop - start < min_frames:
            out[start:stop] = False
    return out


def fill_gaps(mask: np.ndarray, max_frames: int) -> np.ndarray:
    """mask with False gaps shorter than max_frames between two True runs set True."""
    out = mask.copy()
    starts, stops = _runs(mask)
    for stop, start in zip(stops[:-1], starts[1:]):
        if start - stop < max_frames:
            out[stop:start] = True
    return out


def extend(mask: np.ndarray, frames: int) -> np.ndarray:
    """mask with each True run extended by `frames` after its end."""
    out = mask.copy()
    if frames > 0:
        for stop in _runs(mask)[1]:
            out[stop:stop + frames] = True
    return out


@dataclass(kw_only=True)
class DeadBall:
    """Nearest-player team + a tracking-only dead-ball detector (see the module docstring).
    Durations in seconds. speed_mps=0 switches the calm cue off, out_s<0 the ball-out cue."""

    window_s: float = 1.5  # team vote half-width (NearestPlayer)
    speed_s: float = 2.0  # player speed is averaged over +/- this
    speed_mps: float = 1.2  # calm below this
    out_s: float = 2.0  # dead for this long after the ball is last seen off the pitch; < 0: off
    min_dead_s: float = 2.0  # dead runs shorter than this are dropped
    min_live_s: float = 4.0  # live gaps shorter than this between dead runs are filled
    missing: str = "keep"  # where the vote says D and the detector live: "keep" D or "team"
    min_spell_s: float = 2.5  # then drop predicted segments shorter than this (F32); 0: off
    switch_delay_s: float = 0.0  # then move H<->A switches this much later (C2, F43); 0: off

    def __post_init__(self):
        if self.missing not in ("keep", "team"):
            raise ValueError("missing must be 'keep' or 'team', not {!r}".format(self.missing))

    def dead(self, match: Match) -> np.ndarray:
        """(n,) bool: frames the detector calls dead. Never outside the periods."""
        fps = match.fps
        cue = np.zeros(match.n_frames, dtype=bool)
        if self.speed_mps > 0:
            speed = player_speed(match, self.speed_s)
            cue |= np.nan_to_num(speed, nan=np.inf) < self.speed_mps
        if self.out_s >= 0:
            cue |= extend(ball_out(match), to_frames(self.out_s, fps))
        cue &= match.period > 0
        cue = drop_runs(cue, to_frames(self.min_dead_s, fps))
        cue = fill_gaps(cue, to_frames(self.min_live_s, fps))
        return cue & (match.period > 0)

    def predict(self, match: Match) -> np.ndarray:
        codes = NearestPlayer(window_s=self.window_s).predict(match)
        runs = play_runs(match.period)
        dead = self.dead(match)
        if self.missing == "team":
            unseen = (codes == D) & ~dead
            codes[unseen] = X
            codes = pm.fill_abstentions(codes, runs)
            codes[codes == X] = D  # a run with no team at all
        codes[dead] = D
        if self.min_spell_s > 0:
            codes = drop_short(codes, runs, to_frames(self.min_spell_s, match.fps))
        if self.switch_delay_s > 0:
            codes = delay_switches(codes, runs, to_frames(self.switch_delay_s, match.fps))
        return codes.astype(np.int8)
