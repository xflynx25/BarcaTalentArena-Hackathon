"""Frame gates and gap evidence for `learned`'s gated feature groups (local L1/L3, findings F200+).

L1 ("shapef", "shapeo"): the team-shape columns on frames where the ball is seen but free (nobody
within FREE_M), and for "shapef" also moving at FLIGHT_MPS or more (F90's "travelling": there the
ball features are near chance and shape is at its best).

L3 ("deadm"): player motion and set-piece shapes as live-vs-dead evidence on ball-missing frames.

Label-free and deterministic (numpy float64, no randomness). No import of `learned`, to avoid a cycle.
"""

from __future__ import annotations

import numpy as np

from src.models.deadball import player_speed
from src.models.onset import ball_speed
from src.models.setpiece import shape_masks
from src.schema import Match

FLIGHT_MPS = 12.0  # F90's "travelling": 2-D ball speed at least this (onset.ball_speed, +/-0.2 s)
FLIGHT_HALF_S = 0.2  # ball speed smoothed over +/- this
FREE_M = 3.0  # ... and both teams' closest players farther than this

DEAD_SHAPES = ("walk", "throw", "wall", "corner", "kickoff")  # not goalkick: unusable (F94)
DEAD_SPAN_S = 1.0  # player speed from displacement over this span (C5), whatever the decoder's override
DEAD_HALF_S = (2.0, 0.5)  # player speed averaged over +/- these (as the common block's speeds)


def free_ball(match: Match, d_home: np.ndarray, d_away: np.ndarray,
              free_m: float = FREE_M) -> np.ndarray:
    """(n,) bool: ball seen and both teams' closest players farther than free_m. A team with no
    player (NaN distance) counts as far. False wherever the ball is missing."""
    seen = ~np.isnan(match.ball[:, 0])
    near = np.fmin(np.nan_to_num(d_home, nan=np.inf), np.nan_to_num(d_away, nan=np.inf))
    return seen & (near > free_m)


def flight(match: Match, d_home: np.ndarray, d_away: np.ndarray, mps: float = FLIGHT_MPS,
           free_m: float = FREE_M) -> np.ndarray:
    """(n,) bool: free_ball and onset.ball_speed(match, FLIGHT_HALF_S) >= mps (NaN speed -> False)."""
    speed = np.nan_to_num(ball_speed(match, FLIGHT_HALF_S), nan=-np.inf)
    return free_ball(match, d_home, d_away, free_m) & (speed >= mps)


def dead_gap_raw(match: Match) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(speed over +/-2 s, speed over +/-0.5 s, (n, len(DEAD_SHAPES)) float masks), label-free,
    ball-free. Speeds from deadball.player_speed(match, half_s, DEAD_SPAN_S) (NaN where nobody is
    seen); masks from setpiece.shape_masks(match, DEAD_SHAPES), which keeps only the last match's
    masks (keyed on its players array, which ball dropout leaves alone), so a worker reuses them
    across that match's dropout settings without holding every match in memory."""
    s2 = player_speed(match, DEAD_HALF_S[0], DEAD_SPAN_S)
    s05 = player_speed(match, DEAD_HALF_S[1], DEAD_SPAN_S)
    found = shape_masks(match, DEAD_SHAPES)
    masks = np.stack([found[s] for s in DEAD_SHAPES], axis=1).astype(np.float64)
    return s2, s05, masks
