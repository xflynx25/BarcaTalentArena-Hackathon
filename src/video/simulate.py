"""A simulated video ball for matches without video (backlog V2; for V5's refit and V2's value check).

    vb, conf = simulate(clean, hidden, seed=0)              # (n, 2) metres / NaN, (n,) / NaN
    match = attach(degraded, clean, seed=0)                 # degraded with Match.video_ball set

Only SoccerTrack has video, but `learned` trains on every source. So on the frames the dropout grid
hides (src/ingest/degrade.py), the video's view is drawn from the tracked ball the grid took away,
with a detector's measured or assumed behaviour (PROVISIONAL: V2's stack, F137; TARGET: a good
detector, F138), per context:
  - on the 5 Hz grid the real run uses (src.video.run.STEP), the video picks the ball with
    probability RECALL[context] (context: "crowd" >= 3 players within 3 m of the ball, else "fast"
    ball >= 8 m/s, else "open");
  - a pick is right with probability 1 - WRONG[context] and then off by Gaussian noise (NOISE_M per
    axis); a wrong pick lands on the nearest player 2-10 m from the ball (what a detector confuses
    with a ball: a sock, a boot), plus the same noise;
  - between two picks <= 0.4 s apart the ball is interpolated (src/video/track.to_frames).
Seeded by (seed, match id), so a match always gets the same draw. Pass recall / wrong / noise to
override (recall=1, wrong=0, noise_m=0 is the oracle: the true ball on every hidden frame).
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np

from src.schema import Match
from src.video import track

STEP = 5
# PROVISIONAL: what V2's provisional numpy stack does on SoccerTrack's held-out matches (F137, anchor
# frames, mean over the 0.41 grid points): the linked track picks something on 60% of hidden 5 Hz
# frames and 72% of those picks are > 5 m off (moving white kit near a player), ~1 m when right. TARGET: a hypothetical good detector (a
# learned CNN, V1 / V5), for F138's "what would it be worth". RECALL / WRONG / NOISE_M default to
# PROVISIONAL, the stack this repo has.
PROVISIONAL = dict(recall={"open": 0.6, "fast": 0.6, "crowd": 0.6},
                   wrong={"open": 0.72, "fast": 0.72, "crowd": 0.72}, noise_m=1.0)
TARGET = dict(recall={"open": 0.8, "fast": 0.7, "crowd": 0.5},
              wrong={"open": 0.1, "fast": 0.1, "crowd": 0.2}, noise_m=1.0)
RECALL = dict(PROVISIONAL["recall"])
WRONG = dict(PROVISIONAL["wrong"])
NOISE_M = PROVISIONAL["noise_m"]
CROWD_M, CROWD_N, FAST_MPS = 3.0, 3, 8.0


def context(clean: Match) -> np.ndarray:
    """(n,) 0 open, 1 fast, 2 crowd (crowd wins), from the clean ball and players."""
    b = clean.ball
    dx = clean.players[:, :, 0] - b[:, None, 0]
    dy = clean.players[:, :, 1] - b[:, None, 1]
    with np.errstate(invalid="ignore"):
        crowd = np.sum(dx * dx + dy * dy <= CROWD_M ** 2, axis=1) >= CROWD_N
    step = np.diff(b, axis=0, prepend=b[:1])
    speed = np.nan_to_num(np.hypot(step[:, 0], step[:, 1]) * clean.fps, nan=0.0)
    return np.where(crowd, 2, np.where(speed >= FAST_MPS, 1, 0)).astype(np.int64)


def simulate(clean: Match, hidden: np.ndarray, seed: int = 0, recall: dict | None = None,
             wrong: dict | None = None, noise_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    recall = dict(RECALL, **(recall or {}))
    wrong = dict(WRONG, **(wrong or {}))
    noise = NOISE_M if noise_m is None else noise_m
    names = ("open", "fast", "crowd")
    n = clean.n_frames
    rng = np.random.default_rng([seed, zlib.crc32(clean.match_id.encode()), 2])
    draws = rng.random((n, 2))
    jitter = rng.normal(0.0, 1.0, (n, 2)) * noise
    ctx = context(clean)
    frames = np.arange(0, n, STEP)
    keep = hidden[frames] & np.isfinite(clean.ball[frames, 0])
    frames = frames[keep]
    xs = np.full(len(frames), np.nan)
    ys = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        c = names[ctx[f]]
        if draws[f, 0] >= recall[c]:
            continue
        target = clean.ball[f]
        if draws[f, 1] < wrong[c]:
            p = clean.players[f]
            d = np.hypot(p[:, 0] - target[0], p[:, 1] - target[1])
            ok = np.isfinite(d) & (d >= 2.0) & (d <= 10.0)
            if not ok.any():
                continue
            target = p[np.flatnonzero(ok)[np.argmin(d[ok])]]
        xs[i], ys[i] = target + jitter[f]
    tr = {"frame": frames, "x": track.snap(xs, track.GRID_M), "y": track.snap(ys, track.GRID_M),
          "logit": np.where(np.isfinite(xs), 1.0, np.nan)}
    vb, conf = track.to_frames(tr, n, max_gap=2 * STEP)
    vb[~hidden] = np.nan  # the simulation speaks only where the tracking lost the ball
    conf[~hidden] = np.nan
    return vb, conf


def attach(degraded: Match, clean: Match, seed: int = 0, **kw) -> Match:
    hidden = np.isnan(degraded.ball[:, 0]) & np.isfinite(clean.ball[:, 0])
    vb, conf = simulate(clean, hidden, seed, **kw)
    return dataclasses.replace(degraded, video_ball=vb, video_conf=conf)
