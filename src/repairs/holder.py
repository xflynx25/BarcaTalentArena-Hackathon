"""C16 (2): holder persistence. When the player at the ball vanishes and nobody else arrives, a
stand-in moves with the ball for a short while.

Why (backlog C16, owner's error review): players vanish from automatic tracking, and most often in
crowds and duels at the ball (official DATA.md; degrade.fragment simulates breaks likelier in crowds).
When the player who has the ball drops out, the team that has it loses its nearest player, and the
model reads the ball as nobody's or the other team's. C13's link (src/tracks.py) only bridges the gap
once the player comes back in reach; an end it can't link just stops.

apply(match, reach=2, max_s=1.5): for every tracklet end (a slot seen on frame e and not on e+1, same
period) where the ball is seen on e, that player is the nearest to the ball and within `reach` m of
it, the slot is filled from e+1 with the ball's position plus the player's offset from the ball at e,
for at most `max_s` s, and only while: the ball is seen and slower than `speed` m/s (a dribble, not
a pass: after a pass the passer stays behind), the period is the same, the slot is still
empty, and no other player is within `reach` m of the ball (nobody else arrives; that includes the
same person coming back in another slot). Uses the ball, not the labels; deterministic. Runs before
C13's track repair (repairs come first in LearnedDecoder.prepare), so the extended end is what C13
links. Returns the match itself when it fills nothing.

    --param repairs=holder               (defaults)
    --param repairs=holder/reach=2/max_s=1/speed=5
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.schema import Match

REACH_M = 2.0
MAX_S = 1.5
SPEED_MPS = 5.0  # the ball faster than this (over one frame) is passed, not carried


def stand_ins(match: Match, reach: float = REACH_M, max_s: float = MAX_S,
              speed: float = SPEED_MPS) -> np.ndarray:
    """(n, k) bool: the frames and slots apply() fills."""
    players, ball, period = match.players, match.ball, match.period
    n, k = players.shape[:2]
    out = np.zeros((n, k), dtype=bool)
    seen = ~np.isnan(players).any(axis=2)
    ball_seen = ~np.isnan(ball).any(axis=1)
    if n < 2 or not ball_seen.any():
        return out
    dist = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))  # (n, k), NaN if either unseen
    near = np.where(np.isnan(dist), np.inf, dist)
    nearest = near.argmin(axis=1)
    within = (near <= reach).sum(axis=1)  # players within reach of the ball per frame
    longest = max(1, round(max_s * match.fps))
    step = np.hypot(*np.diff(ball, axis=0, prepend=ball[:1]).T) * match.fps  # NaN where unseen
    slow = ~(step > speed)  # the first frame and frames after a missing ball count as slow
    ends = seen[:-1] & ~seen[1:] & (period[:-1] == period[1:])[:, None]
    for e, slot in zip(*np.nonzero(ends)):
        if not ball_seen[e] or nearest[e] != slot or near[e, slot] > reach:
            continue
        for f in range(e + 1, min(n, e + 1 + longest)):
            if (period[f] != period[e] or seen[f, slot] or not ball_seen[f] or within[f] > 0
                    or not slow[f]):
                break
            out[f, slot] = True
    return out


def apply(match: Match, reach: float = REACH_M, max_s: float = MAX_S, speed: float = SPEED_MPS) -> Match:
    fill = stand_ins(match, float(reach), float(max_s), float(speed))
    if not fill.any():
        return match
    players = match.players.copy()
    rows, slots = np.nonzero(fill)
    # the offset from the ball at the slot's last seen frame before each filled run
    order = np.lexsort((rows, slots))  # walk each slot's frames in time order
    rows, slots = rows[order], slots[order]
    last = np.empty_like(rows)
    for i in range(len(rows)):
        cont = i and slots[i - 1] == slots[i] and rows[i - 1] == rows[i] - 1
        last[i] = last[i - 1] if cont else rows[i] - 1
    offset = match.players[last, slots] - match.ball[last]
    players[rows, slots] = match.ball[rows] + offset
    length, width = match.pitch
    players[rows, slots, 0] = np.clip(players[rows, slots, 0], 0.0, length)
    players[rows, slots, 1] = np.clip(players[rows, slots, 1], 0.0, width)
    return dataclasses.replace(match, players=players).validate()
