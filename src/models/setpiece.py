"""Label-free set-piece shapes from the players' positions (backlog B24, findings F94-F95).

A stoppage usually ends in a set piece, and each kind has a shape anyone watching can see:
  corner   a player within CORNER_R_M of a corner flag and at least CORNER_BOX players in the
           penalty area at that end
  wall     at least WALL_MIN still players of one team bunched within WALL_GAP_M of each other
           (a wall), with a still player of the other team WALL_NEAR_M..WALL_FAR_M from its centre
  throw    a still player on or beyond a touchline (within THROW_IN_M inside it, not near a
           corner) while the other players are slow
  goalkick a player in a goal area, nobody of the other team in that penalty area, and at least
           GOALKICK_WAIT of them waiting within GOALKICK_WAIT_M beyond its line
  kickoff  at least KICKOFF_SHARE of the seen players in their own half and somebody within
           KICKOFF_R_M of the centre spot
  walk     everyone walking: the WALK_Q quantile of the players' speeds below WALK_MPS (injuries,
           substitutions, the slow walk to any restart)
Each mask is cleaned to runs of at least SHAPE_MIN_S. Speeds are displacement over SPEED_SPAN_S
(centred), so frame-level position noise cancels (C5). The ball is never used: on BAR-ATH-like
tracking it is missing exactly in the calm stretches these shapes live in (F30).

Positions only, no labels; so a shape can be checked on every source the same way. A shape forms
during the stoppage, after it began, so the cue it gives for the stoppage's start is late; the
end of a shape (it breaks when the kick is taken) is the cue for the restart.
"""

from __future__ import annotations

import numpy as np

from src.models.timing import to_frames
from src.schema import AWAY, HOME, Match

SPEED_SPAN_S = 1.0
STILL_MPS = 1.5
SHAPE_MIN_S = 1.0
BOX_DEPTH_M, BOX_HALF_W_M = 16.5, 20.16
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_W_M = 5.5, 9.16
CORNER_R_M, CORNER_BOX = 3.0, 8
WALL_MIN, WALL_GAP_M, WALL_NEAR_M, WALL_FAR_M = 3, 1.5, 7.0, 12.0
THROW_IN_M, THROW_SLOW_MPS = 0.5, 2.0
GOALKICK_WAIT, GOALKICK_WAIT_M = 2, 15.0
KICKOFF_SHARE, KICKOFF_R_M = 0.9, 3.0
WALK_Q, WALK_MPS = 0.9, 2.0
SHAPES = ("corner", "wall", "throw", "goalkick", "kickoff", "walk")


def player_speeds(match: Match, span_s: float = SPEED_SPAN_S) -> np.ndarray:
    """(n, k) each slot's speed (m/s): displacement over span_s centred on the frame. NaN where
    the slot isn't seen at both ends."""
    p = match.players
    n = len(p)
    k = max(1, to_frames(span_s, match.fps))
    half = k // 2
    out = np.full(p.shape[:2], np.nan)
    if n > k:
        d = np.hypot(*(p[k:] - p[:-k]).transpose(2, 0, 1)) * match.fps / k
        out[half:half + n - k] = d
        out[:half] = d[0]  # the edges take the nearest full span
        out[half + n - k:] = d[-1]
    return out


def attack_sign(match: Match) -> np.ndarray:
    """(n,) +1 where HOME's own goal is at x = 0 (they defend the low end) in that frame's period,
    -1 where it is at x = length. From the teams' mean x over each period; 0 outside periods."""
    out = np.zeros(match.n_frames)
    x = match.players[..., 0]
    for period in np.unique(match.period[match.period > 0]):
        rows = match.period == period
        with np.errstate(invalid="ignore"):
            home = np.nanmean(x[rows][:, match.slot_team == HOME])
            away = np.nanmean(x[rows][:, match.slot_team == AWAY])
        out[rows] = 1.0 if home <= away else -1.0
    return out


def _clean(mask: np.ndarray, period: np.ndarray, min_frames: int) -> np.ndarray:
    mask = mask & (period > 0)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    start, stop = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=bool)
    for a, b in zip(start, stop):
        if b - a >= min_frames:
            out[a:b] = True
    return out


def _in_box(x, y, length, width, end, depth, half_w):
    """Inside the rectangle of the given depth and half width at end 0 (x = 0) or 1 (x = length)."""
    near = x <= depth if end == 0 else x >= length - depth
    return near & (np.abs(y - width / 2) <= half_w)


def corner_mask(match: Match) -> np.ndarray:
    length, width = match.pitch
    x, y = match.players[..., 0], match.players[..., 1]
    out = np.zeros(match.n_frames, dtype=bool)
    with np.errstate(invalid="ignore"):
        for end, cx in ((0, 0.0), (1, length)):
            box = _in_box(x, y, length, width, end, BOX_DEPTH_M, BOX_HALF_W_M).sum(axis=1)
            flag = np.zeros(match.n_frames, dtype=bool)
            for cy in (0.0, width):
                flag |= (np.hypot(x - cx, y - cy) <= CORNER_R_M).any(axis=1)
            out |= flag & (box >= CORNER_BOX)
    return out


def wall_mask(match: Match, speed: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.zeros(match.n_frames, dtype=bool)
    for lo in range(0, match.n_frames, chunk):
        hi = min(match.n_frames, lo + chunk)
        out[lo:hi] = _wall(match.players[lo:hi], speed[lo:hi], match.slot_team)
    return out


def _wall(p: np.ndarray, speed: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    out = np.zeros(len(p), dtype=bool)
    with np.errstate(invalid="ignore"):
        still = speed < STILL_MPS
    for team in (HOME, AWAY):
        mine, theirs = slot_team == team, slot_team != team
        a, s = p[:, mine], still[:, mine]
        b, sb = p[:, theirs], still[:, theirs]
        pair = s[:, :, None] & s[:, None, :]
        with np.errstate(invalid="ignore"):
            d = np.hypot(a[:, :, None, 0] - a[:, None, :, 0], a[:, :, None, 1] - a[:, None, :, 1])
            close = ((d <= WALL_GAP_M) & pair).sum(axis=2) - 1  # minus itself
            group = ((d <= 2 * WALL_GAP_M) & pair).sum(axis=2) - 1
        # a wall member: still, a still team-mate within one gap, WALL_MIN - 1 within two
        member = s & (close >= 1) & (group >= WALL_MIN - 1)
        rows = np.flatnonzero(member.any(axis=1))
        if not len(rows):
            continue
        m = member[rows]
        centre = np.where(m[..., None], a[rows], 0.0).sum(axis=1) / m.sum(axis=1)[:, None]
        with np.errstate(invalid="ignore"):
            dist = np.hypot(b[rows, :, 0] - centre[:, None, 0], b[rows, :, 1] - centre[:, None, 1])
            kicker = ((dist >= WALL_NEAR_M) & (dist <= WALL_FAR_M) & sb[rows]).any(axis=1)
        out[rows[kicker]] = True
    return out


def throw_mask(match: Match, speed: np.ndarray) -> np.ndarray:
    length, width = match.pitch
    x, y = match.players[..., 0], match.players[..., 1]
    with np.errstate(invalid="ignore"):
        on_line = ((y <= THROW_IN_M) | (y >= width - THROW_IN_M)) & (x > CORNER_R_M) \
            & (x < length - CORNER_R_M) & (speed < STILL_MPS)
        taker = on_line.any(axis=1)
        seen = ~np.isnan(speed)
        median = np.nanmedian(np.where(seen.any(axis=1)[:, None], speed, np.inf), axis=1)
        others = median < THROW_SLOW_MPS
    return taker & others


def goalkick_mask(match: Match, sign: np.ndarray) -> np.ndarray:
    length, width = match.pitch
    x, y = match.players[..., 0], match.players[..., 1]
    out = np.zeros(match.n_frames, dtype=bool)
    with np.errstate(invalid="ignore"):
        for end in (0, 1):
            area = _in_box(x, y, length, width, end, GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_W_M)
            box = _in_box(x, y, length, width, end, BOX_DEPTH_M, BOX_HALF_W_M)
            wait = _in_box(x, y, length, width, end, BOX_DEPTH_M + GOALKICK_WAIT_M, width) & ~box
            # the defending team at this end: HOME where HOME's goal is at this end
            home_defends = (sign > 0) if end == 0 else (sign < 0)
            for team in (HOME, AWAY):
                defends = home_defends if team == HOME else ~home_defends & (sign != 0)
                mine, theirs = match.slot_team == team, match.slot_team != team
                keeper = area[:, mine].any(axis=1)
                clear = ~box[:, theirs].any(axis=1)
                waiting = wait[:, theirs].sum(axis=1) >= GOALKICK_WAIT
                out |= defends & keeper & clear & waiting
    return out


def kickoff_mask(match: Match, sign: np.ndarray) -> np.ndarray:
    length, width = match.pitch
    x, y = match.players[..., 0], match.players[..., 1]
    home = match.slot_team == HOME
    with np.errstate(invalid="ignore"):
        # own half: HOME on the low half where sign > 0
        low = x <= length / 2 + 1.0
        high = x >= length / 2 - 1.0
        own = np.where(home[None, :], np.where(sign[:, None] > 0, low, high),
                       np.where(sign[:, None] > 0, high, low))
        seen = ~np.isnan(x)
        share = (own & seen).sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
        spot = (np.hypot(x - length / 2, y - width / 2) <= KICKOFF_R_M).any(axis=1)
    return (share >= KICKOFF_SHARE) & spot & (sign != 0)


def walk_mask(speed: np.ndarray) -> np.ndarray:
    seen = (~np.isnan(speed)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        q = np.nanquantile(np.where(seen[:, None] > 0, speed, 0.0), WALK_Q, axis=1)
        return (q < WALK_MPS) & (seen >= 6)


_CACHE: dict = {}  # the last match's masks: shapes use only players and periods, not the ball


def shape_masks(match: Match, shapes=SHAPES) -> dict:
    """{shape: (n,) bool mask, cleaned to runs of at least SHAPE_MIN_S inside halves}. The last
    match's masks are kept (keyed on its players and period arrays, which hiding the ball leaves
    alone), so asking again for another cue or grid point costs nothing."""
    unknown = set(shapes) - set(SHAPES)
    if unknown:
        raise ValueError("unknown shapes {} (known: {})".format(sorted(unknown), SHAPES))
    same = (_CACHE.get("players") is match.players and _CACHE.get("period") is match.period
            and _CACHE.get("fps") == match.fps and _CACHE.get("pitch") == match.pitch)
    if not same:
        _CACHE.clear()
        _CACHE.update(players=match.players, period=match.period, fps=match.fps,
                      pitch=match.pitch, masks={})
    masks = _CACHE["masks"]
    missing = [s for s in shapes if s not in masks]
    if missing:
        speed = player_speeds(match)
        sign = attack_sign(match)
        makers = {"corner": lambda: corner_mask(match), "wall": lambda: wall_mask(match, speed),
                  "throw": lambda: throw_mask(match, speed),
                  "goalkick": lambda: goalkick_mask(match, sign),
                  "kickoff": lambda: kickoff_mask(match, sign), "walk": lambda: walk_mask(speed)}
        k = to_frames(SHAPE_MIN_S, match.fps)
        for s in missing:
            masks[s] = _clean(makers[s](), match.period, k)
    return {s: masks[s].copy() for s in shapes}
