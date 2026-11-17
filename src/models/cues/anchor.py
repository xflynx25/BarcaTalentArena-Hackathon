"""Cue `anchor` (17 Nov): fill a gap only when the clear moments on both sides agree.

Why: v7 decodes the whole half at once, and a general penalty on short spells already lost (F117:
the model makes fewer short spells than the truth). This is the conditional version. An *anchor*
is a moment where possession is obvious from the tracking alone; between two anchors of the same
team, close together and with nothing that ends a possession between them, the frames most likely
belong to that team too. A phantom spell there is often a loud wrong touch, so this adds evidence
(it can outvote the touch) rather than raising the switch cost (which only makes switches dearer
everywhere and, alone, does not outvote it).

Anchor, label-free, from the ball the decoder was given (`original`: the match as delivered, so a
ball whose gaps were filled or imputed never counts as seen) and the players of the prepared match:
  the ball is seen, the nearest player of one team is within REACH_M (src.models.decoder's
  team_distances), the nearest of the other team is at least MARGIN_M further away (and seen), and
  this holds for at least HOLD_S in a row.
Breaks (the chain between two anchors is broken if any frame strictly between them, or either
anchor's own frame, is one): outside play or a period change, the ball seen off the pitch
(src.models.deadball.ball_out on the delivered ball), or a frame inside any set-piece shape
(src.models.setpiece.shape_masks: corner, wall, throw, goal kick, kick-off, walk). A break frame is
never an anchor, so two same-team anchors with a goal kick between them are two possessions.

Effect: on frame t strictly between anchors p < t < q of the same team with (q - p) / fps < gap
and no break, `gain * w(t) / fps` is added to that team's score (per-second evidence, like every
score in the decoder), with

    w(t) = max(0, 1 - d(t) / gap),   d(t) = seconds from t to the nearer of p and q

so an anchor 1 s away counts more than one 6 s away, and the weight reaches zero at the gap cap.
Nothing is added on anchor frames, where the two anchors disagree, where the chain is broken, or
before the first / after the last anchor of a run. Gain 0 -> None (v7's bytes unchanged).

    cues=...+anchor:GAIN/gap=GAP          (GAP in seconds; fixed from scripts/check_anchor.py)

Runtime: numpy only, no Python loop over frames. Deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.models.cues import Effect
from src.models.deadball import ball_out
from src.models.decoder import team_distances
from src.models.setpiece import SHAPES, shape_masks

REACH_M = 1.5  # the nearest player of the anchor team is within this
MARGIN_M = 2.0  # the other team's nearest is at least this much further away
HOLD_S = 0.3  # for at least this long in a row
GAP_S = 4.0  # default cap on the seconds between the two anchors (Part A picks the real one)
NONE, HOME_T, AWAY_T = 0, 1, 2  # anchor team codes (H column 0, A column 1 of the scores)


def raw_clear(players: np.ndarray, slot_team: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """(n,) int8: HOME_T / AWAY_T where the frame alone is clear for that team, else NONE."""
    view = SimpleNamespace(players=players, slot_team=slot_team, ball=ball)
    d_home, d_away = team_distances(view)
    seen = ~np.isnan(ball[:, 0])
    out = np.zeros(len(ball), dtype=np.int8)
    with np.errstate(invalid="ignore"):
        home = seen & (d_home <= REACH_M) & (d_away - d_home >= MARGIN_M)
        away = seen & (d_away <= REACH_M) & (d_home - d_away >= MARGIN_M)
    out[home] = HOME_T
    out[away] = AWAY_T
    return out


def break_mask(match, original) -> np.ndarray:
    """(n,) bool: frames that end a possession chain (outside play, ball seen off the pitch, any
    set-piece shape). Period changes are handled in chains() through the period array."""
    brk = np.asarray(match.period) == 0
    brk = brk | ball_out(original)
    for m in shape_masks(match, SHAPES).values():
        brk = brk | m
    return brk


def anchors(match, original) -> np.ndarray:
    """(n,) int8 anchor team per frame (NONE where the frame is no anchor): runs of a clear frame
    for one team at least HOLD_S long, inside one period, with no break frame."""
    team = raw_clear(np.asarray(match.players), np.asarray(match.slot_team),
                     np.asarray(original.ball, dtype=np.float64))
    team[break_mask(match, original)] = NONE
    n = len(team)
    if n == 0:
        return team
    period = np.asarray(match.period)
    hold = max(1, int(round(HOLD_S * match.fps)))
    # runs of equal (team, period)
    key = team.astype(np.int64) * 1000 + period.astype(np.int64)
    edges = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [n]))
    out = np.zeros(n, dtype=np.int8)
    for a, b in zip(starts, stops):  # one iteration per run, not per frame
        if team[a] != NONE and b - a >= hold:
            out[a:b] = team[a]
    return out


def chains(anchor: np.ndarray, brk: np.ndarray, period: np.ndarray, fps: float):
    """Per frame: (prev anchor frame or -1, next anchor frame or -1, team of the gap or NONE,
    seconds to the nearer anchor, gap seconds q - p). Team is NONE unless the frame is strictly
    between two anchors of the same team with no break and no period change between them."""
    n = len(anchor)
    idx = np.arange(n)
    is_a = anchor != NONE
    prev = np.where(is_a, idx, -1)
    np.maximum.accumulate(prev, out=prev)
    prev = np.concatenate(([-1], prev[:-1])) if n else prev  # strictly before t
    nxt = np.where(is_a, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    nxt = np.concatenate((nxt[1:], [n])) if n else nxt  # strictly after t
    ok = (prev >= 0) & (nxt < n) & ~is_a
    p, q = np.where(ok, prev, 0), np.where(ok, nxt, 0)
    cb = np.concatenate(([0], np.cumsum(brk.astype(np.int64))))
    pc = np.concatenate(([0], np.cumsum(np.diff(period) != 0))) if n else np.zeros(0, np.int64)
    # breaks on frames p+1 .. q-1, and no period change between p and q
    ok &= (cb[q] - cb[p + 1]) == 0
    ok &= pc[q] == pc[p]
    ok &= anchor[p] == anchor[q]
    team = np.where(ok, anchor[q], NONE).astype(np.int8)
    near = np.where(ok, np.minimum(idx - p, q - idx) / fps, np.inf)
    gap = np.where(ok, (q - p) / fps, np.inf)
    return np.where(ok, prev, -1), np.where(ok, nxt, -1), team, near, gap


_LAST: list = [None, None, None]  # (original ball, players, result): a grid decodes a match often


def gap_team(match, original):
    """chains() for this match (cached on the delivered ball and the prepared players)."""
    if _LAST[0] is not original.ball or _LAST[1] is not match.players:
        anchor = anchors(match, original)
        brk = break_mask(match, original)
        _LAST[:] = [original.ball, match.players,
                    (anchor,) + chains(anchor, brk, np.asarray(match.period), match.fps)]
    return _LAST[2]


def weights(team: np.ndarray, near: np.ndarray, gap: np.ndarray, cap: float) -> np.ndarray:
    """(n, 2) weight in [0, 1] on (H, A): 1 - near / cap inside a same-team gap shorter than cap."""
    w = np.zeros((len(team), 2))
    inside = (team != NONE) & (gap < cap)
    val = np.clip(1.0 - near / cap, 0.0, 1.0)
    w[:, 0] = np.where(inside & (team == HOME_T), val, 0.0)
    w[:, 1] = np.where(inside & (team == AWAY_T), val, 0.0)
    return w


def effect(match, original, decoder, gain, gap: float = GAP_S):
    """gain x w(t) evidence per second on the gap team's H / A score; None at gain 0 or no gap."""
    if not gain:
        return None
    _, _, _, team, near, span = gap_team(match, original)
    w = weights(team, near, span, float(gap))
    if not np.any(w):
        return None
    add = np.zeros((match.n_frames, 3))
    add[:, :2] = float(gain) * w / match.fps
    return Effect(add=add)
