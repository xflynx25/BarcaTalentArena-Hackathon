"""Label-free dead-ball onset cues (backlog B12, findings F76-F79).

A decoder's D stretch starts when its dead evidence (calm players, no ball, ball off the pitch, all
averaged over a few seconds) has built up, which can be after the stoppage began. These are the
instants that mark a stoppage's start on the tracking itself:
  out    the ball is first seen off the pitch (by more than deadball.OUT_MARGIN_M)
  gap    a ball-missing gap of at least GAP_MIN_S starts
  stop   the ball comes to rest: its speed (smoothed over +/- 0.2 s) stays below STOP_MPS for at
         least STOP_S
  calm   the players' mean speed over the next CALM_S drops below CALM_MPS
  whistle  most of both teams brake at once (B21, findings F82): at least WHISTLE_SHARE of each
         team's seen players are WHISTLE_DROP_MPS slower over the next WHISTLE_SPAN_S than over the
         last; the event is the frame where the whole-pitch braking peaks in each such stretch.
         For stoppages where the ball stays on the pitch (fouls, offsides), which `out` can't see.
  corner, wall, throw, goalkick, kickoff, walk
         a set-piece shape starts (src/models/setpiece.py, B24, findings F94-F95). Shapes form
         during the stoppage, so as onset cues they are late (median +0.7 to +3.5 s); in the snap
         a late cue can only start D at the shape, never before it.

Restart cues (the end of a stoppage, for snap_to_restarts, B24): each is an instant play resumes:
  gapend  the ball is seen again after a ball-missing gap of at least GAP_MIN_S
  kick    the ball starts moving after resting (stop's run ends, then KICK_MPS within KICK_S)
  go      most of both teams speed up at once: whistle's mirror (GO_DROP_MPS faster over the next
          WHISTLE_SPAN_S than the last, GO_SHARE of each team), the peak frame
  corner, wall, throw, goalkick, kickoff, walk
          a set-piece shape breaks (the first frame after one of its runs)
"""

from __future__ import annotations

import numpy as np

from src.models.deadball import MAX_SPEED_MPS, _mean_over, ball_out
from src.models.setpiece import SHAPES, shape_masks
from src.models.timing import to_frames
from src.official import D
from src.schema import Match

GAP_MIN_S = 2.0
STOP_MPS, STOP_S = 1.0, 1.0
CALM_MPS, CALM_S = 1.6, 1.0
WHISTLE_SPAN_S, WHISTLE_DROP_MPS, WHISTLE_SHARE = 1.0, 1.0, 0.6
GO_DROP_MPS, GO_SHARE = 1.0, 0.6
KICK_MPS, KICK_S = 3.0, 0.5
CUES = ("out", "gap", "stop", "calm", "whistle") + SHAPES
RESTART_CUES = ("gapend", "kick", "go") + SHAPES


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _starts(mask: np.ndarray, min_frames: int = 1) -> np.ndarray:
    """(n,) bool, True on the first frame of each True run of mask at least min_frames long."""
    out = np.zeros(len(mask), dtype=bool)
    start, stop = _runs(mask)
    out[start[(stop - start) >= max(1, min_frames)]] = True
    return out


def ball_speed(match: Match, half_s: float = 0.2) -> np.ndarray:
    """(n,) ball speed in m/s smoothed over +/- half_s; NaN where the ball is missing."""
    step = np.diff(match.ball, axis=0, prepend=match.ball[:1])
    speed = np.minimum(np.hypot(step[:, 0], step[:, 1]) * match.fps, 40.0)
    seen = ~np.isnan(speed)
    out = _mean_over(np.nan_to_num(speed), seen, to_frames(half_s, match.fps))
    return np.where(np.isnan(match.ball[:, 0]), np.nan, out)


def forward_player_speed(match: Match, span_s: float) -> np.ndarray:
    """(n,) the players' mean speed (m/s) over [t, t + span_s]. NaN where nobody is seen."""
    step = np.diff(match.players, axis=0, prepend=match.players[:1])
    with np.errstate(invalid="ignore"):
        speed = np.minimum(np.hypot(step[..., 0], step[..., 1]) * match.fps, MAX_SPEED_MPS)
    seen = np.sum(~np.isnan(speed), axis=1)
    mean = np.nansum(speed, axis=1) / np.maximum(seen, 1)
    k = to_frames(span_s, match.fps)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen > 0, mean, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen > 0)))
    index = np.arange(len(mean))
    high = np.minimum(len(mean), index + k + 1)
    n = count[high] - count[index]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[index]) / n, np.nan)


def braking(match: Match, span_s: float = WHISTLE_SPAN_S, drop_mps: float = WHISTLE_DROP_MPS,
            sign: float = 1.0):
    """((n,) share of home players braking, (n,) share of away players braking, (n,) mean braking
    in m/s over all seen players). A player brakes at t when their mean speed over [t, t + span_s]
    is at least drop_mps below that over [t - span_s, t] (displacement over the span, so
    frame-level position noise cancels, C5). NaN shares where a team has nobody seen on both sides.
    sign=-1 measures speeding up instead (the restart's `go`)."""
    from src.schema import AWAY, HOME
    k = max(1, to_frames(span_s, match.fps))
    p = match.players
    n = len(p)
    before = np.full(p.shape[:2], np.nan)
    after = np.full(p.shape[:2], np.nan)
    if n > k:
        d = np.hypot(*(p[k:] - p[:-k]).transpose(2, 0, 1)) / span_s
        before[k:] = d
        after[:-k] = d
    drop = sign * (before - after)
    with np.errstate(invalid="ignore"):
        brake = drop >= drop_mps
        seen = ~np.isnan(drop)
        shares = []
        for team in (HOME, AWAY):
            cols = match.slot_team == team
            count = seen[:, cols].sum(axis=1)
            shares.append(np.where(count > 0, brake[:, cols].sum(axis=1) / np.maximum(count, 1),
                                   np.nan))
        mean = np.nansum(np.where(seen, drop, 0.0), axis=1) / np.maximum(seen.sum(axis=1), 1)
    return shares[0], shares[1], mean


def whistle_events(match: Match, span_s: float = WHISTLE_SPAN_S,
                   drop_mps: float = WHISTLE_DROP_MPS, share: float = WHISTLE_SHARE,
                   sign: float = 1.0) -> np.ndarray:
    """(n,) bool: the peak-braking frame of each stretch where both teams brake (see braking);
    sign=-1 gives the peak frame of each stretch where both teams speed up."""
    home, away, mean = braking(match, span_s, drop_mps, sign)
    with np.errstate(invalid="ignore"):
        both = (home >= share) & (away >= share) & (match.period > 0)
    out = np.zeros(len(both), dtype=bool)
    for a, b in zip(*_runs(both)):
        out[a + int(np.argmax(mean[a:b]))] = True
    return out


def cue_starts(match: Match, cues=CUES, combine: bool = False) -> dict:
    """{cue: (n,) bool, True where that cue's event starts}, inside halves only. combine=True
    gives {"any": the union of the requested cues}."""
    fps = match.fps
    play = match.period > 0
    out = {}
    if "out" in cues:
        out["out"] = _starts(ball_out(match) & play)
    if "gap" in cues:
        out["gap"] = _starts(np.isnan(match.ball[:, 0]) & play, to_frames(GAP_MIN_S, fps))
    if "stop" in cues:
        with np.errstate(invalid="ignore"):
            rest = ball_speed(match) < STOP_MPS
        out["stop"] = _starts(rest & play, to_frames(STOP_S, fps))
    if "calm" in cues:
        with np.errstate(invalid="ignore"):
            calm = forward_player_speed(match, CALM_S) < CALM_MPS
        out["calm"] = _starts(calm & play)
    if "whistle" in cues:
        out["whistle"] = whistle_events(match)
    shapes = [c for c in cues if c in SHAPES]
    if shapes:
        for name, mask in shape_masks(match, shapes).items():
            out[name] = _starts(mask)
    if combine:
        return {"any": np.logical_or.reduce(list(out.values())) if out
                else np.zeros(match.n_frames, dtype=bool)}
    return out


def snap_onsets(codes: np.ndarray, cue: np.ndarray, runs, lookback: int, dead: int) -> np.ndarray:
    """Copy of codes where each D segment that starts inside a run (after a non-D segment) starts
    instead at the earliest cue frame in the `lookback` frames before it, if there is one, never
    before the previous segment's first frame. Only ever starts D earlier; lookback 0 changes
    nothing."""
    out = np.array(codes, copy=True)
    if lookback <= 0:
        return out
    cue_index = np.flatnonzero(cue)
    for start, stop in runs:
        window = out[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        prev_start = 0
        for b in breaks:
            if window[b] == dead and window[b - 1] != dead:
                low = max(prev_start + 1, b - lookback)
                hits = cue_index[(cue_index >= start + low) & (cue_index < start + b)]
                if len(hits):
                    window[hits[0] - start:b] = dead
            prev_start = b
    return out


def snap_to_cues(match: Match, codes: np.ndarray, runs, snap_s: float, cues: str) -> np.ndarray:
    """snap_onsets with the union of the named cues ("out+gap"), looking back snap_s seconds."""
    names = tuple(c for c in cues.split("+") if c)
    unknown = set(names) - set(CUES)
    if unknown:
        raise ValueError("unknown onset cues {} (known: {})".format(sorted(unknown), CUES))
    cue = cue_starts(match, names, combine=True)["any"]
    return snap_onsets(codes, cue, runs, to_frames(snap_s, match.fps), D)


def _ends(mask: np.ndarray, min_frames: int = 1) -> np.ndarray:
    """(n,) bool, True on the first frame after each True run of mask at least min_frames long
    (none for a run that reaches the last frame)."""
    out = np.zeros(len(mask), dtype=bool)
    start, stop = _runs(mask)
    stop = stop[((stop - start) >= max(1, min_frames)) & (stop < len(mask))]
    out[stop] = True
    return out


def restart_events(match: Match, cues=RESTART_CUES, combine: bool = False) -> dict:
    """{cue: (n,) bool, True where play visibly resumes by that cue}, inside halves only (see the
    module docstring). combine=True gives {"any": the union}."""
    fps = match.fps
    play = match.period > 0
    out = {}
    if "gapend" in cues:
        out["gapend"] = _ends(np.isnan(match.ball[:, 0]) & play, to_frames(GAP_MIN_S, fps)) & play
    if "kick" in cues:
        speed = ball_speed(match)
        with np.errstate(invalid="ignore"):
            rest_end = np.flatnonzero(_ends((speed < STOP_MPS) & play, to_frames(STOP_S, fps)))
            fast = np.flatnonzero(speed >= KICK_MPS)
        kick = np.zeros(match.n_frames, dtype=bool)
        k = to_frames(KICK_S, fps)
        nxt = np.searchsorted(fast, rest_end)
        ok = nxt < len(fast)
        hit = rest_end[ok][fast[nxt[ok]] - rest_end[ok] <= k]
        kick[hit] = True
        out["kick"] = kick & play
    if "go" in cues:
        out["go"] = whistle_events(match, WHISTLE_SPAN_S, GO_DROP_MPS, GO_SHARE, sign=-1.0)
    shapes = [c for c in cues if c in SHAPES]
    if shapes:
        for name, mask in shape_masks(match, shapes).items():
            out[name] = _ends(mask) & play
    if combine:
        return {"any": np.logical_or.reduce(list(out.values())) if out
                else np.zeros(match.n_frames, dtype=bool)}
    return out


def snap_ends(codes: np.ndarray, cue: np.ndarray, runs, lookback: int, dead: int) -> np.ndarray:
    """Copy of codes where each D segment that ends inside a run (before a non-D segment) ends
    instead at the latest cue frame in the `lookback` frames before its end, if there is one, and
    never before the D segment's second frame. Only ever ends D earlier; lookback 0 changes
    nothing. The mirror of snap_onsets."""
    out = np.array(codes, copy=True)
    if lookback <= 0:
        return out
    cue_index = np.flatnonzero(cue)
    for start, stop in runs:
        window = out[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        seg_start = 0
        for b in breaks:
            if window[b - 1] == dead and window[b] != dead:
                low = max(seg_start + 1, b - lookback)
                hits = cue_index[(cue_index >= start + low) & (cue_index < start + b)]
                if len(hits):
                    window[hits[-1] - start:b] = window[b]
            seg_start = b
    return out


def snap_to_restarts(match: Match, codes: np.ndarray, runs, snap_s: float, cues: str) -> np.ndarray:
    """snap_ends with the union of the named restart cues ("gapend+go"), looking back snap_s."""
    names = tuple(c for c in cues.split("+") if c)
    unknown = set(names) - set(RESTART_CUES)
    if unknown:
        raise ValueError("unknown restart cues {} (known: {})".format(sorted(unknown), RESTART_CUES))
    cue = restart_events(match, names, combine=True)["any"]
    return snap_ends(codes, cue, runs, to_frames(snap_s, match.fps), D)
