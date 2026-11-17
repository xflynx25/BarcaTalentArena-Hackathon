"""Cue `restartw` (wave 5, B35b): restart windows found in the tracking, as extra dead-ball evidence.

Why: v6's restart rule (B35, src/restarts.py) marks the few seconds before each restart kick as dead
ball, but only where it has windows: Smart Tagging's SET PIECES on BAR-ATH (59 in DATA.md
[official]) against ~100-140 restarts in an external match (F175). The owner's error review (14 Nov):
after the ball goes out the model reads the keeper as having it at his feet, although a goal kick
has to be placed in the goal area and kicked with every opponent outside the penalty area (Law 16).
This cue finds restarts from the tracking alone, label-free, and adds D evidence before their kick.

A restart window (detect()):
  1. rest   the ball is still (seen, speed < still_mps), missing, or jumping (faster than JUMP_MPS:
            put on the spot by hand, IDSSE's frozen dead ball) for >= still_s inside one period;
  2. kick   it then moves off: speed > kick_mps within kick_s of the rest's end, near the pitch
            (restarts.dead_stretch's kick test, with a margin so a throw-in from just outside counts);
  3. spot   where it was kicked from, by the ball's last seen position when the ball was seen at the
            end of the rest ("ball" route), else by the players' set-piece shapes in the 2 s before
            the kick (src/models/setpiece.py, the "shape" route, with the first seen ball after the
            kick near that spot):
              goalkick  the ball in a goal area, every player of the attacking team outside that
                        penalty area, a defender within reach of the ball (Law 16)
              corner    the ball at a corner arc
              throw     the ball on or beyond a touchline
              kickoff   the ball on the centre spot, both teams in their own halves
              penalty   the ball on a penalty spot, at most 2 players in that box
              freekick  anywhere, with a wall (setpiece's wall shape, F94)
  4. out    the ball was seen off the pitch, or last seen within out_m of a line before a gap, in
            the out_s before the rest began or during it. An out moves the window's start back to it.
A window is [max(start, kick - back_s), kick) with weight spot (1 or 0) + out_w * out. A window
with no spot is kept only when the ball was seen off the pitch (not merely lost near a line), and a
window whose spot wasn't read on a ball seen over most of the rest also needs calm players (the
median player under calm_mps on at least calm_min of the window's frames). The cue adds gain x weight / fps to D's score on the window's frames (per
frame the largest weight), like restarts.dead_stretch's gain x 1 / fps.

Union with v6's restart rule: where the rule (restart_rule_gain > 0) already marks a frame dead from
its own windows (Smart Tagging on BAR-ATH, the vendor's restart events on external matches), the cue
adds nothing there, so no frame is counted twice; with the rule off or without tags, the cue's
windows act alone. The cue reads the ball and players only, never the labels or tags (the rule's
frames come from restarts.dead_stretch with the decoder's own settings).

    --param "cues=restartw:2"                 gain 2, default options
    --param "cues=restartw:2/out_w=1/back_s=6"

Options (all hand-set before any grid run, B35b): still_s, still_mps, kick_mps, kick_s, back_s,
out_s, out_m, out_w, margin_m. Deterministic: numpy on the match's arrays, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import restarts
from src.models import setpiece
from src.models.cues import Effect
from src.schema import AWAY, HOME, Match

KINDS = ("goalkick", "corner", "throw", "kickoff", "penalty", "freekick")
DEFAULTS = {
    "still_s": 1.0,  # the ball rests (still or missing) at least this long
    "still_mps": 1.0,  # "still": ball speed over 0.2 s below this
    "kick_mps": 4.0,  # the kick: ball speed above this (restarts.dead_stretch's test)
    "kick_s": 0.4,  # the kick must come within this of the rest's end
    "back_s": 8.0,  # a window covers at most this much before the kick
    "out_s": 3.0,  # look this far before the rest for the ball going out
    "out_m": 2.0,  # "last seen near a line before a gap": within this of a line
    "out_w": 0.5,  # weight added by an out
    "margin_m": 2.0,  # the kick may come from this far outside the pitch (throw-ins)
    "calm_min": 1 / 3,  # windows without a spot seen on the ball need calm players on this share
    "calm_mps": 1.5,  # "calm": the median seen player's speed (over 1 s) below this
}
JUMP_MPS = 40.0  # faster than any kick over 0.2 s: the ball was moved, not kicked (IDSSE, F313)
CALM_SEEN = 6  # a frame counts for calm only with at least this many players seen
SPOT_TAIL_S = 0.5  # the ball route: the ball seen still in the last 0.5 s of the rest
SHAPE_S = 2.0  # the shape route: a shape in the 2 s before the kick
GOAL_AREA_MARGIN_M = 1.5  # tracking noise around the goal area's lines
CORNER_M = 2.5
TOUCH_M = 1.5
CENTRE_M = 2.0
PENALTY_M = 1.5
PENALTY_SPOT_M = 11.0
KICKOFF_SHARE = 0.85
DEFENDER_REACH_M = 4.0
BOX_SHRINK_M = 0.5  # an attacker counts as inside the box only this far inside its lines
NEAR_SPOT_M = {"goalkick": 8.0, "corner": 12.0, "throw": 8.0, "kickoff": 10.0, "penalty": 8.0}


@dataclass(frozen=True)
class Window:
    start: int  # first frame the window marks (out moment or rest start, capped at back_s)
    kick: int  # the kick frame (the window ends just before it)
    rest: int  # where the rest began
    kind: str  # one of KINDS, or "" (no spot: kept only because of an out)
    route: str  # "ball" (spot from the ball) or "shape" (from the players), "" with no spot
    out: bool
    weight: float


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _near_line(b: np.ndarray, length: float, width: float, m: float) -> bool:
    x, y = b
    return bool(x <= m or x >= length - m or y <= m or y >= width - m)


def _defends_low(sign: float, team: int) -> bool:
    """Whether `team` defends the goal at x = 0 (setpiece.attack_sign's convention)."""
    return (sign > 0) == (team == HOME)


def _ball_kind(match: Match, b: np.ndarray, kick: int, sign: float, fps: int) -> str:
    length, width = match.pitch
    x, y = float(b[0]), float(b[1])
    p = match.players[kick]
    tail = match.players[max(0, kick - int(round(SPOT_TAIL_S * fps))):kick + 1]
    for cx in (0.0, length):
        for cy in (0.0, width):
            if np.hypot(x - cx, y - cy) <= CORNER_M:
                return "corner"
    for end, gx in ((0, 0.0), (1, length)):
        depth = abs(x - gx)
        if depth <= setpiece.GOAL_AREA_DEPTH_M + GOAL_AREA_MARGIN_M and \
                abs(y - width / 2) <= setpiece.GOAL_AREA_HALF_W_M + GOAL_AREA_MARGIN_M:
            # the team defending this end takes it: nobody of the other team in the box, one of
            # its own players within reach of the ball
            defender = HOME if _defends_low(sign, HOME) == (end == 0) else AWAY
            mine = match.slot_team == defender
            theirs = ~mine
            with np.errstate(invalid="ignore"):
                in_box = setpiece._in_box(tail[..., 0], tail[..., 1], length, width, end,
                                          setpiece.BOX_DEPTH_M - BOX_SHRINK_M,
                                          setpiece.BOX_HALF_W_M - BOX_SHRINK_M)
                reach = np.hypot(p[mine, 0] - x, p[mine, 1] - y) <= DEFENDER_REACH_M
            if not in_box[:, theirs].any() and reach.any():
                return "goalkick"
            return ""
        if abs(depth - PENALTY_SPOT_M) <= PENALTY_M and abs(y - width / 2) <= PENALTY_M:
            with np.errstate(invalid="ignore"):
                box = setpiece._in_box(p[:, 0], p[:, 1], length, width, end,
                                       setpiece.BOX_DEPTH_M, setpiece.BOX_HALF_W_M)
            if int(box.sum()) <= 2:
                return "penalty"
    if (y <= TOUCH_M or y >= width - TOUCH_M) and CORNER_M < x < length - CORNER_M:
        return "throw"
    if np.hypot(x - length / 2, y - width / 2) <= CENTRE_M:
        home = match.slot_team == HOME
        px = tail[..., 0]
        with np.errstate(invalid="ignore"):
            low, high = px <= length / 2 + 1.0, px >= length / 2 - 1.0
            own = np.where(home[None, :], low if sign > 0 else high, high if sign > 0 else low)
        seen = ~np.isnan(px)
        if seen.sum() and (own & seen).sum() / seen.sum() >= KICKOFF_SHARE:
            return "kickoff"
    return ""


class _Shapes:
    """Set-piece shapes (players only) of one match, the cheap ones whole, the wall on demand."""

    def __init__(self, match: Match, speed: np.ndarray):
        self.match = match
        self.speed = speed
        sign = setpiece.attack_sign(match)
        self.masks = {"corner": setpiece.corner_mask(match), "throw": setpiece.throw_mask(match, self.speed),
                      "goalkick": setpiece.goalkick_mask(match, sign),
                      "kickoff": setpiece.kickoff_mask(match, sign)}

    def wall(self, lo: int, hi: int) -> bool:
        if hi <= lo:
            return False
        return bool(setpiece._wall(self.match.players[lo:hi], self.speed[lo:hi], self.match.slot_team).any())

    def kind(self, lo: int, hi: int) -> str:
        for name in ("corner", "goalkick", "kickoff", "throw"):
            if self.masks[name][lo:hi].any():
                return name
        return ""


def _near_spot(match: Match, kind: str, b: np.ndarray) -> bool:
    """The first seen ball after a hidden rest is consistent with the shape's spot."""
    if kind == "freekick" or not np.all(np.isfinite(b)):
        return True
    length, width = match.pitch
    x, y = float(b[0]), float(b[1])
    r = NEAR_SPOT_M[kind]
    if kind == "corner":
        return min(np.hypot(x - cx, y - cy) for cx in (0.0, length) for cy in (0.0, width)) <= r
    if kind == "throw":
        return y <= r or y >= width - r
    if kind == "kickoff":
        return np.hypot(x - length / 2, y - width / 2) <= r
    # goalkick, penalty: within the box plus r of a goal line
    return (x <= setpiece.BOX_DEPTH_M + r or x >= length - setpiece.BOX_DEPTH_M - r) and \
        abs(y - width / 2) <= setpiece.BOX_HALF_W_M + r


def detect(match: Match, **opts) -> list[Window]:
    """The restart windows found in this match's tracking (module docstring), in time order."""
    o = dict(DEFAULTS)
    unknown = set(opts) - set(o)
    if unknown:
        raise ValueError("restartw: unknown options {}".format(sorted(unknown)))
    o.update({k: float(v) for k, v in opts.items()})
    fps, n = match.fps, match.n_frames
    if n < 2:
        return []
    length, width = match.pitch
    ball = match.ball
    seen = np.isfinite(ball).all(axis=1)
    v = restarts.ball_speed(match)
    with np.errstate(invalid="ignore"):
        still = seen & ~(v >= o["still_mps"])  # seen with no measurable speed counts as resting
        m = o["margin_m"]
        near = (ball[:, 0] > -m) & (ball[:, 0] < length + m) & (ball[:, 1] > -m) & (ball[:, 1] < width + m)
        off = ((ball[:, 0] < -0.5) | (ball[:, 0] > length + 0.5)
               | (ball[:, 1] < -0.5) | (ball[:, 1] > width + 0.5))
        jump = seen & (v > JUMP_MPS)  # the ball repositioned (a frozen dead ball moved to the spot)
        kicked = near & (v > o["kick_mps"]) & ~jump
    rest = (still | ~seen | jump) & (match.period > 0)
    change = np.flatnonzero(np.diff(match.period) != 0) + 1
    first = np.zeros(n, dtype=np.int64)  # the first frame of each frame's period run
    first[change] = change
    first = np.maximum.accumulate(first)
    sign = setpiece.attack_sign(match)
    shapes = None
    speed = setpiece.player_speeds(match)
    with np.errstate(invalid="ignore"):
        n_seen = np.isfinite(speed).sum(axis=1)
        med = np.nanmedian(np.where(n_seen[:, None] > 0, speed, 0.0), axis=1)
        calm = (n_seen >= CALM_SEEN) & (med < o["calm_mps"])
    min_rest, kick_k = int(round(o["still_s"] * fps)), int(round(o["kick_s"] * fps))
    back, out_k = int(round(o["back_s"] * fps)), int(round(o["out_s"] * fps))
    tail_k, shape_k = int(round(SPOT_TAIL_S * fps)), int(round(SHAPE_S * fps))
    out: list[Window] = []
    for a, b in zip(*_runs(rest)):
        a, b = int(a), int(b)
        if b - a < min_rest or b >= n:
            continue
        period = match.period[a]
        if np.any(match.period[a:b] != period):
            continue
        hits = np.flatnonzero(kicked[b:min(n, b + kick_k + 1)] & (match.period[b:min(n, b + kick_k + 1)] == period))
        if not len(hits):
            continue
        kick = b + int(hits[0])
        # spot: the ball's last still position when seen at the end of the rest, else the shapes
        # (the last seen frame, not the last still one: a ball moved onto the spot is there)
        tail = np.flatnonzero(seen[max(a, b - tail_k):b])
        kind, route = "", ""
        if len(tail):
            pos = ball[max(a, b - tail_k) + int(tail[-1])]
            kind, route = _ball_kind(match, pos, kick, sign[kick], fps), "ball"
        else:
            if shapes is None:
                shapes = _Shapes(match, speed)
            lo = max(a, kick - shape_k)
            kind = shapes.kind(lo, kick)
            if kind and not _near_spot(match, kind, ball[kick]):
                kind = ""
            route = "shape" if kind else ""
        if not kind:
            if shapes is None:
                shapes = _Shapes(match, speed)
            if shapes.wall(max(a, kick - shape_k), kick):
                kind, route = "freekick", route or ("ball" if len(tail) else "shape")
        if not kind:
            route = ""
        # out: seen off the pitch, or last seen near a line before a gap, before or during the rest
        lo = max(int(first[a]), a - out_k)
        out_at = -1
        offs = np.flatnonzero(off[lo:b])
        if len(offs):
            out_at = lo + int(offs[0])
        gaps = np.flatnonzero(seen[lo:b - 1] & ~seen[lo + 1:b])  # last seen frame before a gap
        for g in gaps:
            f = lo + int(g)
            if _near_line(ball[f], length, width, o["out_m"]):
                out_at = f + 1 if out_at < 0 else min(out_at, f + 1)
                break
        is_out = out_at >= 0
        # with no spot, only a ball seen off the pitch makes a window: a ball merely lost near a
        # line is too common in live play once the ball is hidden at random (F313)
        if not kind and not len(offs):
            continue
        weight = (1.0 if kind else 0.0) + (o["out_w"] if is_out else 0.0)
        start = max(min(a, out_at) if is_out else a, kick - back)
        # a spot read on a ball seen over most of the rest stands alone; anything else (an out
        # alone, a spot from the players' shapes, a mostly hidden ball) also needs calm players (broadcast tracking loses the ball in live play, F315)
        alone = route == "ball" and seen[a:kick].mean() >= 0.5
        if not alone and calm[start:kick].mean() < o["calm_min"]:
            continue
        out.append(Window(start, kick, a, kind, route, is_out, weight))
    return out


_CACHE: list = [None, None, None]  # [match, opts key, windows]: candidates of one grid task share it


def windows(match: Match, **opts) -> list[Window]:
    key = tuple(sorted(opts.items()))
    if _CACHE[0] is not match or _CACHE[1] != key:
        _CACHE[:] = [match, key, detect(match, **opts)]
    return _CACHE[2]


def weights(match: Match, wins: list[Window]) -> np.ndarray:
    """(n,) per frame the largest weight of a window covering it (0 elsewhere)."""
    w = np.zeros(match.n_frames)
    for win in wins:
        seg = w[win.start:win.kick]
        np.maximum(seg, win.weight, out=seg)
    return w


def effect(match: Match, original: Match, decoder, gain: float, **opts) -> Effect | None:
    if not gain:
        return None
    w = weights(match, windows(match, **opts))
    if getattr(decoder, "restart_rule_gain", 0.0):
        # union with v6's restart rule: no frame counted twice
        w[restarts.dead_stretch(match, decoder.restart_rule_back_s, decoder.restart_rule_kick_mps,
                                decoder.restart_rule_search_s, decoder.restart_rule_noise)] = 0.0
    if not w.any():
        return None
    add = np.zeros((match.n_frames, 3))
    add[:, 2] = float(gain) * w / match.fps
    return Effect(add=add)
