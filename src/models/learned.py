"""Learned emissions for the switch decoder (B10, with B5 folded in; touch features B4).

Same Viterbi over {H, A, D} and the same switch costs as `viterbi` (src/models/decoder.py), but the
per-frame evidence is a calibrated log-probability from a small model fit on external matches,
instead of B3's hand-set reach and dead scores.

Features, per frame, label-free, tracking only (FEATURES below; durations in seconds, so one fit
serves every frame rate):
  per team ("own" side, mirrored for the other team)
    distance from the ball to that team's closest player, now, its minimum over the last
    WINDOW_S and over the next WINDOW_S (the receiver is about to reach it, findings F44), and the
    margin to the other team's closest player; each as a piecewise-linear (hat) basis. All zero
    when the ball is missing.
    touches (group "touch", B4): a touch is a jump in the ball's velocity (its change across the
    frame over TOUCH_STEP_S, soft between TOUCH_MPS) credited to the team whose closest player is
    nearer and within TOUCH_REACH_M. Per team: touch strength now, its maximum over the last
    WINDOW_S, the last LONG_WINDOW_S and the next WINDOW_S, the time since that team's last touch
    and how much more recently than the other team it touched (the last-touch rule).
    noise-scaled touches (group "ntouch", B14, findings F68): the same columns, but on a jittery
    (broadcast-smoothed) ball the detector widens per match, label-free: step x s and thresholds
    / s with s = max(1, ball_jitter / NTOUCH_REF_M) ** NTOUCH_EXP. s = 1 on a clean ball, so there
    it equals "touch". Use it instead of "touch" (fit_learned.py --groups ntouch).
    controlled touches (group "ctouch", B18, findings F83): the "touch" columns again, but a touch
    only counts if the touching team then stays nearest the ball on at least CONTROL_KEEP of the
    seen frames over the next CONTROL_S (it kept the ball): label-free, for coarse labels that
    don't count short contacts as possession (F53, F69).
    team shape (group "shape", B25, findings F96-F99; "shapem": the same columns, zero on frames
    where the ball is seen, so shape decides only while the ball is missing; B26, F156-F157:
    "shapeg" / "shapes" are "shapem" twice, x (1 - w) and x w, with w = the gap is long
    (SHAPE_GAP_S) / the ball was seen off the pitch within SHAPE_STOP_S of the gap): ball-free, from src/shape.py at
    SHAPE_RATE_HZ. Per feature of SHAPE["features"], the home-minus-away difference averaged over
    each window (seconds relative to the frame; offline, so windows may look ahead: shape lags a
    switch by ~3.5 s, F89), as a hat basis on symmetric knots; the home block gets hat(d), the away
    block hat(-d), so swapping the teams swaps the blocks. SHAPE["zone"] features are repeated x
    "play is in the middle third" (F90: shape is weakest near the goals). NaN (too few players,
    direction unknown) gives all-zero columns: no evidence.
    Smart Tagging (group "tags", B28, findings F118-F120; off by default): src/smart_tags.columns,
    one column per tag family carrying that team (attack, transition, long ball, defend, defensive
    transition) and its SET PIECES windows as a hat on the position inside the window; common: the
    same hat for N/A set pieces (throw-ins) and "inside any tag". The tags come from Match.tags
    (BAR-ATH's file) or, decode-time, LearnedDecoder.tags_source (stand-ins on external matches).
    hidden ball (group "impute", B29, findings F121-F123; zero where the ball is seen): the ball
    estimated by src/ball_impute.py on ball-missing frames. Per team, the distance from the
    estimate to that team's closest player and the margin to the other team's (hat bases), and
    the margin again x "long gap" (>= IMPUTE_LONG_S, where the estimate is rough); common: the
    estimate is off the pitch. The imputer is cross-fitted: an external match gets the imputer fit
    without it (fold "loso": without its source), BAR-ATH the one fit on every match.
    video ball (group "vball", V2, findings F136-F138; off by default; zero where the tracked ball is
    seen): the ball the video finds on ball-missing frames (Match.video_ball: src.video.run on a real
    video, src/video/simulate.py on external matches). Per team, hat bases of the distance from
    the video ball to that team's closest player and of the margin to the other team's; common: the
    video sees the ball on a missing frame, and that ball is off the pitch.
    broadcast ball (group "bball", V4, findings F160-F161; zero where the ball is seen or no second
    recording maps): Match.ball2, the second recording's ball on this timeline (src/align.py), on
    ball-missing frames: per team the distance from it to that team's closest player and the margin
    to the other team's (hat bases); common: it is there, and it is off the pitch. "bfill" is not
    columns: the features (and the decoder's cues) read the ball with its gaps filled from ball2.
    gap ends (group "gapends", B29; zero where the ball is seen): per team, "this team was nearest
    the ball when it was last seen" x a hat basis on the seconds since, and the same for when it
    is next seen (offline). The carrier at a gap's start and the receiver at its end.
    blind players (group "blind", C13, findings F155; off by default): how far the team's ball
    features can be trusted when players are missing. Per team: its player deficit (11 minus the
    slots seen now, as a hat basis), its "ghosts" near the ball (players lost in the last
    BLIND_GHOST_S, carried with their team's drift, within BLIND_GHOST_M of the ball; hat basis),
    and the margin hat x "a ghost is near" (so a missing nearest player can be discounted). Ball
    columns are zero when the ball is missing. Pairs with the track repair (track_repair=1).
    duels (group "duel", B4): how contested the ball is (the farther of the two teams' closest
    players, as a hat basis), and contested x the last-touch lead (who keeps a contested ball).
    gated shape (groups "shapef" / "shapeo", local L1, findings F200+): the "shape" columns on frames
    where the ball is seen but nobody is within gates.FREE_M ("shapeo"), and also moving at
    gates.FLIGHT_MPS or more ("shapef"); together with "shapem", shape gets separate weights per
    situation. Zero on missing and on controlled frames.
    gap evidence (group "deadm", local L3, common columns): the players' mean speed over +/- 2 s and
    +/- 0.5 s (displacement over gates.DEAD_SPAN_S) and the set-piece shapes gates.DEAD_SHAPES
    (src/models/setpiece.py), zero where the ball is seen: live vs dead inside ball gaps.
  common
    bias, ball missing now, the share of ball-missing frames over +/- DEAD_WINDOW_S, the players'
    mean speed over +/- DEAD_WINDOW_S and +/- 0.5 s (findings F29), the ball off the pitch now and
    for OUT_S after (B2's cues, src/models/deadball.py).

Model: a conditional logit with the home/away symmetry built in (no home advantage can be learned):
  z_H = a.T_home + b.T_away + c.C,   z_A = a.T_away + b.T_home + c.C,   z_D = e.(T_home + T_away)
fit by Newton's method with a small L2 penalty, on frames of every half (labels H/A/D) sampled every
FIT_STRIDE_S, weighted so every source counts the same and every match the same within its source
(findings F40: select on the mean over sources). Plain numpy: no new dependency, no randomness.

Decoding as in `viterbi`, including the optional onset snap (onset_snap_s, B12, findings F78).

Emission score of state k at a frame: log p(k | x) - log prior(k) (prior=1, a scaled likelihood, H
and A sharing their mean prior) or
log p(k | x) (prior=0), divided by fps so the costs are in seconds of evidence at every frame rate.

Regenerable: `scripts/fit_learned.py` fits on the external matches, each match under 5 settings drawn
from the ball-dropout grid (src/dropout.py, C6, findings F66; before C6: F30's single setting), and writes the coefficients to learned_coefs.json next to this file: one fit on all
matches ("all", what src.run uses), one per held-out match (leave-one-match-out across sources) and
one per held-out source. fold="lomo" / "loso" scores each external match with the fit that never saw
it (or its source), so `src.evaluate --model learned --param fold=lomo` gives held-out scores.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import zlib
from dataclasses import dataclass

import numpy as np

from src import restarts, smart_tags
from src.models import deadball, gates
from src.models.deadball import ball_out, extend, player_speed
from src.models.decoder import (B3_SETTINGS, NOISE_MEASURES, SwitchDecoder, ball_jitter,
                                noise_level, shift, team_distances, viterbi, window_mean)
from src.models.nearest import majority_smooth, nearest_team
from src.models.onset import snap_to_cues, snap_to_restarts
from src.models.postprocess import drop_short, play_runs
from src.models.timing import to_frames
from src.official import A, D, H
from src.schema import AWAY, HOME, Match

COEF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_coefs.json")

DIST_KNOTS = (0.0, 0.75, 1.5, 2.5, 4.0, 7.0, 12.0)  # metres
MARGIN_KNOTS = (-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0)  # other team's closest minus own, metres
SHARE_KNOTS = (0.0, 0.5, 1.0)
SPEED_KNOTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.5)  # m/s
WINDOW_S = 0.5  # past / future reach window
LONG_WINDOW_S = 2.0  # past reach window, for long passes
CONTROL_KNOTS = (0.0, 0.75, 1.5, 2.5)  # metres, own distance while the ball is slow
SLOW_MPS = (2.0, 6.0)  # ball speed: fully slow below the first, not slow above the second
NEAR_M = 3.0  # players of a team this close to the ball are counted
COUNT_KNOTS = (0.0, 1.0, 2.0, 3.0)
DEAD_WINDOW_S = 2.0
OUT_S = 2.0
TOUCH_STEP_S = 0.2  # ball velocity over this span before and after the frame
TOUCH_MPS = (2.0, 6.0)  # velocity change: no touch below the first, a full touch above the second
TOUCH_REACH_M = 2.5  # the touching team's closest player is at most this far from the ball
TOUCH_MIN = 0.5  # touch strength that counts as a touch for the time since the last one
SINCE_KNOTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # seconds since the team's last touch
LEAD_KNOTS = (-4.0, -1.0, 0.0, 1.0, 4.0)  # other team's time since its last touch minus own
DUEL_KNOTS = (0.0, 1.0, 2.0, 3.0, 5.0)  # metres, the farther of the two teams' closest players
DUEL_M = (1.5, 3.0)  # contested: fully below the first, not above the second
GAP_KNOTS = (0.0, 1.0, 3.0, 8.0, 20.0)  # seconds, length of the ball-missing gap the frame is in
GAP_SHARE_WINDOW_S = 30.0  # ball-missing share over +/- this: how much this stretch drops the ball
SPEED_SPAN_S = 1.0  # group "speedspan": player speed from displacement across this span (C5)
NTOUCH_REF_M = 0.02  # group "ntouch" (B14): ball_jitter above this widens the touch detector ...
NTOUCH_EXP = 2.0  # ... by s = max(1, jitter / NTOUCH_REF_M) ** this: step x s, thresholds / s
CONTROL_S = 2.0  # group "ctouch" (B18): a touch counts if over the next CONTROL_S ...
CONTROL_KEEP = 0.7  # ... the touching team is nearest the ball on at least this share of seen frames
IMPUTE_LONG_S = (2.0, 6.0)  # group "impute" (B29): a gap is "long" from 2 s (soft) to 6 s (full)
GAPEND_KNOTS = (0.0, 1.0, 3.0, 8.0)  # group "gapends" (B29): seconds since / until the seen ball
BLIND_FULL = 11  # group "blind" (C13): players per team on the pitch
BLIND_DEFICIT_KNOTS = (0.0, 1.0, 2.0, 4.0)  # missing players of a team
BLIND_GHOST_S = 5.0  # a lost player is a ghost this long after its track ends ...
BLIND_GHOST_M = 5.0  # ... and counts when its drifted position is this close to the ball
BLIND_GHOST_KNOTS = (0.0, 1.0, 2.0)
SHAPE_GAP_S = (1.0, 4.0)  # group "shapeg" (B26): a gap is "long" from 1 s (soft) to 4 s (full)
SHAPE_STOP_S = 3.0  # group "shapes" (B26): ball seen off the pitch this close to the gap's ends
SHAPE_RATE_HZ = 5.0  # group "shape" (B25): features sampled at this rate, held to every frame
# name: (symmetric knots for home - away, windows as (from, to) seconds around the frame). The
# (0, t) windows look ahead: B25 step 1 (F96) found +0.03 held-out accuracy on IDSSE from them.
SHAPE = {
    "features": {
        "fb_w": ((-25.0, -15.0, -7.0, 0.0, 7.0, 15.0, 25.0),
                 ((-2.0, 2.0), (-0.5, 0.5), (0.0, 4.0))),
        "goalside": ((-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4), ((-2.0, 2.0), (0.0, 4.0))),
        "press": ((-0.3, -0.15, 0.0, 0.15, 0.3), ((-2.0, 2.0),)),
        "stretch": ((-12.0, -6.0, 0.0, 6.0, 12.0), ((-2.0, 2.0),)),
        "work": ((-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0), ((-0.5, 0.5), (0.0, 1.0))),
        "vx": ((-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0), ((-0.5, 0.5), (0.0, 1.0))),
    },
    "zone": (),  # features whose first window is repeated x middle-third weight
}
def _imputer_id() -> str:
    """The imputer's spec and coefficients (src/ball_impute.py), so "impute" coefficient files go
    stale when the imputer is refit."""
    from src import ball_impute

    path = ball_impute.COEF_FILE
    return ball_impute.SPEC_HASH + "/" + (ball_impute.coef_digest(path) if os.path.isfile(path)
                                          else "missing")


GROUPS = ("shapem", "touch")  # optional groups in the committed coefficients (B4: "duel" didn't help;
# B25: "shapem" since v4)
FEATURES = dict(dist=DIST_KNOTS, margin=MARGIN_KNOTS, share=SHARE_KNOTS, speed=SPEED_KNOTS,
                window_s=WINDOW_S, dead_window_s=DEAD_WINDOW_S, out_s=OUT_S,
                long_window_s=LONG_WINDOW_S, control=CONTROL_KNOTS, slow=SLOW_MPS, near_m=NEAR_M,
                count=COUNT_KNOTS, version=2)
GROUP_FEATURES = {
    "touch": dict(step_s=TOUCH_STEP_S, mps=TOUCH_MPS, reach_m=TOUCH_REACH_M, min=TOUCH_MIN,
                  since=SINCE_KNOTS, lead=LEAD_KNOTS),
    "shape": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, version=1),
    # the same columns, zero where the ball is seen: shape only speaks when the ball is missing
    "shapem": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, gate="missing", version=1),
    # B26: "shapem" split in two by a per-frame weight w in [0, 1] (columns x (1 - w), x w), so
    # shape gets its own weight in short vs long gaps ("shapeg") or away from vs near a stoppage
    # ("shapes"). Use one instead of "shapem".
    "shapeg": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, gate="missing", long_s=SHAPE_GAP_S,
                   version=1),
    "shapes": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, gate="missing", stop_s=SHAPE_STOP_S,
                   out_margin_m=deadball.OUT_MARGIN_M, version=1),
    # local L1 (F200+): the shape columns on frames where the ball is seen but free (nobody within
    # FREE_M); "shapef" also needs it fast (in flight). Not on missing frames: "shapem" has those.
    "shapef": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, gate="flight", flight_mps=gates.FLIGHT_MPS,
                   free_m=gates.FREE_M, version=1),
    "shapeo": dict(rate_hz=SHAPE_RATE_HZ, spec=SHAPE, gate="free", free_m=gates.FREE_M, version=1),
    # local L3 (F200+): player motion and set-piece shapes, common columns, only on missing frames
    "deadm": dict(speed=SPEED_KNOTS, half_s=list(gates.DEAD_HALF_S), span_s=gates.DEAD_SPAN_S,
                  shapes=list(gates.DEAD_SHAPES), gate="missing", version=1),
    "duel": dict(knots=DUEL_KNOTS, contested_m=DUEL_M, lead=LEAD_KNOTS, touch=TOUCH_MIN),
    "impute": dict(dist=DIST_KNOTS, margin=MARGIN_KNOTS, long_s=IMPUTE_LONG_S,
                   imputer=_imputer_id(), version=1),
    "gapends": dict(knots=GAPEND_KNOTS, version=1),
    "blind": dict(full=BLIND_FULL, deficit=BLIND_DEFICIT_KNOTS, ghost_s=BLIND_GHOST_S,
                  ghost_m=BLIND_GHOST_M, ghost=BLIND_GHOST_KNOTS, margin=MARGIN_KNOTS, version=1),
    "vball": dict(dist=DIST_KNOTS, margin=MARGIN_KNOTS, version=1),
    "bball": dict(dist=DIST_KNOTS, margin=MARGIN_KNOTS, version=1),
    # not columns: the ball's gaps filled from Match.ball2 before every feature (V4)
    "bfill": dict(version=1),
    "gap": dict(knots=GAP_KNOTS, share=SHARE_KNOTS, window_s=GAP_SHARE_WINDOW_S),
    # not extra columns: the two player-speed features measured over a span (C5, findings F72)
    "speedspan": dict(span_s=SPEED_SPAN_S),
    "ctouch": dict(step_s=TOUCH_STEP_S, mps=TOUCH_MPS, reach_m=TOUCH_REACH_M, min=TOUCH_MIN,
                   since=SINCE_KNOTS, lead=LEAD_KNOTS, control_s=CONTROL_S, keep=CONTROL_KEEP),
    "tags": dict(families=[[name, list(codes)] for name, codes in smart_tags.FAMILIES],
                 set_knots=list(smart_tags.SET_KNOTS), version=1),
    "ntouch": dict(step_s=TOUCH_STEP_S, mps=TOUCH_MPS, reach_m=TOUCH_REACH_M, min=TOUCH_MIN,
                   since=SINCE_KNOTS, lead=LEAD_KNOTS, noise_ref_m=NTOUCH_REF_M,
                   noise_exp=NTOUCH_EXP),
}


def feature_hash(groups=GROUPS) -> str:
    """Hash of the feature definition for these groups (the base features alone hash as in B10)."""
    spec = dict(FEATURES, **{"group_" + g: GROUP_FEATURES[g] for g in sorted(groups)})
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


FEATURE_HASH = feature_hash()

FIT_STRIDE_S = 0.5  # one training row every this many seconds
L2 = 1e-4  # penalty on the mean weighted log-likelihood
STATES = np.array([H, A, D], dtype=np.int8)


# -- features ---------------------------------------------------------------------------------------

def hat(values: np.ndarray, knots) -> np.ndarray:
    """(n, len(knots)) piecewise-linear basis: each value split between its two nearest knots
    (clamped to the ends). Rows of NaN values are all zero."""
    knots = np.asarray(knots, dtype=float)
    out = np.zeros((len(values), len(knots)))
    seen = ~np.isnan(values)
    v = np.clip(values[seen], knots[0], knots[-1])
    low = np.clip(np.searchsorted(knots, v, side="right") - 1, 0, len(knots) - 2)
    frac = (v - knots[low]) / (knots[low + 1] - knots[low])
    rows = np.flatnonzero(seen)
    out[rows, low] = 1.0 - frac
    out[rows, low + 1] = frac
    return out


def window_min(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """Minimum of values over [t - before, t + after], ignoring NaN. NaN where all are NaN."""
    filled = np.where(np.isnan(values), np.inf, values)
    padded = np.concatenate((np.full(before, np.inf), filled, np.full(after, np.inf)))
    best = np.lib.stride_tricks.sliding_window_view(padded, before + after + 1).min(axis=1)
    return np.where(np.isfinite(best), best, np.nan)


def ball_slow(match: Match) -> np.ndarray:
    """(n,) in [0, 1]: 1 while the ball moves slower than SLOW_MPS[0] (smoothed over +/- 0.2 s),
    0 above SLOW_MPS[1] or when the ball is missing. A ball in flight is fast."""
    step = np.diff(match.ball, axis=0, prepend=match.ball[:1])
    speed = np.hypot(step[:, 0], step[:, 1]) * match.fps
    seen = ~np.isnan(speed)
    speed = window_mean(np.nan_to_num(speed), seen, to_frames(0.2, match.fps))
    low, high = SLOW_MPS
    return np.where(np.isnan(match.ball[:, 0]), 0.0,
                    np.clip((high - np.nan_to_num(speed, nan=high)) / (high - low), 0.0, 1.0))


def near_counts(match: Match) -> tuple[np.ndarray, np.ndarray]:
    """(n,) number of home and away players within NEAR_M of the ball; NaN when the ball is missing."""
    dist = np.hypot(match.players[:, :, 0] - match.ball[:, None, 0],
                    match.players[:, :, 1] - match.ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        near = dist < NEAR_M
    missing = np.isnan(match.ball[:, 0])
    out = []
    for team in (HOME, AWAY):
        count = near[:, match.slot_team == team].sum(axis=1).astype(float)
        out.append(np.where(missing, np.nan, count))
    return out[0], out[1]


def ball_impulse(match: Match, step_s: float = TOUCH_STEP_S) -> np.ndarray:
    """(n,) m/s: how much the ball's velocity changes across each frame (velocity over the
    step_s after it minus over the step_s before it). NaN near a missing ball."""
    m = max(1, to_frames(step_s, match.fps))
    ball = match.ball
    out = np.full(len(ball), np.nan)
    if len(ball) > 2 * m:
        jump = (ball[2 * m:] - 2 * ball[m:-m] + ball[:-2 * m]) * match.fps / m
        out[m:-m] = np.hypot(jump[:, 0], jump[:, 1])
    return out


def gap_lengths(missing: np.ndarray, fps: int) -> np.ndarray:
    """(n,) length in seconds of the ball-missing gap each frame is in (whole gap, past and future:
    offline, label-free); 0 where the ball is seen."""
    out = np.zeros(len(missing))
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = (stop - start) / fps
    return out


def gap_long(missing: np.ndarray, fps: int, long_s=SHAPE_GAP_S) -> np.ndarray:
    """(n,) in [0, 1]: how "long" the ball-missing gap each frame is in (0 below long_s[0] seconds,
    1 above long_s[1], linear between); 0 where the ball is seen."""
    lo, hi = long_s
    return np.clip((gap_lengths(missing, fps) - lo) / (hi - lo), 0.0, 1.0) * missing


def gap_near_stop(match: Match, missing: np.ndarray, stop_s: float = SHAPE_STOP_S) -> np.ndarray:
    """(n,) float 0/1: the frame is in a ball-missing gap whose ball was seen off the pitch within
    stop_s seconds before the gap starts or after it ends, in the same period (B26: a gap at a
    stoppage, where shape is near chance, F96). Offline, label-free."""
    out = ball_out(match)
    near = np.zeros(len(missing))
    reach = to_frames(stop_s, match.fps)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lo, hi = max(0, start - reach), min(len(missing), stop + reach)
        same = match.period[lo:hi] == match.period[start]
        if (out[lo:hi] & same).any():
            near[start:stop] = 1.0
    return near


def gap_block(missing: np.ndarray, period: np.ndarray, fps: int) -> np.ndarray:
    """Common columns for group "gap" (C6): how far to trust "ball missing -> D". The length of
    the current gap (a stoppage loses the ball for long, a duel for a moment), and "ball missing
    now" split by how much the ball goes missing over +/- GAP_SHARE_WINDOW_S (a stretch that drops
    the ball a lot in play makes one more gap weak evidence). All zero where the ball is seen."""
    length = np.where(missing, gap_lengths(missing, fps), np.nan)
    share = window_mean(missing.astype(float), period > 0, to_frames(GAP_SHARE_WINDOW_S, fps))
    return np.hstack([hat(length, GAP_KNOTS), hat(share, SHARE_KNOTS) * missing[:, None]])


def window_max(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """Maximum of values (no NaN) over [t - before, t + after]."""
    return -window_min(-values, before, after)


def since_last(hit: np.ndarray, fps: int) -> np.ndarray:
    """(n,) seconds since the last frame where hit is True (that frame: 0); inf before the first."""
    index = np.arange(len(hit))
    last = np.maximum.accumulate(np.where(hit, index, -1))
    return np.where(last >= 0, (index - last) / fps, np.inf)


def touch_scale(match: Match, noise_ref_m: float = NTOUCH_REF_M,
                noise_exp: float = NTOUCH_EXP) -> float:
    """How much to widen the touch detector on this match (group "ntouch", B14, findings F68):
    s = max(1, ball_jitter / noise_ref_m) ** noise_exp, label-free (B9's noise measure). 1 when
    noise_ref_m is 0 or the jitter can't be measured. Broadcast tracking smooths the ball, so a
    touch's velocity change is spread over more time and looks smaller: a longer step (x s) and
    lower thresholds (/ s) see it again."""
    if not noise_ref_m:
        return 1.0
    jitter = ball_jitter(match)
    return max(1.0, jitter / noise_ref_m) ** noise_exp if np.isfinite(jitter) else 1.0


def touches(match: Match, d_home: np.ndarray, d_away: np.ndarray, step_s: float = TOUCH_STEP_S,
            mps=TOUCH_MPS):
    """(touch_home, touch_away) strengths in [0, 1] per frame: the ball's impulse over step_s
    (soft between mps) credited to the team whose closest player is nearer and within
    TOUCH_REACH_M."""
    low, high = mps
    strength = np.clip((np.nan_to_num(ball_impulse(match, step_s)) - low) / (high - low), 0.0, 1.0)
    home, away = np.nan_to_num(d_home, nan=np.inf), np.nan_to_num(d_away, nan=np.inf)
    t_home = strength * ((home <= away) & (home <= TOUCH_REACH_M))
    t_away = strength * ((away < home) & (away <= TOUCH_REACH_M))
    return t_home, t_away


def controlled(t_own: np.ndarray, d_own: np.ndarray, d_other: np.ndarray, fps: int,
               control_s: float = CONTROL_S, keep: float = CONTROL_KEEP) -> np.ndarray:
    """t_own where the touching team kept the ball (group "ctouch", B18): over the frames
    (t, t + control_s], its closest player is nearer the ball than the other team's on at least
    `keep` of the frames where the ball is seen; 0 elsewhere (and where none is seen)."""
    k = max(1, to_frames(control_s, fps))
    seen = ~np.isnan(d_own)
    near = seen & (d_own < np.nan_to_num(d_other, nan=np.inf))
    cn = np.concatenate(([0], np.cumsum(near)))
    cs = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(t_own))
    high = np.minimum(len(t_own), index + k + 1)
    low = np.minimum(len(t_own), index + 1)
    n_seen = cs[high] - cs[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n_seen > 0, (cn[high] - cn[low]) / np.maximum(n_seen, 1), 0.0)
    return t_own * (share >= keep)


def touch_lead(t_own: np.ndarray, t_other: np.ndarray, fps: int):
    """(seconds since own's last touch, clipped; other's minus own's, clipped)."""
    own = np.minimum(since_last(t_own >= TOUCH_MIN, fps), SINCE_KNOTS[-1])
    other = np.minimum(since_last(t_other >= TOUCH_MIN, fps), SINCE_KNOTS[-1])
    return own, np.clip(other - own, LEAD_KNOTS[0], LEAD_KNOTS[-1])


def touch_block(t_own: np.ndarray, t_other: np.ndarray, fps: int) -> np.ndarray:
    w, long = to_frames(WINDOW_S, fps), to_frames(LONG_WINDOW_S, fps)
    since, lead = touch_lead(t_own, t_other, fps)
    return np.hstack([t_own[:, None], window_max(t_own, w, 0)[:, None],
                      window_max(t_own, long, 0)[:, None], window_max(t_own, 0, w)[:, None],
                      hat(since, SINCE_KNOTS), hat(lead, LEAD_KNOTS)])


def duel_block(own: np.ndarray, other: np.ndarray, t_own: np.ndarray, t_other: np.ndarray,
               fps: int) -> np.ndarray:
    far = np.fmax(own, other)  # NaN only when both are missing
    low, high = DUEL_M
    contested = np.clip((high - np.nan_to_num(far, nan=high)) / (high - low), 0.0, 1.0)
    _, lead = touch_lead(t_own, t_other, fps)
    return np.hstack([hat(far, DUEL_KNOTS), hat(lead, LEAD_KNOTS) * contested[:, None]])


def team_block(own: np.ndarray, other: np.ndarray, count: np.ndarray, slow: np.ndarray,
               fps: int) -> np.ndarray:
    w = to_frames(WINDOW_S, fps)
    margin = np.where(np.isnan(own), np.nan,
                      np.nan_to_num(other, nan=MARGIN_KNOTS[-1]) - own)
    return np.hstack([hat(own, DIST_KNOTS), hat(window_min(own, w, 0), DIST_KNOTS),
                      hat(window_min(own, 0, w), DIST_KNOTS), hat(margin, MARGIN_KNOTS),
                      hat(window_min(own, to_frames(LONG_WINDOW_S, fps), 0), DIST_KNOTS),
                      hat(own, CONTROL_KNOTS) * slow[:, None], hat(count, COUNT_KNOTS)])


def _offset_mean(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of values over [t + lo, t + hi] (samples, lo <= hi), ignoring NaN; NaN where none."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.clip(index + lo, 0, len(values))
    high = np.clip(index + hi + 1, 0, len(values))
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def shape_values(match: Match, spec=None) -> dict:
    """{(name, window) or ("zone", name): (n,) per frame} for group "shape": windowed home - away
    differences of src/shape.py features, sampled at SHAPE_RATE_HZ and held to every frame; and the
    middle-third weight under key "mid". Label-free and ball-free."""
    from src import shape  # local: only this group needs it

    spec = SHAPE if spec is None else spec
    names = tuple(spec["features"])
    step = max(1, int(round(match.fps / SHAPE_RATE_HZ)))
    idx, feats = shape.team_features(match, step=step, features=names + ("cx",))
    diff = shape.differences({k: feats[k] for k in names})
    rate = match.fps / step
    nearest = np.minimum((np.arange(match.n_frames) + step // 2) // step, len(idx) - 1)
    out = {}
    for name in names:
        for window in spec["features"][name][1]:
            lo, hi = (int(round(v * rate)) for v in window)
            out[(name, tuple(window))] = _offset_mean(diff[name], lo, hi)[nearest]
    # where play is: the two outfield centroids' mean, absolute x (cx is in each team's own frame)
    length = match.pitch[0]
    middle = 0.5 * (feats["cx"][:, 0] + (length - feats["cx"][:, 1]))
    with np.errstate(invalid="ignore"):
        mid = np.clip(1.0 - np.abs(middle - length / 2) / (length / 3), 0.0, 1.0)
    out["mid"] = _offset_mean(mid, -int(round(rate)), int(round(rate)))[nearest]
    return out


def shape_blocks(match: Match, spec=None) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) columns for group "shape": hat(d) for home and hat(-d) for away per windowed
    difference d, then the zone features' first window x the middle-third weight."""
    spec = SHAPE if spec is None else spec
    values = shape_values(match, spec)
    mid = np.nan_to_num(values["mid"])[:, None]
    home, away = [], []
    for name, (knots, windows) in spec["features"].items():
        for window in windows:
            d = values[(name, tuple(window))]
            home.append(hat(d, knots))
            away.append(hat(-d, knots))
        if name in spec.get("zone", ()):
            d = values[(name, tuple(windows[0]))]
            home.append(hat(d, knots) * mid)
            away.append(hat(-d, knots) * mid)
    return np.hstack(home), np.hstack(away)


def impute_blocks(match: Match, missing: np.ndarray, fold: str = "auto"):
    """(home, away, common) columns for group "impute": distances to the imputed ball on
    ball-missing frames (zero elsewhere), see the module docstring."""
    from src import ball_impute

    est = Match(match_id=match.match_id, fps=match.fps, pitch=match.pitch, period=match.period,
                ball=ball_impute.impute(match, fold), players=match.players,
                slot_team=match.slot_team)
    i_home, i_away = team_distances(est)
    lo, hi = IMPUTE_LONG_S
    long = np.clip((gap_lengths(missing, match.fps) - lo) / (hi - lo), 0.0, 1.0)[:, None]
    gate = (missing & ~np.isnan(est.ball[:, 0]))[:, None]
    blocks = []
    for own, other in ((i_home, i_away), (i_away, i_home)):
        margin = np.where(np.isnan(own), np.nan, np.nan_to_num(other, nan=MARGIN_KNOTS[-1]) - own)
        m = hat(margin, MARGIN_KNOTS)
        blocks.append(np.hstack([hat(own, DIST_KNOTS), m, m * long]) * gate)
    length, width = match.pitch
    x, y = est.ball[:, 0], est.ball[:, 1]
    with np.errstate(invalid="ignore"):
        off = (x < 0) | (x > length) | (y < 0) | (y > width)
    return blocks[0], blocks[1], (off & gate[:, 0]).astype(float)[:, None]


def vball_blocks(match: Match, missing: np.ndarray):
    """(home, away, common) columns for group "vball": distances to the video's ball on frames the
    tracking lost it (zero elsewhere, and everywhere when the match has no video ball)."""
    n = match.n_frames
    vb = match.video_ball if match.video_ball is not None else np.full((n, 2), np.nan)
    est = Match(match_id=match.match_id, fps=match.fps, pitch=match.pitch, period=match.period,
                ball=vb, players=match.players, slot_team=match.slot_team)
    v_home, v_away = team_distances(est)
    gate = (missing & np.isfinite(vb[:, 0]))[:, None]
    blocks = []
    for own, other in ((v_home, v_away), (v_away, v_home)):
        margin = np.where(np.isnan(own), np.nan, np.nan_to_num(other, nan=MARGIN_KNOTS[-1]) - own)
        blocks.append(np.hstack([hat(own, DIST_KNOTS), hat(margin, MARGIN_KNOTS)]) * gate)
    length, width = match.pitch
    with np.errstate(invalid="ignore"):
        off = (vb[:, 0] < 0) | (vb[:, 0] > length) | (vb[:, 1] < 0) | (vb[:, 1] > width)
    common = np.hstack([gate.astype(float), (off & gate[:, 0]).astype(float)[:, None]])
    return blocks[0], blocks[1], common


def with_ball2(match: Match) -> Match:
    """match with its ball's gaps filled from Match.ball2 (V4); unchanged without ball2."""
    if match.ball2 is None:
        return match
    ball = np.where(np.isnan(match.ball[:, :1]), match.ball2, match.ball)
    return dataclasses.replace(match, ball=ball)


def bball_blocks(match: Match, missing: np.ndarray):
    """(home, away, common) columns for group "bball": distances to Match.ball2 on ball-missing
    frames where it is there (zero elsewhere), see the module docstring."""
    ball2 = match.ball2 if match.ball2 is not None else np.full((match.n_frames, 2), np.nan)
    est = dataclasses.replace(match, ball=ball2, ball2=None)
    b_home, b_away = team_distances(est)
    gate = (missing & ~np.isnan(ball2[:, 0]))[:, None]
    blocks = []
    for own, other in ((b_home, b_away), (b_away, b_home)):
        margin = np.where(np.isnan(own), np.nan, np.nan_to_num(other, nan=MARGIN_KNOTS[-1]) - own)
        blocks.append(np.hstack([hat(own, DIST_KNOTS), hat(margin, MARGIN_KNOTS)]) * gate)
    length, width = match.pitch
    with np.errstate(invalid="ignore"):
        off = (ball2[:, 0] < 0) | (ball2[:, 0] > length) | (ball2[:, 1] < 0) | (ball2[:, 1] > width)
    common = np.hstack([gate.astype(float), (off & gate[:, 0]).astype(float)[:, None]])
    return blocks[0], blocks[1], common


def gapend_blocks(match: Match, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) columns for group "gapends": on ball-missing frames, own team nearest the ball
    when it was last seen x hat(seconds since), and when it is next seen x hat(seconds until)."""
    from src import ball_impute

    last, nxt = ball_impute.gap_context(match.ball, match.period)
    index = np.arange(match.n_frames)
    d_home, d_away = team_distances(match)
    near_home = np.where(np.isnan(d_home), False, d_home <= np.nan_to_num(d_away, nan=np.inf))
    near_away = np.where(np.isnan(d_away), False, d_away < np.nan_to_num(d_home, nan=np.inf))
    out = []
    for near in (near_home, near_away):
        cols = []
        for end, since in ((last, index - last), (nxt, nxt - index)):
            known = (end >= 0) & missing
            own = np.where(known, near[np.maximum(end, 0)], False).astype(float)
            t = np.where(known, since / match.fps, np.nan)
            cols.append(hat(t, GAPEND_KNOTS) * own[:, None])
        out.append(np.hstack(cols))
    return out[0], out[1]


def ghosts(match: Match, ghost_s: float = BLIND_GHOST_S) -> np.ndarray:
    """(n, k, 2) where a slot is not seen, up to ghost_s after it was last seen in the same period:
    its last position moved with its team's drift (src/tracks.py). NaN elsewhere."""
    from src import tracks

    players, period = match.players, match.period
    n, k = players.shape[:2]
    seen = ~np.isnan(players).any(axis=2)
    last = np.maximum.accumulate(np.where(seen, np.arange(n)[:, None], -1), axis=0)
    ref = np.maximum(last, 0)
    age = np.arange(n)[:, None] - last
    use = ~seen & (last >= 0) & (age <= to_frames(ghost_s, match.fps)) & (period[ref] == period[:, None])
    drift = tracks.team_drift(players, match.slot_team, period)  # (2, n, 2)
    team = match.slot_team.astype(int)
    moved = drift[team[None, :], np.arange(n)[:, None]] - drift[team[None, :], ref]
    out = players[ref, np.arange(k)[None, :]] + moved
    return np.where(use[..., None], out, np.nan)


def blind_blocks(match: Match, d_home: np.ndarray, d_away: np.ndarray):
    """(home, away) columns for group "blind" (C13): deficit, ghosts near the ball, margin x ghost."""
    seen = ~np.isnan(match.players).any(axis=2)
    ghost = ghosts(match)
    with np.errstate(invalid="ignore"):
        near = np.hypot(ghost[..., 0] - match.ball[:, None, 0],
                        ghost[..., 1] - match.ball[:, None, 1]) <= BLIND_GHOST_M
    missing = np.isnan(match.ball[:, 0])
    out = []
    for team, own, other in ((HOME, d_home, d_away), (AWAY, d_away, d_home)):
        slots = match.slot_team == team
        deficit = np.maximum(BLIND_FULL - seen[:, slots].sum(axis=1), 0).astype(float)
        count = np.where(missing, np.nan, near[:, slots].sum(axis=1).astype(float))
        margin = np.where(np.isnan(own), np.nan, np.nan_to_num(other, nan=MARGIN_KNOTS[-1]) - own)
        flag = np.minimum(np.nan_to_num(count), 1.0)[:, None]
        out.append(np.hstack([hat(deficit, BLIND_DEFICIT_KNOTS), hat(count, BLIND_GHOST_KNOTS),
                              hat(margin, MARGIN_KNOTS) * flag]))
    return out[0], out[1]


def features(match: Match, groups=GROUPS, speed_span_s: float | None = None,
             shape_gain: float = 1.0, shape_seen_gain: float = 1.0,
             tags: list | None = None,
             impute_fold: str = "auto") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T_home, T_away, C) feature blocks, float32, one row per frame. groups adds the optional
    "touch" and "duel" columns to the team blocks, "gap" to the common block; "speedspan"
    measures the two player-speed features over SPEED_SPAN_S instead of frame steps;
    speed_span_s, when given, overrides that span (0 = frame steps) whatever the groups say.
    shape_gain scales the "shape" columns (decode-time only, B25: shape's windows are
    autocorrelated, so the decoder's per-frame sum over-counts them), times shape_seen_gain
    on frames where the ball is seen. tags (group "tags") overrides match.tags; [] = no tags.
    impute_fold: which imputer fit group "impute" uses (src/ball_impute.py: "auto" = without
    this match if it trained on it)."""
    unknown = set(groups) - set(GROUP_FEATURES)
    if unknown:
        raise ValueError("unknown feature groups {}".format(sorted(unknown)))
    if "bfill" in groups:
        match = with_ball2(match)
    d_home, d_away = team_distances(match)
    missing = np.isnan(match.ball[:, 0])
    half = to_frames(DEAD_WINDOW_S, match.fps)
    span_s = SPEED_SPAN_S if "speedspan" in groups else 0.0
    if speed_span_s is not None:
        span_s = speed_span_s
    share = window_mean(missing.astype(float), np.ones(len(missing), bool), half)
    out = ball_out(match)
    common = np.hstack([
        np.ones((match.n_frames, 1)), missing[:, None].astype(float), hat(share, SHARE_KNOTS),
        hat(player_speed(match, DEAD_WINDOW_S, span_s), SPEED_KNOTS),
        hat(player_speed(match, 0.5, span_s), SPEED_KNOTS),
        out[:, None].astype(float), extend(out, to_frames(OUT_S, match.fps))[:, None].astype(float)])
    if "gap" in groups:
        common = np.hstack([common, gap_block(missing, match.period, match.fps)])
    if "deadm" in groups:  # local L3: player motion and set-piece shapes, only where the ball is missing
        s2, s05, masks = gates.dead_gap_raw(match)
        common = np.hstack([common, np.hstack([hat(s2, SPEED_KNOTS), hat(s05, SPEED_KNOTS), masks])
                            * missing[:, None]])
    slow = ball_slow(match)
    n_home, n_away = near_counts(match)
    home = [team_block(d_home, d_away, n_home, slow, match.fps)]
    away = [team_block(d_away, d_home, n_away, slow, match.fps)]
    if {"touch", "duel"} & set(groups):
        t_home, t_away = touches(match, d_home, d_away)
    if "touch" in groups:
        home.append(touch_block(t_home, t_away, match.fps))
        away.append(touch_block(t_away, t_home, match.fps))
    if "ctouch" in groups:  # B18: the touch columns again, only touches the team kept
        spec = GROUP_FEATURES["ctouch"]
        if "touch" not in groups and "duel" not in groups:
            t_home, t_away = touches(match, d_home, d_away)
        c_home = controlled(t_home, d_home, d_away, match.fps, spec["control_s"], spec["keep"])
        c_away = controlled(t_away, d_away, d_home, match.fps, spec["control_s"], spec["keep"])
        home.append(touch_block(c_home, c_away, match.fps))
        away.append(touch_block(c_away, c_home, match.fps))
    if "ntouch" in groups:  # B14: the touch columns again, detector scaled to the ball's noise
        spec = GROUP_FEATURES["ntouch"]
        scale = touch_scale(match, spec["noise_ref_m"], spec["noise_exp"])
        nt_home, nt_away = touches(match, d_home, d_away, spec["step_s"] * scale,
                                   tuple(v / scale for v in spec["mps"]))
        home.append(touch_block(nt_home, nt_away, match.fps))
        away.append(touch_block(nt_away, nt_home, match.fps))
    s_cache = None
    # this order keeps every committed file's columns unchanged (v4/v5: "shapem" only)
    for group in ("shape", "shapem", "shapeg", "shapes", "shapef", "shapeo"):
        if group not in groups:
            continue
        if s_cache is None:
            s_cache = shape_blocks(match)  # computed once even with several shape groups
        s_home, s_away = s_cache
        if group in ("shapem", "shapeg", "shapes"):
            gate = missing
        elif group == "shapef":  # local L1 (F201): ball seen, free and in flight
            gate = gates.flight(match, d_home, d_away)
        elif group == "shapeo":  # local L1 (F201): ball seen and free
            gate = gates.free_ball(match, d_home, d_away)
        else:
            gate = None
        if gate is not None:
            s_home, s_away = s_home * gate[:, None], s_away * gate[:, None]
        if group in ("shapeg", "shapes"):  # B26: split by gap length / near a stoppage
            w = (gap_long(missing, match.fps) if group == "shapeg"
                 else gap_near_stop(match, missing))[:, None]
            s_home = np.hstack([s_home * (1.0 - w), s_home * w])
            s_away = np.hstack([s_away * (1.0 - w), s_away * w])
        if shape_gain != 1.0 or shape_seen_gain != 1.0:
            gain = shape_gain * np.where(missing, 1.0, shape_seen_gain)[:, None]
            s_home, s_away = s_home * gain, s_away * gain
        home.append(s_home)
        away.append(s_away)
    if "tags" in groups:  # B28
        g_home, g_away, g_common = smart_tags.columns(match.tags if tags is None else tags,
                                                      match.n_frames)
        home.append(g_home)
        away.append(g_away)
        common = np.hstack([common, g_common])
    if "impute" in groups:
        i_home, i_away, i_common = impute_blocks(match, missing, impute_fold)
        home.append(i_home)
        away.append(i_away)
        common = np.hstack([common, i_common])
    if "vball" in groups:
        v_home, v_away, v_common = vball_blocks(match, missing)
        home.append(v_home)
        away.append(v_away)
        common = np.hstack([common, v_common])
    if "bball" in groups:
        b_home, b_away, b_common = bball_blocks(match, missing)
        home.append(b_home)
        away.append(b_away)
        common = np.hstack([common, b_common])
    if "gapends" in groups:
        g_home, g_away = gapend_blocks(match, missing)
        home.append(g_home)
        away.append(g_away)
    if "blind" in groups:  # C13
        b_home, b_away = blind_blocks(match, d_home, d_away)
        home.append(b_home)
        away.append(b_away)
    if "duel" in groups:
        home.append(duel_block(d_home, d_away, t_home, t_away, match.fps))
        away.append(duel_block(d_away, d_home, t_away, t_home, match.fps))
    return (np.hstack(home).astype(np.float32), np.hstack(away).astype(np.float32),
            common.astype(np.float32))


# the order features() stacks the optional groups' columns in: team blocks, then the common block
TEAM_ORDER = ("touch", "ctouch", "ntouch", "shape", "shapem", "shapeg", "shapes", "tags", "impute",
              "vball", "bball", "gapends", "blind", "duel")
COMMON_ORDER = ("gap", "tags", "impute", "vball", "bball")
_WIDTHS: dict = {}


def _layout_match() -> Match:
    """A tiny random match (ball missing for a stretch, every optional input present) to measure
    each group's column count on."""
    rng = np.random.default_rng(0)
    n = 120
    ball = rng.uniform(0, 1, (n, 2)) * [105.0, 68.0]
    ball[40:80] = np.nan
    return Match(match_id="layout:0", fps=10, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=ball, players=rng.uniform(0, 1, (n, 22, 2)) * [105.0, 68.0],
                 slot_team=np.repeat([HOME, AWAY], 11).astype(np.int8), video_ball=ball.copy(),
                 video_conf=np.zeros(n), ball2=ball.copy())


def group_widths(group: str) -> tuple[int, int]:
    """(team-block columns, common columns) the optional group adds to features()."""
    if not _WIDTHS:
        match = _layout_match()
        base = features(match, ())
        _WIDTHS[""] = (base[0].shape[1], base[2].shape[1])
        _WIDTHS["match"] = match
    if group not in _WIDTHS:
        if group not in GROUP_FEATURES:
            raise ValueError("unknown feature group {!r}".format(group))
        own = features(_WIDTHS["match"], (group,), tags=[])
        _WIDTHS[group] = (own[0].shape[1] - _WIDTHS[""][0], own[2].shape[1] - _WIDTHS[""][1])
    return _WIDTHS[group]


def group_layout(groups) -> dict:
    """{group: (team-block slice, common slice)}: where each optional group's columns sit in the
    blocks features(match, groups) returns (an empty slice where it has none). B30: what
    fit_learned.py's --source-zero zeroes and where mixed_rows() writes the tags."""
    team, common = group_widths("")
    out = {}
    for order, which in ((TEAM_ORDER, 0), (COMMON_ORDER, 1)):
        at = (team, common)[which]
        for g in order:
            if g in groups:
                width = group_widths(g)[which]
                out.setdefault(g, [slice(0, 0), slice(0, 0)])[which] = slice(at, at + width)
                at += width
    for g in groups:
        out.setdefault(g, [slice(0, 0), slice(0, 0)])
    return {g: tuple(v) for g, v in out.items()}


# -- the conditional logit --------------------------------------------------------------------------

def _split(theta: np.ndarray, p_team: int):
    a, b = theta[:p_team], theta[p_team:2 * p_team]
    c, e = theta[2 * p_team:-p_team], theta[-p_team:]
    return a, b, c, e


def logits(blocks, theta: np.ndarray) -> np.ndarray:
    """(n, 3) logits for H, A, D."""
    th, ta, cm = (np.asarray(x, dtype=np.float64) for x in blocks)
    a, b, c, e = _split(theta, th.shape[1])
    common = cm @ c
    return np.stack([th @ a + ta @ b + common, ta @ a + th @ b + common, (th + ta) @ e], axis=1)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _design(blocks):
    """Per-class design matrices Phi_k (n, p) such that z_k = Phi_k @ theta."""
    th, ta, cm = (np.asarray(x, dtype=np.float64) for x in blocks)
    zt, zc = np.zeros_like(th), np.zeros_like(cm)
    return (np.hstack([th, ta, cm, zt]), np.hstack([ta, th, cm, zt]),
            np.hstack([zt, zt, zc, th + ta]))


def _objective(blocks, y, w, theta, chunk):
    total = 0.0
    for s in range(0, len(y), chunk):
        part = tuple(x[s:s + chunk] for x in blocks)
        lp = log_softmax(logits(part, theta))
        total += float(np.sum(w[s:s + chunk] * lp[np.arange(len(lp)), y[s:s + chunk]]))
    return total - 0.5 * L2 * float(theta @ theta)


def fit_logit(blocks, y: np.ndarray, w: np.ndarray, iters: int = 30,
              chunk: int = 50_000) -> np.ndarray:
    """theta maximising the weighted log-likelihood (w sums to 1) minus L2/2 |theta|^2.
    Newton's method with step halving; deterministic. y in {0, 1, 2} = H, A, D."""
    p = 3 * blocks[0].shape[1] + blocks[2].shape[1]
    theta = np.zeros(p)
    value = _objective(blocks, y, w, theta, chunk)
    for _ in range(iters):
        grad, hess = -L2 * theta, -L2 * np.eye(p)
        for s in range(0, len(y), chunk):
            part = tuple(x[s:s + chunk] for x in blocks)
            phi = _design(part)
            prob = np.exp(log_softmax(logits(part, theta)))
            ws = w[s:s + chunk]
            ys = y[s:s + chunk]
            mean = sum(prob[:, [k]] * phi[k] for k in range(3))
            for k in range(3):
                grad += phi[k][ys == k].T @ ws[ys == k]
                hess -= phi[k].T @ (phi[k] * (ws * prob[:, k])[:, None])
            grad -= mean.T @ ws
            hess += mean.T @ (mean * ws[:, None])
        step = np.linalg.solve(hess, -grad)
        scale = 1.0
        while scale > 1e-4:
            new = theta + scale * step
            new_value = _objective(blocks, y, w, new, chunk)
            if new_value >= value:
                break
            scale /= 2
        else:
            break
        theta, gain, value = new, new_value - value, new_value
        if gain < 1e-10:
            break
    return theta


# -- training data ----------------------------------------------------------------------------------

def training_rows(match: Match, stride_s: float = FIT_STRIDE_S, groups=GROUPS):
    """(blocks, y) for one labelled match: frames inside a half labelled H/A/D, every stride_s."""
    blocks = features(match, groups)
    step = max(1, to_frames(stride_s, match.fps))
    index = np.arange(0, match.n_frames, step)
    labels = match.labels[index]
    keep = (match.period[index] > 0) & np.isin(labels, STATES)
    index = index[keep]
    y = np.searchsorted(STATES, match.labels[index])  # H, A, D are 0, 1, 2
    return tuple(x[index] for x in blocks), y.astype(np.int64)


def fit(rows: dict) -> dict:
    """Coefficients from {match key ('source:id'): (blocks, y)}. Each source weighs the same, each
    match the same within its source."""
    keys = sorted(rows)
    sources = sorted({k.split(":", 1)[0] for k in keys})
    per_source = {s: [k for k in keys if k.startswith(s + ":")] for s in sources}
    weights = []
    for k in keys:
        n_matches = len(per_source[k.split(":", 1)[0]])
        weights.append(np.full(len(rows[k][1]), 1.0 / (len(sources) * n_matches * len(rows[k][1]))))
    blocks = tuple(np.concatenate([rows[k][0][i] for k in keys]) for i in range(3))
    y = np.concatenate([rows[k][1] for k in keys])
    w = np.concatenate(weights)
    theta = fit_logit(blocks, y, w)
    prior = np.array([w[y == k].sum() for k in range(3)])
    return {"theta": [round(float(v), 6) for v in theta],
            "log_prior": [round(float(v), 6) for v in np.log(prior / prior.sum())],
            "train": keys}


# -- the model --------------------------------------------------------------------------------------

_COEFS: dict = {}


def load_coefs(path: str = COEF_FILE) -> dict:
    if path not in _COEFS:
        with open(path) as handle:
            coefs = json.load(handle)
        groups = tuple(coefs.get("groups", ()))
        expected = feature_hash(groups) if set(groups) <= set(GROUP_FEATURES) else None
        if coefs.get("feature_hash") != expected:
            raise ValueError("{} was fit for other features ({} != {}); rerun "
                             "scripts/fit_learned.py".format(path, coefs.get("feature_hash"),
                                                             expected))
        _COEFS[path] = coefs
    return _COEFS[path]


_REPAIRED: list = [None, None, None]  # the last (match, track_repair, repaired match): one repair
# serves scores() and decode() of the same match


TAG_SOURCES = ("match", "standin", "oracle", "skillcorner", "none")


TAG_CONVENTIONS = ("auto", "as_is")


def tags_for(match: Match, source: str = "match", noise: int = 0, seed: int = 0,
             flip: bool = False, convention: str = "auto") -> list | None:
    """Smart Tagging for the "tags" group (B28): Match.tags, stand-ins made from an external match's
    labels (thinned to DATA.md's rates, or the "oracle" with every candidate), SkillCorner's own
    phases (none on other sources), or none ([]). Stand-ins need labels, so they can't be made on
    BAR-ATH. B32: flip swaps the defending codes' team first (a tagger with the other reading of the
    Team tag, for testing); convention "auto" then puts them in our reading with the label-free
    smart_tags.canonical (BAR-ATH's file included), "as_is" reads them as they are."""
    if source == "match":
        tags = match.tags
    elif source == "none":
        return []
    elif source == "skillcorner":  # SkillCorner's own phases; other sources have none
        if not match.match_id.startswith("skillcorner:"):
            return []
        tags = smart_tags.skillcorner(match)
        if noise:
            tags = smart_tags.perturb(tags, smart_tags.NOISE_LEVELS[noise], match,
                                      np.random.default_rng(seed))
    else:
        tags = smart_tags.stand_in(match, noise, seed, thin=source == "standin")
    if not tags:
        return tags
    if flip:
        tags = smart_tags.flip_defending(tags)
    if convention not in TAG_CONVENTIONS:
        raise ValueError("convention must be one of {}".format(TAG_CONVENTIONS))
    return smart_tags.canonical(tags, match) if convention == "auto" else tags


# -- training on a mix of taggers (B32, moved here from scripts/check_smart_tags.py by B30) ----------

# Tag conditions a training row set can read (B32): stand-ins at each noise level, the unthinned
# oracle, SkillCorner's own phases (SkillCorner matches only), or no tags (whole-tag dropout).
TAG_CONDITIONS = {"n{}".format(k): ("standin", k) for k in smart_tags.NOISE_LEVELS}
TAG_CONDITIONS.update({"oracle": ("oracle", 0), "sc": ("skillcorner", 0), "none": ("none", 0)})
# B32's recipe (findings F132): noise 1-3, SkillCorner's phases and no tags, four per row set, half
# of them read by a tagger with the other defending-code convention
TAG_MIX_B = "n1:1,n2:1,n3:1,sc:1,none:1"


def parse_tag_mix(spec: str) -> dict:
    """"n0:1,n2:2,none:1" -> {"n0": 1.0, "n2": 2.0, "none": 1.0}."""
    out = {}
    for part in (p for p in spec.split(",") if p):
        token, _, weight = part.partition(":")
        if token not in TAG_CONDITIONS:
            raise ValueError("unknown tag condition {!r} (one of {})".format(
                token, sorted(TAG_CONDITIONS)))
        out[token] = float(weight or 1)
    return out


def draw_tag_conditions(key: str, mix: dict, draws: int, flip: float) -> list:
    """[(token, seed, flipped)] for one training row set: `draws` tag conditions drawn from `mix`
    (without replacement while there are enough; "sc" only on SkillCorner), each flipped (the other
    reading of the defending codes' Team) with probability `flip`. Seeded by the key."""
    rng = np.random.default_rng(zlib.crc32("b32|{}".format(key).encode()))
    tokens = [t for t in sorted(mix) if t != "sc" or key.startswith("skillcorner:")]
    p = np.array([mix[t] for t in tokens])
    chosen = rng.choice(len(tokens), size=draws, replace=draws > len(tokens), p=p / p.sum())
    flips = rng.random(draws) < flip
    return [(tokens[c], 1000 + i, bool(f)) for i, (c, f) in enumerate(zip(chosen, flips))]


def mixed_rows(match: Match, groups, conditions, convention: str = "auto",
               stride_s: float = FIT_STRIDE_S):
    """training_rows with one tag condition per interleaved slice of the rows: row i of the match's
    rows reads conditions[i % len(conditions)]. The features are computed once without tags and the
    "tags" columns (wherever group_layout puts them, B30) filled per condition. With one condition
    this equals training_rows on a match carrying those tags."""
    if "tags" not in groups:
        raise ValueError("mixed_rows needs the tags group")
    blocks = features(match, groups, tags=[])
    team_cols, common_cols = group_layout(groups)["tags"]
    step = max(1, to_frames(stride_s, match.fps))
    index = np.arange(0, match.n_frames, step)
    keep = (match.period[index] > 0) & np.isin(match.labels[index], STATES)
    index = index[keep]
    y = np.searchsorted(STATES, match.labels[index]).astype(np.int64)
    out = [x[index] for x in blocks]
    for i, (token, seed, flipped) in enumerate(conditions):
        source, level = TAG_CONDITIONS[token]
        tags = tags_for(match, source, level, seed, flipped, convention)
        cols = smart_tags.columns(tags, match.n_frames)
        rows = np.arange(i, len(index), len(conditions))
        for block, col, where in zip(out, cols, (team_cols, team_cols, common_cols)):
            block[rows, where] = col[index[rows]]
    return tuple(out), y


@dataclass(kw_only=True)
class LearnedDecoder:
    """Viterbi over {H, A, D} with learned emissions. Costs in seconds of evidence (nats / s).
    The defaults are the tuned pick (findings F55: mean over sources, BAR-ATH-like ball, held out;
    F57: still the pick with the touch features)."""

    switch_cost_s: float = 0.75  # cost of an H<->A switch
    dead_cost_s: float = 3.0  # cost of a team<->D switch
    prior: int = 1  # 1: score log p(k|x) - log prior(k); 0: log p(k|x)
    dead_gain: float = 1.0  # multiplies D's score (after the prior)
    lag_s: float = 0.2  # shift the learned team scores this much later
    hand_team: float = 2.0  # add this times `viterbi`'s hand-set team evidence (B3) to H and A
    hand_dead: float = 0.0  # ... and this times its dead evidence to D
    min_spell_s: float | None = 0.5  # then drop segments shorter than this (B6); None/0 = off.
    # 0.5 since C6 (findings F66): best mean over the dropout grid, 15/15 grid points vs 1.0
    fold: str = "all"  # coefficients: "all", "lomo" (fit without this match), "loso" (... source)
    coef_file: str = COEF_FILE
    noise_ref_m: float = 0.02  # ball_jitter above this scales switch_cost_s up (B9/F54 rule; B11/F63 pick); 0 = off
    noise_cost_exp: float = 2.0  # switch_cost_s *= max(1, jitter / noise_ref_m) ** this
    noise_measure: str = "jitter"  # the per-match measure: "jitter" (B9) or "noise" (B19, steady
    # under dropout; pair it with its own noise_ref_m, decoder.noise_level)
    speed_span_s: float | None = 1.0  # player-speed span for the features, overriding the coef
    # file's (C5, findings F72-F73): 1.0 makes the speed features robust to position noise
    # without a refit; None = as fit (frame steps for the committed file). 1.0 since v3 (F80)
    hand_pjitter_ref_m: float = 0.0  # scale the hand evidence's reach by player jitter (C5); 0 = off
    hand_pjitter_exp: float = 0.25  # its reach *= max(1, player_jitter / ref) ** this
    onset_snap_s: float = 3.0  # start decoded D at the earliest onset cue this far before it (B12,
    # F78); 0 = off. 3.0 since v3 (F80)
    onset_cues: str = "out+gap"  # which cues (src/models/onset.py), joined by "+"
    restart_snap_s: float = 0.0  # end decoded D at the latest restart cue this far before its
    # end (B24, findings F94-F95); 0 = off
    restart_cues: str = "gapend+go"  # which restart cues (onset.RESTART_CUES), joined by "+"
    restart_rule_gain: float = 0.0  # add this x the restart rule's dead stretch to D (src/restarts.py,
    # PR #51: the few seconds before a kick found in a restart window); 0 = off
    restart_rule_back_s: float = 6.0  # how far before the kick counts as dead
    restart_rule_kick_mps: float = 4.0  # ball speed inside the pitch that counts as the kick
    restart_rule_search_s: float = 8.0  # look for the kick in this much of each window
    restart_rule_noise: int = 0  # B35 stress test, external matches only: perturb the windows as a
    # tagger would (restarts.noisy, smart_tags.NOISE_LEVELS); 0 = the windows as given
    shape_gain: float = 1.0  # scale the "shape" group's evidence (B25, findings F96-F99); only
    # for coefficient files fit with that group
    shape_seen_gain: float = 1.0  # ... times this on frames where the ball is seen
    tags_source: str = "match"  # group "tags" (B28): "match" = Match.tags (BAR-ATH's file);
    # external matches only: "standin" (smart_tags.stand_in from the labels), "oracle" (the same,
    # not thinned to DATA.md's rates), "skillcorner" (SkillCorner's phases; [] elsewhere); "none"
    tags_noise: int = 0  # stand-in tagger noise, a smart_tags.NOISE_LEVELS key
    tags_seed: int = 0  # stand-in draw
    tags_flip: bool = False  # B32, external tests: swap the defending codes' team before reading
    tags_convention: str = "auto"  # B32: "auto" puts defending codes in our reading, label-free
    # (smart_tags.canonical); "as_is" reads them as given
    priors: str = ""  # duration priors from src/models/priors.json (B27, src/priors.py), e.g.
    # "statsbomb": segments that end shorter than real ones pay an exit penalty; "" = off (flat
    # costs only, decoder.viterbi)
    prior_team_w: float = 0.0  # weight of the possession-spell prior on H / A segments
    prior_dead_w: float = 0.0  # weight of the stoppage prior on D segments
    prior_typed: bool = False  # D segments with a set-piece shape use that restart type's prior
    prior_floor: float = 0.01  # the CDF is floored here: the largest penalty is -log(floor) * w
    track_repair: str = ""  # C13: repair the players before the features and decoding
    # (src/tracks.py: link tracklets, bridge gaps): "1" = its defaults, "max_gap_s/reach_m/speed_mps
    # [/fill]", ""/"0" = off
    ball2_fill: bool = False  # V4: fill the ball's gaps from Match.ball2 (the aligned broadcast
    # ball) before the features and the decoder's cues, whatever the coefficient file; a file fit
    # with group "bfill" does it anyway. The noise scaling always reads the unfilled ball.
    switch_cue_gain: float = 0.0  # local L2 (F200+): per-frame switch costs from team-shape
    # changes (src/models/switch_cue.py); 0 = off
    switch_cue_where: str = "all"  # "all" frames or only where the ball is "missing"
    switch_cue_min_seen: int = 0  # B36: the cue only where both teams have >= this many seen
    # outfield players (switch_cue.gate, C10's coverage); 0 = everywhere
    switch_cue_file: str = os.path.join(os.path.dirname(COEF_FILE), "switch_cue_coefs.json")
    cues: str = ""  # wave 5: decode-time cues, "name[:gain][/k=v...]" joined by "+" (src/models/
    # cues/): per-frame multipliers on the transition costs and extra evidence; "" = off
    repairs: str = ""  # wave 5: run-time tracking repairs before track_repair, "name[/k=v...]"
    # joined by "+" (src/repairs/); "" = off

    def __post_init__(self):
        if self.fold not in ("all", "lomo", "loso"):
            raise ValueError("fold must be all, lomo or loso, not {!r}".format(self.fold))
        if self.noise_measure not in NOISE_MEASURES:
            raise ValueError("noise_measure must be one of {}".format(NOISE_MEASURES))
        if self.tags_source not in TAG_SOURCES:
            raise ValueError("tags_source must be one of {}".format(TAG_SOURCES))
        if self.tags_convention not in TAG_CONVENTIONS:
            raise ValueError("tags_convention must be one of {}".format(TAG_CONVENTIONS))
        if self.switch_cue_where not in ("all", "missing"):
            raise ValueError("switch_cue_where must be all or missing, not {!r}".format(
                self.switch_cue_where))
        if self.switch_cue_min_seen < 0:
            raise ValueError("switch_cue_min_seen must be >= 0")
        if self.switch_cue_gain and self.priors:
            # both are off by default and neither was kept (F203, B27): the aged-segment decoder
            # (src/priors.py) takes one flat switch cost, so combining them needs a new recursion
            raise ValueError("switch_cue_gain and priors can't be combined (yet)")
        if self.cues:
            if self.priors:
                raise ValueError("cues and priors can't be combined (yet)")
            from src.models import cues as cue_mod
            cue_mod.check(self.cues)
        if self.repairs:
            from src import repairs as repair_mod
            repair_mod.check(self.repairs)

    def tags(self, match: Match) -> list | None:
        """The Smart Tagging the "tags" group reads on this match (see tags_source)."""
        return tags_for(match, self.tags_source, self.tags_noise, self.tags_seed, self.tags_flip,
                        self.tags_convention)

    def coefs(self, match: Match) -> dict:
        coefs = load_coefs(self.coef_file)
        source = match.match_id.split(":", 1)[0]
        if self.fold == "all" or source not in coefs.get("fit", {}).get("sources", (source,)):
            return coefs["all"]  # a source the fit never saw: "all" is held out by source (C12)
        key = match.match_id if self.fold == "lomo" else source
        folds = coefs[self.fold]
        if key not in folds:
            raise KeyError("no {} fit without {!r} in {}".format(self.fold, key, self.coef_file))
        return folds[key]

    def prepare(self, match: Match) -> Match:
        """The match the features and the decoder read: repaired when `repairs` (wave 5, first) or
        track_repair (C13, second) is set. The last repair is kept, so scores() and decode() on the
        same match repair once."""
        spec = str(self.track_repair)
        rspec = str(self.repairs)
        if spec in ("", "0") and not rspec:
            return match
        key = spec if not rspec else (rspec, spec)
        if _REPAIRED[0] is not match or _REPAIRED[1] != key:
            out = match
            if rspec:
                from src import repairs as repair_mod
                out = repair_mod.apply(out, rspec)
            if spec not in ("", "0"):
                from src import tracks
                out = tracks.apply(out, spec)
            _REPAIRED[:] = [match, key, out]
        return _REPAIRED[2]

    def fills(self) -> bool:
        """Whether the ball's gaps are filled from Match.ball2 (ball2_fill or a "bfill" fit)."""
        return self.ball2_fill or "bfill" in load_coefs(self.coef_file).get("groups", ())

    def reads_ball2(self) -> bool:
        """Whether this decoder reads Match.ball2 (src.run then attaches BAR-ATH's broadcast ball)."""
        return self.fills() or "bball" in load_coefs(self.coef_file).get("groups", ())

    def scores(self, match: Match) -> np.ndarray:
        """(n, 3) per-second evidence for H, A, D."""
        coefs = self.coefs(match)
        match = self.prepare(match)
        groups = tuple(load_coefs(self.coef_file).get("groups", ()))
        if self.fills():
            match = with_ball2(match)
        tags = self.tags(match) if "tags" in groups else None
        impute_fold = "loso" if self.fold == "loso" else "auto"
        lp = log_softmax(logits(features(match, groups, self.speed_span_s, self.shape_gain,
                                         self.shape_seen_gain, tags=tags, impute_fold=impute_fold),
                                np.array(coefs["theta"])))
        if self.prior:  # H and A share their mean prior, like the logit shares its weights
            prior = np.exp(coefs["log_prior"])
            team = (prior[0] + prior[1]) / 2
            lp = lp - np.log([team, team, prior[2]])
        team = lp[:, :2]
        lag = int(round(self.lag_s * match.fps))
        if lag:
            team = np.stack([shift(team[:, 0], lag), shift(team[:, 1], lag)], axis=1)
        out = np.hstack([team, self.dead_gain * lp[:, 2:]]) / match.fps
        if self.hand_team or self.hand_dead:
            hand = SwitchDecoder(**B3_SETTINGS, pjitter_ref_m=self.hand_pjitter_ref_m,
                                 pjitter_reach_exp=self.hand_pjitter_exp).scores(match)
            out += hand * np.array([self.hand_team, self.hand_team, self.hand_dead])
        return self.add_restart_rule(match, out)

    def add_restart_rule(self, match: Match, out: np.ndarray) -> np.ndarray:
        """out (scores()) with the restart rule's dead stretch added to D's column; out itself
        when restart_rule_gain is 0. Shared with the emission models (emission.combine)."""
        if not self.restart_rule_gain:
            return out
        dead = restarts.dead_stretch(match, self.restart_rule_back_s, self.restart_rule_kick_mps,
                                     self.restart_rule_search_s, self.restart_rule_noise)
        out = out.copy()
        out[:, 2] += self.restart_rule_gain * dead / match.fps
        return out

    def switch_cue_costs(self, match: Match, original: Match, switch_cost: float):
        """(c_ha, c_ah): the switch cue's per-frame H<->A costs (B36) on the prepared match."""
        from src.models import switch_cue  # lazy: switch_cue imports from this module

        cue_fit = switch_cue.fold_fit(switch_cue.load_cue(self.switch_cue_file), self.fold,
                                      match.match_id)
        # the gate counts the tracking as delivered, before C13's repair fills its gaps
        gate = (switch_cue.gate(original, self.switch_cue_min_seen)
                if self.switch_cue_min_seen else None)
        return switch_cue.costs(match, cue_fit, switch_cost, self.switch_cue_gain,
                                self.switch_cue_where, gate)

    def predict(self, match: Match) -> np.ndarray:
        return self.decode(match, self.scores(match))

    def decode(self, match: Match, scores: np.ndarray) -> np.ndarray:
        """Codes from scores() (split out so one set of scores can be decoded several ways)."""
        original = match
        match = self.prepare(match)
        raw = match
        if self.fills():
            match = with_ball2(match)
        codes = majority_smooth(nearest_team(match), to_frames(2.0, match.fps))
        switch_cost = self.switch_cost_s
        if self.noise_ref_m:
            jitter = noise_level(raw, self.noise_measure)
            if np.isfinite(jitter):
                switch_cost *= max(1.0, jitter / self.noise_ref_m) ** self.noise_cost_exp
        runs = play_runs(match.period)
        if self.cues:  # wave 5 (src/models/cues/): per-frame costs for all six transitions
            from src.models import cues as cue_mod

            c_ha = c_ah = switch_cost
            if self.switch_cue_gain:
                c_ha, c_ah = self.switch_cue_costs(match, original, switch_cost)
            base = {"ha": c_ha, "ah": c_ah, "hd": self.dead_cost_s, "ad": self.dead_cost_s,
                    "dh": self.dead_cost_s, "da": self.dead_cost_s}
            costs, add = cue_mod.combine(self.cues, match, original, self, base, len(scores))
            if add is not None:
                scores = scores + add
            for start, stop in runs:
                codes[start:stop] = STATES[cue_mod.viterbi_costs(
                    scores[start:stop], *(costs[k][start:stop] for k in cue_mod.KINDS))]
        elif self.switch_cue_gain:  # local L2 (F203): per-frame switch costs; not with priors (__post_init__)
            from src.models import switch_cue  # lazy: switch_cue imports from this module

            c_ha, c_ah = self.switch_cue_costs(match, original, switch_cost)
            for start, stop in runs:
                codes[start:stop] = STATES[switch_cue.viterbi_var(
                    scores[start:stop], c_ha[start:stop], c_ah[start:stop], self.dead_cost_s)]
        elif self.priors:
            from src import priors
            g, typed_g = priors.exit_costs(self.priors, match.fps, self.prior_team_w,
                                           self.prior_dead_w, self.prior_typed, self.prior_floor)
            types = priors.shape_types(match) if typed_g else None
            for start, stop in runs:
                part = {k: v[start:stop] for k, v in types.items()} if types else None
                codes[start:stop] = STATES[priors.viterbi_aged(
                    scores[start:stop], switch_cost, self.dead_cost_s, g, typed_g, part)]
        else:
            for start, stop in runs:
                codes[start:stop] = STATES[viterbi(scores[start:stop], switch_cost,
                                                   self.dead_cost_s)]
        if self.onset_snap_s:
            codes = snap_to_cues(match, codes, runs, self.onset_snap_s, self.onset_cues)
        if self.restart_snap_s:
            codes = snap_to_restarts(match, codes, runs, self.restart_snap_s, self.restart_cues)
        if self.min_spell_s:
            codes = drop_short(codes, runs, to_frames(self.min_spell_s, match.fps))
        return codes
