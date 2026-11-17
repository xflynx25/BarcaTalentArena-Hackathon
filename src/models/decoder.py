"""Switch-timing decoder: Viterbi over {H, A, D} with per-frame evidence and switching costs (B3 + B7).

Evidence, per frame, label-free, from tracking only:
  team   c_home - c_away, where c_team = max(0, 1 - d_team / reach_m) and d_team is that team's
         closest player to the ball. Exactly 0 while the ball is in flight (nobody within reach) or
         missing, so possession stays with the last team that had it (the competition rule), and
         the decoder switches team only once the new team actually reaches the ball. That puts the
         switch at the receiver's control, not at the moment the ball gets closer to the receiver
         (findings F13: the label switches ~0.7 s after the nearest player changes). On ties
         (the whole flight) the switch goes to the latest frame. lag_s then shifts all team
         evidence later (or earlier, if negative).
  dead   a score in [-1, 1] from the share of frames with no ball and the players' mean speed over
         +/- dead_window_s (findings F29: walking players are the dead-ball cue that holds on every
         source), +1 while the ball is off the pitch and for out_s after (B2's cues, reused from
         src/models/deadball.py). D's evidence is dead_gain * (score - dead_bias). The cues lag
         the truth (findings F52: restarts come ~1.5 s late, onsets late on SkillCorner), so dead
         stretches are then started onset_lead_s earlier and ended restart_lead_s earlier.

Scores are per second (per-frame values / fps), so the costs are in seconds of full evidence and one
config means the same at every frame rate. A switch between H and A costs switch_cost_s, between a
team and D dead_cost_s. Decoded inside each half (play_runs); frames outside halves keep the
baseline's codes. onset_snap_s optionally starts each decoded D at an earlier onset cue
(src/models/onset.py, B12), then min_spell_s optionally runs B6's drop_short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.deadball import ball_out, extend, player_speed
from src.models.nearest import majority_smooth, nearest_team
from src.models.onset import snap_to_cues, snap_to_restarts
from src.models.postprocess import drop_short, play_runs
from src.models.timing import to_frames
from src.official import A, D, H
from src.schema import AWAY, HOME, Match

def team_distances(match: Match) -> tuple[np.ndarray, np.ndarray]:
    """(n,) distances in metres from the ball to the closest home and away player. NaN when the
    ball or every player of that team is missing."""
    dist = np.hypot(match.players[:, :, 0] - match.ball[:, None, 0],
                    match.players[:, :, 1] - match.ball[:, None, 1])
    dist[np.isnan(dist)] = np.inf
    out = []
    for team in (HOME, AWAY):
        cols = dist[:, match.slot_team == team]
        best = cols.min(axis=1) if cols.shape[1] else np.full(len(dist), np.inf)
        out.append(np.where(np.isfinite(best), best, np.nan))
    return out[0], out[1]


def window_mean(values: np.ndarray, valid: np.ndarray, half: int) -> np.ndarray:
    """Mean of values over +/- half frames, counting only valid frames. NaN where none is valid."""
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - half), np.minimum(len(values), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def contact(dist: np.ndarray, reach_m: float, shape: str = "linear") -> np.ndarray:
    """(n,) in [0, 1], how much a team at this distance reaches the ball; 0 where NaN.
    linear: 1 - d / reach_m, 0 beyond reach. gauss: exp(-(d / reach_m)^2 / 2), which decays
    smoothly instead of cutting off (C5): about the same at 1.2 reach_m, a tail to ~3 reach_m."""
    d = np.nan_to_num(dist, nan=np.inf)
    if shape == "linear":
        return np.clip(1.0 - d / reach_m, 0.0, 1.0)
    if shape == "gauss":
        return np.exp(-0.5 * np.square(d / reach_m))
    raise ValueError("reach_shape must be linear or gauss, not {!r}".format(shape))


def team_evidence(d_home: np.ndarray, d_away: np.ndarray, reach_m: float,
                  far_weight: float = 0.0, shape: str = "linear") -> np.ndarray:
    """(n,) in [-1, 1]: + home, - away, 0 when nobody is within reach (or the ball is missing).
    far_weight > 0 gives the nearer team some evidence out of reach too: far_weight times the
    distance gap in units of reach_m, clipped to [-1, 1] (B9). shape: see contact()."""
    c_home = contact(d_home, reach_m, shape)
    c_away = contact(d_away, reach_m, shape)
    out = c_home - c_away
    if far_weight:
        far = (c_home == 0) & (c_away == 0) & ~np.isnan(d_home) & ~np.isnan(d_away)
        out[far] = far_weight * np.clip((d_away[far] - d_home[far]) / reach_m, -1.0, 1.0)
    return out


def ball_jitter(match: Match, window_s: float = 0.4) -> float:
    """Label-free tracking-noise measure (B9): median distance, in metres, of the ball from its own
    mean over +/- window_s / 2, over in-play frames whose whole window has the ball. NaN if none."""
    half = max(1, to_frames(window_s / 2, match.fps))
    width = 2 * half + 1
    seen = ~np.isnan(match.ball[:, 0]) & (match.period > 0)
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(half, len(seen) - half)
    full = index[(count[index + half + 1] - count[index - half]) == width]
    if not len(full):
        return float("nan")
    out = []
    for axis in (0, 1):
        total = np.concatenate(([0.0], np.cumsum(np.where(seen, match.ball[:, axis], 0.0))))
        mean = (total[full + half + 1] - total[full - half]) / width
        out.append(np.abs(match.ball[full, axis] - mean))
    return float(np.median(np.concatenate(out)))


def ball_noise(match: Match, window_s: float = 0.4, band_mps=(1.0, 8.0), fit: str = "mean") -> float:
    """Dropout-robust ball-noise measure (B19, findings F81): ball_jitter's residual, but only on
    frames where the ball moves at band_mps (its speed over the window), as a 2-D distance in
    metres. ball_jitter's median mixes a still ball (exactly 0 on the tactical sources, about half
    their in-play frames) with a moving one, so hiding the calm stretches (ball dropout) moves it;
    within a speed band the residual stays put. fit="quad" measures the residual from a local
    quadratic (Savitzky-Golay) fit instead of the mean, per unit white noise. 2-D because
    SkillCorner rounds positions to 1 cm, which makes per-axis medians jump. NaN if no frame
    qualifies."""
    half = max(2 if fit == "quad" else 1, to_frames(window_s / 2, match.fps))
    width = 2 * half + 1
    j = np.arange(-half, half + 1, dtype=float)
    if fit == "quad":
        smooth = (3 * (3 * half ** 2 + 3 * half - 1) - 15 * j ** 2) / (
            (2 * half - 1) * (2 * half + 1) * (2 * half + 3))
    else:
        smooth = np.full(width, 1.0 / width)
    resid = -smooth
    resid[half] += 1.0  # x_t minus its local fit, a fixed linear filter (symmetric)
    unit = float(np.sqrt(np.sum(resid ** 2))) if fit == "quad" else 1.0
    seen = ~np.isnan(match.ball[:, 0]) & (match.period > 0)
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(half, len(seen) - half)
    full = index[(count[index + half + 1] - count[index - half]) == width]
    if len(full):
        step = match.ball[full + half] - match.ball[full - half]
        speed = np.hypot(step[:, 0], step[:, 1]) * match.fps / (2 * half)
        full = full[(speed >= band_mps[0]) & (speed < band_mps[1])]
    if not len(full):
        return float("nan")
    r = [np.convolve(np.where(seen, match.ball[:, axis], 0.0), resid, mode="same")[full] / unit
         for axis in (0, 1)]
    return float(np.median(np.hypot(r[0], r[1])))


NOISE_MEASURES = ("jitter", "noise")


def noise_level(match: Match, measure: str = "jitter") -> float:
    """The per-match ball-noise measure the noise rules scale by: "jitter" (ball_jitter, B9) or
    "noise" (ball_noise, B19: steady under dropout; pair it with its own noise_ref_m)."""
    if measure == "jitter":
        return ball_jitter(match)
    if measure == "noise":
        return ball_noise(match)
    raise ValueError("noise measure must be one of {}, not {!r}".format(NOISE_MEASURES, measure))


def player_jitter(match: Match, h_s: float = 0.2) -> float:
    """Label-free player-position noise measure (C5): median over players and in-play frames of
    |p(t) - (p(t - h) + p(t + h)) / 2|, h = h_s, in metres (C3's "jitter cm"). Real motion gives
    ~1.5-2 cm on every clean source; frame-to-frame tracking noise adds to it. NaN if none."""
    h = max(1, to_frames(h_s, match.fps))
    players = np.where((match.period > 0)[:, None, None], match.players, np.nan)
    if len(players) <= 2 * h:
        return float("nan")
    resid = players[h:-h] - 0.5 * (players[:-2 * h] + players[2 * h:])
    dist = np.hypot(resid[..., 0], resid[..., 1])
    dist = dist[np.isfinite(dist)]
    return float(np.median(dist)) if len(dist) else float("nan")


def dead_score(match: Match, half_s: float, speed_live: float, speed_dead: float,
               missing_weight: float, out_s: float, speed_span_s: float = 0.0) -> np.ndarray:
    """(n,) in [-1, 1], higher = more like dead time. Mixes the share of ball-missing frames and the
    players' mean speed (both over +/- half_s); +1 where the ball is off the pitch and for out_s
    after it (B2's cues, src/models/deadball.py; out_s < 0 turns that off)."""
    missing = np.isnan(match.ball[:, 0])
    share_missing = window_mean(missing.astype(float), np.ones(len(missing), bool),
                                to_frames(half_s, match.fps))
    mid, span = (speed_live + speed_dead) / 2, (speed_live - speed_dead) / 2
    speed = np.nan_to_num(player_speed(match, half_s, speed_span_s), nan=mid)
    slow = np.clip((mid - speed) / span, -1.0, 1.0)
    score = missing_weight * (2 * share_missing - 1) + (1 - missing_weight) * slow
    if out_s >= 0:
        score[extend(ball_out(match), to_frames(out_s, match.fps))] = 1.0
    return score


def forward_window(values: np.ndarray, frames: int, reduce) -> np.ndarray:
    """(n,) reduce (np.min or np.max) of values over [t, t + frames], clipped at the end. With
    np.min a high stretch keeps its start and ends `frames` earlier; with np.max it starts
    `frames` earlier and keeps its end."""
    if frames <= 0 or len(values) == 0:
        return values
    padded = np.concatenate((values, np.full(frames, values[-1])))
    return reduce(np.lib.stride_tricks.sliding_window_view(padded, frames + 1), axis=1)


def shift(values: np.ndarray, frames: int) -> np.ndarray:
    """values moved `frames` later (earlier if negative), zero-filled."""
    if frames == 0:
        return values
    out = np.zeros_like(values)
    if frames > 0:
        out[frames:] = values[:-frames]
    else:
        out[:frames] = values[-frames:]
    return out


# Scores and costs are snapped to multiples of 2**-GRID_BITS before decoding (findings F87).
# Upstream maths (BLAS matmul, SIMD exp/log) differs in the last bit between machines (Apple
# Accelerate vs OpenBLAS, AVX2 vs AVX-512 kernels), and the path has near-ties that a 1-ulp change
# flips. On the grid every path sum is an exact integer multiple of 2**-GRID_BITS (|sum| < 2**30
# here, far inside float64's 53 bits), so the decoder's arithmetic is exact and order-free, and
# only a value landing within ~1 ulp of a rounding boundary (~1e-9 per value) can still differ.
GRID_BITS = 20


def to_grid(x):
    """x rounded to the nearest multiple of 2**-GRID_BITS (exact: scaling by a power of 2)."""
    return np.round(np.asarray(x, dtype=np.float64) * 2.0 ** GRID_BITS) / 2.0 ** GRID_BITS


def viterbi(scores: np.ndarray, switch_cost: float, dead_cost: float) -> np.ndarray:
    """(n,) best state path (0=H, 1=A, 2=D) for (n, 3) per-frame scores. A change between H and A
    costs switch_cost, between a team and D dead_cost. On ties a change happens as late as possible
    (the recursion prefers arriving from another state, and backtracking follows it). Scores and
    costs go on the 2**-GRID_BITS grid first, so the path is the same on every machine."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    scores = to_grid(scores)
    switch_cost, dead_cost = float(to_grid(switch_cost)), float(to_grid(dead_cost))
    sh, sa, sd = (scores[:, k].tolist() for k in range(3))
    back = bytearray(3 * n)
    vh, va, vd = sh[0], sa[0], sd[0]
    for t in range(1, n):
        # H
        best, arg = vh, 0
        x = va - switch_cost
        if x >= best:
            best, arg = x, 1
        x = vd - dead_cost
        if x >= best:
            best, arg = x, 2
        nh, bh = best + sh[t], arg
        # A
        best, arg = va, 1
        x = vh - switch_cost
        if x >= best:
            best, arg = x, 0
        x = vd - dead_cost
        if x >= best:
            best, arg = x, 2
        na, ba = best + sa[t], arg
        # D
        best, arg = vd, 2
        x = vh - dead_cost
        if x >= best:
            best, arg = x, 0
        x = va - dead_cost
        if x >= best:
            best, arg = x, 1
        nd, bd = best + sd[t], arg
        vh, va, vd = nh, na, nd
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
    state = int(np.argmax([vh, va, vd]))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


# B3/B7's settings (findings F45), before B9 retuned the defaults (F52, F54). `learned` (B10) and
# `spells` (B8) were fit and tuned on these scores, so they pin them to keep their measured results.
B3_SETTINGS = dict(lag_s=0.2, dead_cost_s=0.5, onset_lead_s=0.0, restart_lead_s=0.0,
                   noise_ref_m=0.0)


@dataclass(kw_only=True)
class SwitchDecoder:
    """Viterbi decoder over {H, A, D}. Durations in seconds, distances in metres, scaled per match.
    The defaults are the tuned pick (findings F45, retuned in F52 and F54: mean over sources,
    BAR-ATH-like ball). On jittery tracking (ball_jitter above noise_ref_m: SkillCorner's broadcast
    positions) reach and switch cost grow with the jitter; on tactical tracking nothing is scaled."""

    reach_m: float = 1.5  # a player this close to the ball (or closer) counts as reaching it
    lag_s: float = 0.5  # shift team evidence this much later (findings F13, B7; F52)
    switch_cost_s: float = 0.15  # cost of an H<->A switch, in seconds of full evidence
    dead_cost_s: float = 1.0  # cost of a team<->D switch (findings F54)
    dead_gain: float = 1.0  # weight of the dead score against the team evidence
    dead_bias: float = 0.0  # dead score above this favours D
    dead_window_s: float = 2.0  # half-width of the window for the dead cues
    missing_weight: float = 0.25  # share of the dead score from ball-missing frames (rest: speed)
    speed_live: float = 2.2  # mean player speed typical of live play (m/s, findings F29)
    speed_dead: float = 1.0  # ... and of dead time
    out_s: float = 2.0  # the ball off the pitch, and this long after, counts as dead; < 0: off
    min_spell_s: float | None = 1.0  # then drop segments shorter than this (B6); None/0 = off
    onset_lead_s: float = 1.0  # start dead stretches this much earlier (findings F52); 0 = off
    restart_lead_s: float = 2.0  # end dead stretches this much earlier (findings F52); 0 = off
    far_weight: float = 0.0  # evidence for the nearer team when nobody is within reach (F53: hurts)
    noise_ref_m: float = 0.02  # ball_jitter above this scales reach and switch cost up (F54); 0 = off
    noise_reach_exp: float = 1.0  # reach_m *= (jitter / noise_ref_m) ** this (only when > 1)
    noise_cost_exp: float = 2.0  # switch_cost_s *= (jitter / noise_ref_m) ** this
    speed_span_s: float = 0.0  # player speed over a span this long, not frame steps (C5/F72); 0 = off
    reach_shape: str = "linear"  # "gauss": team evidence decays smoothly with distance (C5)
    pjitter_ref_m: float = 0.0  # player_jitter above this scales reach up (C5); 0 = off
    pjitter_reach_exp: float = 0.5  # reach_m *= max(1, player_jitter / pjitter_ref_m) ** this
    onset_snap_s: float = 0.0  # start decoded D at the earliest onset cue this far before it (B12)
    onset_cues: str = "out+gap"  # which cues (src/models/onset.py), joined by "+"
    restart_snap_s: float = 0.0  # end decoded D at the latest restart cue this far before its
    # end (B24, findings F94-F95); 0 = off
    restart_cues: str = "gapend+go"  # which restart cues (onset.RESTART_CUES), joined by "+"

    def noise_scale(self, match: Match) -> float:
        """max(1, ball_jitter / noise_ref_m), or 1 when the scaling is off or jitter unknown."""
        if not self.noise_ref_m:
            return 1.0
        jitter = ball_jitter(match)
        return max(1.0, jitter / self.noise_ref_m) if np.isfinite(jitter) else 1.0

    def scores(self, match: Match, scale: float = 1.0) -> np.ndarray:
        """(n, 3) per-second evidence for H, A, D."""
        d_home, d_away = team_distances(match)
        reach = self.reach_m * scale ** self.noise_reach_exp
        if self.pjitter_ref_m:
            jitter = player_jitter(match)
            if np.isfinite(jitter):
                reach *= max(1.0, jitter / self.pjitter_ref_m) ** self.pjitter_reach_exp
        team = shift(team_evidence(d_home, d_away, reach, self.far_weight, self.reach_shape),
                     int(round(self.lag_s * match.fps)))
        dead = dead_score(match, self.dead_window_s, self.speed_live, self.speed_dead,
                          self.missing_weight, self.out_s, self.speed_span_s)
        dead = forward_window(dead, to_frames(self.onset_lead_s, match.fps), np.max)
        dead = forward_window(dead, to_frames(self.restart_lead_s, match.fps), np.min)
        return np.stack([team, -team, self.dead_gain * (dead - self.dead_bias)], axis=1) / match.fps

    def predict(self, match: Match) -> np.ndarray:
        codes = majority_smooth(nearest_team(match), to_frames(2.0, match.fps))
        scale = self.noise_scale(match)
        scores = self.scores(match, scale)
        switch_cost = self.switch_cost_s * scale ** self.noise_cost_exp
        states = np.array([H, A, D], dtype=np.int8)
        runs = play_runs(match.period)
        for start, stop in runs:
            codes[start:stop] = states[viterbi(scores[start:stop], switch_cost,
                                               self.dead_cost_s)]
        if self.onset_snap_s:
            codes = snap_to_cues(match, codes, runs, self.onset_snap_s, self.onset_cues)
        if self.restart_snap_s:
            codes = snap_to_restarts(match, codes, runs, self.restart_snap_s, self.restart_cues)
        if self.min_spell_s:
            codes = drop_short(codes, runs, to_frames(self.min_spell_s, match.fps))
        return codes
