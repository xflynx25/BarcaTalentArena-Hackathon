"""Align a second recording of the match with the reference timeline, and carry its ball across
(backlog V4, findings F142-F143, F160-F161).

BAR-ATH was recorded twice (official DATA.md): the tactical camera (the reference: row i of the
submission is its frame i) and the TV broadcast, with its own tracking (164,488 records, 51 slots),
its own frame count, cuts, replays and zooms, and no mapping between the two. Where the TV camera
sees a ball the tactical one lost, the broadcast ball, mapped onto tactical frames, is evidence the
model can use. Everything here is label-free and deterministic: it reads positions in pitch metres
only, so it runs the same on BAR-ATH (owner's machine) and on simulated second recordings of
external matches (src/ingest/degrade.py: second_recording).

    alignment = align.align(tactical, broadcast)    # Alignment: other frame -> ref frame or -1
    fused = align.attach(tactical, broadcast, alignment)   # Match with ball2 (n_ref, 2)
    align.health(tactical, broadcast, alignment, fused)    # aggregates, for the owner check

How (all costs in metres):
  1. Both recordings are sampled at RATE_HZ (5 Hz). A pair cost compares an other-recording sample
     with a reference sample: each player the other recording sees is matched to the reference's
     nearest player of the same team (no identities needed: tracklet ids differ), the ball to the
     ball where both see it (weight BALL_WEIGHT), each term capped at CAP_M, averaged.
  2. Candidate offsets: over sliding WINDOW_S windows of the other recording, the mean squared
     distance between the two balls at every offset into the reference (one FFT correlation per
     window, masked to the frames where both see the ball); each window keeps its TOP_K minima.
     Offsets that win GLOBAL_VOTES windows or more are "global" (the live programme: a TV feed runs
     in real time, so between edits its offset is constant); the others stay local to their window
     (replays, which repeat earlier time).
  3. A Viterbi over the other recording's samples picks one offset per sample, or "unmatched":
     staying on an offset is free, a one-sample step costs DRIFT_COST (clock drift), a jump
     JUMP_COST, entering or leaving "unmatched" CUT_COST; the emission is the pair cost, NEUTRAL_M
     where a sample has no data, plus LOCAL_COST on an offset that isn't global, and "unmatched"
     always costs NEUTRAL_M. So a stretch is matched only where its positions agree better than
     NEUTRAL_M on average, and a replay only where it fits clearly better than the live feed.
  4. Runs of samples on one offset (one-sample steps allowed) are segments. Each segment's offset is
     refined at full frame rate (FINE_BLOCK_S blocks, +/- FINE_RANGE_S), a common clock drift is fit
     on the long segments (>= DRIFT_MIN_S), and each segment's intercept is refit with it; segments
     whose intercepts lie within CLUSTER_S share one joint fit (the live feed keeps its offset
     between edits) unless their own fit is clearly cheaper. The map is evaluated in integers
     (milli-frames, ppm), so it is identical on every machine.
  5. Samples whose cost at the final map (median of 3) exceeds REJECT_M are unmatched, and so are
     segments shorter than MIN_SEGMENT_S or with a median cost above ACCEPT_M.
  Costs are rounded to 1e-4 m before any comparison, FFT scores to 1e-3, so platform-level float
  differences can't change a decision (the submission must regenerate byte for byte).

Orientation: the other recording may use the opposite pitch direction (flip) or the opposite team
labels (swap). align() tries the flip when the unflipped balls give fewer confident windows, and the
team swap when its median pair cost at the global offsets is lower; Alignment records both.

attach() writes Match.ball2: for every reference frame some matched other frame maps to (the lowest
cost one if several do, e.g. a replay and the live feed), that frame's ball in reference
coordinates; reference frames between two consecutive mapped frames of one segment are interpolated
(an other recording at a lower frame rate), and the ball is shifted by the local median difference
between the two balls where both see it (BIAS_S, at least BIAS_MIN frames), since the two
calibrations disagree a little. NaN where nothing maps or the other recording has no ball.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from src.schema import AWAY, HOME, Match

RATE_HZ = 5.0
CAP_M = 5.0
BALL_WEIGHT = 3.0
MIN_TERMS = 3.0  # players + BALL_WEIGHT x ball terms a pair needs, else "no data"
KMAX = 16  # at most this many seen players kept per team and frame (slot order)
FAR = 1e6  # unseen players sit this far away (a sentinel instead of NaN: no masking in the costs)
WINDOW_S = 30.0
HOP_S = 15.0
TOP_K = 3
MIN_JOINT_S = 3.0  # seconds of both balls seen a window offset needs
MIN_SPREAD_M = 3.0  # a window's ball must move this much (std, m), or its minima are ambiguous
CAND_RMS_M = 6.0  # a window's minimum counts only below this root mean squared distance
CONFIDENT_RMS_M = 2.0  # ... and is "confident" below this (orientation check)
EXCLUDE_S = 2.0  # minima of one window at least this far apart
GLOBAL_VOTES = 3  # windows a cluster of top-1 offsets needs to be "global"
LOCAL_PAD_S = 0.0  # a window's candidates are tried this far beyond it
NEUTRAL_M = 2.0
DRIFT_COST = 2.0  # metres x samples
JUMP_COST = 20.0
CUT_COST = 5.0
LOCAL_COST = 0.5  # metres per sample on an offset that isn't global (a replay must fit clearly
# better than the live feed, and doesn't outlive a cut where the two fit alike)
FINE_BLOCK_S = 8.0
FINE_RANGE_S = 0.2
FINE_EVERY = 2  # the fine search reads every this many coarse samples
DRIFT_MIN_S = 60.0
CLUSTER_S = 0.5  # segments whose intercepts are this close (seconds) share one fit ...
JOINT_MARGIN_M = 0.05  # ... unless the segment's own fit costs this much less
MAX_DRIFT_PPM = 2000
REJECT_M = 2.0
ACCEPT_M = 2.5
MIN_SEGMENT_S = 2.0
MAX_BLIND_S = 60.0  # a segment is split where the other recording has no data this long, and its
# blind ends are trimmed: the map stays through cuts and close-ups, not across halftime
BIAS_S = 30.0
BIAS_MIN = 25
NONE = None  # the "unmatched" Viterbi state


@dataclass
class Alignment:
    """Other-recording frame -> reference frame (-1 = unmatched), and how it was found."""
    frame: np.ndarray  # (n_other,) int64
    cost: np.ndarray  # (n_other,) float64: pair cost at the map (m), NaN where unmatched/no data
    segments: list = field(default_factory=list)  # dicts: start, end (other frames), a_milli,
    # drift_ppm, cost (median, m), backward (maps before the previous segment's end: a replay)
    flip: bool = False
    swap: bool = False
    ref_fps: int = 25
    other_fps: int = 25
    drift_ppm: int = 0

    @property
    def matched(self) -> np.ndarray:
        return self.frame >= 0


# -- recordings and pair costs ---------------------------------------------------------------------

def team_points(match: Match, kmax: int = KMAX) -> np.ndarray:
    """(n, 2, k, 2) seen player positions per team (HOME, AWAY) in slot order, NaN padded; k is
    the most players either team shows on one frame, at most kmax."""
    n = match.n_frames
    seen = ~np.isnan(match.players[..., 0])
    counts = [int(seen[:, match.slot_team == team].sum(axis=1).max(initial=0)) for team in (HOME, AWAY)]
    k = max(1, min(kmax, max(counts)))
    out = np.full((n, 2, k, 2), np.nan)
    for team in (HOME, AWAY):
        cols = np.flatnonzero(match.slot_team == team)
        if not len(cols):
            continue
        sub = match.players[:, cols]
        order = np.argsort(~seen[:, cols], axis=1, kind="stable")[:, :k]
        picked = np.take_along_axis(sub, order[..., None], axis=1)
        out[:, team, :picked.shape[1]] = picked
    return out


@dataclass
class Recording:
    """A match's positions in the shape the costs read, optionally flipped / team-swapped."""
    fps: int
    points: np.ndarray  # (n, 2, k, 2), NaN where unseen
    ball: np.ndarray  # (n, 2)

    def __post_init__(self):
        self.seen = ~np.isnan(self.points[..., 0])  # (n, 2, k)
        self.far_ref = np.where(self.seen[..., None], self.points, FAR)  # as the reference
        self.far_other = np.where(self.seen[..., None], self.points, -FAR)  # as the other

    @classmethod
    def of(cls, match: Match, flip: bool = False, swap: bool = False) -> "Recording":
        points, ball = team_points(match), match.ball.astype(np.float64).copy()
        if flip:
            size = np.array(match.pitch, dtype=np.float64)
            points, ball = size - points, size - ball
        if swap:
            points = points[:, ::-1]
        return cls(match.fps, points, ball)

    def has_data(self) -> np.ndarray:
        players = np.sum(~np.isnan(self.points[..., 0]), axis=(1, 2))
        return players + BALL_WEIGHT * ~np.isnan(self.ball[:, 0]) >= MIN_TERMS


def pair_cost(ref: Recording, other: Recording, ri: np.ndarray, oi: np.ndarray,
              chunk: int = 4096) -> np.ndarray:
    """(P,) cost of matching other frames oi with reference frames ri (m, rounded to 1e-4): each
    player the other recording sees to the reference's nearest same-team player, the ball to the
    ball, each capped at CAP_M, averaged with the ball weighted BALL_WEIGHT. NaN where the pair has
    fewer than MIN_TERMS terms."""
    ri, oi = np.asarray(ri, dtype=np.int64), np.asarray(oi, dtype=np.int64)
    out = np.full(len(ri), np.nan)
    for lo in range(0, len(ri), chunk):
        r_idx, o_idx = ri[lo:lo + chunk], oi[lo:lo + chunk]
        o, r = other.far_other[o_idx], ref.far_ref[r_idx]  # (p, 2, k, 2), unseen far away
        dx = o[:, :, :, None, 0] - r[:, :, None, :, 0]
        dy = o[:, :, :, None, 1] - r[:, :, None, :, 1]
        nearest = np.sqrt(np.min(dx * dx + dy * dy, axis=3))
        seen = other.seen[o_idx]
        terms = np.where(seen, np.minimum(nearest, CAP_M), 0.0)
        n_players = seen.sum(axis=(1, 2)).astype(np.float64)
        total = terms.sum(axis=(1, 2))
        db = other.ball[o_idx] - ref.ball[r_idx]
        bd = np.sqrt(db[:, 0] * db[:, 0] + db[:, 1] * db[:, 1])
        has = ~np.isnan(bd)
        total = total + BALL_WEIGHT * np.where(has, np.minimum(bd, CAP_M), 0.0)
        weight = n_players + BALL_WEIGHT * has
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:lo + chunk] = np.where(weight >= MIN_TERMS, np.round(total / weight, 4), np.nan)
    return out


def sample_frames(n_frames: int, fps: int, rate: float = RATE_HZ) -> np.ndarray:
    """Frame index of every coarse sample (one per 1 / rate seconds)."""
    m = int(math.floor((n_frames - 1) * rate / fps)) + 1
    return np.minimum(np.round(np.arange(m) * fps / rate).astype(np.int64), n_frames - 1)


# -- candidate offsets -----------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def ball_candidates(ref_ball: np.ndarray, other_ball: np.ndarray, rate: float = RATE_HZ) -> list:
    """Per window of the other recording's coarse ball: (start, stop, [(offset, rms), ...]), the
    TOP_K offsets (reference sample - other sample) with the smallest mean squared distance between
    the two balls over the samples both see, at least EXCLUDE_S apart, rms below CAND_RMS_M."""
    win, hop = int(round(WINDOW_S * rate)), int(round(HOP_S * rate))
    min_joint, exclude = int(round(MIN_JOINT_S * rate)), int(round(EXCLUDE_S * rate))
    mr, mo = len(ref_ball), len(other_ball)
    size = _next_pow2(mr + win)
    r_mask = ~np.isnan(ref_ball[:, 0])
    r = np.where(r_mask[:, None], ref_ball, 0.0)
    f_m = np.fft.rfft(r_mask.astype(np.float64), size)
    f_r2 = np.fft.rfft(r_mask * (r[:, 0] ** 2 + r[:, 1] ** 2), size)
    f_rx, f_ry = np.fft.rfft(r[:, 0], size), np.fft.rfft(r[:, 1], size)
    # offset k = ref sample - window sample: correlation index k >= 0 at k, k < 0 at size + k
    ks = np.concatenate([np.arange(-(win - 1), 0), np.arange(mr)])
    out = []
    for start in range(0, max(1, mo - win // 2), hop):
        stop = min(mo, start + win)
        seg = other_ball[start:stop]
        mask = ~np.isnan(seg[:, 0])
        if mask.sum() < min_joint or np.sqrt(np.nanvar(seg[:, 0]) + np.nanvar(seg[:, 1])) < MIN_SPREAD_M:
            out.append((start, stop, []))
            continue
        b = np.where(mask[:, None], seg, 0.0)
        a_m = np.conj(np.fft.rfft(mask.astype(np.float64), size))
        a_b2 = np.conj(np.fft.rfft(mask * (b[:, 0] ** 2 + b[:, 1] ** 2), size))
        a_x, a_y = np.conj(np.fft.rfft(b[:, 0], size)), np.conj(np.fft.rfft(b[:, 1], size))
        count = np.round(np.fft.irfft(a_m * f_m, size))
        sq = np.fft.irfft(a_b2 * f_m + a_m * f_r2 - 2.0 * (a_x * f_rx + a_y * f_ry), size)
        count, sq = count[ks % size], sq[ks % size]
        ok = count >= min_joint
        msd = np.where(ok, np.round(np.maximum(sq, 0.0) / np.maximum(count, 1.0), 3), np.inf)
        order = np.lexsort((ks, msd))
        picked = []  # window sample j is other sample start + j and reference sample j + k
        for idx in order:
            if not np.isfinite(msd[idx]) or msd[idx] > CAND_RMS_M ** 2 or len(picked) == TOP_K:
                break
            d = int(ks[idx]) - start
            if all(abs(d - p) >= exclude for p, _ in picked):
                picked.append((d, float(np.sqrt(msd[idx]))))
        out.append((start, stop, picked))
    return out


def global_offsets(windows: list) -> list[int]:
    """Offsets of the clusters of windows' best offsets (consecutive values <= 2 apart) that won at
    least GLOBAL_VOTES windows: every offset in such a cluster, and its neighbours."""
    best = sorted(c[0][0] for _, _, c in windows if c)
    clusters, current = [], []
    for value in best:
        if current and value - current[-1] > 2:
            clusters.append(current)
            current = []
        current.append(value)
    if current:
        clusters.append(current)
    out = set()
    for cluster in clusters:
        if len(cluster) >= GLOBAL_VOTES:
            for value in set(cluster):
                out.update((value - 1, value, value + 1))
    return sorted(out)


def confident_windows(windows: list) -> int:
    return sum(1 for _, _, c in windows if c and c[0][1] <= CONFIDENT_RMS_M)


# -- the Viterbi over offsets ----------------------------------------------------------------------

def candidate_sets(windows: list, glob: list[int], mo: int, mr: int, rate: float = RATE_HZ) -> list:
    """Per other sample, the sorted offsets to try: the global ones plus the candidates (+/- 1) of
    every window within LOCAL_PAD_S, kept where the reference sample exists."""
    pad = int(round(LOCAL_PAD_S * rate))
    local = [set() for _ in range(mo)]
    for start, stop, cands in windows:
        lo, hi = max(0, start - pad), min(mo, stop + pad)
        values = {k + s for k, _ in cands for s in (-1, 0, 1)}
        for j in range(lo, hi):
            local[j].update(values)
    out = []
    base = set(glob)
    for j in range(mo):
        out.append(sorted(d for d in base | local[j] if 0 <= j + d < mr))
    return out


def viterbi_offsets(cands: list, emis: list) -> list:
    """One offset (or NONE) per sample minimising emissions + transition costs (module docstring).
    Ties go to: stay, drift, unmatched, jump, then the smaller offset."""
    prev: dict = {}
    prev_none = 0.0
    back = []
    for j, (cset, e) in enumerate(zip(cands, emis)):
        if prev:
            best_d = min(prev, key=lambda d: (prev[d], d))
            best_v = prev[best_d]
        else:
            best_d, best_v = None, math.inf
        cur, ptr = {}, {}
        for d, cost in zip(cset, e):
            if j == 0:
                options = [(0.0, 0, d)]
            else:
                options = []
                if d in prev:
                    options.append((prev[d], 0, d))
                for dd in (d - 1, d + 1):
                    if dd in prev:
                        options.append((prev[dd] + DRIFT_COST, 1, dd))
                options.append((prev_none + CUT_COST, 2, NONE))
                if best_d is not None:
                    options.append((best_v + JUMP_COST, 3, best_d))
            value, _, source = min(options, key=lambda o: (o[0], o[1], -1 if o[2] is None else o[2]))
            cur[d] = value + cost
            ptr[d] = source
        none_options = [(prev_none, 0, NONE)] if j else [(0.0, 0, NONE)]
        if best_d is not None:
            none_options.append((best_v + CUT_COST, 1, best_d))
        value, _, source = min(none_options, key=lambda o: (o[0], o[1]))
        ptr[NONE] = source
        back.append(ptr)
        prev, prev_none = cur, value + NEUTRAL_M
    # trace back
    if prev:
        best_d = min(prev, key=lambda d: (prev[d], d))
        state = best_d if prev[best_d] < prev_none else NONE
    else:
        state = NONE
    path = [None] * len(cands)
    for j in range(len(cands) - 1, -1, -1):
        path[j] = state
        state = back[j][state]
    return path


# -- refinement ------------------------------------------------------------------------------------

def _segments(path: list, data: np.ndarray, max_blind: int) -> list[tuple[int, int]]:
    """[start, stop) runs of samples on one offset (one-sample steps allowed), split where `data`
    is False for more than max_blind samples, blind ends trimmed."""
    runs, start = [], None
    for j, d in enumerate(path + [NONE]):
        if start is not None and (d is NONE or abs(d - path[j - 1]) > 1):
            runs.append((start, j))
            start = None
        if d is not NONE and start is None:
            start = j
    out = []
    for s0, s1 in runs:
        seen = np.flatnonzero(data[s0:s1]) + s0
        if not seen.size:
            continue
        cuts = np.flatnonzero(np.diff(seen) > max_blind)
        for a, b in zip(np.concatenate([[0], cuts + 1]), np.concatenate([cuts, [len(seen) - 1]])):
            out.append((int(seen[a]), int(seen[b]) + 1))
    return out


def map_frames(frames: np.ndarray, a_milli: int, drift_ppm: int, ref_fps: int,
               other_fps: int) -> np.ndarray:
    """Reference frame of other frames b: round(a + b * ref_fps / other_fps * (1 + ppm 1e-6)), in
    integers (a in milli-frames), so every machine gives the same frames."""
    b = np.asarray(frames, dtype=np.int64)
    den = other_fps * 1_000_000
    num = a_milli * other_fps * 1000 + b * ref_fps * (1_000_000 + drift_ppm)
    return (num + den // 2) // den


def _fit_line(x: np.ndarray, y: np.ndarray, slope: float | None) -> tuple[float, float]:
    """y = a + s x by least squares (s fixed when given), dropping residuals over 1.5 frames once."""
    keep = np.ones(len(x), bool)
    a, s = float(np.median(y - (slope or 0.0) * x)), slope or 0.0
    for _ in range(3):
        if keep.sum() == 0:
            break
        if slope is None and keep.sum() >= 2 and np.ptp(x[keep]) > 0:
            s, a = np.polyfit(x[keep], y[keep], 1)
        else:
            a = float(np.median(y[keep] - s * x[keep]))
        keep = np.abs(y - a - s * x) <= 1.5
    return float(a), float(s)


def _intercept(seg: dict, slope: float) -> float:
    """A segment's own intercept at this slope: its blocks' fit, else its coarse offset."""
    if len(seg["x"]):
        return _fit_line(seg["x"], seg["y"], slope)[0]
    return seg["fallback"]


def _clusters(segs: list, slope: float, tolerance: float) -> list[list[int]]:
    """Indices of segments grouped by own intercept at this slope (consecutive ones <= tolerance
    frames apart), in intercept order."""
    order = sorted(range(len(segs)), key=lambda i: (_intercept(segs[i], slope), i))
    out, current, last = [], [], None
    for i in order:
        a = _intercept(segs[i], slope)
        if current and a - last > tolerance:
            out.append(current)
            current = []
        current.append(i)
        last = a
    if current:
        out.append(current)
    return out


def _shared_slope(groups: list, start: float) -> float:
    """One slope for several (x, y) groups with their own intercepts (least squares on the
    within-group deviations), dropping residuals over 1.5 frames, twice."""
    slope = start
    keeps = [np.ones(len(x), bool) for x, _ in groups]
    for _ in range(3):
        num = den = 0.0
        for (x, y), keep in zip(groups, keeps):
            if keep.sum() < 2:
                continue
            xm, ym = x[keep].mean(), y[keep].mean()
            num += float(np.sum((x[keep] - xm) * (y[keep] - ym)))
            den += float(np.sum((x[keep] - xm) ** 2))
        if den <= 0:
            break
        slope = num / den
        for k, ((x, y), keep) in enumerate(zip(groups, keeps)):
            a = float(np.median(y[keep] - slope * x[keep])) if keep.any() else 0.0
            keeps[k] = np.abs(y - a - slope * x) <= 1.5
    return slope


def _block_offsets(ref: Recording, other: Recording, frames: np.ndarray, guess: np.ndarray,
                   block: int, fine: int) -> list[tuple[float, float, float]]:
    """(other frame, reference frame, cost) at the best fine shift for each block of `frames`
    (every coarse sample's frame in a segment, with its coarse guess)."""
    out = []
    shifts = np.arange(-fine, fine + 1)
    for lo in range(0, len(frames), block):
        f, g = frames[lo:lo + block], guess[lo:lo + block]
        ri = (g[None, :] + shifts[:, None]).ravel()
        oi = np.broadcast_to(f, (len(shifts), len(f))).ravel()
        valid = (ri >= 0) & (ri < len(ref.ball))
        cost = np.full(len(ri), np.nan)
        cost[valid] = pair_cost(ref, other, ri[valid], oi[valid])
        cost = cost.reshape(len(shifts), len(f))
        seen = ~np.isnan(cost)
        n = seen.sum(axis=1)
        if n.max() < 3:
            continue
        mean = np.where(n >= max(3, n.max() // 2), np.round(np.nansum(cost, axis=1) / np.maximum(n, 1), 4),
                        np.inf)
        best = int(np.argmin(mean))  # first minimum: the smallest shift on ties
        if not np.isfinite(mean[best]):
            continue
        sub = 0.0  # parabolic sub-frame refinement of the minimum
        if 0 < best < len(shifts) - 1 and np.isfinite(mean[best - 1]) and np.isfinite(mean[best + 1]):
            curve = mean[best - 1] - 2.0 * mean[best] + mean[best + 1]
            if curve > 0:
                sub = round(float(np.clip(0.5 * (mean[best - 1] - mean[best + 1]) / curve, -0.5, 0.5)), 3)
        centre = int(len(f) // 2)
        out.append((float(f[centre]), float(g[centre] + shifts[best]) + sub, float(mean[best])))
    return out


def align(ref: Match, other: Match, rate: float = RATE_HZ, orientation: str = "auto") -> Alignment:
    """Map every frame of `other` (a second recording of the same match) to a frame of `ref` or -1.
    orientation: "auto" (try the flip and the team swap), or "same" (neither)."""
    ref_frames = sample_frames(ref.n_frames, ref.fps, rate)
    oth_frames = sample_frames(other.n_frames, other.fps, rate)
    mr, mo = len(ref_frames), len(oth_frames)
    ref_rec = Recording.of(ref)
    ref_ball = ref_rec.ball[ref_frames]

    flip = False
    other_rec = Recording.of(other)
    windows = ball_candidates(ref_ball, other_rec.ball[oth_frames], rate)
    searched = sum(1 for _, _, c in windows if c) or 1
    if orientation == "auto" and confident_windows(windows) < 0.5 * searched:
        flipped = Recording.of(other, flip=True)
        alt = ball_candidates(ref_ball, flipped.ball[oth_frames], rate)
        if confident_windows(alt) > confident_windows(windows):
            flip, other_rec, windows = True, flipped, alt
    glob = global_offsets(windows)
    cands = candidate_sets(windows, glob, mo, mr, rate)

    swap = False
    if orientation == "auto" and glob:
        top = max(set(c[0][0] for _, _, c in windows if c),
                  key=lambda d: (sum(1 for _, _, c in windows if c and abs(c[0][0] - d) <= 1), -d))
        js = np.flatnonzero((np.arange(mo) + top >= 0) & (np.arange(mo) + top < mr))[::5]
        swapped = Recording.of(other, flip=flip, swap=True)
        plain = pair_cost(ref_rec, other_rec, ref_frames[js + top], oth_frames[js])
        crossed = pair_cost(ref_rec, swapped, ref_frames[js + top], oth_frames[js])
        if np.isfinite(plain).any() and np.isfinite(crossed).any() and \
                np.nanmedian(crossed) + 0.5 < np.nanmedian(plain):
            swap, other_rec = True, swapped

    # emissions
    js = np.repeat(np.arange(mo), [len(c) for c in cands])
    ds = np.concatenate([np.asarray(c, dtype=np.int64) for c in cands]) if js.size else np.zeros(0, np.int64)
    cost = pair_cost(ref_rec, other_rec, ref_frames[js + ds], oth_frames[js]) if js.size else np.zeros(0)
    cost = np.where(np.isnan(cost), NEUTRAL_M, cost)
    if js.size:
        cost = cost + LOCAL_COST * ~np.isin(ds, np.asarray(glob, dtype=np.int64))
    bounds = np.concatenate([[0], np.cumsum([len(c) for c in cands])])
    emis = [cost[bounds[j]:bounds[j + 1]].tolist() for j in range(mo)]
    path = viterbi_offsets(cands, emis)

    # refine each segment at full frame rate
    ratio = ref.fps / other.fps
    block = max(1, int(round(FINE_BLOCK_S * rate)))
    fine = max(1, int(math.ceil(FINE_RANGE_S * ref.fps)))
    segs = []
    data = other_rec.has_data()[oth_frames]
    for s0, s1 in _segments(path, data, int(round(MAX_BLIND_S * rate))):
        if (s1 - s0) / rate < MIN_SEGMENT_S:
            continue
        js = np.arange(s0, s1)
        guess = ref_frames[js + np.array([path[j] for j in js])]
        blocks = _block_offsets(ref_rec, other_rec, oth_frames[js][::FINE_EVERY],
                                guess[::FINE_EVERY], max(1, block // FINE_EVERY), fine)
        segs.append({"s0": s0, "s1": s1, "blocks": blocks})
    # intercepts: each segment's own, then one per cluster of segments with close intercepts (the
    # live feed keeps its offset between edits, so short pieces borrow the long ones' precision);
    # the clock drift is one slope shared by the clusters that span DRIFT_MIN_S or more
    for s in segs:
        js = np.arange(s["s0"], s["s1"])
        s["x"] = np.array([b[0] for b in s["blocks"]])
        s["y"] = np.array([b[1] for b in s["blocks"]])
        s["fallback"] = float(ref_frames[min(mr - 1, s["s0"] + int(np.median([path[j] for j in js])))]
                              - ratio * oth_frames[s["s0"]])
    drift_ppm = 0
    clusters = _clusters(segs, ratio, CLUSTER_S * ref.fps)
    spans = [(np.concatenate([segs[i]["x"] for i in c]), np.concatenate([segs[i]["y"] for i in c]))
             for c in clusters]
    spans = [(x, y) for x, y in spans if len(x) >= 3 and np.ptp(x) / other.fps >= DRIFT_MIN_S]
    if spans:
        fitted = _shared_slope(spans, ratio)
        drift_ppm = int(np.clip(round((fitted / ratio - 1.0) * 1e6), -MAX_DRIFT_PPM, MAX_DRIFT_PPM))
    slope = ratio * (1.0 + drift_ppm * 1e-6)
    for members in _clusters(segs, slope, CLUSTER_S * ref.fps):
        x = np.concatenate([segs[i]["x"] for i in members])
        y = np.concatenate([segs[i]["y"] for i in members])
        if len(members) >= 2 and len(x):
            joint = _fit_line(x, y, slope)[0]
            for i in members:
                segs[i]["joint"] = joint

    def sample_costs(s, a_milli):
        js = np.arange(s["s0"], s["s1"])
        sample_ref = map_frames(oth_frames[js], a_milli, drift_ppm, ref.fps, other.fps)
        ok = (sample_ref >= 0) & (sample_ref < ref.n_frames)
        c = np.full(len(js), np.nan)
        c[ok] = pair_cost(ref_rec, other_rec, sample_ref[ok], oth_frames[js][ok])
        return c

    frame = np.full(other.n_frames, -1, dtype=np.int64)
    fcost = np.full(other.n_frames, np.nan)
    segments = []
    last_end_ref = -1
    for s in sorted(segs, key=lambda s: s["s0"]):
        js = np.arange(s["s0"], s["s1"])
        a_milli = int(round(_intercept(s, slope) * 1000))
        c = sample_costs(s, a_milli)
        if "joint" in s and int(round(s["joint"] * 1000)) != a_milli:
            a_joint = int(round(s["joint"] * 1000))
            c_joint = sample_costs(s, a_joint)
            own = np.nanmean(c) if np.isfinite(c).any() else math.inf
            joint = np.nanmean(c_joint) if np.isfinite(c_joint).any() else math.inf
            if not own + JOINT_MARGIN_M < joint:
                a_milli, c = a_joint, c_joint
        # the segment's frames: from its first sample's frame to its last's
        f0, f1 = int(oth_frames[s["s0"]]), int(oth_frames[s["s1"] - 1]) + 1
        mapped = map_frames(np.arange(f0, f1), a_milli, drift_ppm, ref.fps, other.fps)
        inside = (mapped >= 0) & (mapped < ref.n_frames)
        med = _median3(c)
        median = float(np.nanmedian(c)) if np.isfinite(c).any() else math.nan
        if not np.isfinite(median) or median > ACCEPT_M:
            continue
        # per frame: the cost of its nearest sample; rejected samples unmatch their frames
        nearest = np.clip(np.round((np.arange(f0, f1) - oth_frames[s["s0"]]) * rate / other.fps)
                          .astype(np.int64), 0, len(js) - 1)
        good = inside & ~(med[nearest] > REJECT_M)
        idx = np.arange(f0, f1)[good]
        frame[idx] = mapped[good]
        fcost[idx] = c[nearest[good]]
        start_ref = int(mapped[inside][0]) if inside.any() else -1
        segments.append({"start": f0, "end": f1, "a_milli": a_milli, "drift_ppm": drift_ppm,
                         "cost": round(median, 4), "backward": bool(start_ref < last_end_ref - 1)})
        if inside.any():
            last_end_ref = max(last_end_ref, int(mapped[inside][-1]))
    return Alignment(frame=frame, cost=fcost, segments=segments, flip=flip, swap=swap,
                     ref_fps=ref.fps, other_fps=other.fps, drift_ppm=drift_ppm)


def _median3(values: np.ndarray) -> np.ndarray:
    """Median of each value and its two neighbours (NaN-aware; NaN where all three are)."""
    if len(values) < 3:
        return values.copy()
    padded = np.concatenate([[np.nan], values, [np.nan]])
    stack = np.stack([padded[:-2], padded[1:-1], padded[2:]])
    out = np.full(len(values), np.nan)
    any_seen = ~np.isnan(stack).all(axis=0)
    out[any_seen] = np.nanmedian(stack[:, any_seen], axis=0)
    return out


# -- carrying the ball across ----------------------------------------------------------------------

def attach(ref: Match, other: Match, alignment: Alignment, bias: bool = True) -> Match:
    """Copy of ref with ball2: the other recording's ball on reference frames (module docstring)."""
    n = ref.n_frames
    ball_o = Recording.of(other, flip=alignment.flip).ball
    seen = alignment.matched & ~np.isnan(ball_o[:, 0])
    idx = np.flatnonzero(seen)
    ball2 = np.full((n, 2), np.nan)
    best = np.full(n, np.inf)
    if idx.size:
        target = alignment.frame[idx]
        cost = np.nan_to_num(alignment.cost[idx], nan=NEUTRAL_M)
        order = np.lexsort((idx, cost))  # lowest cost first, then the earlier other frame
        for i in order[::-1]:  # write the worst first, so the best one is written last
            ball2[target[i]] = ball_o[idx[i]]
            best[target[i]] = cost[i]
        # interpolate between consecutive mapped frames of one segment (a lower other frame rate)
        for seg in alignment.segments:
            inside = idx[(idx >= seg["start"]) & (idx < seg["end"])]
            if inside.size < 2:
                continue
            consecutive = np.flatnonzero(np.diff(inside) == 1)
            t0, t1 = alignment.frame[inside[consecutive]], alignment.frame[inside[consecutive + 1]]
            gap = t1 - t0
            for k in np.flatnonzero((gap >= 2) & (gap <= math.ceil(ref.fps / other.fps) + 1)):
                a, b = ball_o[inside[consecutive[k]]], ball_o[inside[consecutive[k] + 1]]
                steps = np.arange(1, gap[k])
                frac = (steps / gap[k])[:, None]
                frames = t0[k] + steps
                free = ~np.isfinite(best[frames])
                ball2[frames[free]] = (a + (b - a) * frac)[free]
    if bias:
        ball2 = ball2 + local_bias(ref.ball, ball2, ref.fps)
    return dataclasses.replace(ref, ball2=ball2).validate()


def local_bias(ball: np.ndarray, ball2: np.ndarray, fps: int) -> np.ndarray:
    """(n, 2) median of ball - ball2 over the frames both see within +/- BIAS_S (0 with fewer than
    BIAS_MIN such frames), evaluated once a second and held."""
    n = len(ball)
    joint = np.flatnonzero(~np.isnan(ball[:, 0]) & ~np.isnan(ball2[:, 0]))
    out = np.zeros((n, 2))
    if joint.size < BIAS_MIN:
        return out
    diff = ball[joint] - ball2[joint]
    step = max(1, int(fps))
    half = int(round(BIAS_S * fps))
    for t in range(0, n, step):
        lo, hi = np.searchsorted(joint, [t - half, t + half + 1])
        if hi - lo >= BIAS_MIN:
            out[t:t + step] = np.round(np.median(diff[lo:hi], axis=0), 4)
    return out


# -- health (owner check) and truth (simulations) --------------------------------------------------

def health(ref: Match, other: Match, alignment: Alignment, fused: Match | None = None) -> dict:
    """Label-free aggregates of an alignment. On BAR-ATH they describe the match, so they are
    printed to the owner only (scripts/check_broadcast.py --bar-ath) and never leave the machine."""
    rec = Recording.of(other, flip=alignment.flip, swap=alignment.swap)
    data = rec.has_data()
    matched = alignment.matched
    live = [s for s in alignment.segments if not s["backward"]]
    out = {
        "other_frames": int(other.n_frames),
        "data_share": round(float(data.mean()), 4),
        "matched_share_of_data": round(float((matched & data).sum() / max(1, data.sum())), 4),
        "segments": len(alignment.segments),
        "replay_segments": len(alignment.segments) - len(live),
        "drift_ppm": int(alignment.drift_ppm),
        "flip": bool(alignment.flip),
        "swap": bool(alignment.swap),
        "median_cost_m": (round(float(np.nanmedian(alignment.cost)), 3)
                          if np.isfinite(alignment.cost).any() else None),
    }
    if fused is not None and fused.ball2 is not None:
        both = ~np.isnan(fused.ball[:, 0]) & ~np.isnan(fused.ball2[:, 0])
        d = np.hypot(*(fused.ball[both] - fused.ball2[both]).T) if both.any() else np.zeros(0)
        missing = np.isnan(fused.ball[:, 0]) & (fused.period > 0)
        filled = missing & ~np.isnan(fused.ball2[:, 0])
        out.update({
            "ball_agree_median_m": round(float(np.median(d)), 3) if d.size else None,
            "ball_agree_share_2m": round(float(np.mean(d <= 2.0)), 4) if d.size else None,
            "fill_share": round(float(filled.sum() / max(1, missing.sum())), 4),
        })
    return out


def from_truth(ref: Match, other: Match, truth: np.ndarray, live: np.ndarray | None = None) -> Alignment:
    """The alignment a perfect aligner would give a simulated second recording: truth[b] is the
    reference frame (float, NaN = none) the other frame shows (degrade.second_recording)."""
    frame = np.where(np.isnan(truth), -1, np.round(np.nan_to_num(truth, nan=-1))).astype(np.int64)
    frame[(frame < 0) | (frame >= ref.n_frames)] = -1
    cost = np.where(frame >= 0, 0.0, np.nan)
    segments, start = [], None
    jumps = np.flatnonzero(np.abs(np.diff(np.nan_to_num(truth, nan=-1e9))) > 2 * ref.fps / other.fps + 1)
    edges = np.concatenate([[0], jumps + 1, [other.n_frames]])
    for s, e in zip(edges[:-1], edges[1:]):
        if (frame[s:e] >= 0).any():
            segments.append({"start": int(s), "end": int(e), "a_milli": 0, "drift_ppm": 0, "cost": 0.0,
                             "backward": bool(live is not None and not live[s:e].any())})
    return Alignment(frame=frame, cost=cost, segments=segments, ref_fps=ref.fps,
                     other_fps=other.fps)
