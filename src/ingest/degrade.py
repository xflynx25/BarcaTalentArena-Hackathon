"""Make an external match's ball look like BAR-ATH's: hide it in runs, mostly near crowds and fast balls.

Why (findings F3, F14, F21): every external source gets ball visibility wrong in its own way.
IDSSE never loses the ball; Metrica and SkillCorner lose it almost only in dead time. BAR-ATH loses
it on 41% of graded frames, "disproportionately" in crowded boxes, aerial duels and tackles
[official DATA.md, FAQ.md]. Models tuned on clean balls, or on "missing means dead", won't transfer.

drop_ball(match, target=0.41, quiet=q) hides the ball until `target` of the graded frames
(labels != X) have no ball, counting frames that already lack one. Two kinds of gap:
  - quiet stretches (share q of the frames to hide, default 0): the in-period frames where the
    players' mean speed over +/- quiet_s (4 s) is lowest, i.e. mostly stoppages. Deterministic.
    Why (findings F28, F30, retired by C6): an earlier q = 0.75 was chosen so a baseline score
    matched published full-match numbers. That setting is not used to train or to choose the model.
    The grid in src/dropout.py uses ball_quiet 0, 0.5 and 1. The 41% share above is the tracking
    fact in official DATA.md.
Then short random gaps make up the rest, as runs, not single frames:
  - a gap can start on any in-period frame with the ball, with hazard base * risk, where
    risk = 1 + CROWD_WEIGHT * (players within CROWD_RADIUS_M of the ball) + ball speed / SPEED_SCALE.
    Dead balls (still, players spread out) get low risk on their own; the labels are never used.
  - gap lengths are log-normal with median gap_s seconds (sigma GAP_SIGMA), at most MAX_GAP_S.
  - base is found by bisection. The uniform draws and lengths are drawn once per match from the
    seed, so more hazard only adds gaps and the result is deterministic.
The gap shape is a guess: BAR-ATH's own gap lengths can't be measured by a cloud agent (CLAUDE.md).

Use it through src.evaluate as load options, on any source:
    --load ball_missing=0.41 --load ball_quiet=0.5 [--load ball_gap_s=1 --load ball_seed=0]
src.evaluate.load_match pops these before calling the source's loader. Editing this file
invalidates the cached matches that were loaded with ball options.

Players (backlog C11): our match's player coverage is unknown to cloud agents, so training can cover
a range of it. degrade_players(match, ...) / the load options
    camera=<share>        a simulated broadcast view that follows the ball: a trapezoid from the near
                          touchline (narrow) to past the far one (3x wider), as SkillCorner's
                          image_corners_projection (findings F111); <share> is its pitch share at
                          mid-pitch (SkillCorner ~0.34, 0.29 averaged over play). Players outside it are
    camera_fill=nan|estimate   hidden (nan, default) or "estimated": the last position seen in this
                          period, drifting with the mean motion of the team's players in view, like
                          broadcast vendors' guesses (F93).
    player_drop=<share>   tracklet gaps: each slot hidden on ~share of its frames, in runs of
                          player_gap_s (5 s). Moved from check_shape_model.py --drop-players (F99).
    id_swap=<rate>        identity switches: <rate> per team per minute, two teammates swap slots from
                          then on (the closest pair at that moment, as trackers confuse them in contact).
    player_seed=<n>       0 by default; every draw is seeded by the match key, like the ball.
    fragment=<rate>       tracklet fragmentation (C13): <rate> breaks per player per minute of play, more
                          likely in crowds (1 + FRAG_CROWD_WEIGHT x other players within FRAG_CROWD_M);
                          at a break the player is lost for a log-normal gap (median fragment_gap_s,
                          2 s) and comes back in another slot of the team (the lowest one free, never
                          the one just left; a new slot if none), so k grows as on BAR-ATH (46 slots).
    player_fill=<s>       applied last, as a run-time repair would be (C11, F113): fill each slot's gaps
                          of at most <s> seconds that have the slot seen on both sides in the same period
                          (fill_gaps). Uses only the degraded positions, so it could run on any match.
    gk_flip=<rate>        goalkeeper team flips (C15): kit-colour team classification gets keepers wrong
                          (they wear their own kit). <rate> stretches per keeper per 10 minutes of play;
                          for each, the keeper's positions come in a slot of the other team (a free
                          added one, else a new one; k grows) and his own slot is empty, then he goes
                          back. Stretch lengths log-normal, median gk_flip_s (5 s, sigma GK_SIGMA).
                          Keepers are found label-free: per half, each team's slot that is most often
                          its deepest player towards its own goal (keepers()).
    gk_ghost=1            (with gk_flip) for the same stretch an extra detection of the keeper's own
                          team, 1-4 m from where he was, moving little and jittering, so his team still
                          shows its usual count.
    gk_hide=1             (with gk_flip) for the same stretch the other team's own keeper is missing, so
                          the counts stay 11 v 11 that way too (a count check can't see it).
    track_repair=<spec>   after everything else: link tracklets and bridge their gaps (src/tracks.py,
                          C13), as `learned` does at run time with its track_repair parameter.
    clean=1               no degradation at all: drops every other option here (the clean copy).
The camera follows the clean ball, and the ball is hidden on the clean match (its crowd risk sees every
player), so the ball gaps are the same with or without player options: apply() merges the two.

A second recording (backlog V4): BAR-ATH also has broadcast tracking on its own timeline (official
DATA.md). second_recording(match) simulates one from the clean match (camera window, cuts, replays,
a halftime gap, clock drift, its own frame rate, noise, its own ball dropout), and the load options
    broadcast=1           simulate it, align it with the degraded copy (src/align.py, as at run time
                          on BAR-ATH) and attach its ball as Match.ball2; broadcast=oracle uses the
                          true map instead (the ceiling a perfect aligner reaches)
    bc_fps, bc_drift_ppm, bc_cut, bc_replays, bc_ball_missing, bc_noise_m, bc_shift_m, bc_camera,
    bc_seed               second_recording's settings (defaults: BC_* below)
The second recording is seeded by the match id and bc_seed only, so every ball-grid point gets the same
one; the alignment runs against each degraded copy.
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np

from src.official import X
from src.schema import Match

CROWD_RADIUS_M = 3.0
CROWD_WEIGHT = 1.0
SPEED_SCALE = 10.0  # m/s; a 20 m/s ball adds 2 to the risk
GAP_SIGMA = 1.0
MAX_GAP_S = 20.0
QUIET_WINDOW_S = 4.0  # player speed is averaged over +/- this for quiet stretches
BALL_OPTIONS = ("ball_missing", "ball_gap_s", "ball_seed", "ball_quiet", "ball_quiet_s")
PLAYER_OPTIONS = ("camera", "camera_fill", "player_drop", "player_gap_s", "id_swap", "player_seed",
                  "player_fill", "fragment", "fragment_gap_s", "track_repair",
                  "gk_flip", "gk_flip_s", "gk_ghost", "gk_hide")
# V4: a simulated second recording (second_recording), aligned and attached as Match.ball2
BROADCAST_OPTIONS = ("broadcast", "bc_fps", "bc_drift_ppm", "bc_cut", "bc_replays",
                     "bc_ball_missing", "bc_noise_m", "bc_shift_m", "bc_camera", "bc_seed")
OPTIONS = BALL_OPTIONS + PLAYER_OPTIONS + BROADCAST_OPTIONS + ("clean",)

# the simulated second recording (V4, second_recording): BAR-ATH's broadcast tracking per DATA.md
# (homography valid on 65.8% of its frames, ball seen on 50.5%, i.e. ~77% of the tracked ones)
BC_FPS = 25
BC_CUT = 0.30          # share of programme frames with no tracking (close-ups, crowd, graphics)
BC_REPLAYS = 3.0       # replays per 10 minutes of play, put in calm stretches
BC_BALL_MISSING = 0.25 # share of its tracked in-play frames whose ball the second recording loses
BC_NOISE_M = 0.3       # position noise (players and ball): AR(1), sd this, correlation time ...
BC_NOISE_TAU_S = 0.5   # ... this (white noise would fake ball velocity jumps, i.e. touches)
BC_SHIFT_M = 0.7       # a constant calibration offset per shot (sd per axis)
BC_CAMERA = 0.34       # its view's pitch share at mid-pitch (camera_view; SkillCorner-like, F111)
BC_DRIFT_PPM = 100.0   # clock drift drawn uniformly in +/- this unless bc_drift_ppm is given
BC_SHOT_S = 6.0        # median shot length (log-normal, sigma 0.8)
BC_REPLAY_TRACKED = 0.6  # share of replays with tracking (the rest are close-up angles)

# broadcast camera footprint, from SkillCorner's image_corners_projection (findings F111)
CAMERA_NEAR_Y = 2.5     # m: the view's near edge, just inside the near touchline
CAMERA_FAR_PAST = 5.0   # m: its far edge, beyond the far touchline
CAMERA_RATIO = 3.0      # far edge width / near edge width (SkillCorner: ~57 / ~19 m)
CAMERA_SMOOTH_S = 1.0   # the view centre is the ball's x averaged over +/- this
CAMERA_FILLS = ("nan", "estimate")
PLAYER_GAP_S = 5.0
FRAG_GAP_S = 2.0  # median gap at a tracklet break (C13)
FRAG_CROWD_M = 2.0  # breaks are likelier with other players this close ...
FRAG_CROWD_WEIGHT = 1.0  # ... by 1 + this x their number (occlusion)
GK_FLIP_S = 5.0  # C15: median keeper-flip stretch (s), log-normal ...
GK_SIGMA = 0.6  # ... with this sigma, at most MAX_GK_S
MAX_GK_S = 30.0
GK_GHOST_M = (1.0, 4.0)  # the ghost starts this far from the keeper (uniform) ...
GK_GHOST_FOLLOW = 0.2  # ... moves this share of the keeper's motion ...
GK_GHOST_JITTER_M = 0.1  # ... plus white jitter with this sd per axis per frame


def risk(match: Match) -> np.ndarray:
    """(n,) relative hazard of losing the ball, >= 1. NaN-safe: missing players don't count."""
    dx = match.players[:, :, 0] - match.ball[:, None, 0]
    dy = match.players[:, :, 1] - match.ball[:, None, 1]
    with np.errstate(invalid="ignore"):
        crowd = np.sum(dx * dx + dy * dy <= CROWD_RADIUS_M ** 2, axis=1)
    step = np.diff(match.ball, axis=0, prepend=match.ball[:1])
    speed = np.nan_to_num(np.hypot(step[:, 0], step[:, 1]) * match.fps, nan=0.0)
    speed = np.minimum(speed, 40.0)  # tracking glitches
    return 1.0 + CROWD_WEIGHT * crowd + speed / SPEED_SCALE


def calm(match: Match, window_s: float = QUIET_WINDOW_S) -> np.ndarray:
    """(n,) mean player speed in m/s over +/- window_s, inf where no player is seen in the window.
    Players walk during stoppages (dead median ~1.05 m/s vs ~2.2 live, findings F29). Ball speed
    is left out on purpose: IDSSE freezes the ball in dead time, so it would line the stretches
    up with D too exactly on that source only."""
    step = np.diff(match.players, axis=0, prepend=match.players[:1])
    with np.errstate(invalid="ignore"):
        speed = np.minimum(np.hypot(step[..., 0], step[..., 1]) * match.fps, 12.0)
    seen = np.sum(~np.isnan(speed), axis=1)
    mean = np.where(seen > 0, np.nansum(speed, axis=1) / np.maximum(seen, 1), np.nan)
    kernel = np.ones(2 * max(1, round(window_s * match.fps)) + 1)
    valid = np.convolve(~np.isnan(mean), kernel, mode="same")
    total = np.convolve(np.nan_to_num(mean), kernel, mode="same")
    return np.where(valid > 0, total / np.maximum(valid, 1), np.inf)


def _coverage(n: int, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, np.minimum(starts + lengths[starts], n), -1)
    return np.cumsum(diff[:-1]) > 0


def _lengths(rng, n: int, median_s: float, fps: int) -> np.ndarray:
    max_frames = max(1, round(MAX_GAP_S * fps))
    lengths = np.clip(np.rint(rng.lognormal(np.log(median_s * fps), GAP_SIGMA, n)), 1, max_frames)
    return lengths.astype(np.int64)


def quiet_stretches(match: Match, graded: np.ndarray, has_ball: np.ndarray, share: float,
                    window_s: float = QUIET_WINDOW_S) -> np.ndarray:
    """(n,) the calmest in-period frames that still have the ball, as many as `share` of the graded
    frames. Calm is smoothed, so these come as whole stretches (stoppages, mostly), not frames."""
    speed = np.where(has_ball & (match.period > 0), calm(match, window_s), np.inf)
    need = int(round(share * graded.sum()))
    if need <= 0:
        return np.zeros(len(speed), dtype=bool)
    candidates = np.sort(speed[graded & np.isfinite(speed)])
    if not len(candidates):
        return np.zeros(len(speed), dtype=bool)
    return speed <= candidates[min(need, len(candidates)) - 1]


def drop_ball(match: Match, target: float = 0.41, gap_s: float = 1.0, seed: int = 0,
              quiet: float = 0.0, quiet_s: float = QUIET_WINDOW_S) -> Match:
    """Copy of match with the ball set to NaN until `target` of graded frames lack it.
    First `quiet` of the frames still to hide go to the calmest stretches (see quiet_stretches),
    then short random gaps near crowds and fast balls make up the rest."""
    if not 0.0 <= target < 1.0:
        raise ValueError("ball_missing must be in [0, 1), not {}".format(target))
    if not 0.0 <= quiet <= 1.0:
        raise ValueError("ball_quiet must be in [0, 1], not {}".format(quiet))
    graded = match.labels != X if match.labels is not None else match.period > 0
    has_ball = ~np.isnan(match.ball).any(axis=1)
    n = len(has_ball)
    already = 1.0 - has_ball[graded].mean() if graded.any() else 0.0

    out = Match(match_id=match.match_id, fps=match.fps, pitch=match.pitch, period=match.period,
                ball=match.ball.copy(), players=match.players, slot_team=match.slot_team,
                labels=match.labels)
    if already >= target:
        return out.validate()
    if quiet > 0:
        hidden = quiet_stretches(match, graded, has_ball, quiet * (target - already), quiet_s)
        out.ball[hidden] = np.nan
        has_ball = has_ball & ~hidden

    rng = np.random.default_rng([seed, zlib.crc32(match.match_id.encode())])
    draws = rng.random(n)
    lengths = _lengths(rng, n, gap_s, match.fps)
    in_play = match.period > 0
    hazard = np.where(has_ball & in_play, risk(match), 0.0)

    def missing(base: float) -> tuple[float, np.ndarray]:
        starts = np.flatnonzero(draws < base * hazard)
        hidden = _coverage(n, starts, lengths) & in_play
        return 1.0 - (has_ball & ~hidden)[graded].mean(), hidden

    if missing(0.0)[0] < target:
        low, high = 0.0, 1.0  # at base 1 every frame with a ball starts a gap (risk >= 1)
        for _ in range(40):
            mid = (low + high) / 2
            if missing(mid)[0] < target:
                low = mid
            else:
                high = mid
        out.ball[missing(high)[1]] = np.nan
    return out.validate()


# -- players (C11) ------------------------------------------------------------------------------------

def _truthy(value) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def camera_width(pitch, share: float) -> float:
    """Near-edge width (m) of a camera trapezoid covering `share` of the pitch when centred on it."""
    length, width = pitch
    y0, y1 = CAMERA_NEAR_Y, width + CAMERA_FAR_PAST
    depth = width - y0  # on the pitch
    per_metre = depth + (CAMERA_RATIO - 1.0) * depth ** 2 / (2.0 * (y1 - y0))
    return share * length * width / per_metre


def camera_centre(match: Match, ball: np.ndarray | None = None) -> np.ndarray:
    """(n,) x of the view centre: the ball's x, gaps filled with the last (then next) seen value,
    averaged over +/- CAMERA_SMOOTH_S. Mid-pitch if the ball is never seen."""
    x = (match.ball if ball is None else ball)[:, 0]
    seen = ~np.isnan(x)
    if not seen.any():
        return np.full(len(x), match.pitch[0] / 2)
    idx = np.where(seen, np.arange(len(x)), -1)
    last = np.maximum.accumulate(idx)
    first = np.flatnonzero(seen)[0]
    filled = x[np.where(last >= 0, last, first)]
    half = max(1, round(CAMERA_SMOOTH_S * match.fps))
    kernel = np.ones(2 * half + 1)
    padded = np.concatenate([np.full(half, filled[0]), filled, np.full(half, filled[-1])])
    return np.convolve(padded, kernel, mode="valid") / len(kernel)


def camera_view(match: Match, share: float, ball: np.ndarray | None = None) -> np.ndarray:
    """(n, k) bool: player slots inside the simulated broadcast view (False where NaN)."""
    near = camera_width(match.pitch, share)
    y0, y1 = CAMERA_NEAR_Y, match.pitch[1] + CAMERA_FAR_PAST
    x, y = match.players[..., 0], match.players[..., 1]
    with np.errstate(invalid="ignore"):
        half = 0.5 * near * (1.0 + (CAMERA_RATIO - 1.0) * (y - y0) / (y1 - y0))
        return (y >= y0) & (y <= y1) & (np.abs(x - camera_centre(match, ball)[:, None]) <= half)


def estimate(match: Match, seen: np.ndarray) -> np.ndarray:
    """(n, k, 2) players where seen; where a player is present but not seen, the slot's last
    position seen in the same period, moved by the team's mean motion since (players seen on both
    frames), clipped to the pitch. NaN where the slot is absent (a tracklet that has ended stays
    ended) or hasn't been seen yet in the period."""
    players = match.players
    present = ~np.isnan(players).any(axis=2)
    n, k = seen.shape
    out = np.where(seen[..., None], players, np.nan)
    run = np.concatenate([[0], np.cumsum(np.diff(match.period) != 0)])
    step = np.diff(players, axis=0, prepend=players[:1])
    both = seen & np.vstack([np.zeros((1, k), dtype=bool), seen[:-1]])
    both[1:] &= (run[1:] == run[:-1])[:, None]
    idx = np.where(seen, np.arange(n)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)
    for team in np.unique(match.slot_team):
        slots = np.flatnonzero(match.slot_team == team)
        moving = both[:, slots]
        count = moving.sum(axis=1)
        total = np.where(moving[..., None], step[:, slots], 0.0).sum(axis=1)
        drift = np.cumsum(total / np.maximum(count, 1)[:, None], axis=0)
        for slot in slots:
            ref = last[:, slot]
            use = present[:, slot] & ~seen[:, slot] & (ref >= 0)
            use &= run[np.maximum(ref, 0)] == run
            r = ref[use]
            out[use, slot] = players[r, slot] + drift[use] - drift[r]
    length, width = match.pitch
    out[..., 0] = np.clip(out[..., 0], 0.0, length)
    out[..., 1] = np.clip(out[..., 1], 0.0, width)
    return out


def drop_tracklets(players: np.ndarray, fps: int, share: float, rng,
                   gap_s: float = PLAYER_GAP_S) -> np.ndarray:
    """Copy of players with each slot hidden on ~share of its frames, in runs of gap_s (tracklet
    gaps). Same draws as check_shape_model.py's former drop_players (findings F99)."""
    n, k = players.shape[:2]
    length = max(1, int(gap_s * fps))
    out = players.copy()
    for slot in range(k):
        starts = rng.random(n) < share / length
        hidden = np.convolve(starts.astype(float), np.ones(length))[:n] > 0
        out[hidden, slot] = np.nan
    return out


def swap_ids(match: Match, rate: float, rng) -> np.ndarray:
    """Copy of players where, `rate` times per team per minute of play, the two closest teammates
    on screen swap slots from that frame on (identity switches)."""
    players = match.players.copy()
    n = len(players)
    in_play = match.period > 0
    p = rate / (60.0 * match.fps)
    for team in np.unique(match.slot_team):
        slots = np.flatnonzero(match.slot_team == team)
        events = np.flatnonzero((rng.random(n) < p) & in_play)
        for t in events:
            pos = players[t, slots]
            d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
            d[np.isnan(d)] = np.inf
            np.fill_diagonal(d, np.inf)
            if not np.isfinite(d).any():
                continue
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            sa, sb = slots[a], slots[b]
            players[t:, [sa, sb]] = players[t:, [sb, sa]]
    return players


def crowding(players: np.ndarray, radius_m: float, chunk: int = 4096) -> np.ndarray:
    """(n, k) number of other slots within radius_m of each slot (0 where it is NaN)."""
    n, k = players.shape[:2]
    out = np.zeros((n, k), dtype=np.int16)
    for s in range(0, n, chunk):
        p = players[s:s + chunk]
        d2 = (p[:, :, None, 0] - p[:, None, :, 0]) ** 2 + (p[:, :, None, 1] - p[:, None, :, 1]) ** 2
        with np.errstate(invalid="ignore"):
            out[s:s + chunk] = np.maximum(np.sum(d2 <= radius_m ** 2, axis=2) - 1, 0)
    return out


def fragment(match: Match, rate: float, rng, gap_s: float = FRAG_GAP_S):
    """(players, slot_team, person) with each slot's track broken ~rate times per minute of play
    (likelier in crowds), the player lost for a log-normal gap (median gap_s, at most MAX_GAP_S) and
    continuing in another slot of the team: the lowest slot free since before the piece starts,
    never the one it just left, else a new slot (C13). person (n, k') int16 is the input slot each
    output position came from (-1: empty), for checks (scripts/check_tracks.py)."""
    players = match.players
    n, k = players.shape[:2]
    seen = ~np.isnan(players).any(axis=2)
    risk = np.where(seen & (match.period > 0)[:, None],
                    1.0 + FRAG_CROWD_WEIGHT * crowding(players, FRAG_CROWD_M), 0.0)
    mean_risk = risk[risk > 0].mean() if (risk > 0).any() else 1.0
    draws = rng.random((n, k)) < rate / (60.0 * match.fps) * risk / mean_risk
    lengths = _lengths(rng, n * k, gap_s, match.fps).reshape(n, k)
    pieces = []  # (start, stop, input slot, piece number)
    for slot in range(k):
        start, number = 0, 0
        for t in np.flatnonzero(draws[:, slot]):
            if t < start:
                continue  # inside the last gap
            pieces.append((start, int(t), slot, number))
            start, number = min(n, int(t + lengths[t, slot])), number + 1
        pieces.append((start, n, slot, number))
    team = list(match.slot_team)  # per output slot; the input slots come first
    free_from = [0] * k
    col_of = {}
    for start, stop, slot, number in sorted(pieces):
        if number == 0:
            col = slot
        else:
            last = col_of[(slot, number - 1)]
            col = next((c for c in range(len(team)) if team[c] == match.slot_team[slot]
                        and free_from[c] <= start and c != last), len(team))
            if col == len(team):
                team.append(match.slot_team[slot])
                free_from.append(0)
        free_from[col] = max(free_from[col], stop)
        col_of[(slot, number)] = col
    out = np.full((n, len(team), 2), np.nan)
    person = np.full((n, len(team)), -1, dtype=np.int16)
    for start, stop, slot, number in pieces:
        col = col_of[(slot, number)]
        out[start:stop, col] = players[start:stop, slot]
        person[start:stop, col] = np.where(seen[start:stop, slot], slot, -1)
    return out, np.array(team, dtype=np.int8), person


def defending_left(match: Match) -> dict:
    """{period: team that defends the x = 0 goal} for each period > 0, label-free: the team whose
    deepest player (lowest x) is lower in the median over the period's frames (its keeper)."""
    out = {}
    for p in np.unique(match.period[match.period > 0]):
        rows = match.period == p
        low = []
        for team in (0, 1):
            x = match.players[rows][:, match.slot_team == team, 0]
            with np.errstate(all="ignore"):
                m = np.nanmin(np.where(np.isnan(x), np.inf, x), axis=1) if x.shape[1] else np.full(1, np.inf)
            m = m[np.isfinite(m)]
            low.append(float(np.median(m)) if len(m) else np.inf)
        out[int(p)] = 0 if low[0] <= low[1] else 1
    return out


def keepers(match: Match) -> dict:
    """{(period, team): slot} label-free: per period and team, the slot that is most often the team's
    deepest player towards its own goal (defending_left), counted over the frames it is seen."""
    left = defending_left(match)
    length = match.pitch[0]
    out = {}
    for p, lteam in left.items():
        rows = np.flatnonzero(match.period == p)
        for team in (0, 1):
            slots = np.flatnonzero(match.slot_team == team)
            if not len(slots):
                continue
            x = match.players[rows][:, slots, 0]
            depth = x if team == lteam else length - x  # distance from its own goal line
            depth = np.where(np.isnan(depth), np.inf, depth)
            seen = np.isfinite(depth).any(axis=1)
            if not seen.any():
                continue
            counts = np.bincount(np.argmin(depth[seen], axis=1), minlength=len(slots))
            out[(int(p), team)] = int(slots[int(np.argmax(counts))])
    return out


def gk_damage(match: Match, rate: float, rng, median_s: float = GK_FLIP_S, ghost: bool = False,
              hide: bool = False):
    """(players, slot_team, events): keeper team flips (C15). For each keeper (keepers()), stretches
    start on in-period frames where he is seen, `rate` per 10 minutes of play, log-normal lengths
    (median median_s, at most MAX_GK_S, cut at the period's end, never overlapping his last one). In
    a stretch his positions move to a slot of the other team (the lowest added slot of that team free
    since before it starts, else a new one) and his own slot is empty; with ghost, a new detection of
    his own team appears 1-4 m from him, following GK_GHOST_FOLLOW of his motion and jittering; with
    hide, the other team's keeper of that period is missing. events: (start, stop, team, keeper slot,
    flip slot, ghost slot or -1, hidden slot or -1), for checks (scripts/check_gkteam.py)."""
    n, k = match.players.shape[:2]
    gk = keepers(match)
    p_frame = rate / (600.0 * match.fps)
    plan = []  # (start, stop, team, keeper slot, period)
    for (p, team), slot in sorted(gk.items()):
        rows = match.period == p
        seen = rows & ~np.isnan(match.players[:, slot]).any(axis=1)
        draws = rng.random(n) < p_frame
        lengths = np.minimum(np.exp(np.log(median_s * match.fps) + GK_SIGMA * rng.standard_normal(n)),
                             MAX_GK_S * match.fps).astype(np.int64).clip(1)
        stop_of_period = np.flatnonzero(rows)[-1] + 1
        last = 0
        for t in np.flatnonzero(draws & seen):
            if t <= last:
                continue
            stop = int(min(stop_of_period, t + lengths[t]))
            plan.append((int(t), stop, team, slot, p))
            last = stop
    if not plan:
        return match.players, match.slot_team, []
    players = match.players.copy()
    team_of = list(match.slot_team)
    cols = {}  # added column -> free from
    extra = []  # (column, values) for added slots
    events = []

    def column(team, start, stop):
        for c in sorted(cols):
            if team_of[c] == team and cols[c] <= start:
                cols[c] = stop
                return c
        c = len(team_of)
        team_of.append(team)
        cols[c] = stop
        extra.append(np.full((n, 2), np.nan))
        return c

    def put(c, a, b, values):
        extra[c - k][a:b] = values

    for start, stop, team, slot, p in sorted(plan):
        other = 1 - team
        pos = players[start:stop, slot].copy()
        players[start:stop, slot] = np.nan
        flip = column(other, start, stop)
        put(flip, start, stop, pos)
        g = h = -1
        if ghost:
            g = column(team, start, stop)
            radius = rng.uniform(*GK_GHOST_M)
            angle = rng.uniform(0, 2 * np.pi)
            ref = pos[0]
            moved = np.where(np.isnan(pos), 0.0, pos - ref)
            vals = (ref + radius * np.array([np.cos(angle), np.sin(angle)]) + GK_GHOST_FOLLOW * moved
                    + GK_GHOST_JITTER_M * rng.standard_normal((stop - start, 2)))
            vals[:, 0] = np.clip(vals[:, 0], 0, match.pitch[0])
            vals[:, 1] = np.clip(vals[:, 1], 0, match.pitch[1])
            put(g, start, stop, vals)
        if hide and (p, other) in gk:
            h = gk[(p, other)]
            players[start:stop, h] = np.nan
        events.append((start, stop, team, slot, flip, g, h))
    if extra:
        players = np.concatenate([players, np.stack(extra, axis=1)], axis=1)
    return players, np.array(team_of, dtype=np.int8), events


def fill_gaps(match: Match, max_gap_s: float) -> Match:
    """Copy of match with each slot's short gaps filled: a run of missing frames of at most max_gap_s,
    with the slot seen just before and just after it in the same period, gets estimate()'s position
    (last seen + the team's mean motion). Longer gaps and tracklet ends stay NaN. Label-free and
    ball-free, using only the match's own positions."""
    seen = ~np.isnan(match.players).any(axis=2)
    n, k = seen.shape
    run = np.concatenate([[0], np.cumsum(np.diff(match.period) != 0)])
    max_frames = max(1, round(max_gap_s * match.fps))
    inside = np.zeros_like(seen)
    for slot in range(k):
        edges = np.diff(np.concatenate([[0], (~seen[:, slot]).astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a > 0 and b < n and b - a <= max_frames and run[a - 1] == run[b]:
                inside[a:b, slot] = True
    if not inside.any():
        return match
    filled = estimate(dataclasses.replace(match, players=np.where(
        (seen | inside)[..., None], np.nan_to_num(match.players), np.nan)), seen)
    return dataclasses.replace(match, players=np.where(inside[..., None], filled, match.players))


def degrade_players(match: Match, camera: float = 0.0, camera_fill: str = "nan",
                    player_drop: float = 0.0, player_gap_s: float = PLAYER_GAP_S,
                    id_swap: float = 0.0, seed: int = 0, ball: np.ndarray | None = None,
                    fragment_rate: float = 0.0, fragment_gap_s: float = FRAG_GAP_S,
                    gk_flip: float = 0.0, gk_flip_s: float = GK_FLIP_S, gk_ghost: bool = False,
                    gk_hide: bool = False) -> Match:
    """Copy of match with its players degraded: keeper team flips (k may grow), then identity
    switches, then the camera window (players outside hidden or estimated), then tracklet gaps, then
    fragmentation (k may grow). `ball` (default: the match's) steers the camera. Deterministic: seeded
    by the match id (and `seed`)."""
    if camera_fill not in CAMERA_FILLS:
        raise ValueError("camera_fill must be one of {}, not {}".format(CAMERA_FILLS, camera_fill))
    if not 0.0 <= camera <= 1.0:
        raise ValueError("camera must be a pitch share in [0, 1], not {}".format(camera))
    if not 0.0 <= player_drop < 1.0:
        raise ValueError("player_drop must be in [0, 1), not {}".format(player_drop))
    if id_swap < 0:
        raise ValueError("id_swap must be >= 0, not {}".format(id_swap))
    key = zlib.crc32(match.match_id.encode())
    out = match
    if (gk_ghost or gk_hide) and not gk_flip:
        raise ValueError("gk_ghost / gk_hide need gk_flip")
    if gk_flip:
        if gk_flip < 0 or gk_flip_s <= 0:
            raise ValueError("gk_flip must be >= 0 and gk_flip_s > 0")
        rng = np.random.default_rng([seed, key, 4])
        players, slot_team, _ = gk_damage(out, gk_flip, rng, gk_flip_s, gk_ghost, gk_hide)
        out = dataclasses.replace(out, players=players, slot_team=slot_team)
    if id_swap:
        rng = np.random.default_rng([seed, key, 1])
        out = dataclasses.replace(out, players=swap_ids(out, id_swap, rng))
    if camera:
        seen = camera_view(out, camera, ball) & ~np.isnan(out.players).any(axis=2)
        players = (estimate(out, seen) if camera_fill == "estimate"
                   else np.where(seen[..., None], out.players, np.nan))
        out = dataclasses.replace(out, players=players)
    if player_drop:
        rng = (np.random.default_rng(zlib.crc32(("b25drop|" + match.match_id).encode())) if not seed
               else np.random.default_rng([seed, key, 2]))
        out = dataclasses.replace(out, players=drop_tracklets(out.players, match.fps, player_drop,
                                                              rng, player_gap_s))
    if fragment_rate:
        if fragment_rate < 0:
            raise ValueError("fragment must be >= 0, not {}".format(fragment_rate))
        rng = np.random.default_rng([seed, key, 3])
        players, slot_team, _ = fragment(out, fragment_rate, rng, fragment_gap_s)
        out = dataclasses.replace(out, players=players, slot_team=slot_team)
    return out.validate()


# -- a second recording (V4) ----------------------------------------------------------------------

def _play_runs(period: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) runs of one period value > 0."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], period, [0]])) != 0)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if period[a] > 0]


def _interp(values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """values (n, ...) at fractional frames `at` (linear; NaN if either neighbour is NaN)."""
    i0 = np.clip(np.floor(at).astype(np.int64), 0, len(values) - 1)
    i1 = np.minimum(i0 + 1, len(values) - 1)
    frac = (at - i0).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[i0] * (1.0 - frac) + values[i1] * frac


SECOND_KINDS = ("none", "live", "live_untracked", "replay", "replay_untracked")


def second_recording(match: Match, fps: int = BC_FPS, drift_ppm: float | None = None,
                     cut: float = BC_CUT, replays: float = BC_REPLAYS,
                     ball_missing: float = BC_BALL_MISSING, noise_m: float = BC_NOISE_M,
                     shift_m: float = BC_SHIFT_M, camera: float = BC_CAMERA,
                     seed: int = 0) -> tuple[Match, np.ndarray, np.ndarray]:
    """A simulated TV recording of the clean `match` on its own timeline: (Match, truth, kind).
    truth[b] is the (fractional) match frame other frame b shows, NaN where it shows none; kind[b]
    indexes SECOND_KINDS. Label-free (it reads positions and periods only).

    Programme: a studio lead-in with no tracking, each period from a few seconds before its start to
    a few after its end, a halftime gap between periods (so each half has its own offset) and an
    outro; frames at `fps` with the clock running (1 + drift_ppm 1e-6) fast. Shots of log-normal
    length (median BC_SHOT_S): untracked (close-ups, crowd) with probability 2 cut (1 - q), q the
    shot's calmness percentile, so on average `cut` and mostly in stoppages; `replays` per 10 min of
    play replace calm untracked shots with an earlier 3-8 s of play (ending 1-8 s before), at full
    or half speed, tracked with probability BC_REPLAY_TRACKED. Tracked frames show the players in a
    camera window following the ball (camera_view, share `camera`), slots shuffled within each team
    at every shot (tracklets restart at cuts), with a per-shot calibration shift (sd shift_m) and
    smooth noise (AR(1), sd noise_m, BC_NOISE_TAU_S); the ball too, lost on a further
    `ball_missing` of in-play frames by drop_ball's gaps (near crowds and fast balls, so partly where
    the first recording loses it).
    Seeded by the match id and `seed` only, so every ball-grid point gets the same recording."""
    rng = np.random.default_rng([seed, zlib.crc32(match.match_id.encode()), 41])
    if drift_ppm is None:
        drift_ppm = float(rng.uniform(-BC_DRIFT_PPM, BC_DRIFT_PPM))
    clock = 1.0 + drift_ppm * 1e-6
    n, f = match.n_frames, match.fps
    runs = _play_runs(match.period)
    pieces = [(np.nan, int(round(rng.uniform(5, 60) * fps)))]  # (content start frame, frames)
    for i, (start, stop) in enumerate(runs):
        lo = runs[i - 1][1] if i else 0
        hi = runs[i + 1][0] if i + 1 < len(runs) else n
        c0 = max(lo, start - rng.uniform(5, 30) * f)
        c1 = min(hi - 1, stop - 1 + rng.uniform(2, 20) * f)
        pieces.append((c0, int((c1 - c0) / f / clock * fps)))
        pieces.append((np.nan, int(round(rng.uniform(10, 120) * fps)) if i + 1 < len(runs)
                       else int(round(rng.uniform(5, 30) * fps))))
    truth = np.concatenate([np.full(m, np.nan) if np.isnan(c0) else c0 + np.arange(m) / fps * clock * f
                            for c0, m in pieces])
    m = len(truth)
    kind = np.where(np.isnan(truth), 0, 1).astype(np.int8)

    # shots, untracked mostly in calm stretches
    lengths = np.maximum(1, np.round(np.exp(np.log(BC_SHOT_S) + 0.8 * rng.standard_normal(m // fps + 10))
                                     * fps).astype(np.int64))
    starts = np.concatenate([[0], np.cumsum(lengths)])
    starts = starts[starts < m]
    shot = np.repeat(np.arange(len(starts)), np.diff(np.concatenate([starts, [m]])))
    speed = calm(match)
    in_play = match.period > 0
    finite = np.sort(speed[in_play & np.isfinite(speed)])
    content = np.clip(np.round(np.nan_to_num(truth, nan=0)).astype(np.int64), 0, n - 1)
    q = (np.searchsorted(finite, speed[content]) / max(1, len(finite))) if len(finite) else np.full(m, 0.5)
    shot_q = np.bincount(shot, weights=q, minlength=len(starts)) / np.maximum(
        np.bincount(shot, minlength=len(starts)), 1)
    untracked = rng.random(len(starts)) < np.minimum(0.95, 2.0 * cut * (1.0 - shot_q))
    kind[(kind == 1) & untracked[shot]] = 2

    # replays in calm untracked shots of play
    play_minutes = in_play.sum() / f / 60.0
    shot_len = np.diff(np.concatenate([starts, [m]]))
    live_shot = np.array([kind[s] == 2 and in_play[content[s]] for s in starts])
    eligible = np.flatnonzero(untracked & live_shot & (shot_len >= 3 * fps))
    count = min(len(eligible), int(round(replays * play_minutes / 10.0)))
    for s in np.sort(rng.choice(eligible, size=count, replace=False)) if count else []:
        b0 = int(starts[s])
        src_len = rng.uniform(3, 8) * f
        src_end = truth[b0] - rng.uniform(1, 8) * f
        slow = 0.5 if rng.random() < 0.5 else 1.0
        frames = min(m - b0, int(src_len / f / slow * fps))
        if src_end - src_len < 0 or frames < 1:
            continue
        truth[b0:b0 + frames] = src_end - src_len + np.arange(frames) / fps * slow * f
        kind[b0:b0 + frames] = 3 if rng.random() < BC_REPLAY_TRACKED else 4

    # what the tracked frames show
    tracked = (kind == 1) | (kind == 3)
    at = np.where(tracked, np.nan_to_num(truth, nan=0.0), 0.0)
    near = np.clip(np.round(at).astype(np.int64), 0, n - 1)
    visible = camera_view(match, camera)[near] & tracked[:, None]
    players = np.where(visible[..., None], _interp(match.players, at), np.nan)
    ball = _interp(match.ball, at)
    already = ~np.isnan(match.ball[:, 0])
    graded = in_play if match.labels is None else match.labels != X
    base = 1.0 - already[graded].mean() if graded.any() else 0.0
    hidden = np.isnan(drop_ball(match, target=min(0.99, base + ball_missing * (1 - base)), gap_s=1.0,
                                seed=seed + 1000).ball[:, 0]) & already
    ball[~tracked | hidden[near]] = np.nan
    from src.noise import ar1

    shift = rng.normal(0.0, shift_m, (len(starts), 2))[shot]
    rho = float(np.exp(-1.0 / (BC_NOISE_TAU_S * fps)))
    players = players + shift[:, None, :] + ar1(rng, players.shape, noise_m, rho)
    ball = ball + shift + ar1(rng, ball.shape, noise_m, rho)
    out = np.empty_like(players)
    for i, (s0, s1) in enumerate(zip(starts, np.concatenate([starts[1:], [m]]))):
        perm = np.arange(players.shape[1])
        for team in np.unique(match.slot_team):
            cols = np.flatnonzero(match.slot_team == team)
            perm[cols] = cols[rng.permutation(len(cols))]
        out[s0:s1] = players[s0:s1][:, perm]
    other = Match(match_id=match.match_id + "|second", fps=int(fps), pitch=match.pitch,
                  period=np.zeros(m, dtype=np.int8), ball=ball, players=out,
                  slot_team=match.slot_team.copy())
    truth = np.where(kind == 0, np.nan, truth)
    return other.validate(), truth, kind


def second_opts(opts: dict) -> dict:
    """second_recording keyword arguments from bc_* load options."""
    names = {"bc_fps": ("fps", int), "bc_drift_ppm": ("drift_ppm", float), "bc_cut": ("cut", float),
             "bc_replays": ("replays", float), "bc_ball_missing": ("ball_missing", float),
             "bc_noise_m": ("noise_m", float), "bc_shift_m": ("shift_m", float),
             "bc_camera": ("camera", float), "bc_seed": ("seed", int)}
    return {names[k][0]: names[k][1](v) for k, v in opts.items() if k in names}


def with_second(clean: Match, degraded: Match, opts: dict) -> Match:
    """degraded with ball2 from a simulated second recording of `clean` (V4): aligned by
    src/align.py, or with the true map (broadcast=oracle: the ceiling, what a perfect aligner gives)."""
    from src import align

    other, truth, kind = second_recording(clean, **second_opts(opts))
    if str(opts.get("broadcast")).strip().lower() == "oracle":
        alignment = align.from_truth(degraded, other, truth, (kind == 1) | (kind == 2))
        return align.attach(degraded, other, alignment, bias=False)
    return align.attach(degraded, other, align.align(degraded, other))


def apply(match: Match, opts: dict) -> Match:
    """The clean match degraded by load options (OPTIONS). The ball is hidden on the clean match and
    the players degraded from it (the camera follows the clean ball), then merged, so the ball's
    gaps don't depend on the player options; broadcast=1 then attaches a simulated second
    recording's aligned ball (with_second). clean=1 returns the match untouched."""
    unknown = set(opts) - set(OPTIONS)
    if unknown:
        raise ValueError("unknown degrade options: {}".format(sorted(unknown)))
    if _truthy(opts.get("clean", "")):
        return match
    out = match
    if any(k in opts for k in BALL_OPTIONS):
        if "ball_missing" not in opts:
            raise ValueError("ball_gap_s / ball_seed / ball_quiet need ball_missing")
        out = drop_ball(match, target=float(opts["ball_missing"]),
                        gap_s=float(opts.get("ball_gap_s", 1.0)), seed=int(opts.get("ball_seed", 0)),
                        quiet=float(opts.get("ball_quiet", 0.0)),
                        quiet_s=float(opts.get("ball_quiet_s", QUIET_WINDOW_S)))
    if any(k in opts for k in PLAYER_OPTIONS):
        if any(k in opts for k in ("camera_fill",)) and not float(opts.get("camera", 0)):
            raise ValueError("camera_fill needs camera")
        players = degrade_players(match, camera=float(opts.get("camera", 0.0)),
                                  camera_fill=str(opts.get("camera_fill", "nan")),
                                  player_drop=float(opts.get("player_drop", 0.0)),
                                  player_gap_s=float(opts.get("player_gap_s", PLAYER_GAP_S)),
                                  id_swap=float(opts.get("id_swap", 0.0)),
                                  seed=int(opts.get("player_seed", 0)),
                                  fragment_rate=float(opts.get("fragment", 0.0)),
                                  fragment_gap_s=float(opts.get("fragment_gap_s", FRAG_GAP_S)),
                                  gk_flip=float(opts.get("gk_flip", 0.0)),
                                  gk_flip_s=float(opts.get("gk_flip_s", GK_FLIP_S)),
                                  gk_ghost=_truthy(opts.get("gk_ghost", "")),
                                  gk_hide=_truthy(opts.get("gk_hide", "")))
        out = dataclasses.replace(out, players=players.players,
                                  slot_team=players.slot_team).validate()
        if float(opts.get("player_fill", 0)):
            out = fill_gaps(out, float(opts["player_fill"])).validate()
        if "track_repair" in opts:
            from src import tracks
            out = tracks.apply(out, opts["track_repair"]).validate()
    if any(k in opts for k in BROADCAST_OPTIONS[1:]) and not _truthy(opts.get("broadcast", "")):
        raise ValueError("bc_* options need broadcast=1 (or broadcast=oracle)")
    if _truthy(opts.get("broadcast", "")):
        out = with_second(match, out, opts)
    return out
