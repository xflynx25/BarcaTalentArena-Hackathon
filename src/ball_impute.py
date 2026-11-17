"""Where is the hidden ball? An imputer trained on every visible ball (backlog B29, findings F121-F123).

Why: BAR-ATH's tactical tracking has no ball on ~43% of frames [official DATA.md], and every `learned`
ball feature is blind there. But every frame where an external source *does* track the ball is a free
training example: hide it the way the dropout grid does (src/dropout.py, src/ingest/degrade.py), and
learn to put it back from what is still visible. Label-free at run time (players and the seen ball
only); the labels are never used to fit it.

The estimate at a frame t inside a gap (offline: the whole gap is known, past and future):
  I  = the straight line between the last seen ball before t and the next seen ball after t (same
       period), at t's share of the gap. One-sided gaps (a period starts or ends without a ball) copy
       the seen side; a period with no ball at all starts from the players' centroid.
  ball(t) = I + sum_j sum_k beta_jk b_k(t) (A_j(t) - I)
with anchors A_j, each a point the ball could be near:
  last, next      the last / next seen ball
  flight_last     the last seen ball carried on at its velocity (capped at FLIGHT_S): a pass in flight
  flight_next     the next seen ball traced back at its velocity
  carrier         the player nearest the ball when it was last seen, where he is now, plus his offset
                  to the ball then (capped at OFFSET_M): he dribbles it
  receiver        the player nearest the ball when it is next seen, where he is now, plus that offset
  nearest         the player now nearest I
  cluster         the centroid of the NEAR_K players now nearest I: where a crowd gathers
  centroid        the centroid of every player now on the pitch
and b_k a hat basis on (share of the gap elapsed: FRAC_KNOTS) x (gap length, s: GAP_KNOTS) x (the
players' mean speed: CALM_KNOTS, walking in a stoppage vs running in play), so the weights move from
"follow the carrier" early in a gap to "the receiver" late, from "the line" in short gaps to "the
players" in long ones, and to "where the ball reappears" (the restart spot) in a stoppage. A missing anchor (player off-tracking) counts as I.

The model is translation, rotation and reflection equivariant (the same weights for x and y, all
terms differences of points), and it never looks at which team a player is on, so it can't learn
"home has it more". Ridge regression (closed form, float64), rows weighted so every source counts the
same and every match the same within its source, as `learned.fit`. Deterministic; plain numpy.

`impute(match)` returns the ball with every in-period gap filled. Coefficients come from
src/ball_impute_coefs.json (scripts/fit_ball_impute.py): "all" (every external match, what BAR-ATH
gets), "lomo" (one per held-out match) and "loso" (one per held-out source). fold="auto" uses the
lomo fit for an external match it was trained on (so features built from it are cross-fitted and a
held-out score stays held out) and "all" for any other match, BAR-ATH included.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings

import numpy as np

from src.schema import AWAY, HOME, Match

COEF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ball_impute_coefs.json")

ANCHORS = ("last", "next", "flight_last", "flight_next", "carrier", "receiver", "nearest",
           "cluster", "centroid")
FRAC_KNOTS = (0.0, 0.5, 1.0)  # share of the gap elapsed at the frame
GAP_KNOTS = (0.5, 2.0, 6.0, 20.0)  # gap length, seconds
CALM_KNOTS = (1.0, 2.0)  # players' mean speed over +/- CALM_S, m/s: a stoppage (walking) vs play
CALM_S = 2.0
NEAR_K = 4  # players in the "cluster" anchor
FLIGHT_S = 1.5  # carry the seen ball's velocity at most this long
VELOCITY_S = 0.2  # the seen ball's velocity over this span before the gap (after it, for "next")
MAX_MPS = 35.0  # tracking glitches
OFFSET_M = 1.5  # carrier / receiver offset to the ball, capped
PITCH_MARGIN_M = 5.0  # estimates are clipped to the pitch plus this
RIDGE = 1e-3  # on the mean weighted squared error (m^2)
SPEC = dict(anchors=ANCHORS, frac=FRAC_KNOTS, gap=GAP_KNOTS, calm=CALM_KNOTS, calm_s=CALM_S,
            near_k=NEAR_K, flight_s=FLIGHT_S,
            velocity_s=VELOCITY_S, max_mps=MAX_MPS, offset_m=OFFSET_M, margin_m=PITCH_MARGIN_M,
            ridge=RIDGE, version=2)
SPEC_HASH = hashlib.sha1(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:10]
N_WEIGHTS = len(ANCHORS) * len(FRAC_KNOTS) * len(GAP_KNOTS) * len(CALM_KNOTS)
CHUNK = 20_000


def _hat(values: np.ndarray, knots) -> np.ndarray:
    """(n, len(knots)) piecewise-linear basis, clamped to the ends (as learned.hat, no NaN)."""
    knots = np.asarray(knots, dtype=float)
    v = np.clip(values, knots[0], knots[-1])
    low = np.clip(np.searchsorted(knots, v, side="right") - 1, 0, len(knots) - 2)
    frac = (v - knots[low]) / (knots[low + 1] - knots[low])
    out = np.zeros((len(values), len(knots)))
    rows = np.arange(len(values))
    out[rows, low] = 1.0 - frac
    out[rows, low + 1] = frac
    return out


# -- gap context ------------------------------------------------------------------------------------

def gap_context(ball: np.ndarray, period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(last, next): per frame, the index of the last frame at or before it and the first at or after
    it where the ball is seen, within the same period; -1 where there is none (and outside play)."""
    n = len(period)
    seen = ~np.isnan(ball[:, 0]) & (period > 0)
    last, nxt = np.full(n, -1), np.full(n, -1)
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        s = seen[rows]
        pos = np.arange(len(rows))
        li = np.maximum.accumulate(np.where(s, pos, -1))
        ni = np.minimum.accumulate(np.where(s, pos, len(rows))[::-1])[::-1]
        last[rows] = np.where(li >= 0, rows[np.maximum(li, 0)], -1)
        nxt[rows] = np.where(ni < len(rows), rows[np.minimum(ni, len(rows) - 1)], -1)
    return last, nxt


def context(match: Match) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(last, next, calm) for design(): gap_context and the players' mean speed per frame."""
    from src.models.deadball import player_speed  # local: keeps this module light to import

    last, nxt = gap_context(match.ball, match.period)
    return last, nxt, player_speed(match, CALM_S, 1.0)


def _velocity(ball: np.ndarray, index: np.ndarray, step: int, fps: int, sign: int) -> np.ndarray:
    """(m, 2) the seen ball's velocity at `index` over `step` frames before it (sign -1) or after
    it (sign +1), in the direction of time; 0 where the other end isn't seen or is out of range."""
    other = index + sign * step
    ok = (index >= 0) & (other >= 0) & (other < len(ball))
    out = np.zeros((len(index), 2))
    a = ball[np.where(ok, index, 0)]
    b = ball[np.where(ok, other, 0)]
    v = (a - b) if sign < 0 else (b - a)
    v = v * fps / step
    speed = np.hypot(v[:, 0], v[:, 1])
    v = v * np.minimum(1.0, MAX_MPS / np.maximum(speed, 1e-9))[:, None]
    good = ok & np.isfinite(v).all(axis=1)
    out[good] = v[good]
    return out


def _nearest_slot(players: np.ndarray, ball: np.ndarray, index: np.ndarray) -> np.ndarray:
    """(m,) the slot nearest the ball at frame `index` (-1 where index < 0 or no player is seen)."""
    out = np.full(len(index), -1)
    ok = index >= 0
    if not ok.any():
        return out
    idx = index[ok]
    d = np.hypot(players[idx, :, 0] - ball[idx, None, 0], players[idx, :, 1] - ball[idx, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    best = d.argmin(axis=1)
    found = np.isfinite(d[np.arange(len(idx)), best])
    out[np.flatnonzero(ok)] = np.where(found, best, -1)
    return out


def _follow(players, ball, rows, index, slot):
    """(m, 2) where `slot` (chosen at frame `index`) is at `rows`, plus its offset to the ball at
    `index` (capped at OFFSET_M); NaN where the slot is unknown or not seen at `rows`."""
    out = np.full((len(rows), 2), np.nan)
    ok = slot >= 0
    if not ok.any():
        return out
    r, i, s = rows[ok], index[ok], slot[ok]
    offset = ball[i] - players[i, s]
    norm = np.hypot(offset[:, 0], offset[:, 1])
    offset = offset * np.minimum(1.0, OFFSET_M / np.maximum(norm, 1e-9))[:, None]
    out[ok] = players[r, s] + offset
    return out


def design(match: Match, rows: np.ndarray, ctx=None) -> tuple[np.ndarray, np.ndarray]:
    """(start (m, 2), columns (m, 2, N_WEIGHTS)) for frames `rows` (in play, ball missing): the
    straight-line start I and the basis-weighted anchor offsets A_j - I. The estimate is
    start + columns @ beta."""
    ball, players, fps = match.ball, match.players, match.fps
    last, nxt, calm = context(match) if ctx is None else ctx
    lo, hi = last[rows], nxt[rows]
    has_lo, has_hi = lo >= 0, hi >= 0
    # a one-sided gap copies the seen side; distances in frames
    lo_i, hi_i = np.where(has_lo, lo, hi), np.where(has_hi, hi, lo)
    t_lo = np.where(has_lo, rows - lo, hi - rows).astype(float)
    t_hi = np.where(has_hi, hi - rows, rows - lo).astype(float)
    none = ~has_lo & ~has_hi

    p = players[rows]  # (m, k, 2)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid = np.nanmean(p, axis=1) if p.shape[1] else np.full((len(rows), 2), np.nan)
    b_lo = ball[np.maximum(lo_i, 0)]
    b_hi = ball[np.maximum(hi_i, 0)]
    gap = t_lo + t_hi
    frac = np.where(gap > 0, t_lo / np.maximum(gap, 1), 0.5)
    start = b_lo + frac[:, None] * (b_hi - b_lo)
    start[none] = centroid[none]
    frac[none] = 0.5
    gap_s = np.where(none, GAP_KNOTS[-1], gap / fps)
    start = np.where(np.isnan(start), 0.5 * np.array(match.pitch), start)

    step = max(1, int(round(VELOCITY_S * fps)))
    v_lo = _velocity(ball, np.where(has_lo, lo, -1), step, fps, -1)
    v_hi = _velocity(ball, np.where(has_hi, hi, -1), step, fps, +1)
    anchors = {
        "last": np.where(has_lo[:, None], b_lo, np.nan),
        "next": np.where(has_hi[:, None], b_hi, np.nan),
        "flight_last": np.where(has_lo[:, None],
                                b_lo + v_lo * np.minimum(t_lo / fps, FLIGHT_S)[:, None], np.nan),
        "flight_next": np.where(has_hi[:, None],
                                b_hi - v_hi * np.minimum(t_hi / fps, FLIGHT_S)[:, None], np.nan),
        "carrier": _follow(players, ball, rows, lo, _nearest_slot(players, ball, lo)),
        "receiver": _follow(players, ball, rows, hi, _nearest_slot(players, ball, hi)),
        "centroid": centroid,
    }
    d = np.hypot(p[..., 0] - start[:, None, 0], p[..., 1] - start[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1, kind="stable")[:, :NEAR_K]
    near_d = np.take_along_axis(d, order, axis=1)
    near_p = np.take_along_axis(p, order[..., None], axis=1)
    near_p = np.where(np.isfinite(near_d)[..., None], near_p, np.nan)
    anchors["nearest"] = near_p[:, 0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows with no player: NaN, handled below
        anchors["cluster"] = np.nanmean(near_p, axis=1)
    speed = np.nan_to_num(calm[rows], nan=CALM_KNOTS[-1])
    basis = (_hat(frac, FRAC_KNOTS)[:, :, None, None] * _hat(gap_s, GAP_KNOTS)[:, None, :, None]
             * _hat(speed, CALM_KNOTS)[:, None, None, :]).reshape(len(rows), -1)  # (m, B)
    cols = []
    for name in ANCHORS:
        off = anchors[name] - start
        off = np.where(np.isnan(off), 0.0, off)
        cols.append(off[:, :, None] * basis[:, None, :])  # (m, 2, B)
    return start, np.concatenate(cols, axis=2)


def estimate(start: np.ndarray, columns: np.ndarray, beta: np.ndarray, pitch) -> np.ndarray:
    """(m, 2) start + columns @ beta, clipped to the pitch plus PITCH_MARGIN_M."""
    out = start + columns @ np.asarray(beta, dtype=np.float64)
    out[:, 0] = np.clip(out[:, 0], -PITCH_MARGIN_M, pitch[0] + PITCH_MARGIN_M)
    out[:, 1] = np.clip(out[:, 1], -PITCH_MARGIN_M, pitch[1] + PITCH_MARGIN_M)
    return out


# -- fitting ----------------------------------------------------------------------------------------

def gram(start: np.ndarray, columns: np.ndarray, target: np.ndarray, weight: np.ndarray):
    """(G, g, s): the weighted normal-equation pieces sum w X'X, sum w X'r, sum w for residual
    r = target - start, x and y stacked (one weight per row, used for both coordinates)."""
    x = columns.reshape(-1, columns.shape[2])
    r = (target - start).reshape(-1)
    w = np.repeat(weight, 2)
    return x.T @ (x * w[:, None]), x.T @ (w * r), float(weight.sum())


def solve(pieces: list) -> np.ndarray:
    """beta from summed gram() pieces: minimises mean weighted squared error + RIDGE |beta|^2."""
    G = sum(p[0] for p in pieces)
    g = sum(p[1] for p in pieces)
    s = sum(p[2] for p in pieces)
    return np.linalg.solve(G / s + RIDGE * np.eye(len(g)), g / s)


def weights(keys: list[str]) -> dict:
    """{key: total weight}: every source the same, every match the same within its source (as
    learned.fit). Keys are 'source:id'."""
    sources = sorted({k.split(":", 1)[0] for k in keys})
    per = {s: sum(k.split(":", 1)[0] == s for k in keys) for s in sources}
    return {k: 1.0 / (len(sources) * per[k.split(":", 1)[0]]) for k in keys}


# -- the imputer ------------------------------------------------------------------------------------

_COEFS: dict = {}


def load_coefs(path: str = COEF_FILE) -> dict:
    if path not in _COEFS:
        with open(path) as handle:
            coefs = json.load(handle)
        if coefs.get("spec_hash") != SPEC_HASH:
            raise ValueError("{} was fit for another imputer spec ({} != {}); rerun "
                             "scripts/fit_ball_impute.py".format(path, coefs.get("spec_hash"),
                                                                 SPEC_HASH))
        _COEFS[path] = coefs
    return _COEFS[path]


def coef_digest(path: str = COEF_FILE) -> str:
    """Short hash of the coefficient file's content, for learned's feature hash."""
    with open(path, "rb") as handle:
        return hashlib.sha1(handle.read()).hexdigest()[:10]


def beta_for(match_id: str, fold: str = "auto", path: str = COEF_FILE) -> np.ndarray:
    """The coefficients to use on a match: fold "all", "lomo" / "loso" (KeyError if that match or
    source has no held-out fit), or "auto" (lomo for a training match, else all). A match from a
    source the fit never saw gets "all" under every fold: it is held out by source already (B30, as
    `learned` does since C12)."""
    coefs = load_coefs(path)
    if fold == "all" or match_id.split(":", 1)[0] not in coefs.get("fit", {}).get(
            "sources", (match_id.split(":", 1)[0],)):
        return np.array(coefs["all"])
    if fold == "auto":
        return np.array(coefs["lomo"].get(match_id, coefs["all"]))
    if fold == "lomo":
        return np.array(coefs["lomo"][match_id])
    if fold == "loso":
        return np.array(coefs["loso"][match_id.split(":", 1)[0]])
    raise ValueError("fold must be auto, all, lomo or loso, not {!r}".format(fold))


def impute(match: Match, fold: str = "auto", path: str = COEF_FILE, beta=None) -> np.ndarray:
    """(n, 2) the ball with every in-play gap filled by the estimate; seen frames keep the seen
    ball, frames outside play stay as they are. Deterministic."""
    beta = beta_for(match.match_id, fold, path) if beta is None else np.asarray(beta)
    out = match.ball.copy()
    rows = np.flatnonzero((match.period > 0) & np.isnan(match.ball[:, 0]))
    if not len(rows) or not match.players.shape[1]:
        return out
    ctx = context(match)
    for s in range(0, len(rows), CHUNK):
        part = rows[s:s + CHUNK]
        start, columns = design(match, part, ctx)
        out[part] = estimate(start, columns, beta, match.pitch)
    return out


def nearest_team_call(players: np.ndarray, slot_team: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """(m,) 1.0 if a home player is at least as near `ball` (m, 2) as any away player, 0.0 if an away
    player is nearer, NaN if either team or the ball is missing. players (m, k, 2)."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    out = []
    for team in (HOME, AWAY):
        cols = d[:, slot_team == team]
        out.append(cols.min(axis=1) if cols.shape[1] else np.full(len(d), np.inf))
    home, away = out
    call = (home <= away).astype(float)
    return np.where(np.isfinite(home) & np.isfinite(away), call, np.nan)
