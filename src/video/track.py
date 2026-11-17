"""Link per-frame ball candidates into one ball track with a confidence (backlog V2).

    track = link(frames, positions, scores, fps_step)   # one stretch of evenly spaced frames
    ball, conf = to_frames(track, n_frames)             # (n, 2) metres / NaN, (n,) logit / NaN

Viterbi over each frame's candidates plus a "not seen" state. Costs are additive log-odds-like
numbers, float64 elementwise, ties to the lowest index, so the path is the same on every machine:
  - picking candidate j costs -score_j (the scorer's logit: a candidate is worth taking above 0);
  - "not seen" costs 0;
  - a move of d metres between consecutive picks costs (d / MOVE_SCALE_M)^2 above FREE_MOVE_M per
    step and is impossible beyond MAX_SPEED_MPS (a kicked ball reaches ~30 m/s);
  - entering or leaving "not seen" costs SWITCH each time (so a lone spike far from the track loses).
The confidence of a picked frame is its candidate's logit; frames in "not seen" get NaN.
"""

from __future__ import annotations

import numpy as np

MAX_SPEED_MPS = 35.0
FREE_MOVE_M = 0.5
MOVE_SCALE_M = 2.0
SWITCH = 1.0
GRID_M = 0.01  # positions in the output are snapped to 1 cm
GRID_LOGIT = 1e-3


def link(frames: np.ndarray, positions: list, scores: list, step_s: float) -> dict:
    """frames (T,) frame indices of the stretch, positions[t] (m_t, 2) metres, scores[t] (m_t,).
    Returns {"frame", "x", "y", "logit", "pick"} with NaN / -1 where the ball is not seen."""
    T = len(frames)
    max_move = MAX_SPEED_MPS * step_s
    # state 0 = not seen, 1 + j = candidate j
    cost = [np.concatenate([[0.0], -np.asarray(scores[0], dtype=np.float64)])]
    back = [np.zeros(len(cost[0]), dtype=np.int64)]
    for t in range(1, T):
        prev_p = np.asarray(positions[t - 1], dtype=np.float64).reshape(-1, 2)
        cur_p = np.asarray(positions[t], dtype=np.float64).reshape(-1, 2)
        prev_c = cost[-1]
        m = len(cur_p)
        new = np.empty(m + 1)
        arg = np.empty(m + 1, dtype=np.int64)
        # to "not seen": from not seen (0) or from any candidate (+SWITCH)
        from_cands = prev_c[1:] + SWITCH
        if len(from_cands) and from_cands.min() < prev_c[0]:
            k = int(np.argmin(from_cands))
            new[0], arg[0] = from_cands[k], k + 1
        else:
            new[0], arg[0] = prev_c[0], 0
        if m:
            emit = -np.asarray(scores[t], dtype=np.float64)
            if len(prev_p):
                d = np.hypot(cur_p[:, None, 0] - prev_p[None, :, 0], cur_p[:, None, 1] - prev_p[None, :, 1])
                move = np.maximum(d - FREE_MOVE_M, 0.0) / MOVE_SCALE_M
                trans = np.where(d <= max_move, move * move, np.inf) + prev_c[None, 1:]
                k = np.argmin(trans, axis=1)
                best = trans[np.arange(m), k]
            else:
                best = np.full(m, np.inf)
                k = np.zeros(m, dtype=np.int64)
            enter = prev_c[0] + SWITCH
            take_enter = enter <= best
            new[1:] = np.where(take_enter, enter, best) + emit
            arg[1:] = np.where(take_enter, 0, k + 1)
        cost.append(new)
        back.append(arg)
    state = int(np.argmin(cost[-1]))
    picks = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        picks[t] = state - 1
        state = int(back[t][state])
    x = np.full(T, np.nan)
    y = np.full(T, np.nan)
    logit = np.full(T, np.nan)
    for t in range(T):
        j = picks[t]
        if j >= 0:
            p = np.asarray(positions[t], dtype=np.float64).reshape(-1, 2)[j]
            x[t], y[t] = p
            logit[t] = float(np.asarray(scores[t], dtype=np.float64)[j])
    return {"frame": np.asarray(frames), "x": snap(x, GRID_M), "y": snap(y, GRID_M),
            "logit": snap(logit, GRID_LOGIT), "pick": picks}


def snap(values: np.ndarray, grid: float) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64) / grid) * grid


def to_frames(track: dict, n_frames: int, max_gap: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Spread a stretch's picks to every frame: linear between two picks at most max_gap frames
    apart (the 5 Hz grid: 0.4 s), the pick itself on its own frame, NaN elsewhere. The
    confidence between two picks is the lower of the two logits."""
    ball = np.full((n_frames, 2), np.nan)
    conf = np.full(n_frames, np.nan)
    f = np.asarray(track["frame"], dtype=np.int64)
    seen = np.isfinite(track["x"])
    fs, xs, ys, ls = f[seen], track["x"][seen], track["y"][seen], track["logit"][seen]
    for i in range(len(fs)):
        if 0 <= fs[i] < n_frames:
            ball[fs[i]] = (xs[i], ys[i])
            conf[fs[i]] = ls[i]
        if i + 1 < len(fs) and 0 < fs[i + 1] - fs[i] <= max_gap:
            a, b = fs[i], fs[i + 1]
            w = (np.arange(a + 1, b) - a) / float(b - a)
            idx = np.arange(a + 1, b)
            ok = (idx >= 0) & (idx < n_frames)
            ball[idx[ok], 0] = snap(xs[i] + (xs[i + 1] - xs[i]) * w[ok], GRID_M)
            ball[idx[ok], 1] = snap(ys[i] + (ys[i + 1] - ys[i]) * w[ok], GRID_M)
            conf[idx[ok]] = min(ls[i], ls[i + 1])
    return ball, conf
