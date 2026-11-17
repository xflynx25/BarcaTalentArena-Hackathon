"""Training for the bigger emission models (backlog B31; inference in src/models/emission.py).

Plain numpy, deterministic (fixed seeds; one BLAS thread per process in scripts/fit_emission.py), on
external matches only. Never imported at run time: src.run only reads the committed coefficients.

Rows: every external match (each dropout draw a "row set", like scripts/fit_learned.py) becomes a
regular grid of frames every STRIDE_S with the home view u_H of learned.features (v4's groups) on
the input grid, stored as uint16 (value * 2**IN_BITS), the label (-1 where it isn't H/A/D or the
frame is outside a half) and each grid point's run bounds (the temporal taps never cross half-time).

Weights: each source counts the same, each row set the same within its source (learned.fit's rule).
Model selection inside a fit: the fit's own matches are split by match (INNER_SHARE of each source,
by a hash of the key) into train and an inner validation set; boosting rounds / epochs stop where the
inner validation log-loss is lowest. The held-out matches of a fold are never seen.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from src.models import emission
from src.models.emission import IN_BITS, LEAF_BITS, W_BITS, snap
from src.models.timing import to_frames
from src.official import A, D, H
from src.schema import Match

STATES = np.array([H, A, D], dtype=np.int8)
INNER_SHARE = 0.15  # share of a fit's matches (per source) kept for early stopping

BOOSTED_SPEC = dict(depth=6, rate=0.15, rounds=500, l2=1.0, bins=32, subsample=0.3, patience=30,
                    min_rounds=50, base="none")  # base="logit": boost from v4's logit fit on the same rows
# taps in STRIDE_S steps (+/- 4 s); weight decay 1e-3 (B31, F128: the +/- 8 s, 1e-4 net fit the
# sources in training better and transferred to a held-out source worse)
TEMPORAL_SPEC = dict(embed=32, hidden=64, taps=[-8, -4, -2, -1, 0, 1, 2, 4, 8],
                     lr=1e-3, batch=512, epochs=40, patience=6, weight_decay=1e-3)


# -- rows -------------------------------------------------------------------------------------------------

def grid_rows(match: Match, groups, spec: dict) -> dict:
    """One row set: {"u": (n, F) uint16 home view, "y": (n,) int8 (0/1/2 = H/A/D, -1 = no target),
    "first", "last": (n,) int32 run bounds in grid positions, "p": (p_team, p_common)}."""
    blocks = emission.match_blocks(match, groups, spec)
    step = max(1, to_frames(emission.STRIDE_S, match.fps))
    index = np.arange(0, match.n_frames, step)
    u_home, _ = emission.views(tuple(b[index] for b in blocks))
    labels = match.labels[index]
    target = (match.period[index] > 0) & np.isin(labels, STATES)
    y = np.where(target, np.searchsorted(STATES, labels), -1).astype(np.int8)
    first, last = emission.run_bounds(match.period[index])
    return {"u": np.rint(u_home * 2.0 ** IN_BITS).astype(np.uint16), "y": y,
            "first": first.astype(np.int32), "last": last.astype(np.int32),
            "p": np.array([blocks[0].shape[1], blocks[2].shape[1]], dtype=np.int32)}


def set_weights(keys: list[str]) -> dict[str, float]:
    """{row-set key: weight of the whole set}: each source 1 / n_sources, each set equal within."""
    sources = sorted({k.split(":", 1)[0] for k in keys})
    per = {s: sum(k.startswith(s + ":") for k in keys) for s in sources}
    return {k: 1.0 / (len(sources) * per[k.split(":", 1)[0]]) for k in keys}


def inner_split(keys: list[str]) -> tuple[list[str], list[str]]:
    """(train, valid) row-set keys, split by match: per source, the INNER_SHARE of matches first in
    a hash order (at least one if the source has 3+ matches) go to validation. Deterministic."""
    matches = sorted({k.split("|", 1)[0] for k in keys})
    valid = set()
    for source in sorted({m.split(":", 1)[0] for m in matches}):
        own = sorted((m for m in matches if m.startswith(source + ":")),
                     key=lambda m: (zlib.crc32(("b31|" + m).encode()), m))
        n = int(round(INNER_SHARE * len(own))) or (1 if len(own) >= 3 else 0)
        valid.update(own[:n])
    return ([k for k in keys if k.split("|", 1)[0] not in valid],
            [k for k in keys if k.split("|", 1)[0] in valid])


def stack(rows: dict, keys: list[str]) -> dict:
    """Concatenate row sets: u, y, row weight w (each set's weight spread over its targets), and run
    bounds shifted to global positions."""
    weights = set_weights(keys)
    us, ys, ws, firsts, lasts = [], [], [], [], []
    offset = 0
    for k in keys:
        r = rows[k]
        n_target = max(1, int((r["y"] >= 0).sum()))
        us.append(r["u"])
        ys.append(r["y"])
        ws.append(np.where(r["y"] >= 0, weights[k] / n_target, 0.0))
        firsts.append(r["first"] + offset)
        lasts.append(r["last"] + offset)
        offset += len(r["y"])
    p = rows[keys[0]]["p"]
    return {"u": np.concatenate(us), "y": np.concatenate(ys).astype(np.int64),
            "w": np.concatenate(ws), "first": np.concatenate(firsts),
            "last": np.concatenate(lasts), "p_team": int(p[0]), "p_common": int(p[1])}


def class_prior(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted H/A/D shares over targets (what the decoder's prior=1 subtracts)."""
    prior = np.array([w[y == k].sum() for k in range(3)])
    return prior / prior.sum()


def base_dead(prior: np.ndarray) -> float:
    """Initial D logit: log-odds of D against the mean team share."""
    return float(np.log(prior[2]) - np.log((prior[0] + prior[1]) / 2))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    z = z - z.max(axis=1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-(w * lp[np.arange(len(y)), y]).sum() / w.sum())


# -- boosted oblivious trees ----------------------------------------------------------------------------

def bin_edges(u_int: np.ndarray, p_team: int, bins: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Per feature, sorted int thresholds k (a split tests value_int > k). Team columns share edges
    with their mirror column, so the away view's bins are the home view's, permuted. Zero gets its
    own bin; the rest are quantiles of the nonzero values (on a subsample)."""
    n, F = u_int.shape
    sample = u_int[rng.choice(n, size=min(n, 200_000), replace=False)]
    edges = [None] * F
    for f in range(F):
        if f >= 2 * p_team:
            values = sample[:, f]
        elif f < p_team:
            values = np.concatenate([sample[:, f], sample[:, f + p_team]])
        else:
            edges[f] = edges[f - p_team]
            continue
        distinct = np.unique(values)
        if len(distinct) <= bins:
            cuts = distinct[:-1]
        else:
            nonzero = values[values > 0]
            # at most bins - 1 thresholds, so a bin index (thresholds below the value) is < bins
            qs = np.quantile(nonzero, np.linspace(0, 1, bins - 1)[:-1], method="lower") if len(nonzero) else []
            cuts = np.unique(np.concatenate([[0], np.asarray(qs, dtype=np.int64)]))
            cuts = cuts[cuts < distinct[-1]]
        edges[f] = np.asarray(cuts, dtype=np.int64)
    return edges


def to_bins(u_int: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """(F, n) uint8 bin of each value: the number of thresholds below it."""
    return np.stack([np.searchsorted(edges[f], u_int[:, f], side="left").astype(np.uint8)
                     for f in range(u_int.shape[1])])


def _histograms(xb_views, nodes, grads, hesss, sel, n_nodes: int, bins: int):
    """(G, H) histograms (n_nodes, F, bins) over every view's rows (or the rows in sel)."""
    F = xb_views[0].shape[0]
    hg, hh = np.zeros((n_nodes, F, bins)), np.zeros((n_nodes, F, bins))
    size = n_nodes * bins
    for i, (xb, node, g, h) in enumerate(zip(xb_views, nodes, grads, hesss)):
        if sel is not None:
            xb, node, g, h = xb[:, sel[i]], node[sel[i]], g[sel[i]], h[sel[i]]
        base = node * bins
        for f in range(F):
            idx = base + xb[f]
            hg[:, f, :] += np.bincount(idx, weights=g, minlength=size).reshape(n_nodes, bins)
            hh[:, f, :] += np.bincount(idx, weights=h, minlength=size).reshape(n_nodes, bins)
    return hg, hh


def grow_oblivious(xb_views, grads, hesss, edges, depth: int, l2: float, bins: int):
    """One oblivious tree on binned views. xb_views: list of (F, n) uint8 (bins of each view's
    rows), grads / hesss: matching (n,) arrays. Returns (feat, split bin, leaf G, leaf H) where
    a row goes right at level d when its bin of feat[d] > split[d]."""
    F = xb_views[0].shape[0]
    nodes = [np.zeros(x.shape[1], dtype=np.int64) for x in xb_views]
    feat, split = [], []
    hg, hh = _histograms(xb_views, nodes, grads, hesss, None, 1, bins)
    for d in range(depth):
        n_nodes = 1 << d
        if d:  # histogram subtraction: only the rows that went right at level d - 1 are counted
            half = n_nodes >> 1
            sel = [np.flatnonzero(node >= half) for node in nodes]
            rg, rh = _histograms(xb_views, [node - half for node in nodes], grads, hesss, sel, half, bins)
            hg, hh = np.concatenate([hg - rg, rg]), np.concatenate([hh - rh, rh])
        cg, ch = np.cumsum(hg, axis=2), np.cumsum(hh, axis=2)  # left = bins <= s
        tg, th = cg[:, :, -1:], ch[:, :, -1:]
        gain = (cg ** 2 / (ch + l2) + (tg - cg) ** 2 / (th - ch + l2) - tg ** 2 / (th + l2)).sum(axis=0)
        n_cuts = np.array([len(e) for e in edges])
        valid = np.arange(bins)[None, :] < n_cuts[:, None]  # split s needs threshold s to exist
        gain = np.where(valid, gain, -np.inf)
        best = int(np.argmax(gain))
        f_best, s_best = divmod(best, bins)
        if not np.isfinite(gain.flat[best]) or gain.flat[best] <= 1e-12:
            feat.append(-1)
            split.append(-1)
            continue
        feat.append(f_best)
        split.append(s_best)
        for i, xb in enumerate(xb_views):
            nodes[i] = nodes[i] | ((xb[f_best] > s_best).astype(np.int64) << d)
    n_leaves = 1 << depth
    leaf_g, leaf_h = np.zeros(n_leaves), np.zeros(n_leaves)
    for node, g, h in zip(nodes, grads, hesss):
        leaf_g += np.bincount(node, weights=g, minlength=n_leaves)
        leaf_h += np.bincount(node, weights=h, minlength=n_leaves)
    return np.array(feat), np.array(split), leaf_g, leaf_h


def apply_oblivious(xb: np.ndarray, feat, split, leaf: np.ndarray) -> np.ndarray:
    index = np.zeros(xb.shape[1], dtype=np.int64)
    for d, (f, s) in enumerate(zip(feat, split)):
        if f >= 0:
            index |= (xb[f] > s).astype(np.int64) << d
    return leaf[index]


def _float_views(u_int: np.ndarray, perm: np.ndarray):
    u = u_int.astype(np.float64) / 2.0 ** IN_BITS
    return u, u[:, perm]


def fit_linear(u_int: np.ndarray, y: np.ndarray, w: np.ndarray, p_team: int) -> dict:
    """learned's conditional logit (fit_logit: Newton, its L2) on these rows, as exact weights:
    {"team": [a, b, c] on the home view, "dead": e}, on the 2**-LEAF_BITS grid."""
    from src.models import learned

    u = u_int.astype(np.float64) / 2.0 ** IN_BITS
    blocks = (u[:, :p_team], u[:, p_team:2 * p_team], u[:, 2 * p_team:])
    theta = learned.fit_logit(blocks, y, w / w.sum())
    a, b, c, e = learned._split(theta, p_team)
    return {"team": snap(np.concatenate([a, b, c]), LEAF_BITS), "dead": snap(e, LEAF_BITS)}


def fit_boosted(train: dict, valid: dict, spec: dict, seed: int, log=None) -> dict:
    """Symmetric boosted oblivious trees: z_H = f(u_H), z_A = f(u_A), z_D = g(u_H) + g(u_A) + c_D.
    Each round grows one f tree and one g tree from the same probabilities (Newton leaves, L2,
    learning rate, a row subsample), with leaves snapped to the 2**-LEAF_BITS grid."""
    rng = np.random.default_rng(seed)
    p_team = train["p_team"]
    perm = emission.mirror_index(p_team, train["p_common"])
    bins = spec["bins"]
    keep = train["y"] >= 0
    u, y, w = train["u"][keep], train["y"][keep], train["w"][keep]
    w = w / w.mean()
    vkeep = valid["y"] >= 0
    vu, vy, vw = valid["u"][vkeep], valid["y"][vkeep], valid["w"][vkeep]
    edges = bin_edges(u, p_team, bins, rng)
    xb = to_bins(u, edges)
    xb_away = xb[perm]
    vxb = to_bins(vu, edges)
    vxb_away = vxb[perm]
    prior = class_prior(y, w)
    lin = None
    if spec.get("base", "none") == "logit":  # start from the logit's logits; trees learn corrections
        lin = fit_linear(u, y, w, p_team)
        c_d = 0.0
        z = emission.linear_logits(*_float_views(u, perm), lin)
        vz = emission.linear_logits(*_float_views(vu, perm), lin)
    else:
        c_d = float(snap(base_dead(prior), LEAF_BITS))
        z = np.zeros((len(y), 3))
        z[:, 2] = c_d
        vz = np.zeros((len(vy), 3))
        vz[:, 2] = c_d
    onehot = np.eye(3)[y]
    trees = {"f": [], "g": []}
    best_loss, best_round, history = log_loss(vz, vy, vw), 0, []
    started = time.time()
    for r in range(spec["rounds"]):
        p = softmax(z)
        sub = np.flatnonzero(rng.random(len(y)) < spec["subsample"])
        resid = (p - onehot) * w[:, None]
        hess = p * (1 - p) * w[:, None]
        views = [np.ascontiguousarray(xb[:, sub]), np.ascontiguousarray(xb_away[:, sub])]
        for name, cols in (("f", (0, 1)), ("g", (2, 2))):
            feat, split, lg, lh = grow_oblivious(
                views, [resid[sub, cols[0]], resid[sub, cols[1]]],
                [hess[sub, cols[0]], hess[sub, cols[1]]], edges, spec["depth"], spec["l2"], bins)
            leaf = snap(-spec["rate"] * lg / (lh + spec["l2"]), LEAF_BITS)
            trees[name].append((feat, split, leaf))
            if name == "f":
                z[:, 0] += apply_oblivious(xb, feat, split, leaf)
                z[:, 1] += apply_oblivious(xb_away, feat, split, leaf)
                vz[:, 0] += apply_oblivious(vxb, feat, split, leaf)
                vz[:, 1] += apply_oblivious(vxb_away, feat, split, leaf)
            else:
                z[:, 2] += apply_oblivious(xb, feat, split, leaf) + apply_oblivious(xb_away, feat, split, leaf)
                vz[:, 2] += apply_oblivious(vxb, feat, split, leaf) + apply_oblivious(vxb_away, feat, split, leaf)
        loss = log_loss(vz, vy, vw)
        history.append(round(loss, 6))
        if loss < best_loss - 1e-7:
            best_loss, best_round = loss, r + 1
        if log and (r + 1) % 50 == 0:
            log("  round {} valid {:.5f} best {} ({:.0f} s)".format(r + 1, loss, best_round,
                                                                   time.time() - started))
        if r + 1 >= spec["min_rounds"] and r + 1 - best_round >= spec["patience"]:
            break
    best_round = max(best_round, 1)
    model = {"c_d": c_d}
    if lin is not None:
        model["lin"] = lin
    for name in ("f", "g"):
        kept = trees[name][:best_round]
        thr = np.array([[edges[f][s] + 0.5 if f >= 0 else np.inf for f, s in zip(feat, split)]
                        for feat, split, _ in kept]) / 2.0 ** IN_BITS
        model[name] = {"feat": np.array([[max(f, 0) for f in feat] for feat, _, _ in kept]),
                       "thr": thr, "leaf": np.array([leaf for _, _, leaf in kept])}
    return {"model": model, "log_prior": np.log(prior), "best": best_round,
            "valid_loss": best_loss, "history": history}


# -- temporal net ------------------------------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict, lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= (self.lr * mhat / (np.sqrt(vhat) + eps)).astype(params[k].dtype)


def _net_forward(params: dict, x_views, k: int):
    """x_views: [(B*K, F) float32 per view]. Returns z (B, 3) and a cache for backward."""
    cache, outs = [], []
    for x in x_views:
        z1 = x @ params["w1"] + params["b1"]
        a1 = np.maximum(z1, 0)
        a1r = a1.reshape(-1, k * a1.shape[1])
        s = a1r @ params["w2"] + params["b2"]
        h = np.maximum(s, 0)
        outs.append((h @ params["vf"], h @ params["vg"]))
        cache.append((x, z1, a1r, s, h))
    z = np.stack([outs[0][0], outs[1][0], outs[0][1] + outs[1][1] + params["c_d"]], axis=1)
    return z, cache


def _net_backward(params: dict, cache, dz: np.ndarray, k: int, weight_decay: float) -> dict:
    grads = {name: np.zeros_like(v) for name, v in params.items()}
    grads["c_d"] += dz[:, 2].sum()
    for view, (x, z1, a1r, s, h) in enumerate(cache):
        df, dg = dz[:, view], dz[:, 2]
        grads["vf"] += h.T @ df
        grads["vg"] += h.T @ dg
        ds = (np.outer(df, params["vf"]) + np.outer(dg, params["vg"])) * (s > 0)
        grads["w2"] += a1r.T @ ds
        grads["b2"] += ds.sum(axis=0)
        da1 = (ds @ params["w2"].T).reshape(z1.shape) * (z1 > 0)
        grads["w1"] += x.T @ da1
        grads["b1"] += da1.sum(axis=0)
    for name in ("w1", "w2", "vf", "vg"):
        grads[name] += weight_decay * params[name]
    return grads


def _gather(data: dict, rows: np.ndarray, taps: np.ndarray, perm: np.ndarray):
    pos = np.clip(rows[:, None] + taps[None, :], data["first"][rows][:, None], data["last"][rows][:, None])
    x = data["u"][pos.ravel()].astype(np.float32) * np.float32(2.0 ** -IN_BITS)
    return [x, x[:, perm]]


def net_logits(params: dict, data: dict, rows: np.ndarray, taps: np.ndarray, perm: np.ndarray,
               batch: int = 4096) -> np.ndarray:
    out = []
    for s in range(0, len(rows), batch):
        z, _ = _net_forward(params, _gather(data, rows[s:s + batch], taps, perm), len(taps))
        out.append(z)
    return np.concatenate(out) if out else np.zeros((0, 3))


def quantize_net(params: dict) -> dict:
    """The trained weights on the inference grids (float64), clipped to the exactness bounds."""
    def w(x):
        return snap(np.clip(x, -emission.W_MAX, emission.W_MAX), W_BITS)

    return {"w1": w(params["w1"]), "b1": snap(params["b1"], W_BITS + IN_BITS),
            "w2": w(params["w2"]), "b2": snap(params["b2"], W_BITS + IN_BITS),
            "vf": w(params["vf"]), "vg": w(params["vg"]),
            "c_d": float(snap(params["c_d"], W_BITS + IN_BITS)[0])}


def fit_temporal(train: dict, valid: dict, spec: dict, seed: int, log=None) -> dict:
    """The symmetric temporal conv net (src/models/emission.py), trained with Adam on the weighted
    log-loss; the epoch with the lowest inner validation loss is kept."""
    rng = np.random.default_rng(seed)
    n_in = train["u"].shape[1]
    perm = emission.mirror_index(train["p_team"], train["p_common"])
    taps = np.asarray(spec["taps"], dtype=np.int64)
    k, embed, hidden = len(taps), spec["embed"], spec["hidden"]
    rows = np.flatnonzero(train["y"] >= 0)
    vrows = np.flatnonzero(valid["y"] >= 0)
    w = train["w"] / train["w"][rows].mean()
    prior = class_prior(train["y"][rows], w[rows])
    params = {"w1": (rng.standard_normal((n_in, embed)) * np.sqrt(2.0 / 20)).astype(np.float32),
              "b1": np.zeros(embed, np.float32),
              "w2": (rng.standard_normal((k * embed, hidden)) * np.sqrt(2.0 / (k * embed))).astype(np.float32),
              "b2": np.zeros(hidden, np.float32),
              "vf": (rng.standard_normal(hidden) * 0.01).astype(np.float32),
              "vg": (rng.standard_normal(hidden) * 0.01).astype(np.float32),
              "c_d": np.array([base_dead(prior)], np.float32)}
    opt = Adam(params, spec["lr"])
    eye = np.eye(3, dtype=np.float32)
    best = (log_loss(net_logits(params, valid, vrows, taps, perm), valid["y"][vrows], valid["w"][vrows]),
            0, {n: v.copy() for n, v in params.items()})
    history, stale, started = [], 0, time.time()
    for epoch in range(spec["epochs"]):
        order = rows[rng.permutation(len(rows))]
        for s in range(0, len(order), spec["batch"]):
            batch = order[s:s + spec["batch"]]
            z, cache = _net_forward(params, _gather(train, batch, taps, perm), k)
            wb = w[batch].astype(np.float32)
            dz = (softmax(z).astype(np.float32) - eye[train["y"][batch]]) * (wb / wb.sum())[:, None]
            opt.step(params, _net_backward(params, cache, dz, k, spec["weight_decay"]))
        loss = log_loss(net_logits(params, valid, vrows, taps, perm), valid["y"][vrows], valid["w"][vrows])
        history.append(round(loss, 6))
        if log:
            log("  epoch {} valid {:.5f} ({:.0f} s)".format(epoch + 1, loss, time.time() - started))
        if loss < best[0] - 1e-6:
            best, stale = (loss, epoch + 1, {n: v.copy() for n, v in params.items()}), 0
        else:
            stale += 1
            if stale == 2:
                opt.lr /= 2
            if stale >= spec["patience"]:
                break
    return {"model": quantize_net(best[2]), "log_prior": np.log(prior), "best": best[1],
            "valid_loss": best[0], "history": history}


FITTERS = {"boosted": fit_boosted, "temporal": fit_temporal}
SPECS = {"boosted": BOOSTED_SPEC, "temporal": TEMPORAL_SPEC}
