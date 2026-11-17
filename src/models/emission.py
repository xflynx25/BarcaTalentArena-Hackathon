"""Bigger emission models for the switch decoder (backlog B31, findings F127-F129).

`learned` (v4) scores each frame with a ~400-weight conditional logit over hat-basis features (F55).
This module keeps everything else of v4 (LearnedDecoder.decode: the Viterbi costs, onset snap and
short-spell filter, plus the hand-set reach evidence) and swaps in a bigger emission model over the
*same* per-frame features (learned.features with v4's groups):

  boosted   gradient-boosted oblivious trees (every node of a level splits on the same feature and
            threshold, as in CatBoost), so features can interact, which the logit can't do.
  temporal  a small 1-D convolutional network: each frame's features -> EMBED units (ReLU), then one
            dilated temporal layer over those embeddings at the taps around the frame (+/- 8 s by
            default) -> HIDDEN units (ReLU) -> the scores. It sees what happened before and after,
            beyond the logit's fixed 0.5 s / 2 s windows.

Both have the home/away symmetry built in, like the logit: with the home view u_H = [T_home, T_away,
C] and the away view u_A = [T_away, T_home, C] (learned.features' blocks),
    z_H = f(u_H),   z_A = f(u_A),   z_D = g(u_H) + g(u_A) + c_D
so swapping the teams swaps H and A exactly, and no home advantage can be learned.

Run-time determinism across machines (the CSV must regenerate byte for byte, findings F87): inference
is exact fixed-point arithmetic in numpy float64. Inputs are snapped to a 2**-IN_BITS grid, net weights
to 2**-W_BITS and activations to 2**-IN_BITS; tree thresholds sit halfway between input grid points
and tree leaves are multiples of 2**-LEAF_BITS. Every product and partial sum is then an exact float64
(bounded well inside 53 bits; check_bounds() asserts it when a file is written and loaded), so a sum
has the same value in any order: BLAS blocking, SIMD width and thread count can't change it.
log-softmax uses det_exp / det_log, built only from IEEE-exact operations (+, -, *, /, rint, ldexp,
frexp), not the platform's exp / log. The features themselves (shared with v4) are snapped to the
input grid first, so a last-bit difference there only matters within ~1e-16 of a grid boundary.

Training (src/emission_train.py, scripts/fit_emission.py) uses external data only. A coefficient file
holds an "all" fit (what src.run uses), grouped-fold fits ("folds" + "fold_of": each external match
gets the fit that never saw it, read by fold=lomo) and per-source fits ("loso"), like
learned_coefs.json.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from src.models import learned
from src.models.decoder import B3_SETTINGS, SwitchDecoder, shift
from src.models.learned import LearnedDecoder
from src.models.timing import to_frames
from src.schema import Match

HERE = os.path.dirname(os.path.abspath(__file__))
BOOSTED_FILE = os.path.join(HERE, "boosted_coefs.json")
TEMPORAL_FILE = os.path.join(HERE, "temporal_coefs.json")
KINDS = ("boosted", "temporal")

GROUPS = learned.GROUPS  # v4's feature groups: the same features as the shipped logit
STRIDE_S = learned.FIT_STRIDE_S  # training grid step, and the unit of the temporal taps
IN_BITS = 12  # inputs and activations on a 2**-12 grid
W_BITS = 16  # net weights on a 2**-16 grid
LEAF_BITS = 24  # tree leaves (and constants) on a 2**-24 grid
ACT_MAX = 2.0 ** 10  # activations are clipped to [0, ACT_MAX]
W_MAX = 8.0  # net weights are clipped to [-W_MAX, W_MAX]


# -- exact arithmetic ---------------------------------------------------------------------------------

def snap(x, bits: int) -> np.ndarray:
    """x rounded to the nearest multiple of 2**-bits, float64 (exact: scaling by a power of 2)."""
    return np.rint(np.asarray(x, dtype=np.float64) * 2.0 ** bits) / 2.0 ** bits


def to_int(x, bits: int) -> list[int]:
    """x on the 2**-bits grid as plain ints (how a coefficient file stores it)."""
    return [int(v) for v in np.rint(np.asarray(x, dtype=np.float64).ravel() * 2.0 ** bits)]


def from_int(values, bits: int, shape=None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64) / 2.0 ** bits
    return out.reshape(shape) if shape is not None else out


LN2_HI = 6.93147180369123816490e-01  # ln 2 split so that n * LN2_HI is exact for |n| < 2**11
LN2_LO = 1.90821492927058770002e-10
INV_LN2 = 1.44269504088896338700e+00
EXP_COEFS = tuple(1.0 / math.factorial(k) for k in range(14))  # 1/k!, k = 0..13


def det_exp(x: np.ndarray) -> np.ndarray:
    """exp(x), ~1e-16 relative error, from IEEE-exact operations only (same bits on every machine).
    x is clipped to [-700, 700]."""
    x = np.clip(np.asarray(x, dtype=np.float64), -700.0, 700.0)
    n = np.rint(x * INV_LN2)
    r = (x - n * LN2_HI) - n * LN2_LO  # |r| <= 0.35
    poly = np.full_like(r, EXP_COEFS[-1])
    for c in EXP_COEFS[-2::-1]:  # Horner on sum r^k / k!, k = 0..13
        poly = poly * r + c
    return np.ldexp(poly, n.astype(np.int64))


def det_log(x: np.ndarray) -> np.ndarray:
    """log(x) for x > 0, ~1e-16 relative error, from IEEE-exact operations only."""
    m, e = np.frexp(np.asarray(x, dtype=np.float64))  # x = m * 2**e, m in [0.5, 1)
    low = m < 0.70710678118654752440
    m = np.where(low, m * 2.0, m)
    e = np.where(low, e - 1, e).astype(np.float64)
    s = (m - 1.0) / (m + 1.0)  # |s| <= 0.1716
    s2 = s * s
    poly = np.full_like(s, 1.0 / 31.0)
    for k in range(29, 0, -2):  # log m = 2 (s + s^3/3 + ... + s^31/31)
        poly = poly * s2 + 1.0 / k
    return e * LN2_HI + (e * LN2_LO + 2.0 * s * poly)


def det_log_softmax(z: np.ndarray) -> np.ndarray:
    """(n, 3) log-softmax of exact logits, the same bits on every machine."""
    z = np.asarray(z, dtype=np.float64)
    s = z - z.max(axis=1, keepdims=True)
    e = det_exp(s)
    total = (e[:, 0] + e[:, 1]) + e[:, 2]  # fixed order
    return s - det_log(total)[:, None]


# -- inputs ---------------------------------------------------------------------------------------------

def views(blocks) -> tuple[np.ndarray, np.ndarray]:
    """(u_home, u_away), float64 on the input grid: [T_home, T_away, C] and [T_away, T_home, C]."""
    th, ta, cm = (snap(b, IN_BITS) for b in blocks)
    return np.hstack([th, ta, cm]), np.hstack([ta, th, cm])


def mirror_index(p_team: int, p_common: int) -> np.ndarray:
    """Column permutation taking the home view to the away view (and back)."""
    team = np.arange(p_team)
    return np.concatenate([team + p_team, team, 2 * p_team + np.arange(p_common)])


def run_bounds(period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(first, last) index of the run of equal period values each position is in."""
    period = np.asarray(period)
    n = len(period)
    breaks = np.flatnonzero(period[1:] != period[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [n])) - 1
    run = np.searchsorted(starts, np.arange(n), side="right") - 1
    return starts[run], stops[run]


def tap_index(first: np.ndarray, last: np.ndarray, offsets) -> np.ndarray:
    """(n, len(offsets)) positions t + offset, clipped to t's run (so context never crosses half-time)."""
    index = np.arange(len(first))[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
    return np.clip(index, first[:, None], last[:, None])


# -- trees ------------------------------------------------------------------------------------------------

def trees_eval(cols: np.ndarray, ens: dict) -> np.ndarray:
    """Sum of an oblivious-tree ensemble's leaves, exact. cols: (F, n) inputs, one row per feature.
    ens: feat (T, d) int, thr (T, d) float (between input grid points, +inf = no split), leaf
    (T, 2**d) on the leaf grid. A leaf's index has bit d set when level d's test is true."""
    feat, thr, leaf = ens["feat"], ens["thr"], ens["leaf"]
    out = np.zeros(cols.shape[1])
    for t in range(len(feat)):
        index = np.zeros(cols.shape[1], dtype=np.int64)
        for d in range(feat.shape[1]):
            index |= (cols[feat[t, d]] > thr[t, d]).astype(np.int64) << d
        out += leaf[t][index]
    return out


def linear_logits(u_home: np.ndarray, u_away: np.ndarray, lin: dict) -> np.ndarray:
    """The logit's logits (learned.logits' structure) from the views, exact: z_H = w . u_H,
    z_A = w . u_A, z_D = e . (T_home + T_away), weights on the 2**-LEAF_BITS grid."""
    p = len(lin["dead"])
    teams = u_home[:, :p] + u_home[:, p:2 * p]  # exact: two grid values
    return np.stack([u_home @ lin["team"], u_away @ lin["team"], teams @ lin["dead"]], axis=1)


def boosted_logits(u_home: np.ndarray, u_away: np.ndarray, model: dict) -> np.ndarray:
    """Trees (plus the logit they were boosted from, if the file has one: spec base=logit)."""
    f, g = model["f"], model["g"]
    c_home, c_away = np.ascontiguousarray(u_home.T), np.ascontiguousarray(u_away.T)
    g_sum = trees_eval(c_home, g) + trees_eval(c_away, g)
    z = np.stack([trees_eval(c_home, f), trees_eval(c_away, f), g_sum + model["c_d"]], axis=1)
    if "lin" in model:
        z = z + linear_logits(u_home, u_away, model["lin"])
    return z


# -- temporal net -----------------------------------------------------------------------------------------

def net_embed(u: np.ndarray, model: dict) -> np.ndarray:
    """(n, EMBED) exact first layer (per frame) of the temporal net for one view."""
    return np.clip(snap(np.maximum(u @ model["w1"] + model["b1"], 0.0), IN_BITS), 0.0, ACT_MAX)


def net_temporal(a: np.ndarray, taps: np.ndarray, model: dict) -> np.ndarray:
    """(len(taps), HIDDEN) exact temporal layer: a (all frames) at the taps (rows of positions)."""
    stacked = a[taps].reshape(len(taps), -1)  # (n, K * EMBED), tap-major
    return np.clip(snap(np.maximum(stacked @ model["w2"] + model["b2"], 0.0), IN_BITS), 0.0, ACT_MAX)


def net_hidden(u: np.ndarray, taps: np.ndarray, model: dict) -> np.ndarray:
    """(n, HIDDEN) exact hidden layer of the temporal net for one view; taps (n, K) positions."""
    return net_temporal(net_embed(u, model), taps, model)


def temporal_out(h_home: np.ndarray, h_away: np.ndarray, model: dict) -> np.ndarray:
    vf, vg = model["vf"], model["vg"]
    return np.stack([h_home @ vf, h_away @ vf, (h_home @ vg + h_away @ vg) + model["c_d"]], axis=1)


def temporal_logits(u_home: np.ndarray, u_away: np.ndarray, taps: np.ndarray, model: dict) -> np.ndarray:
    return temporal_out(net_hidden(u_home, taps, model), net_hidden(u_away, taps, model), model)


# -- coefficient files --------------------------------------------------------------------------------

def encode_model(kind: str, model: dict) -> dict:
    """A fitted model (float arrays already on their grids) as JSON-ready ints."""
    if kind == "boosted":
        out = {"c_d": to_int(model["c_d"], LEAF_BITS)[0]}
        for name in ("f", "g"):
            ens = model[name]
            thr = np.asarray(ens["thr"], dtype=np.float64)
            # thresholds sit at (k + 0.5) * 2**-IN_BITS: store k; no split (+inf) as None
            k = [[None if not np.isfinite(v) else int(np.floor(v * 2.0 ** IN_BITS)) for v in row]
                 for row in thr]
            out[name] = {"feat": np.asarray(ens["feat"]).astype(int).tolist(), "thr": k,
                         "leaf": [to_int(row, LEAF_BITS) for row in ens["leaf"]]}
        if "lin" in model:
            out["lin"] = {k: to_int(v, LEAF_BITS) for k, v in model["lin"].items()}
        return out
    return {"w1": to_int(model["w1"], W_BITS), "b1": to_int(model["b1"], W_BITS + IN_BITS),
            "w2": to_int(model["w2"], W_BITS), "b2": to_int(model["b2"], W_BITS + IN_BITS),
            "vf": to_int(model["vf"], W_BITS), "vg": to_int(model["vg"], W_BITS),
            "c_d": to_int(model["c_d"], W_BITS + IN_BITS)[0],
            "shape": [int(v) for v in np.asarray(model["w1"]).shape]
            + [int(np.asarray(model["w2"]).shape[1])]}


def decode_model(kind: str, data: dict, spec: dict) -> dict:
    """JSON ints back to the float64 arrays inference uses."""
    if kind == "boosted":
        out = {"c_d": from_int([data["c_d"]], LEAF_BITS)[0]}
        for name in ("f", "g"):
            ens = data[name]
            thr = np.array([[np.inf if k is None else (k + 0.5) / 2.0 ** IN_BITS for k in row]
                            for row in ens["thr"]], dtype=np.float64).reshape(len(ens["thr"]), -1)
            out[name] = {"feat": np.array(ens["feat"], dtype=np.int64).reshape(thr.shape),
                         "thr": thr,
                         "leaf": from_int(ens["leaf"], LEAF_BITS).reshape(thr.shape[0], -1)}
        if "lin" in data:
            out["lin"] = {k: from_int(v, LEAF_BITS) for k, v in data["lin"].items()}
        return out
    n_in, embed, hidden = data["shape"]
    k = len(spec["taps"])
    return {"w1": from_int(data["w1"], W_BITS, (n_in, embed)),
            "b1": from_int(data["b1"], W_BITS + IN_BITS),
            "w2": from_int(data["w2"], W_BITS, (k * embed, hidden)),
            "b2": from_int(data["b2"], W_BITS + IN_BITS),
            "vf": from_int(data["vf"], W_BITS), "vg": from_int(data["vg"], W_BITS),
            "c_d": from_int([data["c_d"]], W_BITS + IN_BITS)[0]}


def check_bounds(kind: str, model: dict, n_in: int) -> None:
    """Raise unless every sum inference forms is exact in float64 (< 2**53 grid units)."""
    if kind == "boosted":
        total = sum(float(np.abs(model[n]["leaf"]).max(axis=1).sum()) for n in ("f", "g")) * 2
        bits = np.log2(max(total + abs(model["c_d"]), 1.0)) + LEAF_BITS
        if "lin" in model:  # inputs <= 1 on 2**-IN_BITS times weights on 2**-LEAF_BITS
            lin = sum(float(np.abs(v).sum()) for v in model["lin"].values()) * 2
            bits = max(bits, np.log2(max(lin + total, 1.0)) + LEAF_BITS + IN_BITS)
    else:
        layer1 = n_in * float(np.abs(model["w1"]).max()) + float(np.abs(model["b1"]).max())
        layer2 = model["w2"].shape[0] * ACT_MAX * float(np.abs(model["w2"]).max()) \
            + float(np.abs(model["b2"]).max())
        out = 2 * model["vf"].shape[0] * ACT_MAX * max(float(np.abs(model["vf"]).max()),
                                                        float(np.abs(model["vg"]).max()))
        bits = np.log2(max(layer1, layer2, out + abs(model["c_d"]), 1.0)) + W_BITS + IN_BITS
    if bits >= 52:
        raise ValueError("{} model too large for exact float64 inference ({:.1f} bits)".format(kind, bits))


HELDOUT_KEYS = ("folds", "fold_of", "loso")


def heldout_path(path: str) -> str:
    """The held-out fits that go with a coefficient file: x.json -> x_heldout.json.gz."""
    return (path[:-5] if path.endswith(".json") else path) + "_heldout.json.gz"


def write_coefs(path: str, payload: dict) -> None:
    """Write a coefficient file as two: `path` (compact JSON: the fit on every match, what src.run
    reads, plus the metadata) and heldout_path(path) (the fold and per-source fits, only for
    scoring external matches held out; gzip with a fixed header, so the same bytes every time).
    The public repo exports `path` only (scripts/export_public_repo.sh)."""
    main = {k: v for k, v in payload.items() if k not in HELDOUT_KEYS}
    with open(path, "w") as handle:
        handle.write(json.dumps(main, sort_keys=True, separators=(",", ":")) + "\n")
    held = {k: payload[k] for k in HELDOUT_KEYS if k in payload}
    if held:
        held.update(kind=payload["kind"], feature_hash=payload["feature_hash"])
        raw = json.dumps(held, sort_keys=True, separators=(",", ":")).encode()
        with open(heldout_path(path), "wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as gz:
                gz.write(raw)


_COEFS: dict = {}


def load_coefs(path: str) -> dict:
    """The coefficient file (and its held-out fits, if present), decoded, checked against today's
    features and bounds (cached)."""
    if path not in _COEFS:
        with open(path) as handle:
            data = json.load(handle)
        groups = tuple(data["groups"])
        if data.get("feature_hash") != learned.feature_hash(groups):
            raise ValueError("{} was fit for other features; rerun scripts/fit_emission.py".format(path))
        held_file = heldout_path(path)
        if os.path.isfile(held_file):
            with gzip.open(held_file, "rb") as handle:
                held = json.loads(handle.read().decode())
            if held.get("feature_hash") != data["feature_hash"] or held.get("kind") != data["kind"]:
                raise ValueError("{} doesn't belong to {}".format(held_file, path))
            data.update({k: held[k] for k in HELDOUT_KEYS if k in held})
        kind, spec = data["kind"], data["spec"]
        if kind not in KINDS:
            raise ValueError("unknown emission kind {!r} in {}".format(kind, path))

        def fit(entry):
            model = decode_model(kind, entry["model"], spec)
            check_bounds(kind, model, spec["n_in"])
            return {"model": model, "log_prior": from_int(entry["log_prior"], LEAF_BITS)}

        _COEFS[path] = {"kind": kind, "spec": spec, "groups": groups,
                        "all": fit(data["all"]) if "all" in data else None,
                        "folds": {k: fit(v) for k, v in data.get("folds", {}).items()},
                        "fold_of": data.get("fold_of", {}),
                        "loso": {k: fit(v) for k, v in data.get("loso", {}).items()},
                        "fit": data.get("fit", {})}
    return _COEFS[path]


def match_blocks(match: Match, groups, spec: dict, impute_fold: str = "auto"):
    """learned.features as the emission models use them, in training and at run time alike: the
    player-speed span is the file's (SPEED_SPAN_S), not the decoder's speed_span_s."""
    return learned.features(match, groups, speed_span_s=spec["speed_span_s"], impute_fold=impute_fold)


CHUNK = 20_000  # frames per inference chunk (bounds memory on a full match; results are per frame)


def logits(match: Match, coefs: dict, fit: dict, blocks=None, impute_fold: str = "auto") -> np.ndarray:
    """(n, 3) exact logits for H, A, D on every frame of the match, in chunks of frames (every
    frame's arithmetic is the same as in one pass, so the bits don't depend on the chunking)."""
    if blocks is None:
        blocks = match_blocks(match, coefs["groups"], coefs["spec"], impute_fold)
    n = len(blocks[0])
    chunks = [(s, min(n, s + CHUNK)) for s in range(0, n, CHUNK)]
    model = fit["model"]
    if coefs["kind"] == "boosted":
        return np.concatenate([boosted_logits(*views(tuple(b[s:e] for b in blocks)), model)
                               for s, e in chunks]) if n else np.zeros((0, 3))
    step = max(1, to_frames(STRIDE_S, match.fps))
    first, last = run_bounds(match.period)
    taps = tap_index(first, last, np.asarray(coefs["spec"]["taps"]) * step)
    embed = [[], []]
    for s, e in chunks:
        for i, u in enumerate(views(tuple(b[s:e] for b in blocks))):
            embed[i].append(net_embed(u, model))
    a_home, a_away = (np.concatenate(x) for x in embed)
    return np.concatenate([temporal_out(net_temporal(a_home, taps[s:e], model),
                                        net_temporal(a_away, taps[s:e], model), model)
                           for s, e in chunks]) if n else np.zeros((0, 3))


def log_prior_term(log_prior: np.ndarray) -> np.ndarray:
    """What LearnedDecoder subtracts (prior=1): H and A share their mean prior. Exact."""
    p = det_exp(log_prior)
    team = (p[0] + p[1]) / 2.0
    return det_log(np.array([team, team, p[2]]))


# -- the model --------------------------------------------------------------------------------------------

@dataclass(kw_only=True)
class EmissionDecoder(LearnedDecoder):
    """`learned`'s decoder (every LearnedDecoder setting applies) with a bigger emission model from
    coef_file. emission_gain scales the model's log-probabilities before the costs (its per-frame
    evidence is more autocorrelated than the logit's, so the decoder may want it damped)."""

    coef_file: str = BOOSTED_FILE
    emission_gain: float = 1.0

    def files(self) -> list[str]:
        """coef_file, or several joined by "+": an ensemble (their log-probabilities averaged)."""
        return [f for f in self.coef_file.split("+") if f]

    # Wave-2 integration: C13's track repair and V4's broadcast-ball fill reach these models as they
    # reach `learned` (LearnedDecoder reads coef_file as one file, so the group checks go through
    # this module's loader, file by file). Neither set (preset b31): the match is read unchanged.
    def fills(self) -> bool:
        """Whether the ball's gaps are filled from Match.ball2 (ball2_fill or a "bfill" fit)."""
        return self.ball2_fill or any("bfill" in load_coefs(f)["groups"] for f in self.files())

    def reads_ball2(self) -> bool:
        """Whether this decoder reads Match.ball2 (src.run then attaches BAR-ATH's broadcast ball)."""
        return self.fills() or any("bball" in load_coefs(f)["groups"] for f in self.files())

    def read(self, match: Match) -> Match:
        """The match the emissions read: repaired when track_repair is set (C13, cached per match),
        then its ball's gaps filled from ball2 when fills() (V4), as LearnedDecoder does."""
        match = self.prepare(match)
        return learned.with_ball2(match) if self.fills() else match

    def coefs(self, match: Match, path: str | None = None) -> dict:
        """The fit that scores this match (fold) in path (default: the first file)."""
        path = path or self.files()[0]
        coefs = load_coefs(path)
        if self.fold == "all":
            if coefs["all"] is None:
                raise KeyError("no fit on every match in {}".format(path))
            return coefs["all"]
        source = match.match_id.split(":", 1)[0]
        if source not in coefs.get("fit", {}).get("sources", (source,)) and coefs["all"] is not None:
            return coefs["all"]  # a source the fit never saw: "all" is held out (C12, B30)
        if self.fold == "lomo":
            fold = coefs["fold_of"].get(match.match_id)
            if fold is None:
                raise KeyError("no fold holds out {!r} in {}".format(match.match_id, path))
            return coefs["folds"][str(fold)]
        if source not in coefs["loso"]:
            raise KeyError("no loso fit without {!r} in {}".format(source, path))
        return coefs["loso"][source]

    def emission(self, match: Match) -> np.ndarray:
        """(n, 3) log p(k | x), minus the log prior when prior=1; exact on every machine. Several
        files: the mean of their log-probabilities, renormalised (a log-linear pool), minus the
        mean of their prior terms."""
        impute_fold = "loso" if self.fold == "loso" else "auto"
        match = self.read(match)
        blocks, lps, priors = {}, [], []
        for path in self.files():
            coefs, fit = load_coefs(path), self.coefs(match, path)
            key = (coefs["groups"], coefs["spec"]["speed_span_s"])
            if key not in blocks:  # the files share features: compute them once
                blocks[key] = match_blocks(match, coefs["groups"], coefs["spec"], impute_fold)
            lps.append(det_log_softmax(logits(match, coefs, fit, blocks[key])))
            priors.append(log_prior_term(fit["log_prior"]))
        lp, prior = lps[0], priors[0]
        if len(lps) > 1:
            total, prior_total = lps[0], priors[0]
            for x, q in zip(lps[1:], priors[1:]):  # fixed order: exact the same on every machine
                total, prior_total = total + x, prior_total + q
            lp, prior = det_log_softmax(total / len(lps)), prior_total / len(lps)
        return lp - prior if self.prior else lp

    def scores(self, match: Match) -> np.ndarray:
        """(n, 3) per-second evidence for H, A, D (as LearnedDecoder.scores)."""
        return self.combine(self.read(match), self.emission(match))

    def combine(self, match: Match, lp: np.ndarray, hand: np.ndarray | None = None) -> np.ndarray:
        """Decoder scores from emission(): lag, gains, per second, plus the hand-set evidence
        (hand: SwitchDecoder's scores if already computed, to decode several settings cheaply)."""
        if self.emission_gain != 1.0:
            lp = lp * self.emission_gain
        team = lp[:, :2]
        lag = int(round(self.lag_s * match.fps))
        if lag:
            team = np.stack([shift(team[:, 0], lag), shift(team[:, 1], lag)], axis=1)
        out = np.hstack([team, self.dead_gain * lp[:, 2:]]) / match.fps
        if self.hand_team or self.hand_dead:
            if hand is None:
                hand = SwitchDecoder(**B3_SETTINGS, pjitter_ref_m=self.hand_pjitter_ref_m,
                                     pjitter_reach_exp=self.hand_pjitter_exp).scores(match)
            out = out + hand * np.array([self.hand_team, self.hand_team, self.hand_dead])
        return self.add_restart_rule(match, out)  # PR #51's rule; unchanged when its gain is 0


# Decoder settings picked held out by match on the dropout grid (scripts/check_emission.py, F127,
# F128); the rest are LearnedDecoder's (v4's) defaults.
@dataclass(kw_only=True)
class BoostedDecoder(EmissionDecoder):
    """B31 (a): gradient-boosted oblivious trees over v4's features."""

    coef_file: str = BOOSTED_FILE
    switch_cost_s: float = 0.5


@dataclass(kw_only=True)
class TemporalDecoder(EmissionDecoder):
    """B31 (b): a small temporal conv net over v4's features at taps around each frame."""

    coef_file: str = TEMPORAL_FILE
    switch_cost_s: float = 0.375
    emission_gain: float = 1.25


@dataclass(kw_only=True)
class BiggerDecoder(EmissionDecoder):
    """B31's candidate: the trees and the temporal net together (a log-linear pool: their
    log-probabilities averaged), F128. Held out by match and by source it beats v4 on every source."""

    coef_file: str = BOOSTED_FILE + "+" + TEMPORAL_FILE
    switch_cost_s: float = 0.5
    emission_gain: float = 1.5
