"""Ball candidates in one video frame: the provisional CPU detector behind one function (backlog V2).

    ctx = Context.build(calib, (height, width))          # once per camera pose (SoccerTrack: per match)
    cands = detect(prev, cur, nxt, ctx, players_xy=...)  # one frame -> Candidates (arrays, m rows)
    score = Scorer.load(path).score(cands)               # (m,) logit: is this the ball?

V1 (the video plan) picks the real stack; this one exists so V2 can measure and wire the rest now.
It is plain numpy, no learned convolution: a white ball on grass is a small round blob brighter and
less saturated than what is around it, about BALL_M x (pixels per metre) across, and usually moving.

Stage 1, candidates (integer sums, so bit-identical everywhere): per scale band (the expected ball
diameter from the calibration, BANDS), the luma mean of a centre cell minus the brightest of its 8
neighbour cells (a 3 x 3 grid of cells, integral images): a blob scores, a pitch line through it
doesn't. Evaluated every STRIDES[b] px inside the pitch (+ ROI_MARGIN_M), block-pooled maxima (POOL),
the top MAX_CANDS per frame.
Stage 2, features per candidate (float64, elementwise only: no BLAS, no SIMD reductions): the blob
contrast at all three scales, luma and saturation inside / in the ring, motion against frames t - 2
and t + 2 (decoded anyway), the expected diameter, a PATCH x PATCH luma patch resampled over
PATCH_SPAN diameters and normalised, and the distance (metres) to the nearest tracked player and
whether it sits on a player's body in the image. FEATURES names the columns.
The scorer (Scorer) is a one-hidden-layer ReLU network on those columns, evaluated as explicit
multiply-adds in a fixed order, so its logit is the same bits on every machine; scripts/video/
v2_train.py fits it on SoccerTrack's train matches.

`prev` / `nxt` may be None (no motion features: zeros). `cur` is (y, u, v) uint8 planes (yuv420p).
"""

from __future__ import annotations

import json
import os

import numpy as np

BALL_M = 0.22  # FIFA size 5: 68-70 cm round
# (largest expected diameter in px, cell side, cell spacing): a 3 x 3 grid of cells, the spacing
# at least the band's largest diameter so the neighbour cells sit on what is around the ball
BANDS = ((5.0, 3, 6), (9.0, 5, 10), (14.0, 7, 15), (np.inf, 11, 22))
PAD = 30  # >= spacing + side // 2 of the largest band
STRIDES = (1, 2, 2, 3)  # stage 1 evaluates the contrast every STRIDES[b] px in band b ...
POOL = (4, 3, 4, 4)  # ... and keeps maxima of POOL[b] x POOL[b] blocks of that grid over their neighbours
MAX_CANDS = 64
MOTION_W = 1.0  # the cut ranks peaks by contrast + MOTION_W x motion (both luma levels)
MIN_CONTRAST = 8.0  # luma levels: centre cell mean - brightest neighbour cell mean
ROI_MARGIN_M = 4.0  # candidates up to this far outside the lines (throw-ins, goal kicks)
PATCH = 7
PATCH_SPAN = 2.5  # the patch covers this many expected diameters
MOTION_STEP = 2  # frames
GRID = 16  # px: the context's coarse grid for diameter and pitch position
BODY_H_M = 1.9  # a player's image extends this far above the foot point ...
BODY_W_M = 0.7  # ... and this wide

FEATURES = (["c_band", "c0", "c1", "c2", "c3", "in_y", "ring_y", "in_sat", "ring_sat", "mot_prev",
             "mot_next", "mot_min", "diam", "near_m", "on_body", "off_pitch_m"]
            + ["p{}".format(i) for i in range(PATCH * PATCH)])


class Context:
    """What detect() needs from the camera, precomputed on a coarse grid: the expected ball
    diameter (px), the scale band, the ROI (pitch + margin) and pixel -> pitch for candidates."""

    def __init__(self, calib, shape, diam, roi, rows):
        self.calib, self.shape, self.diam, self.roi, self.rows = calib, shape, diam, roi, rows

    @classmethod
    def build(cls, calib, shape: tuple[int, int]) -> "Context":
        h, w = shape
        gv, gu = np.meshgrid(np.arange(0, h + GRID, GRID, dtype=np.float64),
                             np.arange(0, w + GRID, GRID, dtype=np.float64), indexing="ij")
        uv = np.stack([gu.ravel(), gv.ravel()], axis=1)
        xy = calib.to_pitch(uv)
        back = calib.to_pixel(xy)
        good = np.hypot(back[:, 0] - uv[:, 0], back[:, 1] - uv[:, 1]) < 1.0  # Newton converged
        length, width = calib.pitch
        inside = ((xy[:, 0] >= -ROI_MARGIN_M) & (xy[:, 0] <= length + ROI_MARGIN_M)
                  & (xy[:, 1] >= -ROI_MARGIN_M) & (xy[:, 1] <= width + ROI_MARGIN_M) & good)
        diam = np.full(len(uv), np.nan)
        if inside.any():
            diam[inside] = BALL_M * calib.scale_x(uv[inside])
        diam_g = diam.reshape(gu.shape)
        roi_g = inside.reshape(gu.shape)
        # nearest-grid upsample to full resolution
        iv = np.minimum(np.rint(np.arange(h) / GRID).astype(np.int64), diam_g.shape[0] - 1)
        iu = np.minimum(np.rint(np.arange(w) / GRID).astype(np.int64), diam_g.shape[1] - 1)
        roi = roi_g[iv][:, iu]
        d = diam_g[iv][:, iu]
        rows = np.flatnonzero(roi.any(axis=1))
        return cls(calib, (h, w), np.where(roi, d, np.nan), roi,
                   (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 0))


def _integral(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(a, axis=0, dtype=np.int64), axis=1, out=out[1:, 1:])
    return out


# neighbour order: N, S, W, E, NW, SE, NE, SW (opposite pairs adjacent)
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1))


def _cell_sums(ii: np.ndarray, r: np.ndarray, c: np.ndarray, side: int, gap: int) -> list:
    """Sums of the 3 x 3 grid of side x side cells, `gap` px apart, centred at (r, c) (index arrays
    into the padded image whose integral is ii); [centre, then NEIGHBOURS]. r, c may be 2-D."""
    h = side // 2
    out = []
    for dr, dc in ((0, 0),) + NEIGHBOURS:
        r0 = r + dr * gap - h
        c0 = c + dc * gap - h
        out.append(ii[r0 + side, c0 + side] - ii[r0, c0 + side] - ii[r0 + side, c0] + ii[r0, c0])
    return out


def _cell_sums_grid(ii: np.ndarray, r0: int, nr: int, c0: int, nc: int, stride: int, side: int,
                    gap: int) -> list:
    """_cell_sums on the regular grid rows r0 + stride * i (i < nr), cols c0 + stride * j (j < nc)
    of the padded image, with strided slices instead of gathers (the hot loop: ~10x faster)."""
    h = side // 2
    out = []
    for dr, dc in ((0, 0),) + NEIGHBOURS:
        a = r0 + dr * gap - h
        b = c0 + dc * gap - h
        ra, rb = slice(a, a + stride * nr, stride), slice(a + side, a + side + stride * nr, stride)
        ca, cb = slice(b, b + stride * nc, stride), slice(b + side, b + side + stride * nc, stride)
        out.append(ii[rb, cb] - ii[ra, cb] - ii[rb, ca] + ii[ra, ca])
    return out


def _contrast(cells: list, side: int) -> np.ndarray:
    """Centre cell mean minus the ring, where the ring is the brightest of the four opposite pairs'
    darker cell: a line through the centre lights both cells of one pair and kills the contrast; a
    bright neighbour on one side only (a boot, a sock next to the ball) doesn't."""
    ring = np.minimum(cells[1], cells[2])
    for k in (3, 5, 7):
        ring = np.maximum(ring, np.minimum(cells[k], cells[k + 1]))
    return (cells[0] - ring).astype(np.float64) / float(side * side)


def _pooled_peaks(score: np.ndarray, block: int):
    """(rows, cols) of block maxima that are >= the maxima of the 8 neighbouring blocks."""
    h, w = score.shape
    hb, wb = h // block, w // block
    if hb < 1 or wb < 1:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    s = score[:hb * block, :wb * block].reshape(hb, block, wb, block).transpose(0, 2, 1, 3)
    s = s.reshape(hb, wb, block * block)
    arg = np.argmax(s, axis=2)  # first maximum: deterministic
    val = np.take_along_axis(s, arg[..., None], axis=2)[..., 0]
    pad = np.pad(val, 1, mode="constant", constant_values=-np.inf)
    keep = np.ones_like(val, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = pad[1 + dr:1 + dr + hb, 1 + dc:1 + dc + wb]
            # ties: keep the earlier block only
            keep &= (val > nb) | ((val == nb) & ((dr > 0) | ((dr == 0) & (dc > 0))))
    br, bc = np.nonzero(keep & np.isfinite(val))
    a = arg[br, bc]
    return br * block + a // block, bc * block + a % block


class Candidates(dict):
    """u, v (pixels, int), x, y (pitch metres), band, X (m, len(FEATURES)) float64."""

    def __len__(self):
        return len(self["u"])


def detect(prev, cur, nxt, ctx: Context, players_xy: np.ndarray | None = None,
           max_cands: int = MAX_CANDS) -> Candidates:
    y, u, v = cur
    h, w = y.shape
    r0, r1 = ctx.rows
    if r1 <= r0:
        return _empty()
    # the ROI slab plus a margin, padded: index (r - top + PAD, c + PAD)
    top, bot = max(0, r0 - PAD), min(h, r1 + PAD)
    slab = np.pad(y[top:bot], PAD, mode="edge")
    yi = _integral(slab)

    # stage 1: line-robust contrast per scale band, on a stride, block-pooled maxima
    found_v, found_r, found_c, found_b = [], [], [], []
    lo = 0.0
    for b, (hi, s_in, gap) in enumerate(BANDS):
        mask = (ctx.diam > lo) & (ctx.diam <= hi) & ctx.roi
        lo = hi
        rows = np.flatnonzero(mask.any(axis=1))
        if not len(rows):
            continue
        stride = STRIDES[b]
        rr = np.arange(int(rows[0]), int(rows[-1]) + 1, stride)
        cc = np.arange(0, w, stride)
        con = _contrast(_cell_sums_grid(yi, int(rr[0]) - top + PAD, len(rr), PAD, len(cc), stride,
                                        s_in, gap), s_in)
        con = np.where(mask[rr][:, cc], con, -np.inf)
        pr, pc = _pooled_peaks(con, POOL[b])
        val = con[pr, pc]
        ok = val >= MIN_CONTRAST
        found_v.append(val[ok])
        found_r.append(rr[pr[ok]])
        found_c.append(cc[pc[ok]])
        found_b.append(np.full(int(ok.sum()), b, dtype=np.int64))
    if not found_v or not sum(len(a) for a in found_v):
        return _empty()
    vals, rr, cc, band = (np.concatenate(found_v), np.concatenate(found_r), np.concatenate(found_c),
                          np.concatenate(found_b))
    # motion of every peak (cheap: small windows), then the cut by contrast + motion: a moving ball
    # beats the many static bright blobs (spots, letters, spare balls, socks) at the same contrast
    mot = {}
    for name, other in (("mot_prev", prev), ("mot_next", nxt)):
        mot[name] = _motion(y, other, rr, cc, band)
    mot_min = np.minimum(mot["mot_prev"], mot["mot_next"])
    key = vals + MOTION_W * mot_min
    order = np.lexsort((cc, rr, -key))[:max_cands]  # ties by position: deterministic
    rr, cc, band = rr[order], cc[order], band[order]

    # stage 2: features from small windows around each candidate
    m = len(rr)
    X = np.zeros((m, len(FEATURES)))
    col = {name: i for i, name in enumerate(FEATURES)}
    R, C = rr - top + PAD, cc + PAD
    for b, (_, s_in, gap) in enumerate(BANDS):
        cells = _cell_sums(yi, R, C, s_in, gap)
        X[:, col["c{}".format(b)]] = _contrast(cells, s_in)
        sel = band == b
        X[sel, col["c_band"]] = X[sel, col["c{}".format(b)]]
        X[sel, col["in_y"]] = cells[0][sel] / float(s_in * s_in)
        ring = sum(cells[1:]) / float(8 * s_in * s_in)
        X[sel, col["ring_y"]] = ring[sel]
    sat = np.abs(u.astype(np.int16) - 128) + np.abs(v.astype(np.int16) - 128)
    sat_i = _integral(np.pad(sat, PAD, mode="edge"))
    for b, (_, s_in, gap) in enumerate(BANDS):
        sel = band == b
        if not sel.any():
            continue
        s_half = max(1, s_in // 2) | 1  # odd, at half resolution
        cells = _cell_sums(sat_i, rr[sel] // 2 + PAD, cc[sel] // 2 + PAD, s_half, max(1, gap // 2))
        X[sel, col["in_sat"]] = cells[0] / float(s_half * s_half)
        X[sel, col["ring_sat"]] = sum(cells[1:]) / float(8 * s_half * s_half)
    X[:, col["mot_prev"]] = mot["mot_prev"][order]
    X[:, col["mot_next"]] = mot["mot_next"][order]
    X[:, col["mot_min"]] = mot_min[order]
    dm = ctx.diam[rr, cc]
    X[:, col["diam"]] = dm
    # patch: PATCH x PATCH bilinear samples over PATCH_SPAN diameters, minus the ring mean, / 32
    offs = (np.arange(PATCH, dtype=np.float64) - (PATCH - 1) / 2.0) / ((PATCH - 1) / 2.0) * (PATCH_SPAN / 2.0)
    ring = X[:, col["ring_y"]]
    k = 0
    for oy in offs:
        for ox in offs:
            X[:, col["p{}".format(k)]] = (_sample(y, rr + oy * dm, cc + ox * dm) - ring) / 32.0
            k += 1
    uv = np.stack([cc.astype(np.float64), rr.astype(np.float64)], axis=1)
    xy = ctx.calib.to_pitch(uv)
    length, width = ctx.calib.pitch
    off = np.maximum.reduce([-xy[:, 0], xy[:, 0] - length, -xy[:, 1], xy[:, 1] - width,
                             np.zeros(m)])
    X[:, col["off_pitch_m"]] = off
    near = np.full(m, 30.0)
    on_body = np.zeros(m)
    if players_xy is not None:
        p = players_xy[np.isfinite(players_xy).all(axis=1)]
        if len(p):
            for j in range(len(p)):  # fixed order: deterministic minimum
                near = np.minimum(near, np.hypot(xy[:, 0] - p[j, 0], xy[:, 1] - p[j, 1]))
            foot = ctx.calib.to_pixel(p)
            sc = dm / BALL_M  # px per metre at the candidate
            for j in range(len(p)):
                du = np.abs(cc - foot[j, 0])
                dv = foot[j, 1] - rr
                on_body = np.maximum(on_body, ((du <= BODY_W_M / 2 * sc)
                                               & (dv >= 0.15 * sc) & (dv <= BODY_H_M * sc)).astype(float))
    X[:, col["near_m"]] = np.minimum(near, 30.0)
    X[:, col["on_body"]] = on_body
    return Candidates(u=cc, v=rr, x=xy[:, 0], y=xy[:, 1], band=band, X=X)


def _motion(y: np.ndarray, other, rr: np.ndarray, cc: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Mean |luma(t) - luma(other)| over each point's centre cell; zeros without the other frame."""
    out = np.zeros(len(rr))
    if other is None:
        return out
    h, w = y.shape
    for b, (_, s_in, _) in enumerate(BANDS):
        sel = np.flatnonzero(band == b)
        if not len(sel):
            continue
        h2 = s_in // 2
        dr, dc = np.meshgrid(np.arange(-h2, h2 + 1), np.arange(-h2, h2 + 1), indexing="ij")
        pr = np.clip(rr[sel][:, None, None] + dr, 0, h - 1)
        pc = np.clip(cc[sel][:, None, None] + dc, 0, w - 1)
        d = np.abs(y[pr, pc].astype(np.int32) - other[0][pr, pc].astype(np.int32))
        out[sel] = d.reshape(len(sel), -1).sum(axis=1).astype(np.float64) / (s_in * s_in)
    return out


def _sample(img: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    h, w = img.shape
    img = img if img.dtype == np.float64 else img
    r = np.clip(r, 0.0, h - 1.000001)
    c = np.clip(c, 0.0, w - 1.000001)
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr, fc = r - r0, c - c0
    i00, i01 = img[r0, c0].astype(np.float64), img[r0, c0 + 1].astype(np.float64)
    i10, i11 = img[r0 + 1, c0].astype(np.float64), img[r0 + 1, c0 + 1].astype(np.float64)
    a = i00 * (1 - fc) + i01 * fc
    b = i10 * (1 - fc) + i11 * fc
    return a * (1 - fr) + b * fr


def _empty() -> Candidates:
    z = np.zeros(0)
    return Candidates(u=z.astype(np.int64), v=z.astype(np.int64), x=z, y=z, band=z.astype(np.int64),
                      X=np.zeros((0, len(FEATURES))))


EXTRA = ["spot_m"]  # computed from the candidate's pitch position (design()), not stored by detect()


def design(X: np.ndarray, x: np.ndarray, y: np.ndarray, pitch=(105.0, 68.0)) -> np.ndarray:
    """The scorer's input: detect()'s FEATURES plus EXTRA. spot_m: the distance to the nearest
    painted spot (centre, the two penalty spots; static white blobs on every pitch), capped at 5."""
    length, width = pitch
    spot = np.full(len(x), 5.0)
    for sx in (length / 2.0, 11.0, length - 11.0):
        spot = np.minimum(spot, np.hypot(np.asarray(x, np.float64) - sx, np.asarray(y, np.float64) - width / 2.0))
    return np.hstack([np.asarray(X, dtype=np.float64), spot[:, None]])


class Scorer:
    """logit = w2 . relu(W1 (x - mu) / sd + b1) + b2, with explicit multiply-adds in a fixed order."""

    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items() if k != "features"}
        self.features = list(params.get("features", FEATURES + EXTRA))
        if self.features != list(FEATURES) + EXTRA:
            raise ValueError("scorer was fit on other features; refit it (scripts/video/v2_train.py)")

    @classmethod
    def load(cls, path: str) -> "Scorer":
        with open(path) as f:
            return cls(json.load(f))

    def score_candidates(self, cands, pitch=(105.0, 68.0)) -> np.ndarray:
        return self.score(design(cands["X"], cands["x"], cands["y"], pitch))

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: design() rows. Returns the logit (float64; snap it before it decides anything)."""
        p = self.p
        Z = (np.asarray(X, dtype=np.float64) - p["mu"]) / p["sd"]
        W1, b1, w2 = p["W1"], p["b1"], p["w2"]
        out = np.full(len(Z), float(p["b2"]))
        for k in range(W1.shape[0]):
            acc = np.full(len(Z), float(b1[k]))
            for j in range(W1.shape[1]):
                acc = acc + Z[:, j] * W1[k, j]
            out = out + np.maximum(acc, 0.0) * w2[k]
        return out


SCORER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ball_scorer.json")
