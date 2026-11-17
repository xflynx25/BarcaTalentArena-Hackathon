"""Pixel <-> pitch for a video frame: one interface, two cameras (backlog V2).

Every calibration speaks the schema's coordinates (src/schema.py: metres, corner origin, y = 0 on the
near touchline for SoccerTrack, the ATD's own orientation for BAR-ATH) and the video's raw pixels
(the decoded frame, no undistortion: the detector never resamples a whole frame).

  calib.to_pixel(xy)   (m, 2) pitch metres -> (m, 2) pixels (u right, v down)
  calib.to_pitch(uv)   (m, 2) pixels -> (m, 2) pitch metres on the ground plane (z = 0)
  calib.scale(uv)      (m,) pixels per metre on the ground at those pixels (the ball's size cue)

SoccerTrackCalib (one per match, the camera is fixed): the release's raw/<id>/<id>_homography.npy maps
pitch metres in its keypoint convention (x 0..105 left to right, y 0 on the FAR touchline) to the
UNDISTORTED image; <id>_mapx.npy / _mapy.npy are OpenCV fixed-point remap tables (CV_16SC2 integer
part + CV_16UC1 5-bit fractions) from undistorted pixels to the raw (distorted) video pixel. Checked on
117093: the homography + maps put the 65 hand keypoints within a median 3 px of where they were
clicked (the homography alone, on raw pixels: 63 px). Schema y = 68 - keypoint y.

FrameHomography (BAR-ATH, one 3x3 per frame, official/DATA.md): normalised pitch (0-1, the ATD's) ->
pixels, no distortion model. Schema metres = normalised x (length, width).

Inversion (pixel -> pitch) for SoccerTrack is Newton on the forward map from a coarse lookup table, so
both directions agree to < 1 mm. Everything is float64 elementwise arithmetic (no BLAS), so it
gives the same bits on every machine.
"""

from __future__ import annotations

import os

import numpy as np

KEYPOINT_W = 68.0  # SoccerTrack keypoint y runs 0 (far touchline) .. 68 (near)
LUT_STEP = 8  # px: the coarse pixel -> pitch table that seeds Newton
NEWTON_ITERS = 6
FD_M = 0.05  # m: finite-difference step for the Jacobian


def _apply_h(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """img sampled at float pixel (u, v), clamped to the edges."""
    hgt, wid = img.shape
    bad = ~(np.isfinite(u) & np.isfinite(v))
    u = np.clip(np.where(bad, 0.0, u), 0.0, wid - 1.000001)
    v = np.clip(np.where(bad, 0.0, v), 0.0, hgt - 1.000001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu, fv = u - u0, v - v0
    a = img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu
    b = img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu
    return np.where(bad, np.nan, a * (1 - fv) + b * fv)


def inv3(m: np.ndarray) -> np.ndarray:
    """3 x 3 inverse by the adjugate: plain float64 arithmetic, the same bits on every machine
    (np.linalg.inv goes through LAPACK, whose last bits can differ between builds)."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    adj = np.array([[e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d]])
    det = a * adj[0, 0] + b * adj[1, 0] + c * adj[2, 0]
    return adj / det


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r * r * np.log(np.where(r > 0, r, 1.0)), 0.0)


class Calib:
    """Base: subclasses define to_pixel(); to_pitch() inverts it by Newton from seed()."""

    pitch = (105.0, 68.0)

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def seed(self, uv: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, xy: np.ndarray) -> np.ndarray:
        """(m, 2, 2) d pixel / d metre, central differences."""
        xy = np.asarray(xy, dtype=np.float64)
        out = np.empty((len(xy), 2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = FD_M
            out[:, :, j] = (self.to_pixel(xy + step) - self.to_pixel(xy - step)) / (2 * FD_M)
        return out

    def to_pitch(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        xy = self.seed(uv)
        for _ in range(NEWTON_ITERS):
            r = self.to_pixel(xy) - uv
            jac = self.jacobian(xy)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (jac[:, 1, 1] * r[:, 0] - jac[:, 0, 1] * r[:, 1]) / det
            dy = (-jac[:, 1, 0] * r[:, 0] + jac[:, 0, 0] * r[:, 1]) / det
            xy = xy - np.stack([dx, dy], axis=1)
        return xy

    def scale(self, uv: np.ndarray) -> np.ndarray:
        """Pixels per metre on the ground: the square root of |det J| (the geometric mean of the two
        directions; the ball is round, its image is about 0.22 m x this across)."""
        jac = self.jacobian(self.to_pitch(uv))
        det = np.abs(jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0])
        return np.sqrt(det)

    def scale_x(self, uv: np.ndarray) -> np.ndarray:
        """Pixels per metre along the image's horizontal at those pixels (the larger direction on
        a low camera: depth is foreshortened, width is not; a sphere's image follows this one)."""
        jac = self.jacobian(self.to_pitch(uv))
        return np.maximum(np.hypot(jac[:, 0, 0], jac[:, 1, 0]), np.hypot(jac[:, 0, 1], jac[:, 1, 1]))


class SoccerTrackCalib(Calib):
    def __init__(self, homography: np.ndarray, map1: np.ndarray, map2: np.ndarray,
                 size: tuple[int, int] | None = None):
        self.h = np.asarray(homography, dtype=np.float64)
        self.h_inv = inv3(self.h)
        self.tps = None  # optional pixel correction fitted to hand keypoints (fit_keypoints)
        # OpenCV CV_16SC2 + CV_16UC1 (INTER_TAB_SIZE 32): source = integer + fraction / 32
        self.src_u = map1[..., 0].astype(np.float64) + (map2 & 31).astype(np.float64) / 32.0
        self.src_v = map1[..., 1].astype(np.float64) + (map2 >> 5).astype(np.float64) / 32.0
        self.size = size or (map1.shape[1], map1.shape[0])  # (width, height) of the raw video
        self._lut = None

    @classmethod
    def load(cls, raw_dir: str, match_id: str) -> "SoccerTrackCalib":
        base = os.path.join(raw_dir, match_id, match_id + "_")
        return cls(np.load(base + "homography.npy", allow_pickle=False),
                   np.load(base + "mapx.npy", allow_pickle=False),
                   np.load(base + "mapy.npy", allow_pickle=False))

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        ux, uy = _apply_h(self.h, xy[:, 0], KEYPOINT_W - xy[:, 1])  # undistorted pixels
        uv = np.stack([_bilinear(self.src_u, ux, uy), _bilinear(self.src_v, ux, uy)], axis=1)
        if self.tps is not None:
            uv = uv + self._tps_delta(xy)
        return uv

    def fit_keypoints(self, keypoints: dict, smooth: float = 1.0) -> "SoccerTrackCalib":
        """Correct to_pixel with a thin-plate spline (in pitch metres) through the residuals at the
        release's hand-clicked keypoints ({"(x,y)": [u, v]}, keypoint convention), smoothed by
        `smooth` (0 = interpolate exactly). Fit / choice: scripts/video/v2_train.py (F136)."""
        self.tps = None
        self._lut = None
        names = sorted(keypoints)
        kxy = np.array([[float(t) for t in k.strip("()").split(",")] for k in names])
        xy = np.stack([kxy[:, 0], KEYPOINT_W - kxy[:, 1]], axis=1)
        target = np.array([keypoints[k] for k in names], dtype=np.float64)
        resid = target - self.to_pixel(xy)
        n = len(xy)
        K = _tps_kernel(np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]))
        P = np.hstack([np.ones((n, 1)), xy])
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K + smooth * np.eye(n)
        A[:n, n:] = P
        A[n:, :n] = P.T
        rhs = np.vstack([resid, np.zeros((3, 2))])
        sol = np.linalg.solve(A, rhs)
        self.tps = (xy, sol[:n], sol[n:])
        return self

    def _tps_delta(self, xy: np.ndarray) -> np.ndarray:
        cxy, w, a = self.tps
        out = a[0][None, :] + xy[:, :1] * a[1][None, :] + xy[:, 1:] * a[2][None, :]
        for j in range(len(cxy)):  # fixed order
            r = np.hypot(xy[:, 0] - cxy[j, 0], xy[:, 1] - cxy[j, 1])
            out = out + _tps_kernel(r)[:, None] * w[j][None, :]
        return out

    def _build_lut(self):
        """Pitch metres at every LUT_STEP-th raw pixel, from a dense forward-mapped pitch grid
        (nearest sample; Newton refines it). Covers the pitch plus a 15 m margin."""
        gx, gy = np.meshgrid(np.arange(-15.0, 120.01, 0.25), np.arange(-15.0, 83.01, 0.25))
        xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        uv = self.to_pixel(xy)
        wid, hgt = self.size
        cu = np.rint(uv[:, 0] / LUT_STEP).astype(np.int64)
        cv = np.rint(uv[:, 1] / LUT_STEP).astype(np.int64)
        nu, nv = wid // LUT_STEP + 1, hgt // LUT_STEP + 1
        ok = (cu >= 0) & (cu < nu) & (cv >= 0) & (cv < nv)
        err = np.hypot(uv[:, 0] - cu * LUT_STEP, uv[:, 1] - cv * LUT_STEP)
        lut = np.full((nv, nu, 2), np.nan)
        best = np.full((nv, nu), np.inf)
        order = np.argsort(-err[ok], kind="stable")  # write the closest sample last
        idx = np.flatnonzero(ok)[order]
        lut[cv[idx], cu[idx]] = xy[idx]
        best[cv[idx], cu[idx]] = err[idx]
        # fill holes (cells no grid sample hit) from the nearest filled cell along the row/column
        for _ in range(64):
            hole = np.isnan(lut[..., 0])
            if not hole.any():
                break
            for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                moved = np.roll(lut, shift, axis=axis)
                fill = hole & ~np.isnan(moved[..., 0])
                lut[fill] = moved[fill]
                hole = np.isnan(lut[..., 0])
        self._lut = lut

    def seed(self, uv: np.ndarray) -> np.ndarray:
        if self._lut is None:
            self._build_lut()
        nv, nu = self._lut.shape[:2]
        cu = np.clip(np.rint(uv[:, 0] / LUT_STEP).astype(np.int64), 0, nu - 1)
        cv = np.clip(np.rint(uv[:, 1] / LUT_STEP).astype(np.int64), 0, nv - 1)
        return self._lut[cv, cu].copy()


class FrameHomography(Calib):
    """BAR-ATH-style: one 3x3 matrix mapping normalised pitch (0-1) to pixels (official/DATA.md)."""

    def __init__(self, matrix: np.ndarray, pitch=(105.0, 68.0)):
        self.h = np.asarray(matrix, dtype=np.float64)
        self.h_inv = inv3(self.h)
        self.pitch = tuple(float(p) for p in pitch)

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        u, v = _apply_h(self.h, xy[:, 0] / self.pitch[0], xy[:, 1] / self.pitch[1])
        return np.stack([u, v], axis=1)

    def seed(self, uv: np.ndarray) -> np.ndarray:
        x, y = _apply_h(self.h_inv, uv[:, 0], uv[:, 1])
        return np.stack([x * self.pitch[0], y * self.pitch[1]], axis=1)

    def to_pitch(self, uv: np.ndarray) -> np.ndarray:  # exact: a homography inverts in closed form
        return self.seed(np.asarray(uv, dtype=np.float64).reshape(-1, 2))
