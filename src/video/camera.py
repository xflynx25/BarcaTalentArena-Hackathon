"""Pitch <-> pixel maps for the two kinds of video we use (V1 interface; V2-V5 build on it).

Pitch coordinates are always the internal schema's (src/schema.py): metres, corner origin, x in
[0, length], y in [0, width], so a video position drops straight into a Match.

  HomographyCamera   BAR-ATH (run time). One 3x3 matrix per video frame from
                     <source>_homography.csv, mapping NORMALISED pitch coordinates (0-1, the ATD's) to
                     pixels (official/DATA.md). The tactical file is valid on every frame. Load it
                     with official/starter/load_homography.load(); this class only does the maths.
  SoccerTrackCamera  SoccerTrack v2 (training). The provider's calibration in raw/<id>/: pitch
                     (corner origin, y = 68 at the near touchline) -> homography.npy -> an undistorted
                     image (camera Knew) -> fisheye distortion (K, D, OpenCV's model) -> the panorama
                     pixel. Checked on 117093: 6.4 px median on the 65 hand-annotated keypoints, 5.5 px
                     on 1.49 M player foot points (F133). Our schema's y is 34 - release y, so the
                     keypoints' y is 68 - schema y.

Both are plain float64 numpy plus cv2.fisheye for SoccerTrack (training only, never at run time).
"""

from __future__ import annotations

import json
import os

import numpy as np


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(h, dtype=np.float64).T
    return q[:, :2] / q[:, 2:]


class HomographyCamera:
    """Per-frame homographies (n, 3, 3), normalised pitch -> pixel; pitch is (length, width) in m."""

    def __init__(self, h: np.ndarray, pitch: tuple[float, float]):
        self.h = np.asarray(h, dtype=np.float64)
        self.scale = np.array(pitch, dtype=np.float64)

    def to_pixel(self, frame: int, xy: np.ndarray) -> np.ndarray:
        return _apply(self.h[frame], np.atleast_2d(xy) / self.scale)

    def to_pitch(self, frame: int, uv: np.ndarray) -> np.ndarray:
        return _apply(np.linalg.inv(self.h[frame]), np.atleast_2d(uv)) * self.scale


class SoccerTrackCamera:
    """The fixed panoramic camera of one SoccerTrack v2 match (both halves share it)."""

    LENGTH, WIDTH = 105.0, 68.0

    def __init__(self, match_id: str, root: str | None = None):
        from src.video import soccertrack as st

        root = root or st.ROOT
        base = "raw/{0}/{0}_".format(match_id)
        paths = {k: os.path.join(root, base + f) for k, f in
                 (("h", "homography.npy"), ("k", "camera_intrinsics.npz"), ("kp", "keypoints.json"))}
        for key, path in paths.items():
            if not os.path.exists(path):
                st._download(os.path.relpath(path, root), root)
        self.h = np.load(paths["h"], allow_pickle=False).astype(np.float64)
        with np.load(paths["k"], allow_pickle=False) as z:
            self.K, self.D, self.Knew = (z[k].astype(np.float64) for k in ("K", "D", "Knew"))
        with open(paths["kp"]) as handle:
            kp = json.load(handle)
        self.keypoints = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])  # provider's
        self.keypoint_pixels = np.array(list(kp.values()), dtype=np.float64)

    def _provider(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        return np.c_[xy[:, 0], self.WIDTH - xy[:, 1]]

    def to_pixel(self, frame: int | None, xy: np.ndarray) -> np.ndarray:
        """Schema metres -> panorama pixels (frame is ignored: the camera is fixed)."""
        import cv2

        und = _apply(self.h, self._provider(xy))
        norm = _apply(np.linalg.inv(self.Knew), und)
        return cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def to_pitch(self, frame: int | None, uv: np.ndarray) -> np.ndarray:
        """Panorama pixels -> schema metres (on the ground plane)."""
        import cv2

        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64)).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(uv, self.K, self.D, P=self.Knew).reshape(-1, 2)
        p = _apply(np.linalg.inv(self.h), und)
        return np.c_[p[:, 0], self.WIDTH - p[:, 1]]

    def keypoint_error(self) -> np.ndarray:
        """Pixel error of the calibration on its own 65 hand-annotated keypoints."""
        kp = self.keypoints
        return np.hypot(*(self.to_pixel(None, np.c_[kp[:, 0], self.WIDTH - kp[:, 1]])
                          - self.keypoint_pixels).T)
