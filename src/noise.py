"""Simulated position noise for robustness tests and training mixes (C3, C5, C7, C8).

Every kind is seeded, so a run is deterministic. Kinds (sigma in metres, tau in seconds):
  ball      AR(1) noise on the ball, sd sigma, correlation time tau (C3)
  players   AR(1) noise on every player, independently (C3). Its frame-to-frame steps are white
            (sd sigma * sqrt(2 / (tau * fps)) per frame), so it inflates frame-step speeds
            (findings F72) and the player jitter measure sees it (F75)
  smooth    the players' AR(1) noise low-passed over SMOOTH_S (a centred moving average, rescaled
            to sd sigma): drifts like calibration / extrapolation errors, with no frame steps (C8)
  offset    one constant offset per player, sd sigma, for the whole match (a biased track) (C8)
  shift     one smooth whole-pitch shift (low-passed AR(1), sd sigma) applied to ball and players
            together: a camera / homography error, which leaves every distance unchanged except to
            the lines (C8)
  calib     smooth + offset + shift, each with sd sigma / sqrt(3), so the total per-player sd is
            about sigma (C8)
"""

from __future__ import annotations

import numpy as np

from src.schema import Match

KINDS = ("ball", "players", "smooth", "offset", "shift", "calib")
SMOOTH_S = 1.0


def ar1(rng: np.random.Generator, shape: tuple, sigma: float, rho: float) -> np.ndarray:
    """(shape) AR(1) series along axis 0 with stationary sd sigma and lag-1 correlation rho."""
    shocks = rng.standard_normal(shape) * sigma * np.sqrt(1 - rho * rho)
    out = np.empty(shape)
    out[0] = rng.standard_normal(shape[1:]) * sigma
    for i in range(1, shape[0]):  # ~150k frames: a second or two per match
        out[i] = rho * out[i - 1] + shocks[i]
    return out


def low_pass(series: np.ndarray, frames: int) -> np.ndarray:
    """Centred moving average over `frames` along axis 0, rescaled to the input's sd."""
    if frames <= 1:
        return series
    flat = series.reshape(len(series), -1)
    kernel = np.ones(frames) / frames
    out = np.stack([np.convolve(flat[:, j], kernel, mode="same") for j in range(flat.shape[1])],
                   axis=1)
    sd_in, sd_out = flat.std(), out.std()
    return (out * (sd_in / sd_out if sd_out > 0 else 1.0)).reshape(series.shape)


def add_noise(match: Match, kind: str, sigma: float, tau_s: float, seed: int) -> Match:
    """match with `kind` noise (see the module docstring) added to every observed position."""
    if kind not in KINDS:
        raise ValueError("noise kind must be one of {}, not {!r}".format(KINDS, kind))
    rng = np.random.default_rng(seed)
    rho = np.exp(-1.0 / (max(tau_s, 1e-9) * match.fps))
    smooth = max(1, int(round(SMOOTH_S * match.fps)))
    ball, players = match.ball, match.players
    n, k = players.shape[:2]
    if kind == "ball":
        ball = ball + ar1(rng, (n, 2), sigma, rho)
    elif kind == "players":
        players = players + ar1(rng, players.shape, sigma, rho)
    else:
        part = sigma / np.sqrt(3) if kind == "calib" else sigma
        if kind in ("smooth", "calib"):
            players = players + low_pass(ar1(rng, players.shape, part, rho), smooth)
        if kind in ("offset", "calib"):
            players = players + rng.standard_normal((1, k, 2)) * part
        if kind in ("shift", "calib"):
            move = low_pass(ar1(rng, (n, 2), part, rho), smooth)
            ball, players = ball + move, players + move[:, None, :]
    return Match(match_id=match.match_id, fps=match.fps, pitch=match.pitch, period=match.period,
                 ball=ball, players=players, slot_team=match.slot_team,
                 labels=match.labels).validate()
