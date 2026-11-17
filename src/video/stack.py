"""The provisional CPU stack for V3: per-video-frame audio features, behind one function.

    frame_features(source, n_frames, fps=25) -> dict[str, np.ndarray]   # the one function V1 swaps

V1 (the video plan, in parallel) picks the real stack and checks cross-machine determinism. Until it
lands, this is the stand-in: the system `ffmpeg` binary decodes the file's audio (no pixels), resampled
to 16 kHz mono s16, and numpy does the rest in float64. Everything downstream (src/video/whistle.py,
scripts/video/v3_*.py) reads only the dict this returns, so swapping the stack is one function.

Features, one row per video frame k (frame k spans [k/fps, (k+1)/fps)); a 1024-sample (64 ms) Hann
window centred on the frame's middle:
  audio_db      total power, dB re a full-scale sine (silence floors at -120)
  band_db       power in the whistle band, BAND_HZ (a referee's whistle sits at ~2.5-4 kHz)
  ratio_db      band_db - audio_db: how much of the sound is in the band
  tonal_db      the band's peak bin over the band's median bin: a whistle is one narrow loud line,
                crowd noise and shouting are broad
  line_db       the strongest *lasting* narrow line in the band: the spectrum in dB minus its median
                over +-K_MED bins (whitening, so broadband bangs and crowd score ~0), max-pooled over
                +-1 bin (a whistle wobbles), then the minimum over LINE_T frames starting at this one
                (it must last LINE_T / fps s). A referee's whistle is one such line at ~2.5-4.5 kHz
  line_hz       that line's frequency
  others        how many other lasting lines (local peaks >= OTHER_DB, more than 4 bins away) the
                frame has in LINES_HZ: a voice is a stack of parallel harmonics, a whistle stands alone
  has_audio     1.0 on every row if the file has an audio stream that isn't digital silence, else 0.0
                (and the rest are 0.0): 2 of SoccerTrack's 10 matches carry an all-zero track (F139)

Every value is rounded to ROUND decimals, so tiny float differences between machines (the AAC decoder
and the resampler are float code) don't reach the decoder's grid; V1 measures whether that is enough.
A file shorter than n_frames pads with the floor; longer is cut.

`source` is a path or an https URL (ffmpeg streams it; nothing is written to disk). `headers` is an
HTTP header block for ffmpeg (e.g. a Hugging Face token, read from the environment by the caller and
never logged).
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STACK = "cpu-ffmpeg-audio-v0"  # provisional: V1 replaces the stack, the dict stays
SR = 16000
WIN = 1024
BAND_HZ = (2000.0, 4800.0)
LINES_HZ = (800.0, 6500.0)
K_MED = 12  # +-12 bins = +-188 Hz
LINE_T = 4  # frames = 0.16 s
OTHER_DB = 10.0
FLOOR_DB = -120.0
ROUND = 3
FEATURES = ("audio_db", "band_db", "ratio_db", "tonal_db", "line_db", "line_hz", "others", "has_audio")


def _ffmpeg_base(headers: str | None) -> list[str]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", headers]
    return cmd


def has_audio(source: str, headers: str | None = None) -> bool:
    cmd = ["ffprobe", "-v", "error"] + (["-headers", headers] if headers else []) + [
        "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", source]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return bool(json.loads(out or b"{}").get("streams"))


def decode_audio(source: str, headers: str | None = None) -> np.ndarray:
    """The file's first audio stream as float64 in [-1, 1], mono, SR Hz."""
    cmd = _ffmpeg_base(headers) + ["-i", source, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR),
                                   "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _spectra(pcm: np.ndarray, n_frames: int, fps: int):
    """Yield (start, power) chunks: power (rows, WIN//2+1) / a full-scale sine's peak, one row per frame."""
    hop = SR / fps
    centres = np.round((np.arange(n_frames) + 0.5) * hop).astype(np.int64)
    pad = np.concatenate([np.zeros(WIN // 2), pcm, np.zeros(WIN)])
    window = np.hanning(WIN)
    ref = (np.sum(window) / 2.0) ** 2
    chunk = 2048
    for s in range(0, n_frames, chunk):
        idx = centres[s:s + chunk, None] + np.arange(WIN)[None, :]  # pad offset WIN//2: c is the centre
        idx = np.minimum(idx, len(pad) - 1)
        yield s, np.abs(np.fft.rfft(pad[idx] * window, axis=1)) ** 2 / ref


def _window_min(a: np.ndarray, t: int) -> np.ndarray:
    """min over rows i..i+t-1 (edge-padded), per column."""
    padded = np.concatenate([a, np.repeat(a[-1:], t - 1, axis=0)])
    return np.min(sliding_window_view(padded, t, axis=0), axis=-1)


def audio_features(pcm: np.ndarray, n_frames: int, fps: int = 25) -> dict[str, np.ndarray]:
    """Per-frame features from SR-Hz mono samples (see the module docstring)."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    lo, hi = np.searchsorted(freqs, LINES_HZ[0]), np.searchsorted(freqs, LINES_HZ[1])
    in_band = band[lo:hi]
    out = {k: np.empty(n_frames) for k in FEATURES}
    white = np.empty((n_frames, hi - lo))
    for s, power in _spectra(pcm, n_frames, fps):
        e = s + len(power)
        tot = power.sum(axis=1)
        bp = power[:, band]
        out["audio_db"][s:e] = 10 * np.log10(np.maximum(tot, 1e-12))
        out["band_db"][s:e] = 10 * np.log10(np.maximum(bp.sum(axis=1), 1e-12))
        out["tonal_db"][s:e] = 10 * np.log10(
            np.maximum(bp.max(axis=1), 1e-12) / np.maximum(np.median(bp, axis=1), 1e-12))
        db = 10 * np.log10(np.maximum(power, 1e-12))
        seg = db[:, lo - K_MED:hi + K_MED]
        white[s:e] = db[:, lo:hi] - np.median(sliding_window_view(seg, 2 * K_MED + 1, axis=1), axis=-1)
    pooled = np.max(sliding_window_view(np.pad(white, ((0, 0), (1, 1)), mode="edge"), 3, axis=1), axis=-1)
    lasting = _window_min(pooled, LINE_T)
    masked = np.where(in_band[None, :], lasting, -np.inf)
    best = masked.argmax(axis=1)
    rows = np.arange(n_frames)
    out["line_db"] = lasting[rows, best]
    out["line_hz"] = freqs[lo:hi][best]
    peaks = (lasting >= np.roll(lasting, 1, axis=1)) & (lasting >= np.roll(lasting, -1, axis=1)) \
        & (lasting >= OTHER_DB)
    far = np.abs(np.arange(hi - lo)[None, :] - best[:, None]) > 4
    out["others"] = (peaks & far).sum(axis=1).astype(np.float64)
    for k in ("audio_db", "band_db"):
        np.maximum(out[k], FLOOR_DB, out=out[k])
    out["ratio_db"] = out["band_db"] - out["audio_db"]
    out["has_audio"] = np.ones(n_frames)
    return {k: np.round(v, ROUND) for k, v in out.items()}


def empty_features(n_frames: int) -> dict[str, np.ndarray]:
    return {k: np.zeros(n_frames) for k in FEATURES}


def frame_features(source: str, n_frames: int, fps: int = 25,
                   headers: str | None = None) -> dict[str, np.ndarray]:
    """THE function the rest of V3 calls. Swap its body when V1 picks the stack."""
    if not has_audio(source, headers):
        return empty_features(n_frames)
    pcm = decode_audio(source, headers)
    if not np.any(pcm):
        return empty_features(n_frames)
    return audio_features(pcm, n_frames, fps)
