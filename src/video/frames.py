"""Decode video frames bit-exactly (V1 interface).

FFmpeg's software H.264 decoder is bit-exact by the standard, so the raw YUV planes are the same on
every machine. Pixel-format conversion and scaling (FFmpeg's swscale) are not: they pick SIMD paths
per CPU. So frames come out as raw yuv420p planes, and `to_rgb` converts and downscales them in
integer numpy, which every machine computes identically. Hardware decoders are never used.

Index i is the i-th decoded frame (0-based, presentation order). BAR-ATH tactical: index i is
submission row i. SoccerTrack v2: release frame k (1-based) is index k - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    y: np.ndarray  # (h, w) uint8, luma
    u: np.ndarray  # (h/2, w/2) uint8, Cb
    v: np.ndarray  # (h/2, w/2) uint8, Cr


def info(path: str) -> dict:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        return {"width": s.width, "height": s.height, "fps": float(s.average_rate), "frames": s.frames,
                "codec": s.codec_context.name, "pix_fmt": s.codec_context.pix_fmt,
                "audio": len(c.streams.audio) > 0}


def iter_frames(path: str, indices=None, start: int = 0, stop: int | None = None,
                threads: int = 0) -> Iterator[tuple[int, Frame]]:
    """Yield (index, Frame) in order. `indices` (sorted, any iterable) picks frames; otherwise
    start..stop. Decodes sequentially (no seeking: keyframe seeks are where decoders disagree on what
    "frame i" is). threads=0 lets FFmpeg choose; the output does not depend on it (F135)."""
    import av

    want = None if indices is None else iter(sorted(set(int(i) for i in indices)))
    nxt = None if want is None else next(want, None)
    if want is not None and nxt is None:
        return
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO" if threads != 1 else "NONE"
        if threads > 1:
            s.thread_count = threads
        for i, f in enumerate(c.decode(s)):
            if want is not None:
                if i < nxt:
                    continue
            elif i < start:
                continue
            if stop is not None and i >= stop:
                return
            if f.format.name != "yuv420p":
                raise ValueError("{}: pixel format {} (expected yuv420p)".format(path, f.format.name))
            planes = [np.frombuffer(p, dtype=np.uint8) for p in f.planes]
            out = []
            for p, plane, (h, w) in zip(planes, f.planes, ((f.height, f.width), (f.height // 2, f.width // 2),
                                                          (f.height // 2, f.width // 2))):
                out.append(p.reshape(-1, plane.line_size)[:h, :w].copy())
            yield i, Frame(*out)
            if want is not None:
                nxt = next(want, None)
                if nxt is None:
                    return


def downscale(plane: np.ndarray, factor: int) -> np.ndarray:
    """Integer area average by `factor` (rounded half up), exact on every machine."""
    if factor == 1:
        return plane
    h, w = (plane.shape[0] // factor) * factor, (plane.shape[1] // factor) * factor
    p = plane[:h, :w].astype(np.int32).reshape(h // factor, factor, w // factor, factor)
    n = factor * factor
    return ((p.sum(axis=(1, 3)) + n // 2) // n).astype(np.uint8)


def crop(f: Frame, box) -> Frame:
    """The frame inside box=(u0, v0, u1, v1), widened to even pixels so the chroma stays aligned."""
    u0, v0 = (int(box[0]) // 2) * 2, (int(box[1]) // 2) * 2
    u1, v1 = min(f.y.shape[1], -(-int(np.ceil(box[2])) // 2) * 2), min(f.y.shape[0], -(-int(np.ceil(box[3])) // 2) * 2)
    return Frame(f.y[v0:v1, u0:u1], f.u[v0 // 2:v1 // 2, u0 // 2:u1 // 2], f.v[v0 // 2:v1 // 2, u0 // 2:u1 // 2])


def to_rgb(f: Frame, factor: int = 1) -> np.ndarray:
    """yuv420p (BT.709, limited range: HD video) -> RGB uint8 (h/factor, w/factor, 3), integer maths.
    factor 2 works at the chroma's own resolution (no chroma upsampling); factor 1 repeats chroma.
    Convert only what the detector reads: `crop` first (a 4096 x 1080 frame costs ~140 ms here)."""
    if factor == 1:
        u = np.repeat(np.repeat(f.u, 2, axis=0), 2, axis=1)[: f.y.shape[0], : f.y.shape[1]]
        v = np.repeat(np.repeat(f.v, 2, axis=0), 2, axis=1)[: f.y.shape[0], : f.y.shape[1]]
        y = f.y
    elif factor % 2 == 0:
        y = downscale(f.y, factor)
        u, v = downscale(f.u, factor // 2), downscale(f.v, factor // 2)
        y = y[: u.shape[0], : u.shape[1]]
    else:
        raise ValueError("factor must be 1 or even")
    c = (y.astype(np.int32) - 16) * 298
    d, e = u.astype(np.int32) - 128, v.astype(np.int32) - 128
    rgb = np.empty(y.shape + (3,), dtype=np.int32)
    rgb[..., 0] = (c + 459 * e + 128) >> 8
    rgb[..., 1] = (c - 55 * d - 136 * e + 128) >> 8
    rgb[..., 2] = (c + 541 * d + 128) >> 8
    return np.clip(rgb, 0, 255).astype(np.uint8)
