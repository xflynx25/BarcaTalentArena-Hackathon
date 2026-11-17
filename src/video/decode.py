"""Decode video frames with the system ffmpeg into numpy, bit-exact on every machine (backlog V2).

    for index, y, u, v in frames(path, start=1000, count=250):   # 0-based frame index
        ...   # y (h, w) uint8 luma, u / v (h/2, w/2) uint8 chroma (yuv420p as stored)

Why this shape: H.264 decoding is specified bit-exactly, so ffmpeg's software decoder gives the same
pixels on Linux and macOS whatever its version, as long as nothing converts or scales the frames.
So we ask for the stream's own pixel format (yuv420p) as raw video and do every conversion in numpy;
never -hwaccel (a hardware decoder is not bound by that), never a -vf scale. Both SoccerTrack v2 and
the BAR-ATH videos are H.264 yuv420p (ffprobe, official/DATA.md); probe() refuses anything else.

Seeking: `-ss` before `-i` lands on the first frame whose time is >= ss + the container's start time,
and the video stream can start later than the container (SoccerTrack: video 1.000 s, container 0.021
s). seek_time() turns a frame index into that time, half a frame early, so the first frame out is the
one asked for; checked against a sequential decode by frame MD5 (tests/test_video.py on a synthetic clip; on
SoccerTrack at 7 seek points, F136).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError("{} not found: install ffmpeg (macOS: brew install ffmpeg)".format(name))
    return path


def probe(path: str) -> dict:
    """width, height, fps (float), frames, pix_fmt, codec, video and container start times."""
    out = subprocess.run([_which(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,start_time"
                          ":format=start_time", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    s, f = info["streams"][0], info.get("format", {})
    num, den = s["r_frame_rate"].split("/")
    meta = dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                frames=int(s.get("nb_frames") or 0), pix_fmt=s["pix_fmt"], codec=s["codec_name"],
                video_start=float(s.get("start_time") or 0.0),
                container_start=float(f.get("start_time") or 0.0))
    if meta["pix_fmt"] != "yuv420p":
        raise ValueError("{}: pixel format {} (expected yuv420p)".format(path, meta["pix_fmt"]))
    return meta


def seek_time(index: int, meta: dict) -> float:
    return max(0.0, (index - 0.5) / meta["fps"] + meta["video_start"] - meta["container_start"])


def frames(path: str, start: int = 0, count: int | None = None, threads: int = 0,
           meta: dict | None = None):
    """Yield (index, y, u, v) for frames start .. start + count - 1 (to the end if count is None)."""
    meta = meta or probe(path)
    w, h = meta["width"], meta["height"]
    size = w * h + 2 * (w // 2) * (h // 2)
    cmd = [_which(FFMPEG), "-v", "error", "-nostdin", "-threads", str(threads)]
    if start > 0:
        cmd += ["-ss", "{:.6f}".format(seek_time(start, meta))]
    cmd += ["-i", path, "-map", "0:v:0", "-an", "-sn", "-fps_mode", "passthrough"]
    if count is not None:
        cmd += ["-frames:v", str(int(count))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 2)
    index = start
    try:
        while count is None or index < start + count:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            a = np.frombuffer(buf, dtype=np.uint8)
            y = a[:w * h].reshape(h, w)
            u = a[w * h:w * h + (w // 2) * (h // 2)].reshape(h // 2, w // 2)
            v = a[w * h + (w // 2) * (h // 2):].reshape(h // 2, w // 2)
            yield index, y, u, v
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def frame_md5s(path: str, start: int, count: int) -> list[str]:
    """MD5 of each decoded frame (ffmpeg's framemd5 muxer), for decode checks."""
    meta = probe(path)
    cmd = [_which(FFMPEG), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", "{:.6f}".format(seek_time(start, meta))]
    cmd += ["-i", path, "-map", "0:v:0", "-frames:v", str(count), "-f", "framemd5", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]
