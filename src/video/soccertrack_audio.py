"""SoccerTrack v2's half videos -> per-frame audio features (V3), cached in $MTAH_HF_CACHE.

    .venv/bin/python -m src.video.soccertrack_audio fetch            # pull v3/audio-v0, else build + push
    .venv/bin/python -m src.video.soccertrack_audio build --jobs 3 [--push]

build streams each half's video from Hugging Face (`atomscott/soccertrack-v2`, release v1.2, read with
$HF_TOKEN) straight into ffmpeg, keeps only the audio (mono AAC, 48 kHz in the release), and writes
data/raw/external/soccertrack/v3audio/<id>_<half>.npz with src.video.stack's features, one row per
video frame 1..N of the half (N = the compact game state's frames, so row k = the loader's frame k of
that half). Nothing touches the disk but the npz. ~35 s per half on one core, network bound; the 20
halves ~5 min with 3 at a time. External data only.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

from src.ingest import soccertrack
from src.video import stack

OUT = os.path.join(soccertrack.ROOT, "v3audio")
CACHE_NAME = "v3/audio-v0"  # bump with stack.STACK or the feature list
URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/videos/{mid}/{mid}_panorama_{half}_half.mp4"


def video_url(match_id: str, half: str) -> str:
    return URL.format(repo=soccertrack.HF_REPO, rev=soccertrack.HF_REVISION, mid=match_id, half=half)


def _headers() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (the release is gated)")
    return "Authorization: Bearer {}\r\n".format(token)


def path(match_id: str, half: str, out: str = OUT) -> str:
    return os.path.join(out, "{}_{}.npz".format(match_id, half))


def n_frames(match_id: str, half: str) -> int:
    comp = soccertrack.read_compact(os.path.join(soccertrack.compact_dir(), "{}_{}.npz".format(match_id, half)))
    return int(len(comp["status"]))


def build_half(task) -> tuple[str, float]:
    match_id, half, out = task
    t = time.time()
    n = n_frames(match_id, half)
    feats = stack.frame_features(video_url(match_id, half), n, soccertrack.FPS, headers=_headers())
    tmp = path(match_id, half, out) + ".tmp.npz"
    np.savez_compressed(tmp, stack=np.array(stack.STACK), **feats)
    os.replace(tmp, path(match_id, half, out))
    return "{}_{}".format(match_id, half), time.time() - t


def build(matches=soccertrack.MATCHES, jobs: int = 3, out: str = OUT, push: bool = False) -> None:
    os.makedirs(out, exist_ok=True)
    todo = [(m, h, out) for m in matches for h in soccertrack.HALVES if not os.path.isfile(path(m, h, out))]
    with multiprocessing.Pool(max(1, jobs)) as pool:
        for name, secs in pool.imap_unordered(build_half, todo):
            print("  {} {:.0f} s".format(name, secs), flush=True)
    if push:
        from src import hf_cache
        print("pushed {} files to {}".format(hf_cache.push(out, CACHE_NAME), CACHE_NAME))


def ready(out: str = OUT) -> list[str]:
    return [m for m in soccertrack.MATCHES if all(os.path.isfile(path(m, h, out)) for h in soccertrack.HALVES)]


def fetch(out: str = OUT, jobs: int = 3) -> list[str]:
    from src import hf_cache
    try:
        hf_cache.pull(CACHE_NAME, out)
    except Exception as exc:  # empty cache, no token: build instead
        print("cache: {} ({})".format(CACHE_NAME, type(exc).__name__))
    if len(ready(out)) < len(soccertrack.MATCHES):
        build(jobs=jobs, out=out, push=True)
    return ready(out)


def load(match_id: str, out: str = OUT, gap_s: int = soccertrack.GAP_S) -> dict[str, np.ndarray]:
    """Features on the loader's Match timeline: half 1, a gap of GAP_S s (zeros, has_audio 0), half 2."""
    halves = [np.load(path(match_id, h, out)) for h in soccertrack.HALVES]
    gap = gap_s * soccertrack.FPS
    return {k: np.concatenate([halves[0][k], np.zeros(gap), halves[1][k]]) for k in stack.FEATURES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "fetch"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--matches", default="all")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print("v3 audio: {} matches ready".format(len(fetch(jobs=a.jobs))))
        return 0
    matches = soccertrack.MATCHES if a.matches == "all" else tuple(a.matches.split(","))
    build(matches, a.jobs, push=a.push)
    return 0


if __name__ == "__main__":
    sys.exit(main())
