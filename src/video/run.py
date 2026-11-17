"""Run the video ball finder on a match video and save the ball track (backlog V2).

On BAR-ATH (owner's machine only, in your own terminal; never in a cloud session):

    export MTAH_DATA_DIR=$PWD/data/mtah2027
    .venv/bin/python -m src.video.run --out outputs/video_ball.npz [--jobs 4]
    .venv/bin/python -m src.video.run --out outputs/video_ball_2.npz --expect outputs/video_ball.npz

It reads the tactical video, its per-frame homography and the ATD players (for the player features),
finds ball candidates on every STEP-th frame inside the published halves, scores them
(src/video/ball_scorer.json), links them per half (src/video/track.py) and writes
  ball (n, 2) float64 pitch metres (the ATD's frame), NaN where the video doesn't see the ball,
  conf (n,) float64 the linked candidate's logit, NaN where unseen,
  stack (str) the detector + scorer version, so a stale file is refused.
It prints words only: OK and the time it took, or with --expect IDENTICAL / DIFFERENT (the same bytes
as an earlier file: run it twice, or on two machines, for the determinism check). Nothing else about
the match is printed, and nothing should be reported back but those words (CLAUDE.md hard rule).

Cost (measured on SoccerTrack's 4096 x 1080 video, one core): ~0.3 s per processed frame; at 1920 x
1080 about half that. BAR-ATH's halves are ~159 k frames, so at 5 Hz ~32 k detections: ~1.5 h on one
core, ~25 min with --jobs 4. Needs ffmpeg on the PATH (macOS: brew install ffmpeg).
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import sys
import time

import numpy as np

from src.video import ball, decode, track
from src.video.calib import FrameHomography

STEP = 5  # frames: 5 Hz
STACK = "v2-provisional-1"  # bump when detect(), FEATURES or the scorer format change


def stack_id(scorer_path: str = ball.SCORER_FILE) -> str:
    with open(scorer_path, "rb") as f:
        return "{}/{}".format(STACK, hashlib.sha256(f.read()).hexdigest()[:12])


def frame_jobs(n_frames: int, blocks: list) -> list[int]:
    """Frames to process: every STEP-th frame of each half block (first..last inclusive)."""
    out = []
    for _, first, last in blocks:
        out.extend(range(int(first), int(last) + 1, STEP))
    return [f for f in out if ball.MOTION_STEP <= f < n_frames - ball.MOTION_STEP]


def _detect_chunk(task):
    path, frames, homographies, valid, players, pitch, scorer_path = task
    scorer = ball.Scorer.load(scorer_path)
    meta = decode.probe(path)
    want = set(frames)
    out = {}
    buf = {}
    first, last = frames[0] - ball.MOTION_STEP, frames[-1] + ball.MOTION_STEP
    for idx, y, u, v in decode.frames(path, start=first, count=last - first + 1, threads=1, meta=meta):
        buf[idx] = (y.copy(), u.copy(), v.copy())
        f = idx - ball.MOTION_STEP
        if f in want and f - ball.MOTION_STEP in buf:
            if valid[f]:
                calib = FrameHomography(homographies[f], pitch)
                ctx = ball.Context.build(calib, (meta["height"], meta["width"]))
                c = ball.detect(buf[f - ball.MOTION_STEP], buf[f], buf[idx], ctx, players_xy=players[f])
                xy = np.stack([c["x"], c["y"]], axis=1) if len(c) else np.zeros((0, 2))
                out[f] = (xy, scorer.score_candidates(c, pitch) if len(c) else np.zeros(0))
            else:
                out[f] = (np.zeros((0, 2)), np.zeros(0))
        for old in [k for k in buf if k < idx - 2 * ball.MOTION_STEP]:
            del buf[old]
    return out


def run(path: str, homographies: np.ndarray, valid: np.ndarray, players: np.ndarray, blocks: list,
        pitch, jobs: int = 1, scorer_path: str = ball.SCORER_FILE) -> dict:
    n = len(homographies)
    frames = frame_jobs(n, blocks)
    chunks = [c for c in np.array_split(np.array(frames, dtype=np.int64), max(1, jobs * 4)) if len(c)]
    tasks = [(path, c.tolist(), homographies, valid, players, pitch, scorer_path) for c in chunks]
    found = {}
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            for part in pool.imap(_detect_chunk, tasks):
                found.update(part)
    else:
        for t in tasks:
            found.update(_detect_chunk(t))
    ball_xy = np.full((n, 2), np.nan)
    conf = np.full(n, np.nan)
    for _, first, last in blocks:
        fs = [f for f in frames if first <= f <= last]
        if not fs:
            continue
        tr = track.link(np.array(fs), [found[f][0] for f in fs], [found[f][1] for f in fs],
                        STEP / 25.0)
        b, c = track.to_frames(tr, n, max_gap=2 * STEP)
        sel = np.isfinite(b[:, 0])
        ball_xy[sel] = b[sel]
        conf[sel] = c[sel]
    return {"ball": ball_xy, "conf": conf, "stack": stack_id(scorer_path)}


def save(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # np.savez (not compressed) of fixed arrays: same values -> same bytes
    with open(path, "wb") as f:
        np.savez(f, ball=result["ball"], conf=result["conf"], stack=np.array(result["stack"]))


def load(path: str, n_frames: int | None = None) -> dict:
    """The saved track; refuses a file made by another detector / scorer version."""
    with np.load(path) as z:
        out = {"ball": z["ball"], "conf": z["conf"], "stack": str(z["stack"])}
    if out["stack"] != stack_id():
        raise ValueError("{} was made by {}, this code is {}: re-run src.video.run".format(
            path, out["stack"], stack_id()))
    if n_frames is not None and len(out["ball"]) != n_frames:
        raise ValueError("{}: {} rows, the match has {}".format(path, len(out["ball"]), n_frames))
    return out


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=None, help="the bundle (default $MTAH_DATA_DIR)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--expect", default=None, help="compare with an earlier output: IDENTICAL / DIFFERENT")
    a = p.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "official"))
    import games  # noqa: E402  (official/games.py)
    from starter import load_homography  # noqa: E402

    from src.ingest import bar_ath

    t0 = time.time()
    match = bar_ath.load(a.data_dir, tags=False)
    video = games.require(["video"], games.GAME, a.data_dir, "tactical")[0]
    hom, _, valid = load_homography.load(games.GAME, a.data_dir, "tactical")
    if len(hom) != match.n_frames:
        print("FAIL: homography rows != tracking frames")
        return 1
    blocks = games.half_video_blocks(games.GAME)
    result = run(video, hom, valid, match.players, blocks, match.pitch, a.jobs)
    save(a.out, result)
    print("OK ({:.0f} min)".format((time.time() - t0) / 60))
    if a.expect:
        print("IDENTICAL" if _file_digest(a.out) == _file_digest(a.expect) else "DIFFERENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
