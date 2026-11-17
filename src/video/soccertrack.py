"""SoccerTrack v2 video, one half at a time (V1). External data only.

The panoramic half videos are 6.7 GB each (4096 x 1080, H.264, 25 fps, ~67k frames) and the cloud
disk is 30 GB, so every tool here works on one half: `fetch_half` downloads the video, the ball
track, the game state and the ball actions; `drop_half` deletes the big two. Video frame k (1-based,
as the release counts) is decoded frame k - 1 of the mp4, game-state frame k and ball row k
(release v1.2: "game state frame k is video frame k in every half").

Ground truth for the ball in pixels: the ball track through the provider's calibration
(src/video/camera.py SoccerTrackCamera; polynomial fits on keypoints or players miss by 50+ px, the
panorama is a fisheye). The game state also carries each player's box in the panorama (`bbox_image`)
next to his pitch position, which checks the calibration and scores person detection. The ball track
is the provider's reconstruction (exact at the annotated actions, straight lines in between, on a
1.05 x 0.68 m grid, F108), so ball detection is scored at the action frames, where it is exact.

Files go to data/raw/external/soccertrack/video/ (gitignored): the download layout of the release,
plus cache/<id>_<half>_points.npz (players' pixel + pitch points, ~30 MB) so the 2.7 GB game state
can be deleted.
"""

from __future__ import annotations

import os

import numpy as np

HF_REPO = "atomscott/soccertrack-v2"
HF_REVISION = "v1.2"
ROOT = os.path.join("data", "raw", "external", "soccertrack", "video")
# Paper labels M1..M10 = ids in ascending order; V-task split from the release (docs/external-datasets.md).
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
TEST = ["128057", "132831"]  # M7, M9
VALIDATION = ["117093", "132877"]  # M2, M10


def _download(filename: str, root: str = ROOT) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, repo_type="dataset", revision=HF_REVISION, local_dir=root,
                           token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def half_files(match_id: str, half: str) -> dict:
    """Release paths of one half (half is "1st" or "2nd")."""
    return {
        "video": "videos/{0}/{0}_panorama_{1}_half.mp4".format(match_id, half),
        "gsr": "gsr/{0}/{0}_{1}.json".format(match_id, half),
        "ball": "ball/{0}_{1}_ball.npz".format(match_id, half),
        "events": "bas/{0}/{0}_12_class_events.json".format(match_id),
    }


def clip_files(match_id: str) -> dict:
    """The 4-minute MOT clip of a match (~300 MB, players' boxes only): small enough for an owner check."""
    return {"video": "mot/clips/{}.mp4".format(match_id), "mot": "mot/{}.txt".format(match_id)}


def fetch_half(match_id: str, half: str, root: str = ROOT, video: bool = True) -> dict:
    """Download one half (video 6.7 GB in ~30 s from a cloud session; game state 2.7 GB in ~5 s)."""
    files = half_files(match_id, half)
    if not video:
        files.pop("video")
    return {k: _download(v, root) for k, v in files.items()}


def drop_half(match_id: str, half: str, root: str = ROOT) -> None:
    """Delete the big files of a half (video, game state); keeps the small ones and the cache."""
    for key in ("video", "gsr"):
        path = os.path.join(root, half_files(match_id, half)[key])
        if os.path.exists(path):
            os.remove(path)


def points_path(match_id: str, half: str, root: str = ROOT) -> str:
    return os.path.join(root, "cache", "{}_{}_points.npz".format(match_id, half))


def player_points(match_id: str, half: str, root: str = ROOT) -> dict:
    """Every player's box in the panorama with his pitch position, for one half, cached.

    Returns arrays over objects: frame (1-based video frame), u (box bottom-centre x, px), v (box
    bottom y, px), w, h (box size, px), x, y (pitch metres, centre origin, as the release).
    """
    out = points_path(match_id, half, root)
    if os.path.exists(out):
        with np.load(out, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    import ijson

    path = os.path.join(root, half_files(match_id, half)["gsr"])
    if not os.path.exists(path):
        path = fetch_half(match_id, half, root, video=False)["gsr"]
    cols = {k: [] for k in ("frame", "u", "v", "w", "h", "x", "y")}
    with open(path, "rb") as handle:
        for a in ijson.items(handle, "annotations.item", use_float=True):
            if a.get("supercategory") != "object":
                continue
            box, pitch = a.get("bbox_image") or {}, a.get("bbox_pitch") or {}
            x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
            if x is None or y is None or box.get("w") is None:
                continue
            cols["frame"].append(int(str(a["image_id"])[-6:]))
            cols["u"].append(box["x"] + box["w"] / 2.0)
            cols["v"].append(box["y"] + box["h"])
            cols["w"].append(box["w"])
            cols["h"].append(box["h"])
            cols["x"].append(x)
            cols["y"].append(y)
    arrays = {"frame": np.array(cols["frame"], dtype=np.int32)}
    arrays.update({k: np.array(cols[k], dtype=np.float32) for k in ("u", "v", "w", "h", "x", "y")})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    return arrays


def ball_track(match_id: str, half: str, root: str = ROOT) -> dict:
    """The release's ball track for a half: frame (1-based), x, y (metres, centre origin)."""
    path = os.path.join(root, half_files(match_id, half)["ball"])
    if not os.path.exists(path):
        path = _download(half_files(match_id, half)["ball"], root)
    with np.load(path, allow_pickle=False) as z:
        return {"frame": z["frame"].astype(np.int64), "x": z["x"].astype(np.float64),
                "y": z["y"].astype(np.float64)}


def action_frames(match_id: str, half: str, root: str = ROOT) -> np.ndarray:
    """1-based video frames of the half's ball actions: where the ball track is exact (an anchor)."""
    import json

    path = os.path.join(root, half_files(match_id, half)["events"])
    if not os.path.exists(path):
        path = _download(half_files(match_id, half)["events"], root)
    with open(path) as handle:
        actions = json.load(handle)["actions"]
    want = "1" if half == "1st" else "2"
    return np.array(sorted({int(a["frame"]) for a in actions if str(a["gameTime"]).split(" ")[0] == want}),
                    dtype=np.int64)
