"""SoccerTrack v2 (10 university matches, fixed panoramic camera, 25 fps) -> Match with labels.

Source: Hugging Face dataset `atomscott/soccertrack-v2`, release v1.2, CC BY 4.0, gated (read with
$HF_TOKEN; the owner accepted the terms). Paper labels M1..M10 = ids in ascending order. Layout and
quirks are in docs/external-datasets.md and findings F108-F110.

Two stages, because the game state is ~54 GB of JSON:

  build   originals -> compact files, once:  python -m src.ingest.soccertrack build [--jobs 3]
          per half: download gsr/<id>/<id>_<half>.json (2.7 GB), stream it with ijson (~20 s), write
          compact/<id>_<half>.npz (float32), delete the JSON. Plus the small files: ball/ (ball track
          and the provider's per-frame ball status), bas/ (ball actions), raw/<id>_tracker_box_
          metadata.xml (teams, players). --push uploads compact/ to $MTAH_HF_CACHE (src/hf_cache.py).
  fetch   python -m src.ingest.soccertrack fetch: pull compact/ from the cache; build only if the
          cache is empty (scripts/fetch_external_data.sh soccertrack runs this).

The loader reads compact/ only: data/raw/external/soccertrack/compact/
  <id>_1st.npz, <id>_2nd.npz   rows = video frames 1..N of the half (game state frame k = video frame
                               k = ball row k = event `frame` k). Fields: xy (N, k, 2) players,
                               ball (N, 2), status (N,), labeled (N,), player_id, side (0 left,
                               1 right, per half), gk, jersey (k,). Coordinates as the source: metres,
                               centre origin, x to the right half as seen from the camera, y towards
                               the near touchline.
  <id>_meta.json               home / away team ids, player -> team, format version.
  <id>_events.json             the ball-action file (bas), unchanged.

Match layout. Half 1's frames, a 60 s gap of period-0 rows (GAP_S; the halves are separate videos, and
the gap keeps time windows from reaching across half-time, as idsse.py does with the real break), half
2's frames. Period 1 / 2 on every annotated frame; the 10 unannotated leading frames of 132877's first
half are period 0. Coordinates go to the schema's corner origin: x + 52.5, 34 - y (so y = 0 is the
near touchline). Pitch 105 x 68 m (the dataset's README). Teams keep their real ends, so they swap at
half-time (src.shape finds the direction label-free). Player slots: one per player id over both
halves, home first, by id; the game state has all 22 players on 98.7% of frames.

Ball. One row per frame, but it is the provider's reconstruction, not a measurement: anchored at the
ball actions and moved in straight constant-speed lines between them, on the provider's 1.05 x 0.68 m
grid, clamped to the pitch in 132831 and 132877 (README). It rarely goes past a line (F108).

Labels (labels=, decided in docs/decisions.md, A17):
  "actions" (default): from the ball actions, following the competition's rule. The team of the last
     controlled action holds possession: Pass, Drive, High Pass, Cross, Throw In, Shot, Free Kick,
     Player Successful Tackle (a won tackle switches the team). Header and Ball Player Block don't
     change possession by themselves (a touch, a deflection or an aerial duel without control, rule
     in official/CHALLENGE.md); the next controlled action decides. Dead ball: D from each Out or
     Goal event to the frame before the next action, which is the restart (Throw In, Free Kick, or a
     Pass/High Pass/Cross for corners, goal kicks and kick-offs). Every Free Kick in the release
     follows an Out, so fouls are covered too (F109). Before the first action of a half: D. After the
     last: its team holds to the end of the half's video (the video stops at the provider's half end),
     or D if the last action is Out/Goal.
  "status": the provider's per-frame ball status from ball/ (0 ball out -> D, 1 home -> H, 2 away ->
     A, 3 neutral -> the last team, the competition's "loose ball stays with the last team"; -1 no
     record -> X). 132831 and 132877 have no status (4 on every frame -> X), so only 8 matches are
     graded with labels="status". Closer in kind to how the competition's truth is made (a tracking vendor's flag),
     but its D starts ~0.9 s before the Out event (F109), like SkillCorner (F84).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from src.official import A, D, H, X
from src.schema import AWAY, HOME, Match

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "raw", "external", "soccertrack")
HF_REPO = "atomscott/soccertrack-v2"
HF_REVISION = "v1.2"  # the release tag the loader was written against
CACHE_NAME = "soccertrack/compact-v1"  # bump the suffix when the compact format changes
FORMAT = 1
MATCHES = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831",
           "132877")
HALVES = ("1st", "2nd")
FPS = 25
PITCH = (105.0, 68.0)
GAP_S = 60

SWITCH = {"PASS", "DRIVE", "HIGH PASS", "CROSS", "THROW IN", "SHOT", "FREE KICK",
          "PLAYER SUCCESSFUL TACKLE"}
TOUCH = {"HEADER", "BALL PLAYER BLOCK"}
STOP = {"OUT", "GOAL"}
LABEL_KINDS = ("actions", "status")
BALL_KINDS = ("shipped", "none", "smooth")  # load(ball=...), A18
SMOOTH_S = 0.3  # ball="smooth": centred window, s: the smallest that brings its touch rate into the other sources' range (A18, F151)
STATUS_DEAD, STATUS_HOME, STATUS_AWAY, STATUS_NEUTRAL, STATUS_NONE = 0, 1, 2, 3, -1
STATUS_ABSENT = 4  # every frame of 132831 and 132877: those two matches have no vendor status


def compact_dir(root: str = ROOT) -> str:
    return os.path.join(root, "compact")


def _compact_paths(match_id: str, root: str) -> dict:
    c = compact_dir(root)
    out = {h: os.path.join(c, "{}_{}.npz".format(match_id, h)) for h in HALVES}
    out["meta"] = os.path.join(c, "{}_meta.json".format(match_id))
    out["events"] = os.path.join(c, "{}_events.json".format(match_id))
    return out


def list_matches(root: str = ROOT) -> list[str]:
    """Ids whose compact files are all present."""
    return [m for m in MATCHES if all(os.path.isfile(p) for p in _compact_paths(m, root).values())]


# -- labels ------------------------------------------------------------------------------------------

def half_events(events: list[dict], half: int, side_team: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """(frames 1-based, teams HOME/AWAY, labels) of one half's actions, in time order (file order on
    ties). side_team maps the event's 'left'/'right' to HOME/AWAY for that half."""
    rows = [e for e in events if e["gameTime"].split(" - ")[0].strip() == str(half)]
    unknown = {e["label"] for e in rows} - (SWITCH | TOUCH | STOP)
    if unknown:
        raise ValueError("unknown ball-action classes: {}".format(sorted(unknown)))
    order = np.argsort([int(e["frame"]) for e in rows], kind="stable")
    rows = [rows[i] for i in order]
    frames = np.array([int(e["frame"]) for e in rows], dtype=np.int64)
    teams = np.array([side_team.get(e.get("team"), -1) for e in rows], dtype=np.int8)
    return frames, teams, [e["label"] for e in rows]


def action_labels(frames: np.ndarray, teams: np.ndarray, kinds: list, n: int) -> np.ndarray:
    """(n,) H/A/D codes for one half of n frames from its time-ordered actions (module docstring)."""
    out = np.full(n, D, dtype=np.int8)
    if not len(frames):
        return out
    rows = np.clip(frames - 1, 0, n - 1)
    switch = np.array([k in SWITCH for k in kinds]) & (teams >= 0)
    stop = np.array([k in STOP for k in kinds])
    if switch.any():
        owner = np.full(n, -1, dtype=np.int64)  # index of the last switching action at or before
        owner[rows[switch]] = np.flatnonzero(switch)  # later actions on the same frame win
        owner = np.maximum.accumulate(owner)
        first = np.flatnonzero(switch)[0]
        team = np.where(owner >= 0, teams[np.maximum(owner, 0)], teams[first])
        out = np.where(team == HOME, H, A).astype(np.int8)
    dead = np.zeros(n, dtype=bool)
    dead[:rows[0]] = True  # before the first action of the half
    live_rows = rows[~stop]
    for r in rows[stop]:  # the restart is the first action on a later frame (a Shot logged on the
        nxt = np.searchsorted(live_rows, r, side="right")  # Goal's own frame is not a restart)
        dead[r:live_rows[nxt] if nxt < len(live_rows) else n] = True
    out[dead] = D
    return out


def status_labels(status: np.ndarray) -> np.ndarray:
    """(n,) codes from the provider's ball status: 0 -> D, 1 -> H, 2 -> A, 3 (neutral) -> the last
    team (D if no team yet in the half), -1 (no record) and 4 (no status in this match) -> X."""
    out = np.full(len(status), D, dtype=np.int8)
    team = np.where(status == STATUS_HOME, H, np.where(status == STATUS_AWAY, A, -1))
    idx = np.where(team >= 0, np.arange(len(status)), -1)
    idx = np.maximum.accumulate(idx)
    held = np.where(idx >= 0, team[np.maximum(idx, 0)], D)
    out[(status == STATUS_HOME) | (status == STATUS_AWAY)] = team[(status == STATUS_HOME) |
                                                                  (status == STATUS_AWAY)]
    neutral = status == STATUS_NEUTRAL
    out[neutral] = held[neutral]
    out[(status == STATUS_NONE) | (status == STATUS_ABSENT)] = X
    unknown = ~np.isin(status, (STATUS_DEAD, STATUS_HOME, STATUS_AWAY, STATUS_NEUTRAL, STATUS_NONE,
                                STATUS_ABSENT))
    if unknown.any():
        raise ValueError("ball status has values other than -1..4")
    return out


# -- loading -----------------------------------------------------------------------------------------

def read_compact(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def side_teams(half: dict, player_team: dict) -> dict:
    """{'left': HOME/AWAY, 'right': ...} for one half, from its players' sides and the match's teams.
    Raises if a side holds players of both teams."""
    out = {}
    for code, name in ((0, "left"), (1, "right")):
        teams = {player_team[int(p)] for p, s in zip(half["player_id"], half["side"]) if s == code}
        if len(teams) != 1:
            raise ValueError("side {} holds {} teams".format(name, len(teams)))
        out[name] = teams.pop()
    if out["left"] == out["right"]:
        raise ValueError("both sides are the same team")
    return out


def smooth_ball(ball: np.ndarray, period: np.ndarray, fps: int, span_s: float) -> np.ndarray:
    """The ball averaged over a centred window of span_s within each period (NaN-aware; frames outside
    a period untouched). SoccerTrack's ball climbs its 1.05 x 0.68 m grid one cell at a time and stands
    still on ~3/4 of frames (F151), which the touch detector reads as kicks; the average turns the
    staircase back into the straight line it was drawn from."""
    half = max(1, int(round(span_s * fps / 2)))
    out = ball.copy()
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        seg = ball[rows]
        seen = np.isfinite(seg[:, 0])
        vals = np.where(seen[:, None], seg, 0.0)
        cs = np.vstack([np.zeros((1, 2)), np.cumsum(vals, axis=0)])
        cn = np.concatenate(([0], np.cumsum(seen)))
        i = np.arange(len(rows))
        lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(rows))
        n = (cn[hi] - cn[lo])[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows] = np.where(seen[:, None] & (n > 0), (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out


def load(match_id: str | int, root: str = ROOT, labels: str = "actions", ball: str = "shipped",
         smooth_s: float = SMOOTH_S) -> Match:
    """One SoccerTrack v2 match as a Match (module docstring). labels="actions" (default) or "status".
    ball (A18): "shipped" (default) as released; "none" hidden on every frame (train on its players and
    labels only, since its ball is reconstructed from the actions, F108); "smooth" averaged over a
    centred smooth_s window (smooth_ball, F151)."""
    if labels not in LABEL_KINDS:
        raise ValueError("labels must be one of {}, not {!r}".format(LABEL_KINDS, labels))
    if ball not in BALL_KINDS:
        raise ValueError("ball must be one of {}, not {!r}".format(BALL_KINDS, ball))
    ball_kind = ball
    match_id = str(match_id)
    paths = _compact_paths(match_id, root)
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing; run scripts/fetch_external_data.sh soccertrack"
                                    .format(path))
    with open(paths["meta"]) as handle:
        meta = json.load(handle)
    if meta.get("format") != FORMAT:
        raise ValueError("{}: compact format {} != {}; re-run the fetch".format(
            match_id, meta.get("format"), FORMAT))
    player_team = {int(p): (HOME if v["team"] == "home" else AWAY) for p, v in meta["players"].items()}
    with open(paths["events"]) as handle:
        events = json.load(handle)["actions"]
    halves = [read_compact(paths[h]) for h in HALVES]

    ids = sorted({int(p) for h in halves for p in h["player_id"]}, key=lambda p: (player_team[p], p))
    slot_of = {p: i for i, p in enumerate(ids)}
    gap = GAP_S * FPS
    n = sum(len(h["ball"]) for h in halves) + gap
    players = np.full((n, len(ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    period = np.zeros(n, dtype=np.int8)
    codes = np.full(n, X, dtype=np.int8)
    start = 0
    for number, half in enumerate(halves, start=1):
        size = len(half["ball"])
        rows = slice(start, start + size)
        slots = np.array([slot_of[int(p)] for p in half["player_id"]], dtype=np.int64)
        players[rows, slots] = half["xy"]
        ball[rows] = half["ball"]
        period[rows] = np.where(half["labeled"], number, 0)
        if labels == "status":
            out = status_labels(half["status"])
        else:
            out = action_labels(*half_events(events, number, side_teams(half, player_team)), size)
        codes[rows] = np.where(half["labeled"], out, X)
        start += size + gap

    def to_corner(xy):
        return np.stack([xy[..., 0] + PITCH[0] / 2, PITCH[1] / 2 - xy[..., 1]], axis=-1)

    ball = to_corner(ball)
    if ball_kind == "smooth":
        ball = smooth_ball(ball, period, FPS, float(smooth_s))
    return Match(
        match_id="soccertrack:{}".format(match_id),
        fps=FPS,
        pitch=PITCH,
        period=period,
        ball=np.full((n, 2), np.nan) if ball_kind == "none" else ball,
        players=to_corner(players),
        slot_team=np.array([player_team[p] for p in ids], dtype=np.int8),
        labels=codes,
    ).validate()


# -- build: originals -> compact ---------------------------------------------------------------------

def parse_gsr(path: str, n: int) -> dict:
    """Stream one game-state half (SoccerNet-COCO object, ~2.7 GB) into arrays for n frames.
    Positions are bbox_pitch.x/y_bottom_middle (the three bottom points are identical in v1.2)."""
    import ijson

    labeled = np.zeros(n, dtype=bool)
    with open(path, "rb") as handle:  # images come first; stop at the annotations
        image_id, is_labeled = None, True
        for prefix, event, value in ijson.parse(handle):
            if prefix == "images.item.image_id":
                image_id = value
            elif prefix == "images.item.is_labeled":
                is_labeled = bool(value)
            elif prefix == "images.item" and event == "end_map":
                frame = int(str(image_id)[-6:])
                if 1 <= frame <= n:
                    labeled[frame - 1] = is_labeled
                image_id, is_labeled = None, True
            elif prefix == "annotations" and event == "start_array":
                break
    frames, pids, xs, ys, sides, gks, jerseys = [], [], [], [], [], [], []
    with open(path, "rb") as handle:
        for a in ijson.items(handle, "annotations.item", use_float=True):
            if a.get("supercategory") != "object":
                continue
            attrs = a.get("attributes") or {}
            pitch = a.get("bbox_pitch") or {}
            x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
            if x is None or y is None or attrs.get("player_id") is None:
                continue
            frames.append(int(str(a["image_id"])[-6:]))
            pids.append(int(attrs["player_id"]))
            xs.append(x)
            ys.append(y)
            sides.append(attrs.get("team"))
            gks.append(attrs.get("role") == "goalkeeper")
            jerseys.append(attrs.get("jersey"))
    frames = np.array(frames, dtype=np.int64)
    pids = np.array(pids, dtype=np.int64)
    if len(frames) and (frames.min() < 1 or frames.max() > n):
        raise ValueError("{}: object frames outside 1..{}".format(path, n))
    ids = np.unique(pids)
    slot = np.searchsorted(ids, pids)
    key = (frames - 1) * len(ids) + slot
    _, first = np.unique(key, return_index=True)  # duplicates (same frame, same player): keep the first
    xy = np.full((n, len(ids), 2), np.nan, dtype=np.float32)
    xy[frames[first] - 1, slot[first], 0] = np.array(xs)[first]
    xy[frames[first] - 1, slot[first], 1] = np.array(ys)[first]

    side = np.zeros(len(ids), dtype=np.int8)
    gk = np.zeros(len(ids), dtype=bool)
    jersey = np.full(len(ids), -1, dtype=np.int16)
    mismatch = 0
    sides_a = np.array([{"left": 0, "right": 1}.get(s, -1) for s in sides], dtype=np.int8)
    gks_a = np.array(gks, dtype=bool)
    for i in range(len(ids)):
        mine = slot == i
        counts = Counter(sides_a[mine].tolist())
        side[i] = max((c, s) for s, c in counts.items())[1]
        mismatch += int(mine.sum() - counts[int(side[i])])
        gk[i] = gks_a[mine].mean() > 0.5
        js = Counter(j for j, m in zip(jerseys, mine) if m and j not in (None, ""))
        if js:
            jersey[i] = int(js.most_common(1)[0][0])
    if (side < 0).any():
        raise ValueError("{}: players without a team side".format(path))
    return {"labeled": labeled, "xy": xy, "player_id": ids, "side": side, "gk": gk, "jersey": jersey,
            "objects": len(frames), "duplicates": len(frames) - len(first), "side_mismatch": mismatch}


def read_meta_xml(path: str) -> dict:
    """Teams and players from raw/<id>/<id>_tracker_box_metadata.xml (names are empty in the release
    and are not kept)."""
    root = ET.parse(path).getroot()
    team_side = {t.get("id"): t.get("side") for t in root.iter("team")}
    if sorted(team_side.values()) != ["away", "home"]:
        raise ValueError("{}: expected one home and one away team".format(path))
    players = {p.get("id"): {"team": team_side[p.get("teamId")], "shirt": p.get("shirtNumber"),
                             "position": p.get("position")} for p in root.find("players").iter("player")}
    return {"teams": {side: tid for tid, side in team_side.items()}, "players": players}


def _download(filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, repo_type="dataset", revision=HF_REVISION,
                           local_dir=local_dir,
                           token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def build_half(match_id: str, half: str, root: str = ROOT, keep_json: bool = False) -> dict:
    """Download one game-state half, write compact/<id>_<half>.npz, delete the JSON. Returns stats."""
    t0 = time.time()
    originals = os.path.join(root, "originals")
    ball_path = _download("ball/{}_{}_ball.npz".format(match_id, half), originals)
    with np.load(ball_path) as z:
        if bool(z["y_flipped"]):
            raise ValueError("{} {}: ball y_flipped is set; the loader assumes it isn't".format(
                match_id, half))
        frame = z["frame"]
        ball = np.stack([z["x"], z["y"]], axis=1).astype(np.float32)
        status = z["status"].astype(np.int8)
    n = len(frame)
    if not np.array_equal(frame, np.arange(1, n + 1)):
        raise ValueError("{} {}: ball rows are not frames 1..N".format(match_id, half))
    gsr = _download("gsr/{0}/{0}_{1}.json".format(match_id, half), originals)
    t1 = time.time()
    parsed = parse_gsr(gsr, n)
    t2 = time.time()
    out = os.path.join(compact_dir(root), "{}_{}.npz".format(match_id, half))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    partial = out + ".partial.npz"
    np.savez_compressed(partial, ball=ball, status=status,
                        **{k: parsed[k] for k in ("labeled", "xy", "player_id", "side", "gk", "jersey")})
    os.replace(partial, out)
    if not keep_json:
        os.remove(gsr)
    stats = {k: parsed[k] for k in ("objects", "duplicates", "side_mismatch")}
    stats.update(frames=n, players=len(parsed["player_id"]), download_s=round(t1 - t0, 1),
                 parse_s=round(t2 - t1, 1), mb=round(os.path.getsize(out) / 1e6, 1),
                 full22=float((np.isfinite(parsed["xy"][:, :, 0]).sum(axis=1) == 22).mean()))
    return stats


def build_match_files(match_id: str, root: str = ROOT) -> None:
    """The small per-match files: meta json (from the metadata xml) and the ball-action file."""
    originals = os.path.join(root, "originals")
    meta = read_meta_xml(_download("raw/{0}/{0}_tracker_box_metadata.xml".format(match_id), originals))
    meta.update(format=FORMAT, match_id=match_id, source="{} {} (CC BY 4.0)".format(HF_REPO, HF_REVISION))
    events = _download("bas/{0}/{0}_12_class_events.json".format(match_id), originals)
    paths = _compact_paths(match_id, root)
    os.makedirs(compact_dir(root), exist_ok=True)
    with open(paths["meta"], "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    shutil.copyfile(events, paths["events"])


README = """# SoccerTrack v2, compact copy (src/ingest/soccertrack.py)

Built from Hugging Face `{repo}` release `{rev}` (CC BY 4.0; please cite the SoccerTrack v2 paper).
Atom Scott et al. Players and ball in pitch metres per video frame, float32; ball actions unchanged.
Rebuild: `python -m src.ingest.soccertrack build`. External data only.
""".format(repo=HF_REPO, rev=HF_REVISION)


def _build_job(task):
    match_id, half, root, keep_json = task
    return match_id, half, build_half(match_id, half, root, keep_json)


def build(matches: list[str], root: str = ROOT, jobs: int = 2, keep_json: bool = False) -> list:
    """Build compact files for the given matches. Returns [(match, half, stats)]."""
    for m in matches:
        build_match_files(m, root)
    with open(os.path.join(compact_dir(root), "README.md"), "w") as handle:
        handle.write(README)
    tasks = [(m, h, root, keep_json) for m in matches for h in HALVES]
    if jobs <= 1:
        return [_build_job(t) for t in tasks]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        return pool.map(_build_job, tasks, chunksize=1)


def fetch(root: str = ROOT, jobs: int = 2, push: bool = True) -> str:
    """Compact files from the cache, or built from the originals if the cache has none (then pushed,
    when push and the cache is set). Returns 'cache' or 'built'."""
    from src import hf_cache

    try:
        got = hf_cache.pull(CACHE_NAME, compact_dir(root))
        print("soccertrack: pulled {} files from the cache ({})".format(got, CACHE_NAME))
        return "cache"
    except (FileNotFoundError, hf_cache.Refused) as error:
        print("soccertrack: cache unavailable ({}); building from the originals".format(error))
    build(list(MATCHES), root, jobs)
    if push and os.environ.get(hf_cache.REPO_ENV):
        print("soccertrack: pushed {} files to the cache".format(
            hf_cache.push(compact_dir(root), CACHE_NAME, message="soccertrack compact v{}".format(FORMAT))))
    return "built"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SoccerTrack v2: build or fetch the compact files")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--matches", default="all", help="all, or comma-separated ids")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--keep-json", action="store_true", help="keep the 2.7 GB game-state files")
    p.add_argument("--push", action="store_true", help="upload compact/ to $MTAH_HF_CACHE afterwards")
    p = sub.add_parser("fetch")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        t0 = time.time()
        how = fetch(jobs=args.jobs, push=not args.no_push)
        print("soccertrack: {} matches ready ({}, {:.0f} s)".format(len(list_matches()), how,
                                                                     time.time() - t0))
        return 0
    matches = list(MATCHES) if args.matches == "all" else args.matches.split(",")
    t0 = time.time()
    for m, h, s in build(matches, jobs=args.jobs, keep_json=args.keep_json):
        print("{} {}: {frames} frames, {players} players, all 22 on {full22:.3f}, {duplicates} dup, "
              "{side_mismatch} side mismatches, download {download_s} s, parse {parse_s} s, {mb} MB"
              .format(m, h, **s))
    print("built {} matches in {:.0f} s".format(len(matches), time.time() - t0))
    if args.push:
        from src import hf_cache

        print("pushed {} files to {}".format(hf_cache.push(compact_dir(), CACHE_NAME), CACHE_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
