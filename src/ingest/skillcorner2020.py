"""SkillCorner Open Data 2019/20 (backlog A15): 9 top-league matches, broadcast tracking at 10 fps.

SkillCorner's first open release (MIT), deleted from their repo in Oct 2025 and fetched from its
history, pinned to commit 1b99d401dd (scripts/fetch_external_data.sh skillcorner2020). Its own
source, separate from `skillcorner` (A-League 24/25): an older tracking version with no
extrapolation, so off-screen players and the ball are simply missing.

Files per match, under data/raw/external/skillcorner2020/data/matches/<id>/:
  match_data.json       teams, players and referees (by trackable_object), ball id, pitch size
  structured_data.json  one JSON list (~56 MB), one entry per frame: frame, period (1, 2 or null),
                        time, possession {trackable_object, group}, data [{trackable_object or
                        group_name, track_id, x, y[, z]}]
The whole file is one JSON array, so it is read in one go (~1 GB peak per match); the match cache
in src.evaluate makes that a one-off.

Labels: as A1 (src/ingest/skillcorner.py, decisions 2026-10-30): possession.group "home team" -> H,
"away team" -> A, null -> D; frames with period null (before kick-off, half-time, after the end)
-> X. Same options as `skillcorner`: cuts="ungraded" sets broadcast cuts (in-period runs of frames
with no object at all, at least min_cut_s long) to X (A7); onsets="out" moves D onsets to the ball
seen off the pitch (B20). The ball here is only where it is detected, so "out" can only fire when
the broadcast shows it.

Player slots: identified objects (trackable_object in match_data's players) get one slot each,
ordered by (team, trackable_object). Unidentified detections (only a group_name: "home team", "away
team" or that team's goalkeeper; ~1% of player detections, a few hundred short tracks per match) are packed into a few extra
slots per team: a track keeps its slot while it lasts, and a new track takes the lowest free slot of
its team (deterministic, in frame then track_id order). Slots are tracklets, not people (src/schema).
anonymous=False drops them. Referees (identified or group_name "referee") are dropped.

Coordinates: centre origin in metres; shifted to the schema's corner origin.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.ingest.skillcorner import GROUP_LABEL, cut_mask, out_onsets
from src.official import X
from src.schema import AWAY, HOME, Match

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "raw", "external", "skillcorner2020")
FPS = 10
TEAM_GROUP = {"home team": HOME, "away team": AWAY, "home goalkeeper": HOME, "away goalkeeper": AWAY}
OTHER_GROUPS = {"referee"}
BALL_GROUPS = {"ball", "balls"}


def _paths(match_id: str, root: str) -> tuple[str, str]:
    folder = os.path.join(root, "data", "matches", str(match_id))
    return (os.path.join(folder, "match_data.json"), os.path.join(folder, "structured_data.json"))


def list_matches(root: str = ROOT) -> list[str]:
    """Ids of matches whose tracking file is present, sorted."""
    folder = os.path.join(root, "data", "matches")
    if not os.path.isdir(folder):
        return []
    return sorted(m for m in os.listdir(folder) if os.path.isfile(_paths(m, root)[1]))


def read_meta(meta_path: str) -> dict:
    """{pitch, player_team {trackable_object: HOME/AWAY}, ball (trackable_object), team_names}."""
    with open(meta_path) as handle:
        meta = json.load(handle)
    team_side = {meta["home_team"]["id"]: HOME, meta["away_team"]["id"]: AWAY}
    return {
        "pitch": (float(meta["pitch_length"]), float(meta["pitch_width"])),
        "player_team": {p["trackable_object"]: team_side[p["team_id"]] for p in meta["players"]
                        if p.get("team_id") in team_side},
        "ball": meta["ball"]["trackable_object"],
        "referees": {r["trackable_object"] for r in meta.get("referees") or ()},
        "team_names": (meta["home_team"]["name"], meta["away_team"]["name"]),
    }


def _pack_anonymous(rows, teams, tracks) -> tuple[np.ndarray, int, int]:
    """Slot index (within its team's anonymous slots) per anonymous detection, and the number of
    anonymous slots per team (home, away). Inputs are in frame order."""
    out = np.zeros(len(rows), dtype=np.int64)
    owner = {HOME: [], AWAY: []}  # slot -> track id or None
    last_seen = {}  # (team, track) -> frame row it was last seen
    slot_of = {}
    for j, (row, team, track) in enumerate(zip(rows, teams, tracks)):
        key = (team, track)
        if key not in slot_of:
            slots = owner[team]
            free = [s for s, t in enumerate(slots)
                    if t is None or last_seen[(team, t)] < row]
            slot = free[0] if free else len(slots)
            if slot == len(slots):
                slots.append(None)
            elif slots[slot] is not None:
                del slot_of[(team, slots[slot])]
            slots[slot] = track
            slot_of[key] = slot
        out[j] = slot_of[key]
        last_seen[key] = row
    return out, len(owner[HOME]), len(owner[AWAY])


def read_tracking(raw_path: str, meta: dict, anonymous: bool = True) -> dict:
    """{frame, period, ball (n, 2), labels, tracked (any object seen), player_rows/keys/xy}, where a
    player key is ("id", trackable_object) or ("anon", team, track_id). Centre-origin metres."""
    with open(raw_path) as handle:
        records = json.load(handle)
    n = len(records)
    frame = np.empty(n, dtype=np.int64)
    period = np.zeros(n, dtype=np.int8)
    labels = np.empty(n, dtype=np.int8)
    ball = np.full((n, 2), np.nan)
    tracked = np.zeros(n, dtype=bool)
    rows, keys, xy = [], [], []
    for i, record in enumerate(records):
        frame[i] = record["frame"]
        period[i] = record["period"] or 0
        group = (record.get("possession") or {}).get("group")
        if group not in GROUP_LABEL:
            raise ValueError("{}: unknown possession group at frame {}".format(raw_path, frame[i]))
        labels[i] = GROUP_LABEL[group] if record["period"] else X
        objects = record.get("data") or ()
        tracked[i] = bool(objects)
        for obj in objects:
            obj_id = obj.get("trackable_object")
            if obj_id == meta["ball"]:
                ball[i] = (obj["x"], obj["y"])
            elif obj_id is None and obj.get("group_name") in BALL_GROUPS:
                if np.isnan(ball[i, 0]):  # unidentified ball (18 frames in one match)
                    ball[i] = (obj["x"], obj["y"])
            elif obj_id in meta["player_team"]:
                rows.append(i)
                keys.append(("id", obj_id))
                xy.append((obj["x"], obj["y"]))
            elif obj_id is None and obj.get("group_name") in TEAM_GROUP:
                if anonymous:
                    rows.append(i)
                    keys.append(("anon", TEAM_GROUP[obj["group_name"]], obj["track_id"]))
                    xy.append((obj["x"], obj["y"]))
            elif obj_id not in meta["referees"] and not (
                    obj_id is None and obj.get("group_name") in OTHER_GROUPS):
                raise ValueError("{}: unknown object {!r} at frame {}".format(
                    raw_path, obj_id if obj_id is not None else obj.get("group_name"), frame[i]))
    if n == 0:
        raise ValueError("{}: no frames".format(raw_path))
    if not np.array_equal(frame, np.arange(frame[0], frame[0] + n)):
        raise ValueError("{}: frame numbers are not consecutive".format(raw_path))
    return {"frame": frame, "period": period, "ball": ball, "labels": labels, "tracked": tracked,
            "player_rows": np.asarray(rows, dtype=np.int64), "player_keys": keys,
            "player_xy": np.asarray(xy, dtype=np.float64).reshape(-1, 2)}


def load(match_id: str | int, root: str = ROOT, cuts: str = "keep", min_cut_s: float = 1.0,
         onsets: str = "label", onset_s: float = 3.0, anonymous: bool = True) -> Match:
    """One 2019/20 SkillCorner match as a Match with per-frame labels (H/A/D, X outside periods).
    Options as src/ingest/skillcorner.load; anonymous=False drops unidentified player detections."""
    if cuts not in ("keep", "ungraded"):
        raise ValueError("cuts must be 'keep' or 'ungraded', not {!r}".format(cuts))
    if onsets not in ("label", "out"):
        raise ValueError("onsets must be 'label' or 'out', not {!r}".format(onsets))
    meta_path, raw_path = _paths(str(match_id), root)
    for path in (meta_path, raw_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing; run scripts/fetch_external_data.sh skillcorner2020"
                                    .format(path))
    meta = read_meta(meta_path)
    raw = read_tracking(raw_path, meta, anonymous=bool(anonymous))
    length, width = meta["pitch"]

    keys = raw["player_keys"]
    ids = sorted({k[1] for k in keys if k[0] == "id"}, key=lambda i: (meta["player_team"][i], i))
    anon = [j for j, k in enumerate(keys) if k[0] == "anon"]
    anon_slot, n_home, n_away = _pack_anonymous(
        raw["player_rows"][anon].tolist(), [keys[j][1] for j in anon], [keys[j][2] for j in anon])
    # layout: identified home, anonymous home, identified away, anonymous away
    home_ids = [i for i in ids if meta["player_team"][i] == HOME]
    away_ids = [i for i in ids if meta["player_team"][i] == AWAY]
    base = {HOME: len(home_ids), AWAY: len(home_ids) + n_home + len(away_ids)}
    slot_of = {i: s for s, i in enumerate(home_ids)}
    slot_of.update({i: len(home_ids) + n_home + s for s, i in enumerate(away_ids)})
    slot_team = np.array([HOME] * (len(home_ids) + n_home) + [AWAY] * (len(away_ids) + n_away),
                         dtype=np.int8)

    slots = np.empty(len(keys), dtype=np.int64)
    for j, k in enumerate(keys):
        if k[0] == "id":
            slots[j] = slot_of[k[1]]
    if anon:
        slots[anon] = [base[keys[j][1]] + s for j, s in zip(anon, anon_slot)]

    n = len(raw["frame"])
    players = np.full((n, len(slot_team), 2), np.nan)
    players[raw["player_rows"], slots] = raw["player_xy"]

    offset = np.array([length / 2, width / 2])
    labels = raw["labels"]
    if onsets == "out":
        labels = out_onsets(labels, raw["ball"] + offset, (length, width), raw["period"],
                            round(float(onset_s) * FPS))
    if cuts == "ungraded":
        labels = labels.copy()
        labels[cut_mask(raw["tracked"], raw["period"], round(float(min_cut_s) * FPS))] = X

    return Match(
        match_id="skillcorner2020:{}".format(match_id),
        fps=FPS,
        pitch=(length, width),
        period=raw["period"],
        ball=raw["ball"] + offset,
        players=players + offset,
        slot_team=slot_team,
        labels=labels,
    ).validate()
