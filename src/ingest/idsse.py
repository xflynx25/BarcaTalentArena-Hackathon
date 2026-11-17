"""IDSSE / Sportec (DFL Bundesliga 1 & 2, 22/23, TRACAB optical tracking at 25 fps) -> Match with labels.

Files per match, flat under data/raw/external/idsse/ (see docs/external-datasets.md), match id
DFL-MAT-XXXXXX:
  DFL_02_01_matchinformation_<competition>_<match>.xml     teams (Role home/guest), players, pitch size
  DFL_04_03_positions_raw_observed_<competition>_<match>.xml  one FrameSet per (half, person); the ball's
                                                         FrameSet (TeamId "BALL") carries the labels
A match is usable only once both files are complete; list_matches() skips the others (and .part files).

Label mapping (decided in docs/decisions.md, 2026-10-30), per ball frame:
  BallStatus 0 -> D; otherwise BallPossession 1 -> H, 2 -> A.
  Rows outside a half (the half-time gap, see below) -> X, which the official scorer treats as
  ungraded, so each half is scored as its own run, like BAR-ATH.
DFL's BallPossession never says "nobody": it keeps the last team through passes and loose balls,
which is the competition rule. A live frame with any other BallPossession value is an error.

Rows: one per 25 fps frame. Inside a half, rows cover every frame number N from the ball's first to
last frame (a missing N gets NaN positions and the previous frame's label). Between halves we insert
period-0 rows for the real break, measured from the timestamps of the last and first ball frames,
so time-based windows never reach across half-time, as on BAR-ATH.

Coordinates: DFL puts the origin at the centre spot, x along the length; we shift to the schema's
corner origin (x + length/2, y + width/2). Referees are dropped. Player slots: one per person seen
in the tracking, ordered by (team, person id), so the layout is deterministic.

Parsing streams the XML with lxml's iterparse and frees each FrameSet as it ends; a match is ~400 MB
and ~3.5M Frame elements.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import re

import numpy as np
from lxml import etree

from src.official import A, D, H, X
from src.schema import AWAY, HOME, Match

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "raw", "external", "idsse")
FPS = 25
SECTIONS = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")  # period = index + 1
BALL_TEAM, REFEREE_TEAM = "BALL", "referee"
POSSESSION_LABEL = {1: H, 2: A}


def match_key(match_id: str) -> str:
    """'J03WMX', 'DFL-MAT-J03WMX' and 'idsse:DFL-MAT-J03WMX' all -> 'DFL-MAT-J03WMX'."""
    key = str(match_id).split(":")[-1]
    return key if key.startswith("DFL-MAT-") else "DFL-MAT-" + key


def _paths(match_id: str, root: str) -> tuple[str | None, str | None]:
    """(match information, positions) paths, None where absent."""
    key = match_key(match_id)
    found = []
    for kind in ("DFL_02_01_matchinformation", "DFL_04_03_positions_raw_observed"):
        hits = sorted(glob.glob(os.path.join(root, "{}_*_{}.xml".format(kind, key))))
        found.append(hits[0] if hits else None)
    return found[0], found[1]


def list_matches(root: str = ROOT) -> list[str]:
    """Full ids (DFL-MAT-...) of matches whose match information and positions are present, sorted."""
    if not os.path.isdir(root):
        return []
    keys = {m.group(1) for name in os.listdir(root)
            for m in [re.fullmatch(r"DFL_04_03_positions_raw_observed_.+_(DFL-MAT-\w+)\.xml", name)] if m}
    return sorted(k for k in keys if all(_paths(k, root)))


def read_meta(meta_path: str) -> dict:
    """{pitch (length, width), player_team {person id: HOME/AWAY}, team_names (home, away)}."""
    info = etree.parse(meta_path).getroot().find("MatchInformation")
    env = info.find("Environment")
    side = {"home": HOME, "guest": AWAY}
    player_team, names = {}, {}
    for team in info.iterfind("Teams/Team"):
        role = side[team.get("Role")]
        names[role] = team.get("TeamName")
        for player in team.iterfind("Players/Player"):
            player_team[player.get("PersonId")] = role
    if set(names) != {HOME, AWAY}:
        raise ValueError("{}: expected one home and one guest team".format(meta_path))
    return {
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
        "player_team": player_team,
        "team_names": (names[HOME], names[AWAY]),
    }


def _timestamp(text: str) -> float:
    return dt.datetime.fromisoformat(text).timestamp()


def read_positions(raw_path: str) -> dict:
    """Stream the positions XML. {section name: {"ball": {N, xy, possession, status, t_first, t_last},
    "players": {person id: (team id, N, xy)}}}. Centre-origin metres, DFL frame numbers."""
    sections: dict = {}
    current = None  # (section, team, person) of the open FrameSet
    n, xy, possession, status, t_first, t_last = [], [], [], [], None, None
    for event, elem in etree.iterparse(raw_path, events=("start", "end"), tag=("FrameSet", "Frame")):
        if elem.tag == "Frame":
            if event == "end":
                n.append(int(elem.get("N")))
                xy.append((float(elem.get("X")), float(elem.get("Y"))))
                if current[1] == BALL_TEAM:
                    possession.append(int(elem.get("BallPossession", 0)))
                    status.append(int(elem.get("BallStatus", 0)))
                    t_last = elem.get("T")
                    if t_first is None:
                        t_first = t_last
                elem.clear()
            continue
        if event == "start":
            current = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
            n, xy, possession, status, t_first, t_last = [], [], [], [], None, None
            continue
        section, team, person = current
        if section not in SECTIONS:
            raise ValueError("{}: unknown GameSection {!r}".format(raw_path, section))
        store = sections.setdefault(section, {"ball": None, "players": {}})
        frames = np.asarray(n, dtype=np.int64)
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if team == BALL_TEAM:
            if store["ball"] is not None:
                raise ValueError("{}: two ball FrameSets in {}".format(raw_path, section))
            store["ball"] = {"N": frames, "xy": points,
                             "possession": np.asarray(possession, dtype=np.int8),
                             "status": np.asarray(status, dtype=np.int8),
                             "t_first": t_first, "t_last": t_last}
        elif team != REFEREE_TEAM:
            store["players"][person] = (team, frames, points)
        elem.clear()
        while elem.getprevious() is not None:  # drop finished FrameSets from the tree
            del elem.getparent()[0]
    return sections


def ball_labels(possession: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Per ball frame: BallStatus 0 -> D, else BallPossession 1 -> H, 2 -> A."""
    labels = np.full(len(status), D, dtype=np.int8)
    live = status != 0
    bad = live & ~np.isin(possession, list(POSSESSION_LABEL))
    if bad.any():
        raise ValueError("{} live ball frames have BallPossession other than 1/2".format(int(bad.sum())))
    labels[live & (possession == 1)] = H
    labels[live & (possession == 2)] = A
    return labels


def load(match_id: str, root: str = ROOT) -> Match:
    """One IDSSE match as a Match with per-frame labels (H/A/D codes, X in the half-time gap)."""
    key = match_key(match_id)
    meta_path, raw_path = _paths(key, root)
    if meta_path is None or raw_path is None:
        raise FileNotFoundError("{}: match information or positions missing under {}; run "
                                "scripts/fetch_external_data.sh idsse".format(key, root))
    meta = read_meta(meta_path)
    sections = read_positions(raw_path)
    length, width = meta["pitch"]

    team_ids = {team for s in sections.values() for team, _, _ in s["players"].values()}
    people = {p for s in sections.values() for p in s["players"]}
    unknown = people - set(meta["player_team"])
    if unknown:
        raise ValueError("{}: {} tracked persons are not in the match information".format(key, len(unknown)))
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams in the tracking, found {}".format(key, len(team_ids)))
    slot_ids = sorted(people, key=lambda p: (meta["player_team"][p], p))
    slot_of = {person: slot for slot, person in enumerate(slot_ids)}

    # Row layout: each half's frame range, separated by the real break in period-0 rows.
    blocks, row, previous_end = [], 0, None
    for period, section in enumerate(SECTIONS, start=1):
        if section not in sections:
            continue
        ball = sections[section]["ball"]
        if ball is None or len(ball["N"]) == 0:
            raise ValueError("{}: no ball frames in {}".format(key, section))
        if previous_end is not None:
            gap_s = _timestamp(ball["t_first"]) - previous_end
            row += max(1, int(round(gap_s * FPS)) - 1)
        first, last = int(ball["N"].min()), int(ball["N"].max())
        blocks.append((period, section, first, row))
        row += last - first + 1
        previous_end = _timestamp(ball["t_last"])

    n = row
    period_arr = np.zeros(n, dtype=np.int8)
    labels = np.full(n, X, dtype=np.int8)
    ball_xy = np.full((n, 2), np.nan)
    players = np.full((n, len(slot_ids), 2), np.nan)
    for period, section, first, start in blocks:
        ball = sections[section]["ball"]
        size = int(ball["N"].max()) - first + 1
        period_arr[start:start + size] = period
        rows = start + ball["N"] - first
        ball_xy[rows] = ball["xy"]
        # forward-fill labels over any missing frame numbers inside the half
        frame_labels = np.full(size, -1, dtype=np.int8)
        frame_labels[ball["N"] - first] = ball_labels(ball["possession"], ball["status"])
        filled = np.maximum.accumulate(np.where(frame_labels >= 0, np.arange(size), 0))
        labels[start:start + size] = frame_labels[filled]
        for person, (_, frames, points) in sections[section]["players"].items():
            inside = (frames >= first) & (frames < first + size)
            players[start + frames[inside] - first, slot_of[person]] = points[inside]

    offset = np.array([length / 2, width / 2])
    return Match(
        match_id="idsse:{}".format(key),
        fps=FPS,
        pitch=(length, width),
        period=period_arr,
        ball=ball_xy + offset,
        players=players + offset,
        slot_team=np.array([meta["player_team"][p] for p in slot_ids], dtype=np.int8),
        labels=labels,
    ).validate()
