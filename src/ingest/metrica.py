"""Metrica Sports sample data (3 anonymised matches, optical tracking at 25 fps) -> Match with labels.

Files, under data/raw/external/metrica/data/ (see docs/external-datasets.md):
  Sample_Game_1, Sample_Game_2   CSV: RawTrackingData_{Home,Away}_Team.csv + RawEventsData.csv
  Sample_Game_3                  EPTS: _metadata.xml + _tracking.txt, events in _events.json
Match ids are "1", "2", "3"; list_matches() gives the ones whose files are present.

There is no possession column, so the truth is derived from the event data (decided in
docs/decisions.md, 2026-10-30). derive_labels() does it in two steps:

  Dead ball. Every restart (a SET PIECE: kick-off, throw-in, free kick, corner; or anything whose
  subtype says GOAL KICK, which the CSV games log as a PASS) means the ball was dead just before it:
  from the last frame any earlier event in that period still covers (max of start and end frame;
  CARD is skipped because cards are given while play is stopped) up to the frame before the restart.
  Frames of a period before its first restart, or after its last event, are dead too. So stoppages
  are detected by what follows them (ball out, foul, offside, goal all end in a restart), and a foul
  played on with advantage, which has no restart, stays live. Stoppages without a restart event
  (drop balls, injuries) are missed and stay live.
  Possession. Live frames take the team of the last on-ball event that started at or before them:
  PASS, CARRY, SET PIECE, RECOVERY, SHOT, BALL LOST, and the restarts. CHALLENGE, BALL OUT, FAULT
  RECEIVED and CARD do not change possession. So a pass keeps its team while the ball travels, and an
  interception switches at the RECOVERY start frame: the competition's "loose ball stays with the
  last team" rule. Frames outside a period -> X, like SkillCorner.

Frame rate 25 fps, one row per tracking frame. Periods: the tracking's Period column (CSV games) or
the metadata's half start/end frames (game 3). Both cover every frame, so the gap before the second
half's kick-off is in period 2 and labelled D, as on BAR-ATH.

Coordinates: normalised 0-1 with the origin at a corner, scaled by the pitch size. Game 3's metadata
says 105 x 68 m. Games 1-2 give no size; we assume the same 105 x 68 (PITCH_DEFAULT).

Player slots: one per player, home first. Game 3's tracking has 22 columns whose player changes at
substitutions (one DataFormatSpecification per frame range in the metadata); each column is moved
to its player's slot range by range.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.ingest import epts
from src.official import A, D, H, X
from src.schema import AWAY, HOME, Match

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "raw", "external", "metrica")
FPS = 25
PITCH_DEFAULT = (105.0, 68.0)
GAMES = ("1", "2", "3")

POSSESSION_TYPES = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}
NOT_PLAY_TYPES = {"CARD"}  # given during stoppages; must not extend live time


def _folder(match_id: str, root: str) -> str:
    return os.path.join(root, "data", "Sample_Game_{}".format(match_id))


def _paths(match_id: str, root: str) -> dict:
    stem = os.path.join(_folder(match_id, root), "Sample_Game_{}_".format(match_id))
    if match_id == "3":
        return {"meta": stem + "metadata.xml", "raw": stem + "tracking.txt",
                "events": stem + "events.json"}
    return {"home": stem + "RawTrackingData_Home_Team.csv",
            "away": stem + "RawTrackingData_Away_Team.csv", "events": stem + "RawEventsData.csv"}


def list_matches(root: str = ROOT) -> list[str]:
    """Ids of games whose files are all present."""
    return [g for g in GAMES if all(os.path.isfile(p) for p in _paths(g, root).values())]


# -- events -----------------------------------------------------------------------------------------

def is_restart(kind: str, subtype: str) -> bool:
    return kind == "SET PIECE" or "GOAL KICK" in subtype


def read_events_csv(path: str) -> pd.DataFrame:
    """Games 1-2. Columns: team (HOME/AWAY/-1), type, subtype ('' if none), period, start, end."""
    raw = pd.read_csv(path)
    side = {"Home": HOME, "Away": AWAY}
    return pd.DataFrame({
        "team": raw["Team"].map(side).fillna(-1).astype(np.int8),
        "type": raw["Type"].astype(str),
        "subtype": raw["Subtype"].fillna("").astype(str),
        "period": raw["Period"].astype(np.int8),
        "start": raw["Start Frame"].astype(np.int64),
        "end": raw["End Frame"].fillna(0).astype(np.int64),
    })


def _subtype_name(subtypes) -> str:
    if not subtypes:
        return ""
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return "-".join(s["name"] for s in subtypes)


def read_events_json(path: str, home_team_id: str) -> pd.DataFrame:
    """Game 3 (Metrica JSON events), same columns as read_events_csv."""
    with open(path) as handle:
        records = json.load(handle)["data"]
    rows = []
    for e in records:
        team = (e.get("team") or {}).get("id")
        rows.append({
            "team": -1 if team is None else (HOME if team == home_team_id else AWAY),
            "type": e["type"]["name"],
            "subtype": _subtype_name(e.get("subtypes")),
            "period": e["period"],
            "start": (e.get("start") or {}).get("frame") or 0,
            "end": (e.get("end") or {}).get("frame") or 0,
        })
    return pd.DataFrame(rows).astype({"team": np.int8, "period": np.int8,
                                      "start": np.int64, "end": np.int64})


def derive_labels(events: pd.DataFrame, frame: np.ndarray, period: np.ndarray) -> np.ndarray:
    """(n,) label codes from events, for tracking rows with frame numbers `frame` (consecutive) and
    `period` (0 outside play). See the module docstring for the rules."""
    labels = np.full(len(frame), X, dtype=np.int8)
    for half in sorted(set(np.unique(period).tolist()) - {0}):
        rows = np.flatnonzero(period == half)
        lo, hi = int(frame[rows[0]]), int(frame[rows[-1]])  # frame range of the period
        ev = events[(events["period"] == half) & (events["start"] >= lo) & (events["start"] <= hi)]
        ev = ev.sort_values("start", kind="stable")

        team = np.full(hi - lo + 1, -1, dtype=np.int8)  # indexed by frame - lo
        restart = np.array([is_restart(t, s) for t, s in zip(ev["type"], ev["subtype"])], bool)
        owners = ev[ev["type"].isin(POSSESSION_TYPES).values | restart]
        owners = owners[owners["team"] >= 0]
        for start, side in zip(owners["start"], owners["team"]):
            team[start - lo:] = side  # later events overwrite: forward fill
        if len(owners):
            team[team < 0] = owners["team"].iloc[0]  # before the first event (dead anyway)
        out = np.where(team == HOME, H, A).astype(np.int8)

        # covered[i]: last frame that play events 0..i reach. Only events that start strictly
        # before a restart count for it: the RECOVERY or PASS logged on the restart's own frame
        # (often listed before the SET PIECE) must not close the gap.
        play = ~ev["type"].isin(NOT_PLAY_TYPES).values
        starts = ev["start"].values[play]
        covered = np.maximum.accumulate(np.maximum(starts, np.minimum(ev["end"].values[play], hi))) \
            if play.any() else np.zeros(0, dtype=np.int64)
        dead = np.zeros(hi - lo + 1, dtype=bool)
        for start in ev["start"].values[restart]:
            before = np.searchsorted(starts, start, side="left")
            last_live = covered[before - 1] if before else lo - 1
            dead[last_live + 1 - lo:start - lo] = True
        dead[(covered[-1] if len(covered) else lo - 1) + 1 - lo:] = True
        if not len(owners):
            dead[:] = True
        out[dead] = D
        labels[rows] = out[frame[rows] - lo]
    if (labels[period > 0] == X).any():
        raise ValueError("derive_labels left in-period frames unlabelled")
    return labels


# -- tracking ---------------------------------------------------------------------------------------

def read_tracking_csv(home_path: str, away_path: str) -> dict:
    """Games 1-2: {frame, period, ball (n, 2), players (n, k, 2), slot_team (k,)}, normalised 0-1."""
    tables, slot_team = [], []
    for path, side in ((home_path, HOME), (away_path, AWAY)):
        table = pd.read_csv(path, skiprows=2)
        names = list(table.columns)
        x_cols = [i for i, c in enumerate(names) if c.startswith("Player")]
        if names[-2] != "Ball" or any(i + 1 >= len(names) or not names[i + 1].startswith("Unnamed")
                                      for i in x_cols):
            raise ValueError("{}: unexpected column layout".format(path))
        tables.append((table, x_cols))
        slot_team += [side] * len(x_cols)
    (home, home_x), (away, away_x) = tables
    if not (np.array_equal(home["Frame"].values, away["Frame"].values)
            and np.array_equal(home["Period"].values, away["Period"].values)):
        raise ValueError("home and away tracking files do not share frames")
    frame = home["Frame"].values.astype(np.int64)
    if not np.array_equal(frame, np.arange(frame[0], frame[0] + len(frame))):
        raise ValueError("{}: frame numbers are not consecutive".format(home_path))
    players = np.concatenate([
        np.stack([t.iloc[:, [i, i + 1]].values.astype(np.float64) for i in xs], axis=1)
        for t, xs in tables], axis=1)
    return {
        "frame": frame,
        "period": np.nan_to_num(home["Period"].values.astype(np.float64)).astype(np.int8),
        "ball": home.iloc[:, -2:].values.astype(np.float64),
        "players": players,
        "slot_team": np.array(slot_team, dtype=np.int8),
    }


def read_metadata_epts(meta_path: str) -> dict:
    """Game 3: {home_team_id, pitch, halves [(period, first, last)], player_team {id: side},
    specs [(first, last, [player id per column])]}."""
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 <Team> elements, found {}".format(meta_path, len(teams)))
    player_team = {p.get("id"): HOME if p.get("teamId") == teams[0] else AWAY
                   for p in root.iter("Player")}

    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    halves = []
    for half, name in enumerate(("first_half", "second_half", "first_extra_half",
                                 "second_extra_half"), start=1):
        if params.get(name + "_start") and params.get(name + "_end"):
            halves.append((half, int(params[name + "_start"]), int(params[name + "_end"])))

    size = root.find(".//FieldSize")
    pitch = (float(size.findtext("Width")), float(size.findtext("Height"))) if size is not None \
        else PITCH_DEFAULT

    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [channel[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")]
        if refs[0::2] != refs[1::2]:
            raise ValueError("{}: a player group is not one player's x,y".format(meta_path))
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), refs[0::2]))
    if not specs or len({len(s[2]) for s in specs}) != 1:
        raise ValueError("{}: data format specifications differ in column count".format(meta_path))
    return {"home_team_id": teams[0], "pitch": pitch, "halves": halves,
            "player_team": player_team, "specs": specs}


def read_tracking_epts(meta: dict, raw_path: str) -> dict:
    """Game 3, same keys as read_tracking_csv. Columns are moved to per-player slots per spec."""
    raw = epts.read_raw(raw_path, len(meta["specs"][0][2]))
    frame = raw["frame_id"]
    if not np.array_equal(frame, np.arange(frame[0], frame[0] + len(frame))):
        raise ValueError("{}: frame numbers are not consecutive".format(raw_path))

    seen = {pid for _, _, pids in meta["specs"] for pid in pids}
    order = {pid: i for i, pid in enumerate(meta["player_team"])}  # metadata order within a team
    slot_ids = sorted(seen, key=lambda pid: (meta["player_team"][pid], order[pid]))
    slot_of = {pid: i for i, pid in enumerate(slot_ids)}

    players = np.full((len(frame), len(slot_ids), 2), np.nan)
    covered = np.zeros(len(frame), dtype=bool)
    for first, last, pids in meta["specs"]:
        rows = np.flatnonzero((frame >= first) & (frame <= last))
        covered[rows] = True
        players[rows[:, None], [slot_of[p] for p in pids]] = raw["players"][rows]
    if not covered.all():
        raise ValueError("{}: some frames are in no data format specification".format(raw_path))

    period = np.zeros(len(frame), dtype=np.int8)
    for half, first, last in meta["halves"]:
        period[(frame >= first) & (frame <= last)] = half
    return {
        "frame": frame,
        "period": period,
        "ball": raw["ball"],
        "players": players,
        "slot_team": np.array([meta["player_team"][p] for p in slot_ids], dtype=np.int8),
    }


def load(match_id: str | int, root: str = ROOT) -> Match:
    """One Metrica sample game as a Match with event-derived labels (H/A/D, X outside periods)."""
    match_id = str(match_id)
    if match_id not in GAMES:
        raise ValueError("unknown Metrica game {!r}; have {}".format(match_id, GAMES))
    paths = _paths(match_id, root)
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing; run scripts/fetch_external_data.sh metrica"
                                    .format(path))
    if match_id == "3":
        meta = read_metadata_epts(paths["meta"])
        track = read_tracking_epts(meta, paths["raw"])
        events = read_events_json(paths["events"], meta["home_team_id"])
        pitch = meta["pitch"]
    else:
        track = read_tracking_csv(paths["home"], paths["away"])
        events = read_events_csv(paths["events"])
        pitch = PITCH_DEFAULT

    scale = np.array(pitch)
    return Match(
        match_id="metrica:{}".format(match_id),
        fps=FPS,
        pitch=pitch,
        period=track["period"],
        ball=track["ball"] * scale,
        players=track["players"] * scale,
        slot_team=track["slot_team"],
        labels=derive_labels(events, track["frame"], track["period"]),
    ).validate()
