"""SkillCorner Open Data (A-League 24/25, broadcast tracking at 10 fps) -> Match with labels.

Files per match, under data/raw/external/skillcorner/data/matches/<id>/ (see docs/external-datasets.md):
  <id>_match.json                    teams, players, pitch size
  <id>_tracking_extrapolated.jsonl   one frame per line: frame, period, ball_data, possession,
                                     player_data[{player_id, x, y, is_detected}]
A match is usable only once its tracking file exists; list_matches() skips the others.

Label mapping (decided in docs/decisions.md, 2026-10-30):
  possession.group "home team" -> H, "away team" -> A, null -> D.
  Frames outside a period (before kick-off, half-time) -> X, which the official scorer treats as
  ungraded (possession_metrics.valid_runs), so each half is scored as its own run, like BAR-ATH.
  Why null means dead ball: on matches 1874553 and 1886347, about 95% of in-period null frames fall
  outside SkillCorner's own phases-of-play file (which covers live play only), and null runs are
  long (median ~17 s) like stoppages. While the ball travels, SkillCorner keeps the group and drops
  only the player id. That is the competition's rule too: loose balls stay with the last team.
  Frames with no tracking at all (broadcast replays and close-ups, ~25% of a match) keep whatever
  the group says, which is almost always null, so D.

Frame rate: kept at the native 10 fps (Match.fps = 10, one row per source frame). BAR-ATH is 25 fps,
so models must size windows from match.fps (seconds, not frames), or resample with
src.resample.to_fps. Scoring at 10 fps is fine: spellF1 is a temporal IoU, so it does not depend on
the frame rate beyond 0.1 s boundary rounding.

Broadcast cuts (findings F3): frames with no ball and no player at all, i.e. replays and close-ups.
They are ~26% of in-period frames and almost all D, a shortcut BAR-ATH's fixed camera doesn't have.
cuts="keep" (default) leaves their labels alone; cuts="ungraded" sets them to X, so the official
scorer skips them and scores each stretch between cuts as its own run (backlog A7). Score both ways:
src.evaluate --load cuts=ungraded. Cuts shorter than min_cut_s (default 1 s) stay graded, so a few
dropped frames in live play don't split a possession.

Dead-ball onsets (B20, findings F76, F84): SkillCorner starts D ~1.5 s before its ball is seen leaving
the pitch, the tactical sources (and, by the docs, BAR-ATH) at the line. onsets="out" moves each D
onset that follows a team label to the first frame the ball is seen off the pitch (more than
OUT_MARGIN_M) when that comes within onset_s after it; the frames in between keep the team label.
Label-side only: the tracking is unchanged. Default onsets="label" keeps SkillCorner's own timing.

Coordinates: SkillCorner puts the origin at the centre spot; we shift to the schema's corner
origin (x + length/2, y + width/2). Off-screen players and the ball are extrapolated by SkillCorner
(is_detected false). They are kept by default; pass extrapolated=False to set them to NaN.

Player slots: one per player id seen in the tracking, ordered by (team, player id), so the layout is
deterministic. The jsonl is streamed line by line; a match is ~90 MB.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.official import A, D, H, X
from src.schema import AWAY, HOME, Match

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "raw", "external", "skillcorner")
GROUP_LABEL = {"home team": H, "away team": A, None: D}
OUT_MARGIN_M = 0.5  # as src/models/deadball.py: tracking noise around the lines


def out_onsets(labels: np.ndarray, ball: np.ndarray, pitch, period: np.ndarray,
               within: int) -> np.ndarray:
    """Copy of labels where each D run inside a period that follows H or A starts instead at the
    first frame in its first `within` frames where the ball is seen off the pitch (more than
    OUT_MARGIN_M), if the ball isn't already off at the onset; frames moved keep the team label."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
               | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))
    labels = labels.copy()
    dead = (labels == D) & (period > 0)
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or labels[start - 1] not in (H, A) or period[start - 1] != period[start]:
            continue
        if out[start]:
            continue
        hits = np.flatnonzero(out[start:min(stop, start + within + 1)])
        if len(hits):
            labels[start:start + hits[0]] = labels[start - 1]
    return labels


def _paths(match_id: str, root: str) -> tuple[str, str]:
    folder = os.path.join(root, "data", "matches", str(match_id))
    return (os.path.join(folder, "{}_match.json".format(match_id)),
            os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)))


def list_matches(root: str = ROOT) -> list[str]:
    """Ids of matches whose tracking file is present, sorted."""
    folder = os.path.join(root, "data", "matches")
    if not os.path.isdir(folder):
        return []
    return sorted(m for m in os.listdir(folder) if os.path.isfile(_paths(m, root)[1]))


def read_meta(meta_path: str) -> dict:
    """{pitch (length, width), player_team {player id: HOME/AWAY}, team_names}."""
    with open(meta_path) as handle:
        meta = json.load(handle)
    team_side = {meta["home_team"]["id"]: HOME, meta["away_team"]["id"]: AWAY}
    player_team = {p["id"]: team_side[p["team_id"]] for p in meta["players"]
                   if p.get("team_id") in team_side}
    return {
        "pitch": (float(meta["pitch_length"]), float(meta["pitch_width"])),
        "player_team": player_team,
        "team_names": (meta["home_team"]["name"], meta["away_team"]["name"]),
    }


def read_tracking(raw_path: str, extrapolated: bool = True) -> dict:
    """Stream the jsonl. {frame (n,), period (n,), ball (n, 2), labels (n,), tracked (n,): the source
    has a ball or any player position (extrapolated or not), False on broadcast cuts,
    player_rows/player_ids/player_xy: one entry per (frame, player) seen}. Centre-origin metres."""
    frame, period, ball, labels, tracked = [], [], [], [], []
    rows, ids, xy = [], [], []
    with open(raw_path) as handle:
        for row, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            frame.append(record["frame"])
            period.append(record["period"] or 0)
            group = (record.get("possession") or {}).get("group")
            if group not in GROUP_LABEL:
                raise ValueError("{}: unknown possession group at line {}".format(raw_path, row + 1))
            labels.append(GROUP_LABEL[group] if record["period"] else X)

            b = record.get("ball_data") or {}
            seen = b.get("x") is not None and b.get("y") is not None
            if seen and not extrapolated and not b.get("is_detected"):
                seen = False
            ball.append((b["x"], b["y"]) if seen else (np.nan, np.nan))
            players = record.get("player_data") or ()
            tracked.append(b.get("x") is not None or any(p.get("x") is not None for p in players))

            for p in players:
                if p.get("x") is None or (not extrapolated and not p.get("is_detected")):
                    continue
                rows.append(len(frame) - 1)
                ids.append(p["player_id"])
                xy.append((p["x"], p["y"]))

    frame = np.asarray(frame, dtype=np.int64)
    if len(frame) == 0:
        raise ValueError("{}: no frames".format(raw_path))
    if not np.array_equal(frame, np.arange(frame[0], frame[0] + len(frame))):
        raise ValueError("{}: frame numbers are not consecutive".format(raw_path))
    return {
        "frame": frame,
        "period": np.asarray(period, dtype=np.int8),
        "ball": np.asarray(ball, dtype=np.float64).reshape(-1, 2),
        "labels": np.asarray(labels, dtype=np.int8),
        "tracked": np.asarray(tracked, dtype=bool),
        "player_rows": np.asarray(rows, dtype=np.int64),
        "player_ids": np.asarray(ids, dtype=np.int64),
        "player_xy": np.asarray(xy, dtype=np.float64).reshape(-1, 2),
    }


def cut_mask(tracked: np.ndarray, period: np.ndarray, min_frames: int) -> np.ndarray:
    """(n,) bool: in-period runs of untracked frames at least min_frames long."""
    untracked = (~tracked) & (period > 0)
    edges = np.diff(np.concatenate(([0], untracked.astype(np.int8), [0])))
    mask = np.zeros(len(tracked), dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= min_frames:
            mask[start:stop] = True
    return mask


def load(match_id: str | int, root: str = ROOT, extrapolated: bool = True, cuts: str = "keep",
         min_cut_s: float = 1.0, onsets: str = "label", onset_s: float = 3.0) -> Match:
    """One SkillCorner match as a Match with per-frame labels (H/A/D codes, X outside periods).
    cuts="ungraded" sets broadcast cuts of at least min_cut_s to X; onsets="out" moves D onsets to
    the ball leaving the pitch within onset_s (B20; see the module docstring)."""
    if cuts not in ("keep", "ungraded"):
        raise ValueError("cuts must be 'keep' or 'ungraded', not {!r}".format(cuts))
    if onsets not in ("label", "out"):
        raise ValueError("onsets must be 'label' or 'out', not {!r}".format(onsets))
    meta_path, raw_path = _paths(str(match_id), root)
    for path in (meta_path, raw_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing; run scripts/fetch_external_data.sh skillcorner:{}"
                                    .format(path, match_id))
    meta = read_meta(meta_path)
    raw = read_tracking(raw_path, extrapolated=extrapolated)
    length, width = meta["pitch"]

    unknown = set(np.unique(raw["player_ids"]).tolist()) - set(meta["player_team"])
    if unknown:
        raise ValueError("{}: {} tracked player ids are not in the match file".format(
            match_id, len(unknown)))
    slot_ids = sorted(set(raw["player_ids"].tolist()), key=lambda i: (meta["player_team"][i], i))
    slot_of = {player_id: slot for slot, player_id in enumerate(slot_ids)}

    n = len(raw["frame"])
    players = np.full((n, len(slot_ids), 2), np.nan)
    slots = np.fromiter((slot_of[i] for i in raw["player_ids"].tolist()), dtype=np.int64,
                        count=len(raw["player_ids"]))
    players[raw["player_rows"], slots] = raw["player_xy"]

    offset = np.array([length / 2, width / 2])
    labels = raw["labels"]
    if onsets == "out":
        labels = out_onsets(labels, raw["ball"] + offset, (length, width), raw["period"],
                            round(float(onset_s) * 10))
    if cuts == "ungraded":
        labels = labels.copy()
        labels[cut_mask(raw["tracked"], raw["period"], round(float(min_cut_s) * 10))] = X

    return Match(
        match_id="skillcorner:{}".format(match_id),
        fps=10,
        pitch=(length, width),
        period=raw["period"],
        ball=raw["ball"] + offset,
        players=players + offset,
        slot_team=np.array([meta["player_team"][i] for i in slot_ids], dtype=np.int8),
        labels=labels,
    ).validate()
