"""Restart rule: the few seconds before a restart kick are a dead ball, used as extra D evidence.

Windows per source (label-free): BAR-ATH = Smart Tagging SET PIECES, SkillCorner = the 4 s before a
dynamic event with game_interruption_before, IDSSE = restart events -2 s to +3 s, Metrica = set pieces
and goal kicks -2 s to +3 s. External windows are looked up from match_id so they survive the ball
hiding, and only apply at the source's own frame rate.
"""

from __future__ import annotations

import csv
import datetime as dt
import functools
import glob
import os
import re
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from src.schema import Match
from src.smart_tags import NA, NOISE_LEVELS, SET_PIECES, Tag, perturb

IDSSE_RESTARTS = ("KickOff", "ThrowIn", "FreeKick", "GoalKick", "CornerKick", "Penalty", "RefereeBall")


def windows(match: Match) -> np.ndarray:
    """(k, 2) restart windows as [start, stop) rows, empty when the source has none."""
    if match.tags is not None:
        out = [(t.start, t.end) for t in match.tags if t.code == SET_PIECES]
        return np.array(out, dtype=np.int64).reshape(-1, 2)
    source, _, key = match.match_id.partition(":")
    found = {"skillcorner": (10, _skillcorner), "idsse": (25, _idsse), "metrica": (25, _metrica)}.get(source)
    if found is None or found[0] != match.fps:
        return np.zeros((0, 2), dtype=np.int64)
    return found[1](key).copy()


def ball_speed(match: Match, span_s: float = 0.2) -> np.ndarray:
    """(n,) ball speed in m/s over span_s, NaN where it can't be measured."""
    k = max(1, int(round(span_s * match.fps)))
    v = np.full(match.n_frames, np.nan)
    v[k:] = np.linalg.norm(match.ball[k:] - match.ball[:-k], axis=1) * match.fps / k
    return v


def noisy(match: Match, wins: np.ndarray, noise: int, seed: int = 0) -> np.ndarray:
    """B35 stress test, external matches only: wins with a tagger's errors (smart_tags.perturb at
    NOISE_LEVELS[noise]: timing jitter, missed windows, false windows), seeded by the match id, so
    the rule can be judged with windows as imperfect as BAR-ATH's Smart Tagging may be."""
    if not noise or not len(wins):
        return wins
    rng = np.random.default_rng(zlib.crc32("b35|{}|{}".format(match.match_id, seed).encode()))
    tags = [Tag(int(a), int(b), SET_PIECES, NA) for a, b in wins]
    out = [(t.start, t.end) for t in perturb(tags, NOISE_LEVELS[noise], match, rng)]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def dead_stretch(match: Match, back_s: float = 6.0, kick_mps: float = 4.0,
                 search_s: float = 8.0, noise: int = 0) -> np.ndarray:
    """(n,) bool: back_s before each restart kick found in a window (noise: B35's stress test)."""
    dead = np.zeros(match.n_frames, dtype=bool)
    wins = noisy(match, windows(match), noise)
    if not len(wins):
        return dead
    length, width = match.pitch
    with np.errstate(invalid="ignore"):
        inside = ((match.ball[:, 0] > 0) & (match.ball[:, 0] < length)
                  & (match.ball[:, 1] > 0) & (match.ball[:, 1] < width))
        kicked = inside & (ball_speed(match) > kick_mps)
    for lo, hi in wins:
        lo, hi = max(0, int(lo)), min(len(dead), int(hi), int(lo) + int(round(search_s * match.fps)))
        hits = np.flatnonzero(kicked[lo:hi])
        if len(hits):
            kick = lo + int(hits[0])
            dead[max(0, kick - int(round(back_s * match.fps))):kick] = True
    return dead


@functools.lru_cache(maxsize=None)
def _skillcorner(match_id: str) -> np.ndarray:
    from src.ingest import skillcorner
    folder = os.path.join(skillcorner.ROOT, "data", "matches", match_id)
    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id))) as handle:
        first = int(re.search(r'"frame":\s*(\d+)', handle.readline()).group(1))
    with open(os.path.join(folder, "{}_dynamic_events.csv".format(match_id))) as handle:
        frames = sorted({int(e["frame_start"]) for e in csv.DictReader(handle) if e["game_interruption_before"]})
    return np.array([(f - first - 40, f - first + 1) for f in frames], dtype=np.int64).reshape(-1, 2)


@functools.lru_cache(maxsize=None)
def _metrica(match_id: str) -> np.ndarray:
    from src.ingest import metrica
    paths = metrica._paths(match_id, metrica.ROOT)
    if match_id == "3":
        events = metrica.read_events_json(paths["events"], metrica.read_metadata_epts(paths["meta"])["home_team_id"])
    else:
        events = metrica.read_events_csv(paths["events"])
    restart = (events["type"] == "SET PIECE") | ((events["type"] == "PASS") & events["subtype"].str.contains("GOAL KICK"))
    #tracking frames count from 1 and rows from 0
    starts = events.loc[restart, "start"].values - 1
    return np.array([(f - 50, f + 75) for f in starts], dtype=np.int64).reshape(-1, 2)


@functools.lru_cache(maxsize=None)
def _idsse(match_id: str) -> np.ndarray:
    from src.ingest import idsse
    key = idsse.match_key(match_id)
    _, positions = idsse._paths(key, idsse.ROOT)
    events = sorted(glob.glob(os.path.join(idsse.ROOT, "DFL_03_02_events_raw_*_{}.xml".format(key))))
    if positions is None or not events:
        return np.zeros((0, 2), dtype=np.int64)
    #the ball's frame range and first / last timestamps per half, laid out like the loader's rows
    halves, current = {}, None
    with open(positions) as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                current = None
                if 'TeamId="BALL"' in line:
                    current = re.search(r'GameSection="([^"]+)"', line).group(1)
            elif current is not None and line.startswith("<Frame "):
                n = int(re.search(r' N="(\d+)"', line).group(1))
                t = re.search(r' T="([^"]+)"', line).group(1)
                h = halves.setdefault(current, [n, n, t, t])
                h[0], h[1], h[3] = min(h[0], n), max(h[1], n), t
    blocks, row, previous_end = [], 0, None
    for section in idsse.SECTIONS:
        if section not in halves:
            continue
        first, last, t_first, t_last = halves[section]
        if previous_end is not None:
            row += max(1, int(round((idsse._timestamp(t_first) - previous_end) * idsse.FPS)) - 1)
        blocks.append((row, row + last - first + 1, dt.datetime.fromisoformat(t_first)))
        row += last - first + 1
        previous_end = idsse._timestamp(t_last)
    out = []
    for e in ET.parse(events[0]).getroot().iter("Event"):
        if not any(c.tag in IDSSE_RESTARTS for c in e):
            continue
        t = dt.datetime.fromisoformat(e.get("EventTime"))
        for start, stop, t0 in blocks:
            f = start + int(round((t - t0).total_seconds() * idsse.FPS))
            if start <= f < stop:
                out.append((f - 2 * idsse.FPS, f + 3 * idsse.FPS))
    return np.array(out, dtype=np.int64).reshape(-1, 2)
