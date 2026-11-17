"""Repair `gkteam` (C15): hand a goalkeeper delivered as the other team back to his team, drop his ghost.

Why (backlog C15): kit-colour team classification has a known weak spot, goalkeepers, who wear
their own kit. When it fails, the keeper comes as a detection of the other team, and possession goes
wrong exactly when he has the ball (catches, goal kicks, distributions). src/ingest/degrade.py
simulates it (gk_flip, gk_ghost, gk_hide); this file finds and undoes it, label-free and ball-free,
from the match's own player positions, deterministically.

Signatures (thresholds below, chosen on external data only: scripts/check_gkteam.py, F316-F319):
  1. box: a tracklet of team X (a run of frames a slot is seen, split at period changes) that stays
     inside the penalty area of the goal X attacks (that goal: the one team Y defends in the period,
     found label-free as degrade.defending_left does) on >= BOX_SHARE of its frames, and
  2. depth: nearer that goal line than every team-Y detection more than NEAR_M away from it on >=
     DEEP_SHARE of its frames (a keeper's place; Y detections close by are excluded so his ghost or a
     defender contesting him can't hide him), and
  3. link: it continues the end of a team-Y tracklet (a cross-team link, allowed only there): a Y
     tracklet ends at most LINK_GAP_S before it starts, its last position in the same penalty area and
     within LINK_REACH_M (+ LINK_SPEED_MPS per second of gap) of the X tracklet's first; or the same
     at its other end (a Y tracklet starts just after it ends, where it ended).
Repair: the X tracklet's positions move to a team-Y slot free over its frames (the linked tracklet's
slot first, then the lowest free Y slot, else a new Y slot). Then the ghost: a team-Y tracklet that
lives inside the flipped stretch (+/- GHOST_SLACK_S), stays within GHOST_M of the keeper (median) and
moves little (median speed over 1 s below GHOST_MPS), is dropped.

Self-guard: if the repair would move more than MAX_SHARE of the in-play frames (the frames with any
detection relabelled), it does nothing and prints a note, so it can't run away on an unfamiliar
match. The match comes back as the same object when nothing is found (clean matches keep their bytes).

    --param repairs=gkteam                 (src/repairs/__init__.py)
    --param "repairs=gkteam/ghost=0"       keep ghosts; other options: the constants below in lower case
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.schema import Match

BOX_DEPTH_M = 30.0  # the keeper zone: this deep from the goal line (keepers step out of the box; F317) ...
BOX_HALF_WIDTH_M = 20.16  # ... and this far either side of the pitch's centre line
BOX_SHARE = 0.9  # 1. share of the tracklet's frames inside the box
DEEP_SHARE = 0.8  # 2. share of its frames nearer the goal line than every Y detection ...
NEAR_M = 5.0  # ... ignoring Y detections this close to it
LINK_GAP_S = 1.0  # 3. a Y tracklet ends at most this long before it starts (or starts after it ends)
LINK_REACH_M = 1.0  # ... this close to its first (last) position ...
LINK_SPEED_MPS = 3.0  # ... plus this per second of gap
ENDS = 2  # links needed: 1 = at either end, 2 = at both ends (the keeper leaves and comes back)
MIN_S = 0.4  # tracklets shorter than this are left alone (too short to tell)
GHOST = 1  # drop the ghost (0: keep it)
GHOST_SLACK_S = 0.5  # the ghost lives inside the flipped stretch +/- this
GHOST_M = 5.0  # median distance to the keeper at most this
GHOST_MPS = 1.0  # median speed over 1 s at most this
MAX_SHARE = 0.15  # self-guard: more relabelled in-play frames than this -> do nothing


def defending_left(match: Match) -> dict:
    """{period: team that defends the x = 0 goal}: the team whose deepest player (lowest x) is lower
    in the median over the period's frames (its keeper). Label-free."""
    out = {}
    for p in np.unique(match.period[match.period > 0]):
        rows = match.period == p
        low = []
        for team in (0, 1):
            x = match.players[rows][:, match.slot_team == team, 0]
            if not x.shape[1]:
                low.append(np.inf)
                continue
            m = np.min(np.where(np.isnan(x), np.inf, x), axis=1)
            m = m[np.isfinite(m)]
            low.append(float(np.median(m)) if len(m) else np.inf)
        out[int(p)] = 0 if low[0] <= low[1] else 1
    return out


def _tracklets(seen: np.ndarray, period: np.ndarray) -> list[tuple[int, int, int]]:
    """(slot, start, stop) for each run of frames a slot is seen, split where the period changes."""
    brk = np.concatenate([[False], period[1:] != period[:-1]])
    out = []
    for slot in range(seen.shape[1]):
        s = seen[:, slot]
        starts = np.flatnonzero(s & (np.concatenate([[True], ~s[:-1]]) | brk))
        stops = np.flatnonzero(s & (np.concatenate([~s[1:], [True]]) | np.concatenate([brk[1:], [True]])))
        out += [(slot, int(a), int(b) + 1) for a, b in zip(starts, stops)]
    return out


def find(match: Match, box_depth_m: float = BOX_DEPTH_M, ends: int = ENDS, box_share: float = BOX_SHARE, deep_share: float = DEEP_SHARE,
         near_m: float = NEAR_M, link_gap_s: float = LINK_GAP_S, link_reach_m: float = LINK_REACH_M,
         link_speed_mps: float = LINK_SPEED_MPS, min_s: float = MIN_S, ghost: int = GHOST,
         ghost_slack_s: float = GHOST_SLACK_S, ghost_m: float = GHOST_M,
         ghost_mps: float = GHOST_MPS) -> list[dict]:
    """The flips found: dicts with slot, start, stop (the X tracklet), team (Y, the keeper's team),
    to (the Y slot it goes to, -1 = a new slot), ghosts [(slot, start, stop)]. No changes made."""
    players, period, fps = match.players, match.period, match.fps
    n, k = players.shape[:2]
    length, width = match.pitch
    seen = ~np.isnan(players).any(axis=2)
    left = defending_left(match)
    segs = _tracklets(seen, period)
    by_team_end = {0: [], 1: []}
    by_team_start = {0: [], 1: []}
    for slot, a, b in segs:
        by_team_end[int(match.slot_team[slot])].append((b - 1, slot, a))
        by_team_start[int(match.slot_team[slot])].append((a, slot, b))
    gap = max(1, round(link_gap_s * fps))
    found = []
    for slot, a, b in segs:
        p = int(period[a])
        if p <= 0 or b - a < max(1, round(min_s * fps)):
            continue
        x_team = int(match.slot_team[slot])
        y_team = 1 - x_team
        if p not in left:
            continue
        goal_left = left[p] == y_team  # the goal X attacks (Y defends) is at x = 0

        def depth(x):
            return x if goal_left else length - x

        def in_box(pos):
            return (depth(pos[..., 0]) <= box_depth_m) & (np.abs(pos[..., 1] - width / 2)
                                                         <= BOX_HALF_WIDTH_M)

        pos = players[a:b, slot]
        if np.mean(in_box(pos)) < box_share:
            continue
        ys = np.flatnonzero(match.slot_team == y_team)
        other = players[a:b][:, ys]
        dist = np.hypot(other[..., 0] - pos[:, None, 0], other[..., 1] - pos[:, None, 1])
        od = np.where(np.isnan(dist) | (dist <= near_m), np.inf, depth(other[..., 0]))
        if np.mean(depth(pos[:, 0]) < np.min(od, axis=1)) < deep_share:
            continue
        # 3. a cross-team link at either end, in the same box
        link_slot = after_slot = -1
        for end, yslot, ya in by_team_end[y_team]:
            g = a - end
            if 1 <= g <= gap and period[end] == p:
                last = players[end, yslot]
                if (in_box(last) and np.hypot(*(last - pos[0])) <= link_reach_m
                        + link_speed_mps * g / fps):
                    link_slot = yslot
                    break
        for start, yslot, yb in by_team_start[y_team]:
            g = start - (b - 1)
            if 1 <= g <= gap and period[start] == p:
                first = players[start, yslot]
                if (in_box(first) and np.hypot(*(first - pos[-1])) <= link_reach_m
                        + link_speed_mps * g / fps):
                    after_slot = yslot
                    break
        if (link_slot >= 0) + (after_slot >= 0) < ends:
            continue
        if link_slot < 0:
            link_slot = after_slot
        # where it goes: the linked slot if free over [a, b), else the lowest free Y slot
        to = -1
        for cand in [link_slot] + [int(s) for s in ys if s != link_slot]:
            if not seen[a:b, cand].any():
                to = int(cand)
                break
        ghosts = []
        if ghost:
            slack = round(ghost_slack_s * fps)
            w = max(1, int(fps))
            for gslot, ga, gb in segs:
                if (match.slot_team[gslot] != y_team or gslot == to or ga < a - slack
                        or gb > b + slack or gb - ga < 2):
                    continue
                lo, hi = max(a, ga), min(b, gb)
                if hi <= lo:
                    continue
                gp = players[ga:gb, gslot]
                d = np.hypot(*(players[lo:hi, gslot] - players[lo:hi, slot]).T)
                step = gp[min(w, len(gp) - 1):] - gp[:len(gp) - min(w, len(gp) - 1)]
                speed = np.hypot(*step.T) * fps / min(w, len(gp) - 1)
                if np.median(d) <= ghost_m and np.median(speed) <= ghost_mps:
                    ghosts.append((int(gslot), int(ga), int(gb)))
        found.append({"slot": int(slot), "start": int(a), "stop": int(b), "team": y_team,
                      "to": to, "ghosts": ghosts})
    # each Y slot can take only one stretch at a time: a later find that wants an occupied slot
    # gets a new one (resolved in apply)
    return found


def apply(match: Match, max_share: float = MAX_SHARE, **opts) -> Match:
    """The match with each flip found (find()) handed back and its ghost dropped; the match itself
    when nothing is found or the self-guard trips (more than max_share of in-play frames moved)."""
    found = find(match, **opts)
    if not found:
        return match
    n = match.n_frames
    moved = np.zeros(n, dtype=bool)
    for f in found:
        moved[f["start"]:f["stop"]] = True
        for _, ga, gb in f["ghosts"]:
            moved[ga:gb] = True
    in_play = match.period > 0
    share = float(moved[in_play].mean()) if in_play.any() else 0.0
    if share > max_share:
        print("gkteam: self-guard: would relabel {:.1%} of in-play frames (> {:.0%}); nothing "
              "repaired".format(share, max_share))
        return match
    players = match.players.copy()
    team = list(match.slot_team)
    extra = []
    for f in found:
        a, b, slot, y = f["start"], f["stop"], f["slot"], f["team"]
        to = f["to"]
        if to < 0 or not np.isnan(players[a:b, to]).all():
            to = next((c for c in range(len(team)) if team[c] == y
                       and np.isnan(players[a:b, c] if c < players.shape[1]
                                    else extra[c - players.shape[1]][a:b]).all()), -1)
        if to < 0:
            team.append(y)
            extra.append(np.full((n, 2), np.nan))
            to = len(team) - 1
        values = players[a:b, slot].copy()
        if to < players.shape[1]:
            players[a:b, to] = values
        else:
            extra[to - players.shape[1]][a:b] = values
        players[a:b, slot] = np.nan
        for gslot, ga, gb in f["ghosts"]:
            players[ga:gb, gslot] = np.nan
    if extra:
        players = np.concatenate([players, np.stack(extra, axis=1)], axis=1)
    return dataclasses.replace(match, players=players,
                               slot_team=np.array(team, dtype=match.slot_team.dtype)).validate()
