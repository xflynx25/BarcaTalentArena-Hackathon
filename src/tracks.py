"""Repair broken player tracks before the model (backlog C13): link tracklets, then bridge the gaps.

Why (findings F112-F113): missing players near the ball cost `learned` far more than shape checks
showed, and filling a slot's short gaps from its last position plus the team's drift won some of it
back. BAR-ATH's slots are tracklets (46 for 22 players, ~20.5 tracked [official DATA.md]): a track
*ends* and the person comes back in another slot, so a fill inside one slot misses most gaps.

repair(match) is label-free and ball-free, uses only the match's own player positions, and is
deterministic (no randomness, stable sorts):
  1. tracklets: every run of frames a slot is seen, split at period changes.
  2. link: a tracklet's end to a later tracklet's start of the same team in the same period, at most
     max_gap_s later, where the start is within reach_m + speed_mps x gap of where the end would be
     after moving with its team (the mean motion of the team's players seen on both frames). Same
     slot or another one: a slot coming back after a gap is linked under the same test. Candidate
     pairs are taken greedily by (distance / reach, gap, end, start), each end and each start once;
     tracklets that overlap in time are never linked (duplicates and real teammates).
  3. bridge: inside each chain of linked tracklets, the frames between an end and its next start
     get the end position moved with the team's drift, plus a linear correction so the bridge lands
     exactly on the start ("bridge"; "drift" = F113's fill: last seen + drift, no correction).
  4. pack: each chain goes back into its first tracklet's slot when that slot is free from before
     the chain starts, else the team's lowest free slot, else a new slot (k never shrinks; a match
     with nothing to link comes back with the same players array).
Unlinked ends stay ended: a gap longer than max_gap_s, or a start too far away, is not bridged,
unless the hold is on:
  5. hold (C16, `hold_s` > 0, off by default): after packing, a chain's last end that nothing
     continues keeps moving with its team's drift for up to hold_s seconds, and a chain's first start
     that continues nothing is extended backwards the same way, both only into free frames of the
     chain's slot in the same period and marked filled. The team keeps its count for a short while
     when a player drops out (off camera, lost in a crowd) instead of losing him at once. With
     hold_s = 0 nothing here runs, so `track_repair=1` keeps its bytes.

The repaired positions are marked (Repair.filled) so features can tell estimates from tracking.
Through the model: `learned` with track_repair=1 (src/models/learned.py) repairs the match before
its features and decoding; as a load option (degrade.apply's track_repair=1) for fits and checks.
Thresholds (MAX_GAP_S, REACH_M, SPEED_MPS) are chosen on external data only (scripts/check_tracks.py,
findings F153).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.schema import AWAY, HOME, Match

MAX_GAP_S = 10.0  # longest gap bridged (s); F113's fill used 10 s
REACH_M = 5.0  # a start this close to the drifted end is linked at any gap ...
SPEED_MPS = 2.0  # ... plus this many metres per second of gap (movement relative to the team)
HOLD_S = 0.0  # C16: unlinked ends / starts held with the team's drift this long (s); 0 = off
FILLS = ("bridge", "drift")
VERSION = 1  # bump when the repair changes: coefficient files fit on repaired matches record it


@dataclass
class Repair:
    match: Match  # the repaired match (players re-packed into chains, k' >= k)
    filled: np.ndarray  # (n, k') bool: positions that are bridges, not tracking
    links: int  # tracklet ends linked to a later start
    tracklets: int


def tracklets(players: np.ndarray, period: np.ndarray) -> np.ndarray:
    """(m, 3) int64 rows (slot, start, stop): each run of frames where a slot is seen, split where the
    period changes. Sorted by slot, then start."""
    seen = ~np.isnan(players).any(axis=2)
    n, k = seen.shape
    brk = np.concatenate([[False], period[1:] != period[:-1]])  # a new run starts here
    out = []
    for slot in range(k):
        s = seen[:, slot]
        starts = np.flatnonzero(s & (np.concatenate([[True], ~s[:-1]]) | brk))
        stops = np.flatnonzero(s & (np.concatenate([~s[1:], [True]]) | np.concatenate([brk[1:], [True]])))
        out.append(np.stack([np.full(len(starts), slot), starts, stops + 1], axis=1))
    return np.concatenate(out).astype(np.int64) if out else np.zeros((0, 3), np.int64)


def team_drift(players: np.ndarray, slot_team: np.ndarray, period: np.ndarray) -> np.ndarray:
    """(2, n, 2) cumulative mean motion of each team (HOME, AWAY): per frame, the mean step of the
    team's slots seen on that frame and the one before (in the same period), summed. Differences
    give how far a team moved between two frames."""
    step = np.diff(players, axis=0, prepend=players[:1])
    seen = ~np.isnan(players).any(axis=2)
    both = seen & np.vstack([np.zeros((1, seen.shape[1]), bool), seen[:-1]])
    both[1:] &= (period[1:] == period[:-1])[:, None]
    out = np.zeros((2, len(players), 2))
    for team in (HOME, AWAY):
        slots = slot_team == team
        moving = both[:, slots]
        total = np.where(moving[..., None], np.nan_to_num(step[:, slots]), 0.0).sum(axis=1)
        out[team] = np.cumsum(total / np.maximum(moving.sum(axis=1), 1)[:, None], axis=0)
    return out


def link(match: Match, max_gap_s: float = MAX_GAP_S, reach_m: float = REACH_M,
         speed_mps: float = SPEED_MPS, segs: np.ndarray | None = None,
         drift: np.ndarray | None = None) -> np.ndarray:
    """(m,) int64: for each tracklet (rows of tracklets()), the index of the tracklet it continues
    into, -1 if none. Deterministic greedy matching; see the module docstring."""
    segs = tracklets(match.players, match.period) if segs is None else segs
    drift = team_drift(match.players, match.slot_team, match.period) if drift is None else drift
    nxt = np.full(len(segs), -1, dtype=np.int64)
    max_gap = max(1, round(max_gap_s * match.fps))
    if not len(segs):
        return nxt
    team = match.slot_team[segs[:, 0]]
    cands = []  # (cost, gap, end index, start index)
    for t in (HOME, AWAY):
        idx = np.flatnonzero(team == t)
        if not len(idx):
            continue
        by_start = idx[np.argsort(segs[idx, 1], kind="stable")]
        starts = segs[by_start, 1]
        for i in idx:
            end = segs[i, 2] - 1  # last seen frame
            lo = np.searchsorted(starts, end + 1)
            hi = np.searchsorted(starts, end + max_gap + 1)
            if lo >= hi:
                continue
            j = by_start[lo:hi]
            s = segs[j, 1]
            j, s = j[match.period[s] == match.period[end]], s[match.period[s] == match.period[end]]
            if not len(j):
                continue
            expect = match.players[end, segs[i, 0]] + drift[t, s] - drift[t, end]
            got = match.players[s, segs[j, 0]]
            dist = np.hypot(*(got - expect).T)
            gap_s = (s - end) / match.fps
            cost = dist / (reach_m + speed_mps * gap_s)
            ok = cost <= 1.0
            cands += [(float(c), int(g), int(i), int(jj)) for c, g, jj in
                      zip(cost[ok], s[ok] - end, j[ok])]
    cands.sort()
    used = np.zeros(len(segs), dtype=bool)
    for _, _, i, j in cands:
        if nxt[i] < 0 and not used[j]:
            nxt[i] = j
            used[j] = True
    return nxt


def repair(match: Match, max_gap_s: float = MAX_GAP_S, reach_m: float = REACH_M,
           speed_mps: float = SPEED_MPS, fill: str = "bridge", hold_s: float = HOLD_S) -> Repair:
    """The match with tracklets linked into chains, the gaps inside chains bridged, and the chains
    packed into slots (see the module docstring). Tags and labels are kept."""
    if fill not in FILLS:
        raise ValueError("fill must be one of {}, not {!r}".format(FILLS, fill))
    players, period = match.players, match.period
    n = len(period)
    segs = tracklets(players, period)
    drift = team_drift(players, match.slot_team, period)
    nxt = link(match, max_gap_s, reach_m, speed_mps, segs, drift)
    has_prev = np.zeros(len(segs), dtype=bool)
    has_prev[nxt[nxt >= 0]] = True
    chains = []
    for head in np.flatnonzero(~has_prev):
        chain = [int(head)]
        while nxt[chain[-1]] >= 0:
            chain.append(int(nxt[chain[-1]]))
        chains.append(chain)
    # pack: chains by first frame, each into its first tracklet's slot if that is free since before
    # it starts, else the team's lowest free slot, else a new slot (so a match with nothing to
    # repair comes back unchanged)
    k = len(match.slot_team)
    team_of = list(match.slot_team)
    free_from = [0] * k
    layout = []  # (out slot, chain)
    for c in sorted(chains, key=lambda c: (segs[c[0], 1], c[0])):
        first, own = segs[c[0], 1], int(segs[c[0], 0])
        col = own if free_from[own] <= first else next(
            (j for j in range(len(team_of)) if team_of[j] == team_of[own] and free_from[j] <= first),
            len(team_of))
        if col == len(team_of):
            team_of.append(team_of[own])
            free_from.append(0)
        free_from[col] = segs[c[-1], 2]
        layout.append((col, c))
    k_out = len(team_of)
    out = np.full((n, k_out, 2), np.nan)
    filled = np.zeros((n, k_out), dtype=bool)
    for col, c in layout:
        t = team_of[col]
        for a, b in zip(c, c[1:] + [None]):
            src, start, stop = segs[a]
            out[start:stop, col] = players[start:stop, src]
            if b is None:
                continue
            end, begin = stop - 1, segs[b, 1]
            if begin - end <= 1:
                continue
            frames = np.arange(end + 1, begin)
            p_end, p_start = players[end, src], players[begin, segs[b, 0]]
            moved = drift[t, frames] - drift[t, end]
            est = p_end + moved
            if fill == "bridge":
                miss = p_start - (p_end + drift[t, begin] - drift[t, end])
                est = est + ((frames - end) / (begin - end))[:, None] * miss
            out[frames, col] = est
            filled[frames, col] = True
    if hold_s > 0:
        _hold(out, filled, layout, segs, players, period, drift, team_of, round(hold_s * match.fps))
    length, width = match.pitch
    out[..., 0] = np.where(filled, np.clip(out[..., 0], 0.0, length), out[..., 0])
    out[..., 1] = np.where(filled, np.clip(out[..., 1], 0.0, width), out[..., 1])
    slot_team = np.array(team_of, dtype=np.int8)
    repaired = dataclasses.replace(match, players=out, slot_team=slot_team).validate()
    return Repair(repaired, filled, int((nxt >= 0).sum()), len(segs))


def _hold(out, filled, layout, segs, players, period, drift, team_of, hold: int) -> None:
    """C16's hold, in place: each chain's unlinked last end runs on with its team's drift for up to
    `hold` frames, then each chain's unlinked first start runs back the same way; only frames of the
    chain's slot that are still empty, in the same period. Forward holds first, so a start held back
    stops where the slot's previous chain (or its hold) is."""
    if hold < 1:
        return
    n = len(period)
    for direction in (1, -1):
        for col, c in layout:
            t = team_of[col]
            src, start, stop = segs[c[-1]] if direction > 0 else segs[c[0]]
            at = stop - 1 if direction > 0 else start
            frames = np.arange(at + direction, at + direction * (hold + 1), direction)
            frames = frames[(frames >= 0) & (frames < n)]
            if not len(frames):
                continue
            ok = (period[frames] == period[at]) & np.isnan(out[frames, col, 0])
            run = len(frames) if ok.all() else int(np.argmin(ok))  # stop at the first busy frame
            frames = frames[:run]
            if not len(frames):
                continue
            out[frames, col] = players[at, src] + drift[t, frames] - drift[t, at]
            filled[frames, col] = True


def apply(match: Match, spec: str | float | int = "1") -> Match:
    """repair(match) from a load option / decoder parameter: "1" (defaults), "0"/"" (off), or
    "max_gap_s[/reach_m[/speed_mps[/fill]]]", e.g. "10/5/2/bridge" (the defaults); options as
    "/key=value" parts in any position: "1/hold=2" (C16: the defaults plus a 2 s hold), "hold=2"."""
    text = str(spec).strip().lower()
    if text in ("", "0", "false", "none", "no"):
        return match
    parts = text.split("/")
    opts = dict(p.split("=", 1) for p in parts if "=" in p)
    parts = [p for p in parts if "=" not in p]
    kwargs = {}
    for key, value in opts.items():
        if key != "hold":
            raise ValueError("track_repair: unknown option {!r} in {!r}".format(key, spec))
        kwargs["hold_s"] = float(value)
    if not parts or parts in (["1"], ["true"], ["yes"]):
        return repair(match, **kwargs).match
    args = [float(parts[0])] + [float(p) for p in parts[1:3]]
    if len(parts) > 3:
        kwargs["fill"] = parts[3]
    return repair(match, *args, **kwargs).match
