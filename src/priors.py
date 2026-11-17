"""Possession and stoppage priors from event data, and a duration-aware decoder (backlog B27).

Event-only datasets have no tracking but thousands of matches. From their timestamps we measure
how long stoppages last (by restart type) and how long possession spells last, under the
competition's rule (D while the ball is dead, otherwise the last team in possession). The same
numbers come from our labelled sources' frame labels (label_segments), so the two can be compared.

Event parsers (external data only, data/raw/external/{wyscout,statsbomb}):
  wyscout    Pappalardo et al. 2019, 1,941 matches 2017/18 (CC BY 4.0). Restarts are "Free Kick"
             events (throw-in, corner, free kick, goal kick, penalty); the kick-off after a goal is
             the first event after it. The ball going dead is an "Interruption" (ball out, whistle),
             a foul or an offside, at its own timestamp.
  statsbomb  StatsBomb open data, a fixed subset of ~580 men's matches. Restarts are pass / shot
             types; the ball going dead is the last on-ball event before the restart (its `out`
             flag, a foul, an offside, a goal).
Matches between Barcelona and Athletic Club are skipped (none are in the StatsBomb subset), so no
event data about any Barcelona v Athletic game is ever read.

Stoppage = (restart time) - (time of the event that made the ball dead, else the last on-ball
event before the restart), inside one period. Kick-offs at the start of a period are not
stoppages. Spell = a run of on-ball events of one team inside live play, from its first event to
the next team's first event (or the next stoppage). Event data can't see what isn't an event:
sub-second touches, deflections, the ball's travel to the line (stoppages start at the last touch,
so they come out ~1 s long), or the exact moment of a restart if the clock is coarse.

Priors table (PRIOR_FILE, committed): per source, quantiles of spell and stoppage lengths (all
stoppages, and per restart type). A prior turns into an exit penalty for the decoder:
  g(L) = weight * -log(max(F(L), floor))
for a segment of length L seconds, F its CDF. Long segments (F near 1) cost nothing extra; a
segment shorter than real ones ever are costs up to weight * -log(floor). That is a soft minimum
length learnt from data, on top of v4's flat switch costs.

viterbi_aged() is the decoder: the same Viterbi over {H, A, D} with the same flat costs as
src.models.decoder.viterbi, plus g(L) charged when a segment of length L ends. Exact: a segment's
length is tracked up to MAX_AGE_S (longer ones pay g at that length); all values are on the
decoder's 2**-GRID_BITS grid, so the arithmetic is exact and the path is the same on every machine.
With typed stoppage priors, a D segment that contains a set-piece shape (src/models/setpiece.py,
players only) pays its type's g instead of the generic one.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np

from src.models.decoder import to_grid
from src.official import A, D, H, X

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXTERNAL = os.path.join(ROOT, "data", "raw", "external")
PRIOR_FILE = os.path.join(ROOT, "src", "models", "priors.json")

RESTART_TYPES = ("throw", "corner", "freekick", "goalkick", "kickoff", "penalty", "drop")
QUANTILES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
             0.8, 0.9, 0.95, 0.99)
MAX_AGE_S = 40.0  # segment lengths tracked exactly up to this; longer ones pay g(MAX_AGE_S)
FLOOR = 0.01  # F floored here, so the largest penalty is weight * -log(FLOOR) = 4.6 * weight
EXCLUDED_TEAMS = ("barcelona", "athletic")  # skip any match between these two (CLAUDE.md)

# stream kinds: an on-ball action, the ball going dead (ON_STOP: an on-ball action that ends play,
# e.g. a pass out), a restart. A STOP followed by an on-ball event rather than a restart was no
# stoppage (advantage played) and is forgotten.
ON, STOP, RESTART, ON_STOP = 0, 1, 2, 3


def _excluded(names) -> bool:
    low = [n.lower() for n in names]
    return all(any(t in n for n in low) for t in EXCLUDED_TEAMS)


# -- Wyscout -----------------------------------------------------------------------------------------

WY_RESTART = {"Throw in": "throw", "Corner": "corner", "Free Kick": "freekick",
              "Free kick cross": "freekick", "Free kick shot": "freekick", "Goal kick": "goalkick",
              "Penalty": "penalty"}
WY_ON = ("Pass", "Shot", "Others on the ball", "Save attempt")
WY_STOP = ("Interruption", "Foul", "Offside")
WY_PERIODS = {"1H": 1, "2H": 2, "E1": 3, "E2": 4}
WY_GOAL_TAGS = {101, 102}  # goal, own goal


def wyscout_streams(competition: str):
    """Yield (match_id, stream) for one Wyscout competition file (e.g. "England"). A stream is a
    dict of arrays per event: period, t (s), team (0 home / 1 away), kind (ON / STOP / RESTART),
    rtype (index into RESTART_TYPES, -1 if none)."""
    base = os.path.join(EXTERNAL, "wyscout")
    with open(os.path.join(base, "teams.json")) as fh:
        teams = {t["wyId"]: t["name"] for t in json.load(fh)}
    with open(os.path.join(base, "matches", "matches_{}.json".format(competition))) as fh:
        matches = {}
        for m in json.load(fh):
            side = {int(k): v["side"] for k, v in m["teamsData"].items()}
            if not _excluded([teams.get(k, "") for k in side]):
                matches[m["wyId"]] = side
    with open(os.path.join(base, "events", "events_{}.json".format(competition))) as fh:
        events = json.load(fh)
    by_match: dict = {}
    for e in events:
        if e["matchId"] in matches and e["matchPeriod"] in WY_PERIODS:
            by_match.setdefault(e["matchId"], []).append(e)
    for mid in sorted(by_match):
        side = matches[mid]
        rows = []
        goal_pending = False
        for e in sorted(by_match[mid], key=lambda e: (WY_PERIODS[e["matchPeriod"]], e["eventSec"])):
            period, t = WY_PERIODS[e["matchPeriod"]], float(e["eventSec"])
            team = 0 if side.get(e["teamId"]) == "home" else 1
            name, sub = e["eventName"], e["subEventName"]
            tags = {tag["id"] for tag in e.get("tags", [])}
            if goal_pending:  # the kick-off: the next pass from the centre spot
                pos = e.get("positions") or [{}]
                if name == "Pass" and 45 <= pos[0].get("x", 0) <= 55 and 40 <= pos[0].get("y", 0) <= 60:
                    if rows and rows[-1][0] == period:
                        rows.append((period, t, team, RESTART, RESTART_TYPES.index("kickoff")))
                    goal_pending = False
                    continue
                if rows and rows[-1][0] != period:
                    goal_pending = False
                else:
                    continue  # the keeper's save attempt, celebrations: not play
            if name == "Free Kick" and sub in WY_RESTART:
                rows.append((period, t, team, RESTART, RESTART_TYPES.index(WY_RESTART[sub])))
            elif name in WY_STOP:
                rows.append((period, t, team, STOP, -1))
            elif name in WY_ON:
                rows.append((period, t, team, ON, -1))
            if tags & WY_GOAL_TAGS and name in ("Shot", "Free Kick"):
                rows.append((period, t, team, STOP, -1))
                goal_pending = True
        yield "wyscout:{}".format(mid), _stream(rows)


# -- StatsBomb ---------------------------------------------------------------------------------------

SB_RESTART = {"Throw-in": "throw", "Corner": "corner", "Free Kick": "freekick",
              "Goal Kick": "goalkick", "Kick Off": "kickoff", "Penalty": "penalty"}
SB_ON = ("Pass", "Carry", "Ball Receipt*", "Shot", "Dribble", "Clearance", "Ball Recovery",
         "Interception", "Goal Keeper", "Miscontrol", "Dispossessed")
SB_STOP = ("Foul Committed", "Offside", "Injury Stoppage", "Half End")


def _sb_seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return 3600 * int(h) + 60 * int(m) + float(s)


def _sb_row(e: dict, team: int):
    """(kind, rtype) for one StatsBomb event, or None if it says nothing about the ball."""
    name = e["type"]["name"]
    if name == "Pass" and e["pass"].get("type", {}).get("name") in SB_RESTART:
        return RESTART, RESTART_TYPES.index(SB_RESTART[e["pass"]["type"]["name"]])
    if name == "Shot" and e["shot"].get("type", {}).get("name") in SB_RESTART:
        return RESTART, RESTART_TYPES.index(SB_RESTART[e["shot"]["type"]["name"]])
    if name == "Referee Ball-Drop":
        return RESTART, RESTART_TYPES.index("drop")
    if name in SB_STOP:
        if name == "Foul Committed" and e.get("foul_committed", {}).get("advantage"):
            return None
        return STOP, -1
    if name not in SB_ON:
        return None
    detail = e.get(name.lower().replace(" ", "_").rstrip("*"), {})
    if name == "Ball Receipt*" and "outcome" in detail:
        return None  # incomplete: never had it
    if name == "Ball Recovery" and detail.get("recovery_failure"):
        return None
    if name == "Interception" and detail.get("outcome", {}).get("name") in ("Lost", "Lost In Play",
                                                                             "Lost Out"):
        return None
    if name == "Goal Keeper" and detail.get("type", {}).get("name") not in ("Collected",
                                                                             "Keeper Sweeper",
                                                                             "Smother"):
        return None
    ended = (e.get("out") or (name == "Pass" and detail.get("outcome", {}).get("name") in
                              ("Out", "Pass Offside"))
             or (name == "Shot" and detail.get("outcome", {}).get("name") in ("Goal", "Off T",
                                                                              "Wayward")))
    return (ON_STOP if ended else ON), -1


def statsbomb_streams():
    """Yield (match_id, stream) for every fetched StatsBomb match (see wyscout_streams). A Shot
    off target or wide ends play only if StatsBomb says so; a goal always does (and its kick-off
    is the restart)."""
    base = os.path.join(EXTERNAL, "statsbomb")
    matches = {}
    for path in sorted(glob.glob(os.path.join(base, "matches", "*", "*.json"))):
        with open(path) as fh:
            for m in json.load(fh):
                names = (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"])
                if not _excluded(names):
                    matches[m["match_id"]] = m["home_team"]["home_team_id"]
    for mid in sorted(matches):
        path = os.path.join(base, "events", "{}.json".format(mid))
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            events = json.load(fh)
        rows = []
        for e in events:
            if e.get("period", 0) not in (1, 2, 3, 4) or "team" not in e:
                continue
            team = 0 if e["team"]["id"] == matches[mid] else 1
            row = _sb_row(e, team)
            if row is not None:
                rows.append((e["period"], _sb_seconds(e["timestamp"]), team) + row)
        rows.sort(key=lambda r: (r[0], r[1]))
        yield "statsbomb:{}".format(mid), _stream(rows)


def _stream(rows) -> dict:
    arr = np.array(rows, dtype=float).reshape(-1, 5)
    return {"period": arr[:, 0].astype(int), "t": arr[:, 1], "team": arr[:, 2].astype(int),
            "kind": arr[:, 3].astype(int), "rtype": arr[:, 4].astype(int)}


# -- measuring ---------------------------------------------------------------------------------------

def stream_segments(stream: dict) -> dict:
    """{"dead": [(length_s, rtype)], "team": [length_s], "live_s": total live seconds,
    "changes": team changes inside live play}. Per period: a restart (not the period's first
    event) closes a stoppage that began at the STOP event since the last on-ball event, else at
    that on-ball event. Spells run between team changes inside live stretches; events of two teams
    at the same instant (StatsBomb logs a duel's both sides) are no spell. The first spell of a
    period and its last are dropped (cut by the period, as label_segments drops them)."""
    out = {"dead": [], "team": [], "live_s": 0.0, "changes": 0}
    for period in np.unique(stream["period"]):
        sel = stream["period"] == period
        t, team, kind, rtype = (stream[k][sel] for k in ("t", "team", "kind", "rtype"))
        if not len(t):
            continue
        stretch, first, stop_at = [(t[0], int(team[0]))], True, None
        for i in range(1, len(t)):
            if kind[i] == RESTART:
                begin = stop_at if stop_at is not None else stretch[-1][0]
                out["dead"].append((float(t[i] - begin), int(rtype[i])))
                _close(stretch, begin, first, False, out)
                stretch, first, stop_at = [(t[i], int(team[i]))], False, None
                continue
            if kind[i] == STOP:
                if stop_at is None:
                    stop_at = t[i]
                continue
            stop_at = None  # an on-ball event, not a restart: play went on (advantage)
            stretch.append((t[i], int(team[i])))
            if kind[i] == ON_STOP:
                stop_at = t[i]
        _close(stretch, stop_at if stop_at is not None else stretch[-1][0], first, True, out)
    return out


def _runs(events) -> list:
    """[(start, team)] runs of one team from [(t, team)] on-ball events, without zero-length runs."""
    runs = []
    for t, team in events:
        if not runs or runs[-1][1] != team:
            runs.append((t, team))
    while True:
        short = [i for i in range(len(runs) - 1) if runs[i + 1][0] - runs[i][0] < 0.05]
        if not short:
            return runs
        del runs[short[0]]
        merged = []
        for r in runs:
            if not merged or merged[-1][1] != r[1]:
                merged.append(r)
        runs = merged


def _close(events, end: float, cut_first: bool, cut_last: bool, out: dict) -> None:
    runs = _runs(events)
    out["live_s"] += max(0.0, end - events[0][0])
    out["changes"] += len(runs) - 1
    for i, (start, _) in enumerate(runs):
        if (i == 0 and cut_first) or (i == len(runs) - 1 and cut_last):
            continue
        stop = runs[i + 1][0] if i + 1 < len(runs) else end
        if stop - start >= 0.05:  # a touch that ends play at once (e.g. a clearance out) has no length
            out["team"].append(float(stop - start))


def label_segments(labels: np.ndarray, period: np.ndarray, fps: int, types=None) -> dict:
    """The same numbers from frame labels: {"dead": [(length_s, rtype)], "team": [length_s],
    "live_s", "changes"}. Segments touching a period's start or end are dropped (cut). types:
    optional {rtype: (n,) bool mask} (shape_types); a D segment gets the first type in
    SHAPE_TYPES order whose mask is on inside it, else -1."""
    from src.models.postprocess import play_runs
    dead, team_spells, live, changes = [], [], 0.0, 0
    for start, stop in play_runs(period):
        w = labels[start:stop]
        live += float(np.sum((w == H) | (w == A))) / fps
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        codes = w[bounds[:-1]]
        for j in range(1, len(codes)):
            if codes[j] in (H, A) and codes[j - 1] in (H, A) and codes[j] != codes[j - 1]:
                changes += 1
        for j in range(1, len(codes) - 1):
            a, b = bounds[j], bounds[j + 1]
            length = (b - a) / fps
            if codes[j] == D:
                dead.append((length, segment_type(types, start + a, start + b)))
            elif codes[j] in (H, A):
                team_spells.append(length)
    return {"dead": dead, "team": team_spells, "live_s": live, "changes": changes}


# shape cue -> restart type, in priority order (the first one seen in a stoppage names it)
SHAPE_TYPES = (("kickoff", "kickoff"), ("corner", "corner"), ("wall", "freekick"),
               ("throw", "throw"))


def segment_type(types, a: int, b: int) -> int:
    if not types:
        return -1
    for _, rtype in SHAPE_TYPES:
        k = RESTART_TYPES.index(rtype)
        if k in types and types[k][a:b].any():
            return k
    return -1


def shape_types(match) -> dict:
    """{rtype index: (n,) bool} from the set-piece shapes (players only)."""
    from src.models.setpiece import shape_masks
    masks = shape_masks(match, tuple(s for s, _ in SHAPE_TYPES))
    return {RESTART_TYPES.index(r): masks[s] for s, r in SHAPE_TYPES}


def merge(parts) -> dict:
    out = {"dead": [], "team": [], "live_s": 0.0, "changes": 0, "matches": 0}
    for p in parts:
        out["dead"] += p["dead"]
        out["team"] += p["team"]
        out["live_s"] += p["live_s"]
        out["changes"] += p["changes"]
        out["matches"] += 1
    return out


def quantiles(values) -> list:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return []
    return [round(float(q), 3) for q in np.quantile(values, QUANTILES)]


def summary(seg: dict) -> dict:
    """Prior-table entry: quantiles of team spells, of all stoppages and per restart type, plus
    counts and rates."""
    dead = np.array([d for d, _ in seg["dead"]]) if seg["dead"] else np.zeros(0)
    kinds = np.array([k for _, k in seg["dead"]], dtype=int) if seg["dead"] else np.zeros(0, int)
    keep = dead < 300  # drop absurd gaps (missing events, VAR, injuries over 5 min)
    out = {"matches": seg["matches"], "team": quantiles(seg["team"]), "n_team": len(seg["team"]),
           "dead": quantiles(dead[keep]), "n_dead": int(keep.sum()),
           "changes_per_live_min": round(60 * seg["changes"] / max(seg["live_s"], 1e-9), 3),
           "dead_per_live_min": round(60 * int(keep.sum()) / max(seg["live_s"], 1e-9), 3),
           "by_type": {}}
    for k, name in enumerate(RESTART_TYPES):
        sel = keep & (kinds == k)
        if sel.sum() >= 20:
            out["by_type"][name] = {"n": int(sel.sum()), "q": quantiles(dead[sel])}
    sel = keep & (kinds < 0)
    if sel.sum() >= 20:
        out["by_type"]["none"] = {"n": int(sel.sum()), "q": quantiles(dead[sel])}
    return out


# -- penalties and the duration-aware decoder --------------------------------------------------------

@lru_cache(maxsize=8)
def load_priors(path: str = PRIOR_FILE) -> dict:
    with open(path) as fh:
        return json.load(fh)


def cdf(q: list, lengths: np.ndarray) -> np.ndarray:
    """F(L) from a quantile list (QUANTILES), linear between knots, linear from (0, 0) to the first
    knot, 1 beyond the last."""
    xs = np.concatenate(([0.0], np.asarray(q, float)))
    ps = np.concatenate(([0.0], QUANTILES))
    xs = np.maximum.accumulate(xs + 1e-9 * np.arange(len(xs)))  # strictly increasing
    out = np.interp(lengths, xs, ps)
    return np.where(lengths > xs[-1], 1.0, out)


def penalty(q: list, fps: int, weight: float, floor: float = FLOOR,
            max_age_s: float = MAX_AGE_S) -> np.ndarray:
    """(K + 1,) exit penalty by segment length in frames L = 0..K (index 0 unused), on the grid."""
    k = int(round(max_age_s * fps))
    lengths = np.arange(k + 1) / fps
    g = weight * -np.log(np.maximum(cdf(q, lengths), floor))
    g[0] = 0.0
    return to_grid(g)


def exit_costs(source: str, fps: int, team_w: float, dead_w: float, typed: bool,
               floor: float = FLOOR, path: str = PRIOR_FILE):
    """(G, typed_G): G (3, K + 1) exit penalties for H, A, D by length; typed_G {rtype: (K + 1,)}
    for D segments that contain that type's shape (empty unless typed)."""
    table = load_priors(path)[source]
    k = int(round(MAX_AGE_S * fps))
    g = np.zeros((3, k + 1))
    if team_w:
        g[0] = g[1] = penalty(table["team"], fps, team_w, floor)
    if dead_w:
        g[2] = penalty(table["dead"], fps, dead_w, floor)
    typed_g = {}
    if typed and dead_w:
        for _, rtype in SHAPE_TYPES:
            if rtype in table["by_type"]:
                typed_g[RESTART_TYPES.index(rtype)] = penalty(table["by_type"][rtype]["q"], fps,
                                                              dead_w, floor)
    return g, typed_g


NEG = -2.0 ** 40  # "impossible", exactly on the grid


def viterbi_aged(scores: np.ndarray, switch_cost: float, dead_cost: float, g: np.ndarray,
                 typed_g: dict | None = None, types: dict | None = None) -> np.ndarray:
    """(n,) best state path (0=H, 1=A, 2=D) for (n, 3) per-frame scores, like decoder.viterbi
    (flat switch_cost between H and A, dead_cost between a team and D), plus g[s, L] charged when a
    segment of state s and length L frames ends (L capped at K = g.shape[1] - 1). The final segment
    pays nothing. typed_g / types: a D segment containing a frame of types[k] pays typed_g[k]
    instead (the first k in SHAPE_TYPES order that it contains). One approximation: for D segments
    longer than K the running max ignores which of them contain a shape (it takes the best entry,
    then charges the type that entry's segment has), so typed paths are exact up to K frames.

    Formulation: h_s(tau) = (best value entering s at tau) - P_s(tau), with P_s the prefix sums of
    s's scores. Leaving s at t from entry tau is worth P_s(t) + h_s(tau) - g_s(t - tau): a max over
    the last K entries (one vector op per state and frame), plus a running max for older entries.
    Exact on the 2**-GRID_BITS grid; ties go to the latest entry (the latest change, as viterbi)."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    scores = to_grid(scores)
    g = to_grid(g)
    k = g.shape[1] - 1
    sw, dc = float(to_grid(switch_cost)), float(to_grid(dead_cost))
    cost = np.array([[0.0, sw, dc], [sw, 0.0, dc], [dc, dc, 0.0]])
    prefix = np.vstack([np.zeros((1, 3)), np.cumsum(scores, axis=0)]).T  # (3, n + 1), P_s(t)
    h = np.full((3, n), NEG)
    h[:, 0] = 0.0  # free start in any state
    # g reversed, so that window position j (tau = t - k + 1 + j ... ) lines up with L = t - tau
    grev = g[:, :0:-1]  # (3, k): L = k .. 1
    grev = grev[:, 1:]  # L = k - 1 .. 1, for tau in [t - k + 1, t - 1]
    typed_rev = {r: v[:0:-1][1:] for r, v in (typed_g or {}).items()}
    order = [r for _, r in SHAPE_TYPES_IDX if typed_rev and r in typed_rev and types and r in types]
    last_seen = {}
    if order:
        idx = np.arange(n)
        for r in order:
            last_seen[r] = np.maximum.accumulate(np.where(types[r], idx, -1))
    cap = np.full(3, NEG)
    cap_tau = np.zeros(3, dtype=np.int64)
    src_state = np.zeros((n, 3), dtype=np.int8)
    src_tau = np.zeros((n, 3), dtype=np.int64)
    leave = np.empty(3)
    leave_tau = np.empty(3, dtype=np.int64)
    for t in range(1, n):
        old = t - k  # entries at or before this have L >= k: move into the running max
        if old >= 0:
            better = h[:, old] >= cap  # later entry wins ties
            cap = np.where(better, h[:, old], cap)
            cap_tau = np.where(better, old, cap_tau)
        lo = max(0, t - k + 1)
        g_win = grev[:, k - 1 - (t - lo):]
        if order:
            g_win = g_win.copy()
            for r in reversed(order):  # highest priority last, so it wins
                ls = last_seen[r][t - 1]
                if ls >= lo:  # entries tau <= ls contain the shape
                    g_win[2, :ls - lo + 1] = typed_rev[r][k - 1 - (t - lo):][:ls - lo + 1]
        win = h[:, lo:t] - g_win
        if win.shape[1]:
            j = win.shape[1] - 1 - np.argmax(win[:, ::-1], axis=1)  # last max = latest entry
            best = win[np.arange(3), j]
        else:
            j, best = np.zeros(3, dtype=np.int64), np.full(3, NEG)
        cap_g = g[:, k].copy()
        if order:
            for r in reversed(order):
                if last_seen[r][t - 1] >= 0 and cap_tau[2] <= last_seen[r][t - 1]:
                    cap_g[2] = typed_g[r][k]
        cap_val = cap - cap_g
        use_win = best >= cap_val
        leave[:] = np.where(use_win, best, cap_val) + prefix[:, t]
        leave_tau[:] = np.where(use_win, lo + j, cap_tau)
        for s2 in range(3):
            bs, bv = -1, NEG
            for s in (2, 1, 0) if s2 != 2 else (1, 0):  # fixed order: deterministic ties
                if s == s2:
                    continue
                v = leave[s] - cost[s, s2]
                if v > bv:
                    bs, bv = s, v
            h[s2, t] = bv - prefix[s2, t]
            src_state[t, s2] = bs
            src_tau[t, s2] = leave_tau[bs]
    # final: best open segment, no exit penalty
    final = h + prefix[:, n][:, None]
    flat = final[:, ::-1]
    tau_rev = np.argmax(flat, axis=1)
    vals = flat[np.arange(3), tau_rev]
    s = int(np.argmax(vals))
    tau = n - 1 - int(tau_rev[s])
    path = np.empty(n, dtype=np.int8)
    end = n
    while True:
        path[tau:end] = s
        if tau == 0:
            break
        s, tau, end = int(src_state[tau, s]), int(src_tau[tau, s]), tau
    return path


SHAPE_TYPES_IDX = tuple((s, RESTART_TYPES.index(r)) for s, r in SHAPE_TYPES)
