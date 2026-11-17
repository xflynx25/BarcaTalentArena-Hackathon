"""Are the players there? Per-frame player coverage, for team-shape features (backlog C10, F92-F93).

Team shape (B23-B25) needs both teams' outfield players. This module says, per frame and team, how
many are present, which one is the goalkeeper, and which way the team attacks, all label-free from
a Match, so the shape code can mask or down-weight frames with missing players:

    cov = coverage(match)              # or coverage(match, seen=detected) for SkillCorner, see below
    cov.mask(min_outfield=10)          # (n,) bool: both teams have >= 10 outfield players
    cov.weight()                       # (n,) float in [0, 1]: min over teams of outfield / 10

Conventions it relies on (src/schema.py): metres, corner origin, NaN = not observed, slots are
tracklets that keep one team. Nothing here assumes a slot is a person, so it works on BAR-ATH's 46
tracklet slots as well as on the external sources' one-slot-per-player layout.

Attack direction, per period: the team whose players' mean x is lower defends the x = 0 goal. Teams
swap ends at half-time on every source we have, but this is measured per period, not assumed.

Goalkeeper, label-free: per frame, each team's player nearest the centre of its own goal is its
"deepest" player. A slot is a keeper slot when it is the deepest on at least KEEPER_SHARE of the
in-period frames it is present (and present for at least KEEPER_MIN_S seconds). A frame has its
keeper when a keeper slot of that team is present, or, failing that (tracklet sources lose the
keeper's slot), when the deepest player is within KEEPER_BOX_M of the goal centre.

SkillCorner is broadcast tracking: off-camera players are estimated (`is_detected` false), not
missing, so a plain count says everyone is there. skillcorner_extras() reads the detection flags
and the camera view (`image_corners_projection`) in the loader's slot layout; pass its `detected`
as `seen` to count only players the camera saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.schema import AWAY, HOME, Match

KEEPER_SHARE = 0.5   # a keeper slot is the deepest on at least this share of its frames
KEEPER_MIN_S = 10.0  # ... and present at least this long
KEEPER_BOX_M = 20.0  # fallback: deepest player this close to the goal centre counts as the keeper
OUTFIELD = 10        # a full team's outfield players
OFF_PITCH_M = 3.0    # positions further than this outside the lines count as off the pitch


@dataclass
class Coverage:
    present: np.ndarray     # (n, 2) int: players present per team (HOME, AWAY), keeper included
    outfield: np.ndarray    # (n, 2) int: present minus the keeper when the keeper is there
    keeper: np.ndarray      # (n, 2) bool: the team's keeper is present
    keeper_slot: np.ndarray  # (n, 2) int: slot of the keeper, -1 when not identified
    deepest: np.ndarray     # (n, 2) int: slot of the player nearest the own goal centre, -1 if none
    own_goal_x: np.ndarray  # (n, 2) float: x of each team's own goal line, NaN outside periods
    keeper_slots: tuple[np.ndarray, np.ndarray]  # keeper slot indices, per team
    in_period: np.ndarray   # (n,) bool

    @property
    def attack_sign(self) -> np.ndarray:
        """(n, 2): +1 where the team attacks towards +x, -1 towards x = 0, 0 outside periods."""
        sign = np.where(self.own_goal_x == 0.0, 1, -1)
        return np.where(np.isnan(self.own_goal_x), 0, sign).astype(np.int8)

    def mask(self, min_outfield: int = OUTFIELD) -> np.ndarray:
        """(n,) bool: in a period and both teams have at least min_outfield outfield players."""
        return self.in_period & (self.outfield.min(axis=1) >= min_outfield)

    def weight(self, full: int = OUTFIELD) -> np.ndarray:
        """(n,) float in [0, 1]: min over teams of outfield / full, 0 outside periods."""
        w = np.clip(self.outfield.min(axis=1) / float(full), 0.0, 1.0)
        return np.where(self.in_period, w, 0.0)


def _observed(match: Match, seen: np.ndarray | None) -> np.ndarray:
    ok = ~np.isnan(match.players).any(axis=2)
    if seen is not None:
        if seen.shape != ok.shape:
            raise ValueError("seen shape {} != players {}".format(seen.shape, ok.shape))
        ok &= seen
    return ok


def own_goal_x(match: Match, observed: np.ndarray) -> np.ndarray:
    """(n, 2): x of each team's own goal line per frame (0 or pitch length), NaN outside periods
    or in a period where a team has no positions."""
    length = float(match.pitch[0])
    out = np.full((match.n_frames, 2), np.nan)
    x = np.where(observed, match.players[:, :, 0], np.nan)
    for p in np.unique(match.period[match.period > 0]):
        rows = match.period == p
        means = []
        for team in (HOME, AWAY):
            values = x[rows][:, match.slot_team == team]
            means.append(np.nanmean(values) if np.isfinite(values).any() else np.nan)
        if np.isnan(means).any():
            continue
        home_left = means[HOME] < means[AWAY]
        out[rows, HOME] = 0.0 if home_left else length
        out[rows, AWAY] = length if home_left else 0.0
    return out


def _deepest(match: Match, observed: np.ndarray, goal_x: np.ndarray) -> np.ndarray:
    """(n, 2) slot nearest each team's own goal centre, -1 if the team has nobody there."""
    out = np.full((match.n_frames, 2), -1, dtype=np.int64)
    centre_y = match.pitch[1] / 2.0
    for team in (HOME, AWAY):
        slots = np.flatnonzero(match.slot_team == team)
        if len(slots) == 0:
            continue
        xy = match.players[:, slots]
        dist = np.hypot(xy[:, :, 0] - goal_x[:, team:team + 1], xy[:, :, 1] - centre_y)
        dist = np.where(observed[:, slots] & np.isfinite(dist), dist, np.inf)
        best = np.argmin(dist, axis=1)
        found = np.isfinite(dist[np.arange(len(best)), best])
        out[found, team] = slots[best[found]]
    return out


def coverage(match: Match, seen: np.ndarray | None = None) -> Coverage:
    """Per-frame coverage of a Match. seen (n, k) bool optionally narrows which positions count
    (e.g. SkillCorner's detected flags); NaN positions never count."""
    observed = _observed(match, seen)
    in_period = match.period > 0
    observed &= in_period[:, None]
    goal_x = own_goal_x(match, observed)
    deepest = _deepest(match, observed, goal_x)

    k = len(match.slot_team)
    frames_present = observed.sum(axis=0)
    frames_deepest = np.zeros(k, dtype=np.int64)
    for team in (HOME, AWAY):
        d = deepest[:, team]
        frames_deepest += np.bincount(d[d >= 0], minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = frames_deepest / frames_present
    is_keeper = (frames_present >= KEEPER_MIN_S * match.fps) & (share >= KEEPER_SHARE)
    keeper_slots = tuple(np.flatnonzero(is_keeper & (match.slot_team == t)) for t in (HOME, AWAY))

    n = match.n_frames
    present = np.zeros((n, 2), dtype=np.int64)
    keeper = np.zeros((n, 2), dtype=bool)
    keeper_slot = np.full((n, 2), -1, dtype=np.int64)
    centre_y = match.pitch[1] / 2.0
    rows = np.arange(n)
    for team in (HOME, AWAY):
        present[:, team] = observed[:, match.slot_team == team].sum(axis=1)
        ks = keeper_slots[team]
        if len(ks):
            here = observed[:, ks]
            has = here.any(axis=1)
            keeper[:, team] = has
            keeper_slot[has, team] = ks[np.argmax(here[has], axis=1)]
        d = deepest[:, team]
        fallback = ~keeper[:, team] & (d >= 0)
        xy = match.players[rows, np.maximum(d, 0)]
        near = np.hypot(xy[:, 0] - goal_x[:, team], xy[:, 1] - centre_y) <= KEEPER_BOX_M
        fallback &= np.nan_to_num(near, nan=False).astype(bool)
        keeper[fallback, team] = True
        keeper_slot[fallback, team] = d[fallback]
    outfield = present - keeper.astype(np.int64)
    return Coverage(present=present, outfield=outfield, keeper=keeper, keeper_slot=keeper_slot,
                    deepest=deepest, own_goal_x=goal_x, keeper_slots=keeper_slots,
                    in_period=in_period)


def off_pitch_share(match: Match, margin: float = OFF_PITCH_M) -> float:
    """Share of in-period player positions more than margin metres outside the lines."""
    xy = match.players[match.period > 0].reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) == 0:
        return float("nan")
    length, width = match.pitch
    out = ((xy[:, 0] < -margin) | (xy[:, 0] > length + margin)
           | (xy[:, 1] < -margin) | (xy[:, 1] > width + margin))
    return float(out.mean())


def summary(cov: Coverage, fps: int) -> dict:
    """Label-free coverage numbers over in-period frames (means over the two teams)."""
    rows = cov.in_period
    if not rows.any():
        raise ValueError("no in-period frames")
    of = cov.outfield[rows]
    sizes = [len(cov.keeper_slots[t]) for t in (HOME, AWAY)]
    return {
        "present": float(cov.present[rows].mean()),
        "outfield>=9": float((of >= 9).mean()),
        "outfield>=10": float((of >= 10).mean()),
        "outfield>10": float((of > 10).mean()),
        "both>=10": float((of.min(axis=1) >= 10).mean()),
        "keeper": float(cov.keeper[rows].mean()),
        "keeper slots": float(np.mean(sizes)),
        "minutes": float(rows.sum()) / fps / 60.0,
    }


# -- SkillCorner: detection flags and camera view, in the loader's slot layout ----------------------

def _inside_quads(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(n, m) bool: points (m, 2) inside each frame's quad (n, 4, 2), any winding; NaN quads -> False.
    A point is inside a convex quad when it is on the same side of all four edges."""
    a = quads
    b = np.roll(quads, -1, axis=1)
    edge = (b - a)[:, :, None, :]                       # (n, 4, 1, 2)
    rel = points[None, None, :, :] - a[:, :, None, :]    # (n, 4, m, 2)
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    with np.errstate(invalid="ignore"):
        inside = (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)
    return inside & np.isfinite(quads).all(axis=(1, 2))[:, None]


def skillcorner_extras(match_id: str | int, root: str | None = None) -> dict:
    """{detected (n, k) bool, view (n, 4, 2) camera corners in corner-origin metres (NaN where none),
    keeper_ids {player id} with role GK, slot_ids, home_side {period: side string or None}} for one
    SkillCorner match, in the same row and slot layout as skillcorner.load()."""
    from src.ingest import skillcorner

    meta_path, raw_path = skillcorner._paths(str(match_id), root or skillcorner.ROOT)
    meta = skillcorner.read_meta(meta_path)
    with open(meta_path) as handle:
        raw_meta = json.load(handle)
    keeper_ids = {p["id"] for p in raw_meta["players"]
                  if (p.get("player_role") or {}).get("acronym") == "GK"}
    length, width = meta["pitch"]
    offset = np.array([length / 2, width / 2])
    corners = ("top_left", "bottom_left", "bottom_right", "top_right")

    views, rows, ids, flags = [], [], [], []
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row = len(views)
            cam = record.get("image_corners_projection") or {}
            quad = [(cam.get("x_" + c), cam.get("y_" + c)) for c in corners]
            views.append([(np.nan, np.nan) if None in q else q for q in quad])
            for p in record.get("player_data") or ():
                if p.get("x") is None:
                    continue
                rows.append(row)
                ids.append(p["player_id"])
                flags.append(bool(p.get("is_detected")))
    slot_ids = sorted(set(ids), key=lambda i: (meta["player_team"][i], i))
    slot_of = {player_id: slot for slot, player_id in enumerate(slot_ids)}
    detected = np.zeros((len(views), len(slot_ids)), dtype=bool)
    detected[np.asarray(rows, dtype=np.int64),
             np.fromiter((slot_of[i] for i in ids), dtype=np.int64, count=len(ids))] = flags
    view = np.asarray(views, dtype=np.float64).reshape(-1, 4, 2) + offset
    return {"detected": detected, "view": view, "keeper_ids": keeper_ids, "slot_ids": slot_ids,
            "home_side": raw_meta.get("home_team_side")}


def view_share(view: np.ndarray, pitch, step: float = 3.0) -> np.ndarray:
    """(n,) share of the pitch inside the camera view per frame (grid of points every step m)."""
    length, width = pitch
    gx, gy = np.meshgrid(np.arange(step / 2, length, step), np.arange(step / 2, width, step))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    out = np.empty(len(view))
    for start in range(0, len(view), 5000):
        out[start:start + 5000] = _inside_quads(grid, view[start:start + 5000]).mean(axis=1)
    return out
