"""Smart Tagging (B28): Metrica's automatic phase and set-piece tags as intervals on a Match, stand-in
tags for external matches, SkillCorner's own phases mapped to the same codes, and the feature
columns the `learned` model's optional "tags" group reads.

BAR-ATH ships Smart Tagging (official/DATA.md: 638 phases and set pieces, SportsCode XML, seconds
from the first video frame). External matches don't, so a model can only learn to read tags from
tags we make ourselves:

  read(...)          the real file -> [Tag], through official/starter/load_smart_tagging.py. Run
                     time input for BAR-ATH only (src/ingest/bar_ath.py fills Match.tags).
  stand_in(match)    stand-in tags from an external match's labels and tracking, with DATA.md's
                     quirks (Team N/A on every throw-in; phases are not possessions; a tagger that
                     misses most restarts) and rates scaled to DATA.md's published counts per code.
                     Needs labels, so it can never run on BAR-ATH.
  perturb(tags, noise)  timing jitter, missed tags, false tags, wrong team (NOISE_LEVELS).
  skillcorner(match) SkillCorner's automatic phases_of_play.csv mapped to the same codes: a second,
                     real tagger, to test whether reading tags transfers between taggers.
  columns(tags, ...) (home, away, common) feature columns for learned's "tags" group.

Our reading: a DEFENDING ... or DEFENSIVE TRANSITION tag carries the defending team, the other
phases the team in possession. Stand-ins follow it. B32: nothing H24 sees comes back to an agent, so
the run time checks it label-free (convention / canonical: the ball's third under each reading) and
flips the defending codes if the file names the other team; line_up is the owner's local check that
the tags parse and line up with the tracking.

Everything here is label-free at run time; nothing is fit on BAR-ATH (backlog G rules).
"""

from __future__ import annotations

import csv
import difflib
import json
import os
import re
import zlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.official import A, D, H, games
from src.schema import AWAY, HOME, Match

NA = -1  # Tag.team for "N/A" or no Team tag

ATTACK = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "CREATING CHANCES",
          "PLAYERS IN THE BOX", "GOALS")
TRANSITION = ("ATTACKING TRANSITION",)
LONG = ("LONG BALL",)
DEFEND = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN ATTACKING THIRD")
DEF_TRANSITION = ("DEFENSIVE TRANSITION",)
SET_PIECES = "SET PIECES"
# Team-block families of the "tags" feature group: (name, codes). A family column is 1 on frames
# inside a tag of one of its codes carrying that block's team.
FAMILIES = (("attack", ATTACK), ("transition", TRANSITION), ("long", LONG), ("defend", DEFEND),
            ("deftransition", DEF_TRANSITION))
SET_KNOTS = (0.0, 0.5, 1.0)  # position inside a SET PIECES window, as a hat basis

# official/DATA.md [official]: instances per code and per set-piece Type over the whole tagged
# file. Stand-ins are thinned to these rates per graded minute (158,857 graded frames at 25 fps).
PUBLISHED = {"PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
             "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
             "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
             "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
             "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4}
PUBLISHED_TYPES = {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                   "Kick Off (After Goal)": 4, "Kick Off (Start)": 2}
PUBLISHED_MINUTES = 158_857 / 25 / 60

# Stand-in geometry and timing.
SET_BEFORE_S, SET_AFTER_S = 3.0, 2.0  # SET PIECES window around the restart (DATA.md: ~5 s)
MIN_PHASE_S = 2.0  # zone runs shorter than this are not tagged
TRANSITION_S = 6.0  # (attacking / defensive) transition tag after an open-play turnover
LONG_BALL_M, LONG_BALL_S = 30.0, 3.0  # a long ball: the ball moves this far within this long
CHANCE_BEFORE_S, CHANCE_AFTER_S = 5.0, 2.0  # CREATING CHANCES around the ball entering the box
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
TOUCHLINE_M, CORNER_M, GOAL_AREA_M, CENTRE_M = 1.5, 3.0, 6.5, 2.5
RESTART_LOOKBACK_S = 2.0  # restart position: last ball seen this far before the restart
BOX_MIN_S, BOX_AFTER_S = 0.4, 1.0  # B32: BALL IN THE BOX from this short a stay, held this long after
PLAYERS_BOX_S = 1.0  # B32: PLAYERS IN THE BOX runs at least this long


class Tag(NamedTuple):
    start: int  # first frame
    end: int  # first frame after (half-open, as the starter loader)
    code: str  # upper case, e.g. "SET PIECES"
    team: int  # H, A or NA
    kind: str = ""  # the Type tag ("Throw-in", ...), "" when none
    labels: tuple = ()  # other (group, text) pairs, sorted


@dataclass(frozen=True)
class Noise:
    """How a tagger can be wrong. jitter_s: sd (s) of independent shifts of each start and end;
    miss: share of tags dropped; false: tags added at random times, as a share of the kept count
    (code, Type and length copied from a random kept tag, team random); wrong: share of team tags
    flipped H <-> A (N/A stays N/A)."""

    jitter_s: float = 0.0
    miss: float = 0.0
    false: float = 0.0
    wrong: float = 0.0


NOISE_LEVELS = {0: Noise(), 1: Noise(1.0, 0.15, 0.1, 0.05), 2: Noise(2.0, 0.3, 0.2, 0.1),
                3: Noise(3.0, 0.45, 0.3, 0.2), 4: Noise(4.0, 0.6, 0.5, 0.3)}


# -- the real file (BAR-ATH) ------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower())


# DATA.md's Team values [official]; BAR-ATH is "BAR vs ATH", Barcelona at home. The fallback when
# the tracking's team names don't decide it.
PUBLISHED_TEAMS = {"barcelona": H, "atleticclub": A}


def team_side(tag_names, team_names) -> dict:
    """{Team tag value: H / A / NA}. Tag spellings differ from the tracking's team names (DATA.md:
    "Atletic Club"), so each value goes to the closer of the two names, else to PUBLISHED_TEAMS;
    "N/A" to NA. Raises (without printing values) if a value can't be placed or both values land on
    the same side."""
    home, away = (_norm(n) for n in team_names)
    out = {}
    for value in tag_names:
        key = _norm(value)
        if key in ("na", ""):
            out[value] = NA
            continue
        score = [difflib.SequenceMatcher(None, key, n).ratio() for n in (home, away)]
        if max(score) >= 0.6 and score[0] != score[1]:
            out[value] = H if score[0] > score[1] else A
        elif key in PUBLISHED_TEAMS:
            out[value] = PUBLISHED_TEAMS[key]
        else:
            raise ValueError("a Smart Tagging Team value matches neither team name")
    sides = [v for v in out.values() if v != NA]
    if len(set(sides)) < len(sides):
        raise ValueError("two Smart Tagging Team values map to the same side")
    return out


def read(team_names, data_dir: str | None = None, game: str = "BAR-ATH",
         fps: int | None = None) -> list[Tag]:
    """The Smart Tagging file of `game` (tactical source) as [Tag], sorted by start. Frames from
    official/starter/load_smart_tagging.py (round(seconds * fps), end exclusive)."""
    from starter import load_smart_tagging  # official/ is on sys.path (src.official)

    events = load_smart_tagging.load(game, data_dir, fps)
    side = team_side({e["tags"].get("Team") for e in events if e["tags"].get("Team")}, team_names)
    out = []
    for e in events:
        tags = dict(e["tags"])
        team = side.get(tags.pop("Team", None), NA)
        kind = tags.pop("Type", "") or ""
        out.append(Tag(int(e["start_frame"]), int(e["end_frame"]), (e["code"] or "").strip().upper(),
                       team, kind.strip(), tuple(sorted(tags.items()))))
    return sorted(out)


def smart_tagging_path(data_dir: str | None = None, game: str = "BAR-ATH") -> str | None:
    """Path of the tactical Smart Tagging file, or None if the bundle has none."""
    path = games.asset_paths(game, data_dir=data_dir, source="tactical").get("smart_tagging")
    return path if path and os.path.isfile(path) else None


# -- stand-in tags ----------------------------------------------------------------------------------

def _runs(values: np.ndarray):
    """[(start, stop, value)] of equal consecutive values."""
    if len(values) == 0:
        return []
    values = np.asarray(values)
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(starts, stops)]


def _filled_ball(match: Match) -> np.ndarray:
    """Ball carried forward over gaps within each period (backward at a period's start)."""
    ball = match.ball.copy()
    for start, stop, p in _runs(match.period):
        if p <= 0:
            continue
        seg = ball[start:stop]
        seen = ~np.isnan(seg[:, 0])
        if not seen.any():
            continue
        idx = np.where(seen, np.arange(len(seg)), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = np.flatnonzero(seen)[0]
        ball[start:stop] = seg[idx]
    return ball


def _progress(x: np.ndarray, sign: np.ndarray, length: float) -> np.ndarray:
    """How far up the pitch x is for a team attacking +x where sign > 0 and -x where sign < 0: 0 at
    its own goal line, 1 at the goal it attacks; NaN where sign is 0 or x is NaN. x is (n,) or
    (n, k) with sign (n,)."""
    sign = sign.reshape(sign.shape + (1,) * (x.ndim - 1)).astype(float)
    u = np.where(sign > 0, x / length, 1.0 - x / length)
    return np.where(sign == 0, np.nan, u)


def _in_box(u: np.ndarray, y: np.ndarray, pitch) -> np.ndarray:
    """In the box the team attacks, from its progress u and the absolute y."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return (u > 1 - BOX_DEPTH_M / length) & (np.abs(y - width / 2) < BOX_HALF_WIDTH_M)


def _restart_kind(match: Match, ball: np.ndarray, r: int, period_start: int, fps: int) -> str:
    """Set-piece Type from where the ball is at the restart (last seen within RESTART_LOOKBACK_S)."""
    length, width = match.pitch
    lo = max(period_start, r - int(RESTART_LOOKBACK_S * fps))
    seen = np.flatnonzero(~np.isnan(match.ball[lo:r + 1, 0]))
    x, y = match.ball[lo + seen[-1]] if len(seen) else ball[r]
    if np.isnan(x):
        return "Free Kick"
    if (min(x, length - x) < CORNER_M) and (min(y, width - y) < CORNER_M):
        return "Corner Kick"
    if min(y, width - y) < TOUCHLINE_M:
        return "Throw-in"
    if min(x, length - x) < GOAL_AREA_M and abs(y - width / 2) < 10.0:
        return "Goal Kick"
    if abs(x - length / 2) < CENTRE_M and abs(y - width / 2) < CENTRE_M:
        return "Kick Off (After Goal)"
    return "Free Kick"


def _zone_tags(code_of: np.ndarray, start: int, team: int, min_len: int) -> list[Tag]:
    """Tags for runs of the same zone code (strings, "" = none) at least min_len long."""
    return [Tag(start + a, start + b, str(c), team) for a, b, c in _runs(code_of)
            if c and b - a >= min_len]


def _players_in_box(match: Match, sign: np.ndarray, s: int, e: int, team: int,
                    final_third: np.ndarray) -> list[Tag]:
    """PLAYERS IN THE BOX candidates (B32): runs of at least PLAYERS_BOX_S where `team` (in
    possession over [s, e)) has the ball in its final third and at least one outfield player in the
    box it attacks, with "Max Players in the box" over the run."""
    side = HOME if team == H else AWAY
    length = match.pitch[0]
    xy = match.players[s:e, match.slot_team == side]
    with np.errstate(invalid="ignore"):
        ux = _progress(xy[:, :, 0], sign[s:e, side], length)
        n_box = _in_box(ux, xy[:, :, 1], match.pitch).sum(axis=1)
    on = final_third & (n_box > 0)
    out = []
    for a, b, inside in _runs(on.astype(np.int8)):
        if inside and b - a >= int(PLAYERS_BOX_S * match.fps):
            most = int(n_box[a:b].max())
            out.append(Tag(s + a, s + b, "PLAYERS IN THE BOX", team, "",
                           (("Max Players in the box", "7+" if most >= 7 else str(most)),)))
    return out


def candidates(match: Match, box_v2: bool = True) -> list[Tag]:
    """Every tag a perfect tagger following our reading of DATA.md would emit, from labels and
    tracking (before thinning to published rates). See stand_in. box_v2 (B32, default): BALL IN
    THE BOX for every stay in the box of at least BOX_MIN_S (not only zone runs of MIN_PHASE_S),
    PLAYERS IN THE BOX whenever attackers are in the box with the ball in the final third (not
    only while the ball is in the box); False is B28's tagger, whose box codes ran at a third / half
    of DATA.md's counts (F118)."""
    if match.labels is None:
        raise ValueError("stand-in tags need labels: external matches only")
    fps, labels, period = match.fps, match.labels, match.period
    length, width = match.pitch
    ball = _filled_ball(match)
    from src.shape import attack_sign

    sign = attack_sign(match)
    side = {H: HOME, A: AWAY}
    u = {t: _progress(ball[:, 0], sign[:, side[t]], length) for t in (H, A)}
    box = {t: _in_box(u[t], ball[:, 1], match.pitch) for t in (H, A)}
    min_len = max(1, int(MIN_PHASE_S * fps))
    tags = []
    for p_start, p_stop, p in _runs(period):
        if p <= 0:
            continue
        lab = labels[p_start:p_stop]
        runs = [(p_start + a, p_start + b, int(v)) for a, b, v in _runs(lab)]
        live = [r for r in runs if r[2] in (H, A)]
        if live:  # kick-off at the first live frame of the period
            r = live[0][0]
            tags.append(Tag(max(p_start, r - int(SET_BEFORE_S * fps)),
                            min(p_stop, r + int(SET_AFTER_S * fps)), SET_PIECES, live[0][2],
                            "Kick Off (Start)"))
        for i, (s, e, v) in enumerate(runs):
            prev = runs[i - 1] if i else None
            if v == D or v not in (H, A):
                continue
            other = A if v == H else H
            if prev is not None and prev[2] == D:  # a restart
                kind = _restart_kind(match, ball, s, p_start, fps)
                tags.append(Tag(max(p_start, s - int(SET_BEFORE_S * fps)),
                                min(p_stop, s + int(SET_AFTER_S * fps)), SET_PIECES,
                                NA if kind == "Throw-in" else v, kind))
            if prev is not None and prev[2] == other:  # open-play turnover
                end = min(e, s + int(TRANSITION_S * fps))
                tags.append(Tag(s, end, "ATTACKING TRANSITION", v))
                tags.append(Tag(s, end, "DEFENSIVE TRANSITION", other))
            # in possession: where the ball is, for the team with it
            uv, bv = u[v][s:e], box[v][s:e]
            zone = np.full(e - s, "", dtype=object)
            with np.errstate(invalid="ignore"):
                zone[uv < 1 / 3] = "BUILD UP"
                zone[(uv >= 1 / 3) & (uv < 2 / 3)] = "PROGRESSION"
                zone[uv >= 2 / 3] = "BALL IN FINAL THIRD"
            if not box_v2:
                zone[bv] = "BALL IN THE BOX"
            tags += _zone_tags(zone, s, v, min_len)
            # out of possession: where the ball is, in the defending team's frame
            uo = u[other][s:e]
            zone = np.full(e - s, "", dtype=object)
            with np.errstate(invalid="ignore"):
                zone[uo < 1 / 3] = "DEFENDING IN DEFENSIVE THIRD"
                zone[(uo >= 1 / 3) & (uo < 2 / 3)] = "DEFENDING IN MIDDLE THIRD"
                zone[uo >= 2 / 3] = "DEFENDING IN ATTACKING THIRD"
            tags += _zone_tags(zone, s, other, min_len)
            # the box: creating chances on the way in, players in the box while there
            for a, b, inside in _runs(bv.astype(np.int8)):
                if not inside:
                    continue
                tags.append(Tag(max(s, s + a - int(CHANCE_BEFORE_S * fps)),
                                min(e, s + b + int(CHANCE_AFTER_S * fps)), "CREATING CHANCES", v))
                if box_v2:
                    if b - a >= int(BOX_MIN_S * fps):
                        tags.append(Tag(s + a, min(e, s + b + int(BOX_AFTER_S * fps)),
                                        "BALL IN THE BOX", v))
                    continue
                if b - a >= fps:
                    xy = match.players[s + a:s + b, match.slot_team == side[v]]
                    ux = _progress(xy[:, :, 0], sign[s + a:s + b, side[v]], length)
                    in_box = _in_box(ux, xy[:, :, 1], match.pitch)
                    most = int(in_box.sum(axis=1).max()) if len(in_box) else 0
                    if most:
                        tags.append(Tag(s + a, s + b, "PLAYERS IN THE BOX", v, "",
                                        (("Max Players in the box", "7+" if most >= 7
                                          else str(most)),)))
            if box_v2:  # B32: attackers in the box while the ball is in the final third
                tags += _players_in_box(match, sign, s, e, v, uv >= 2 / 3)
            # long balls: the (seen) ball moves LONG_BALL_M within LONG_BALL_S
            k = int(LONG_BALL_S * fps)
            seg = match.ball[s:e]
            if len(seg) > k:
                with np.errstate(invalid="ignore"):
                    far = np.hypot(*(seg[k:] - seg[:-k]).T) >= LONG_BALL_M
                last = -10 ** 9
                for j in np.flatnonzero(far):
                    if j - last >= 2 * k:
                        tags.append(Tag(s + int(j), s + int(j) + k, "LONG BALL", v))
                        last = j
    return sorted(tags)


def _thin(tags: list[Tag], minutes: float, rng) -> list[Tag]:
    """Keep each code (each Type, for SET PIECES) at its published rate per graded minute, chosen at
    random where there are more candidates; Kick Off (Start) always stays."""
    out = []
    groups = {}
    for t in tags:
        key = (t.code, t.kind) if t.code == SET_PIECES else (t.code, "")
        groups.setdefault(key, []).append(t)
    for key in sorted(groups):
        items = groups[key]
        code, kind = key
        if code == SET_PIECES and kind == "Kick Off (Start)":
            out += items
            continue
        rate = (PUBLISHED_TYPES.get(kind, 0) if code == SET_PIECES else PUBLISHED.get(code, 0))
        target = int(round(rate / PUBLISHED_MINUTES * minutes))
        if len(items) > target:
            keep = np.sort(rng.choice(len(items), size=target, replace=False))
            items = [items[i] for i in keep]
        out += items
    return sorted(out)


def perturb(tags: list[Tag], noise: Noise, match: Match, rng) -> list[Tag]:
    """tags with the tagger's errors: misses, wrong teams, timing jitter, false tags."""
    fps, n = match.fps, match.n_frames
    out = []
    for t in tags:
        if noise.miss and rng.random() < noise.miss:
            continue
        team = t.team
        if noise.wrong and team != NA and rng.random() < noise.wrong:
            team = A if team == H else H
        start, end = t.start, t.end
        if noise.jitter_s:
            start += int(round(rng.normal(0, noise.jitter_s) * fps))
            end += int(round(rng.normal(0, noise.jitter_s) * fps))
        start, end = max(0, start), min(n, end)
        if end - start < fps:
            end = min(n, start + fps)
        if end > start:
            out.append(t._replace(start=start, end=end, team=team))
    if noise.false and out:
        playing = np.flatnonzero(match.period > 0)
        for _ in range(int(round(noise.false * len(out)))):
            src = out[int(rng.integers(len(out)))]
            start = int(playing[int(rng.integers(len(playing)))])
            team = NA if src.team == NA else int(rng.choice([H, A]))
            out.append(src._replace(start=start, end=min(n, start + src.end - src.start), team=team))
    return sorted(out)


def stand_in(match: Match, noise: Noise | int = 0, seed: int = 0, thin: bool = True,
             box_v2: bool = True) -> list[Tag]:
    """Stand-in Smart Tagging for a labelled external match: candidates() thinned to DATA.md's
    published rate per code (thin=False keeps every candidate: the oracle), then perturb()ed by
    `noise` (a Noise or a NOISE_LEVELS key). Deterministic: seeded by the match id and `seed`.
    Defending codes carry the defending team (our reading; flip_defending gives the other one)."""
    noise = NOISE_LEVELS[noise] if isinstance(noise, int) else noise
    rng = np.random.default_rng(zlib.crc32("b28|{}|{}".format(match.match_id, seed).encode()))
    tags = candidates(match, box_v2)
    if thin:
        tags = _thin(tags, float((match.period > 0).sum()) / match.fps / 60, rng)
    return perturb(tags, noise, match, rng)


# -- SkillCorner's own phases of play: a second, real tagger ----------------------------------------

SC_IN = {"build_up": "BUILD UP", "create": "PROGRESSION", "finish": "BALL IN FINAL THIRD",
         "direct": "LONG BALL", "transition": "ATTACKING TRANSITION",
         "quick_break": "ATTACKING TRANSITION", "set_play": SET_PIECES}  # chaotic: no tag
SC_OUT = {"high_block": "DEFENDING IN ATTACKING THIRD", "medium_block": "DEFENDING IN MIDDLE THIRD",
          "low_block": "DEFENDING IN DEFENSIVE THIRD",
          "defending_transition": "DEFENSIVE TRANSITION",
          "defending_quick_break": "DEFENSIVE TRANSITION"}


def skillcorner(match: Match, root: str | None = None) -> list[Tag]:
    """SkillCorner's phases_of_play.csv for a SkillCorner match, as [Tag]: each phase gives the team in
    possession a tag (SC_IN; "finish" ending in the box -> BALL IN THE BOX) and the other team one
    (SC_OUT). Frames are SkillCorner frame numbers minus the tracking's first frame (Match row 0)."""
    from src.ingest import skillcorner as sc

    source, match_id = match.match_id.split(":", 1)
    if source != "skillcorner":
        raise ValueError("SkillCorner phases exist for SkillCorner matches only")
    root = root or sc.ROOT
    meta_path, raw_path = sc._paths(match_id, root)
    with open(meta_path) as handle:
        meta = json.load(handle)
    side = {meta["home_team"]["id"]: H, meta["away_team"]["id"]: A}
    with open(raw_path) as handle:
        first = json.loads(handle.readline())["frame"]
    path = os.path.join(os.path.dirname(meta_path), "{}_phases_of_play.csv".format(match_id))
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            team = side.get(int(row["team_in_possession_id"]), NA)
            start = int(row["frame_start"]) - first
            end = int(row["frame_end"]) - first
            if team == NA or end <= start or start < 0 or end > match.n_frames:
                continue
            code = SC_IN.get(row["team_in_possession_phase_type"])
            if code == "BALL IN FINAL THIRD" and row.get("penalty_area_end") == "True":
                code = "BALL IN THE BOX"
            if code:
                out.append(Tag(start, end, code, team))
            code = SC_OUT.get(row["team_out_of_possession_phase_type"])
            if code:
                out.append(Tag(start, end, code, A if team == H else H))
    return sorted(out)


# -- which team a defending tag names (B32) ---------------------------------------------------------
#
# DATA.md doesn't say whether DEFENDING IN <third> and DEFENSIVE TRANSITION carry the defending team
# (our reading, what stand-ins and SkillCorner's mapping emit) or the team in possession. Nothing
# H24 sees on our match comes back to an agent (CLAUDE.md), so the run time decides it label-free
# from the tags and the tracking, like attack_sign decides the attack direction: under the right
# reading a DEFENDING IN DEFENSIVE THIRD tag has the ball in its team's own third and DEFENDING IN
# ATTACKING THIRD in its team's final third, and a defending tag overlaps the other team's
# attacking tags. Middle-third tags can't tell the readings apart and don't vote.

CONVENTIONS = ("defender", "attacker")
DEFENDING_CODES = DEFEND + DEF_TRANSITION


def flip_defending(tags: list[Tag]) -> list[Tag]:
    """tags with H <-> A swapped on the defending codes: the other reading of the Team tag."""
    return [t._replace(team=A if t.team == H else H)
            if t.code in DEFENDING_CODES and t.team in (H, A) else t for t in tags]


def convention_votes(tags: list[Tag], match: Match) -> tuple[float, float]:
    """(defender, attacker): seen-ball frames inside DEFENDING IN DEFENSIVE / ATTACKING THIRD tags
    whose ball zone fits each reading, plus frames where a defending tag overlaps an attacking-family
    tag of the other team (defender) or of its own team (attacker). Label-free."""
    from src.shape import attack_sign

    n = match.n_frames
    sign = attack_sign(match)
    length = match.pitch[0]
    side = {H: HOME, A: AWAY}
    attacking = {c for _, codes in FAMILIES[:3] for c in codes}
    own = {H: np.zeros(n, bool), A: np.zeros(n, bool)}
    for t in tags:
        if t.code in attacking and t.team in (H, A):
            own[t.team][max(0, t.start):min(n, t.end)] = True
    vote_def = vote_att = 0.0
    for t in tags:
        if t.code not in DEFENDING_CODES or t.team not in (H, A):
            continue
        a, b = max(0, t.start), min(n, t.end)
        if b <= a:
            continue
        other = A if t.team == H else H
        vote_def += float(own[other][a:b].sum())
        vote_att += float(own[t.team][a:b].sum())
        if t.code not in ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN ATTACKING THIRD"):
            continue
        with np.errstate(invalid="ignore"):
            u = _progress(match.ball[a:b, 0], sign[a:b, side[t.team]], length)
        u = u[~np.isnan(u)]
        own_third, final_third = float((u < 1 / 3).sum()), float((u > 2 / 3).sum())
        if t.code == "DEFENDING IN DEFENSIVE THIRD":
            vote_def, vote_att = vote_def + own_third, vote_att + final_third
        else:
            vote_def, vote_att = vote_def + final_third, vote_att + own_third
    return vote_def, vote_att


def convention(tags: list[Tag] | None, match: Match) -> str:
    """"defender" or "attacker": the reading of the defending codes' Team tag the votes favour
    (convention_votes); "defender" (ours) on a tie or without votes."""
    if not tags:
        return "defender"
    vote_def, vote_att = convention_votes(tags, match)
    return "attacker" if vote_att > vote_def else "defender"


def canonical(tags: list[Tag] | None, match: Match) -> list[Tag] | None:
    """tags in our reading (defending codes carry the defending team): flipped if convention() says
    the tagger names the team in possession. None stays None."""
    if not tags:
        return tags
    return flip_defending(tags) if convention(tags, match) == "attacker" else tags


# -- do the tags line up with the tracking? (B32, the owner's local check) --------------------------

ZONES = {"BUILD UP": 0, "PROGRESSION": 1, "BALL IN FINAL THIRD": 2, "BALL IN THE BOX": 2,
         "DEFENDING IN ATTACKING THIRD": 2, "DEFENDING IN MIDDLE THIRD": 1,
         "DEFENDING IN DEFENSIVE THIRD": 0}  # the ball's third, in the tag team's attack direction
KICKOFF_NEAR_S = 60.0  # every Kick Off (Start) tag starts within this of a period's first frame
# (stand-ins put it at the first live frame: up to 32 s after the period start on Metrica)
ZONE_FLOOR = 0.5  # zone tags agree with the ball's third on at least this share (chance ~1/3) ...
ZONE_SHIFT_S = 10.0  # ... and better than with the tags shifted this far either way: on 30
# external matches offset 0 beats +-10 s on every match for clean, noise-4 and SkillCorner tags (F130)


def zone_agreement(tags: list[Tag], match: Match, shift: int = 0) -> float:
    """Share of seen-ball frames inside zone tags (ZONES, team H or A, shifted by `shift` frames)
    where the ball is in the tag's third, in the tag team's attack direction. Label-free."""
    from src.shape import attack_sign

    n, length = match.n_frames, match.pitch[0]
    sign = attack_sign(match)
    side = {H: HOME, A: AWAY}
    hit = tot = 0
    for t in tags:
        if t.code not in ZONES or t.team not in (H, A):
            continue
        a, b = max(0, t.start + shift), min(n, t.end + shift)
        if b <= a:
            continue
        with np.errstate(invalid="ignore"):
            u = _progress(match.ball[a:b, 0], sign[a:b, side[t.team]], length)
        u = u[~np.isnan(u)]
        third = np.minimum((u * 3).astype(int), 2)
        hit += int((third == ZONES[t.code]).sum())
        tot += len(u)
    return hit / tot if tot else float("nan")


def line_up(tags: list[Tag] | None, match: Match, kickoffs=None) -> dict:
    """The owner's local check (B32): {check: bool} for "parse" (tags read, inside the match, known
    codes), "teams" (both teams named), "kickoff" (with `kickoffs`, the true kickoff frames, each
    lies inside a Kick Off (Start) tag: the official starter's check_time_base; without, there are
    Kick Off (Start) tags and each starts within KICKOFF_NEAR_S of a period's first frame) and "zones" (after canonical(), zone tags agree with the
    ball's third on at least ZONE_FLOOR of seen frames and better than shifted ZONE_SHIFT_S either
    way). Label-free; the caller prints only the check names and PASS / FAIL."""
    known = set(PUBLISHED) | {SET_PIECES}
    ok = {"parse": bool(tags) and all(0 <= t.start < t.end <= match.n_frames for t in tags)
          and sum(t.code in known for t in tags) >= 0.9 * len(tags)}
    tags = tags or []
    ok["teams"] = {H, A} <= {t.team for t in tags}
    starts = [t.start for t in tags if t.code == SET_PIECES and t.kind == "Kick Off (Start)"]
    near = int(KICKOFF_NEAR_S * match.fps)
    firsts = [a for a, _, p in _runs(match.period) if p > 0]
    if kickoffs is not None:
        spans = [(t.start, t.end) for t in tags if t.code == SET_PIECES and t.kind == "Kick Off (Start)"]
        ok["kickoff"] = bool(kickoffs) and all(any(a <= k < b for a, b in spans) for k in kickoffs)
    else:
        ok["kickoff"] = bool(starts) and all(any(abs(s - f) <= near for f in firsts) for s in starts)
    tags = canonical(tags, match) or []
    shift = int(ZONE_SHIFT_S * match.fps)
    here = zone_agreement(tags, match)
    away = [zone_agreement(tags, match, d) for d in (-shift, shift)]
    ok["zones"] = bool(here >= ZONE_FLOOR and all(not here <= v for v in away))
    return ok


# -- feature columns --------------------------------------------------------------------------------

def columns(tags: list[Tag] | None, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(home (n, p), away (n, p), common (n, c)) for learned's "tags" group. Team blocks: one column
    per FAMILIES entry (inside a tag of that family carrying the block's team), then that team's SET
    PIECES windows as a hat basis on the position inside the window (SET_KNOTS; the restart is
    near 0.6). Common: the same hat for SET PIECES with Team N/A (throw-ins), and "inside any tag".
    No tags (None or []) gives all-zero columns: no evidence."""
    p = len(FAMILIES) + len(SET_KNOTS)
    home, away = np.zeros((n, p)), np.zeros((n, p))
    common = np.zeros((n, len(SET_KNOTS) + 1))
    family_of = {c: i for i, (_, codes) in enumerate(FAMILIES) for c in codes}
    k = len(SET_KNOTS)
    for t in tags or ():
        a, b = max(0, t.start), min(n, t.end)
        if b <= a:
            continue
        common[a:b, -1] = 1.0
        if t.code == SET_PIECES:
            pos = (np.arange(a, b) - t.start + 0.5) / max(1, t.end - t.start)
            basis = _hat(pos, SET_KNOTS)
            if t.team == NA:
                target = common[a:b, :k]
            else:
                target = (home if t.team == H else away)[a:b, len(FAMILIES):]
            np.maximum(target, basis, out=target)
        elif t.code in family_of and t.team != NA:
            (home if t.team == H else away)[a:b, family_of[t.code]] = 1.0
    return home, away, common


def _hat(values: np.ndarray, knots) -> np.ndarray:
    from src.models.learned import hat  # local: learned imports this module

    return hat(values, knots)


def counts(tags: list[Tag], minutes: float) -> dict:
    """{code: instances per PUBLISHED_MINUTES} for comparing a tagger with DATA.md's counts."""
    out = {}
    for t in tags:
        out[t.code] = out.get(t.code, 0) + 1
    return {c: v * PUBLISHED_MINUTES / minutes for c, v in out.items()}


def team_agreement(tags: list[Tag], labels: np.ndarray) -> dict:
    """External matches only. For tags with a team, the share of their H/A frames where the tag's
    team (attacking families) or the other team (defending families) has the ball; and the share of
    H/A frames inside such a tag. A tagger quality measure, label-based."""
    n = len(labels)
    live = (labels == H) | (labels == A)
    attack = {c for _, codes in FAMILIES[:3] for c in codes} | {SET_PIECES}
    hit = tot = 0
    covered = np.zeros(n, bool)
    for t in tags:
        if t.team == NA or (t.code not in attack and t.code not in DEFEND + DEF_TRANSITION):
            continue
        a, b = max(0, t.start), min(n, t.end)
        seg = labels[a:b][live[a:b]]
        holder = t.team if t.code in attack else (A if t.team == H else H)
        hit += int((seg == holder).sum())
        tot += len(seg)
        covered[a:b] = True
    return {"agree": hit / tot if tot else float("nan"),
            "coverage": float(covered[live].mean()) if live.any() else float("nan")}
