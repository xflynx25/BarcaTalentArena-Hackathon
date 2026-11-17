"""The ball-dropout grid (backlog C6): train and select across a range of simulated dropout instead of
one setting fit to BAR-ATH's published scores (findings F30, flagged by the organisers' rules email).

Each grid point is a set of src.evaluate load options for src/ingest/degrade.py:
    ball_missing  share of graded frames without a ball: 0.30 / 0.41 / 0.50. 0.41 is BAR-ATH's
                  documented tracking fact [official DATA.md]; 0.30 and 0.50 bracket it.
    ball_quiet    share of the hidden frames that go to the calmest stretches (mostly stoppages):
                  0 (all random, near crowds and fast balls), 0.5, 1 (all in calm stretches first).
    ball_gap_s    median length of the random gaps: 1 s (short, duels and tackles, as DATA.md
                  describes) or 4 s (long). 15 points.
Nothing here was chosen to match a number computed from BAR-ATH's ground truth.

draws(key, k) picks k grid points for a training match, seeded by the match key only, so a fit on
the mix is deterministic and every lomo fold sees the same settings for the matches it keeps.

Players (C11): PLAYER_GRID is a range of player coverage (all players; tracklet gaps; identity
switches; a broadcast camera with off-camera players hidden or estimated), since our match's coverage
is unknown to agents. MIXED is the training mix: every ball point x every player point (the first 15
are GRID itself, with all players), then the clean copy (nothing hidden) last. mixed_draws(key, k)
picks k of them with weights (MIXED_WEIGHTS: clean 1/6, full players 5/12, degraded players 5/12),
seeded by the key. Scoring sets: GRID (ball, full players), PLAYER_EVAL (each degraded player point
at three ball points, 15), CLEAN_EVAL. scripts/fit_mixed.py and scripts/check_player_grid.py use them.
GRID and its 15 points are unchanged, so old comparisons still hold. FRAG_EVAL (C13): tracklet
fragmentation at the same three ball points, 9 points, for scoring only.
"""

from __future__ import annotations

import itertools
import zlib

import numpy as np

MISSING = (0.30, 0.41, 0.50)
QUIET = (0.0, 0.5, 1.0)
GAP_S = (1.0, 4.0)
# At quiet 1 no random gaps are drawn, so the gap length doesn't matter: one point, not two.
GRID = tuple({"ball_missing": m, "ball_quiet": q, "ball_gap_s": g}
             for m, q, g in itertools.product(MISSING, QUIET, GAP_S) if q < 1 or g == GAP_S[0])


# -- players (C11) ------------------------------------------------------------------------------------

CAMERA_SHARE = 0.34  # SkillCorner's view at mid-pitch (0.29 averaged over play, findings F111)
PLAYER_GRID = (
    {},                                                     # every player, as IDSSE and Metrica
    {"player_drop": 0.1},                                   # tracklet gaps, 10% of each slot
    {"player_drop": 0.25},
    {"id_swap": 1.0, "player_drop": 0.1},                   # plus identity switches, 1/team/min
    {"camera": CAMERA_SHARE, "camera_fill": "nan"},         # broadcast view, off-camera hidden
    {"camera": CAMERA_SHARE, "camera_fill": "estimate"},    # ... or estimated (SkillCorner-like)
)
CLEAN = {"clean": 1}
MIXED = tuple(dict(b, **p) for p in PLAYER_GRID for b in GRID) + (CLEAN,)
MIXED_WEIGHTS = np.array([(5 / 12) / len(GRID) if not p else (5 / 12) / (len(PLAYER_GRID) - 1)
                          / len(GRID) for p in PLAYER_GRID for _ in GRID] + [1 / 6])
# scoring: each degraded player point at the middle quiet / short-gap ball point of each missing level
BALL_FOR_PLAYERS = tuple(p for p in GRID if p["ball_quiet"] == 0.5 and p["ball_gap_s"] == 1.0)
PLAYER_EVAL = tuple(dict(b, **p) for p in PLAYER_GRID[1:] for b in BALL_FOR_PLAYERS)
CLEAN_EVAL = (CLEAN,)
# C13: tracklet fragmentation (a track ends and continues in another slot after a gap, as BAR-ATH's
# 46 slots for 22 players), scoring only: not in MIXED, so C11's training draws don't move
FRAG_GRID = (
    {"fragment": 1.0},                                      # 1 break / player / min, 2 s gaps
    {"fragment": 2.0, "fragment_gap_s": 5.0},               # 2 / min, 5 s gaps
    {"fragment": 1.0, "player_drop": 0.1},                  # plus in-slot gaps
)
FRAG_EVAL = tuple(dict(b, **p) for p in FRAG_GRID for b in BALL_FOR_PLAYERS)
EVAL = {"ball": GRID, "player": PLAYER_EVAL, "clean": CLEAN_EVAL, "frag": FRAG_EVAL}


def name(point: dict) -> str:
    """'m0.41 q0.5 g1' for a ball grid point; player options add e.g. ' drop0.1', ' swap1',
    ' cam0.34e' (estimated) / ' cam0.34n' (hidden); the clean copy is 'clean'."""
    if point.get("clean"):
        return "clean"
    # load options added on the command line (check_player_grid.py --add) arrive as strings
    point = {k: (float(v) if isinstance(v, str) and k != "camera_fill" else v) for k, v in point.items()}
    parts = []
    if "ball_missing" in point:
        parts.append("m{:g} q{:g} g{:g}".format(point["ball_missing"], point.get("ball_quiet", 0.0),
                                               point.get("ball_gap_s", 1.0)))
    if point.get("camera"):
        parts.append("cam{:g}{}".format(point["camera"], point.get("camera_fill", "nan")[0]))
    if point.get("id_swap"):
        parts.append("swap{:g}".format(point["id_swap"]))
    if point.get("player_drop"):
        parts.append("drop{:g}".format(point["player_drop"]))
    if point.get("fragment"):
        parts.append("frag{:g}g{:g}".format(point["fragment"], point.get("fragment_gap_s", 2.0)))
    return " ".join(parts) or "full"


def draws(key: str, k: int) -> list[int]:
    """k distinct grid indices for the match `key`, deterministic (seeded by the key alone)."""
    rng = np.random.default_rng(zlib.crc32(("c6|" + key).encode()))
    return sorted(int(i) for i in rng.choice(len(GRID), size=min(k, len(GRID)), replace=False))


def mixed_draws(key: str, k: int) -> list[int]:
    """k distinct MIXED indices for the match `key`, weighted by MIXED_WEIGHTS, seeded by the key."""
    rng = np.random.default_rng(zlib.crc32(("c11|" + key).encode()))
    p = MIXED_WEIGHTS / MIXED_WEIGHTS.sum()
    return sorted(int(i) for i in rng.choice(len(MIXED), size=min(k, len(MIXED)), replace=False, p=p))
