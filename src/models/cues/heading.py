"""Cue `heading` (session of 16 Nov, Part B): a dearer H<->A switch while both teams keep running
the same way.

Why: on BAR-ATH the ball is missing on ~41% of graded frames, and there v7 still invents turnovers
(long balls, a switch in the middle of a spell). After a real turnover both teams turn around; in a
long ball or a mid-possession stretch nobody does. Part A (scripts/check_player_cues.py, findings)
measured how often both teams keep a coherent, unchanged heading at v7's false switches and at its
correct ones; this cue is that "stable" flag, with nothing fitted.

STABLE on a sample (src/player_cues.py at 10 Hz, each team in its own attack frame):
  coh_min    >= STABLE_COH   both teams' outfielders run the same way (speed-weighted mean resultant
                             length of their velocities: |sum v| / sum |v|)
  dhead1_max <= 30 deg       neither team's mean heading moves across [t - 0.5 s, t + 0.5 s]
  dhead2_max <= 45 deg       ... nor across [t - 1 s, t + 1 s]
A team's heading is unknown when its mean velocity is under 1 m/s (standing still has no heading)
or when it has fewer than 7 seen outfielders; then the flag is unknown.

Effect: both H<->A costs times (1 + gain) on frames whose nearest sample is STABLE, 1 elsewhere
(unknown included). Only a dearer switch: never a cheaper one (v7's switch cue already does that,
and switch precision on SkillCorner is 0.33, F339). Players only, so a missing ball does not turn
it off; SkillCorner 2024/25's estimated players stay masked (player_cues.seen_players). gain 0 is
no effect (None); a match where the flag is never known gets multipliers of exactly 1, so v7's
bytes. No fit, no fold: thresholds are Part A's, fixed before any grid.

Runtime: player_cues.team_values once per match (cached on the players array, which hiding the ball
leaves alone), ~1-2 s on a 90-minute match. Deterministic (float64, no randomness).
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.models.cues import Effect

STABLE_COH = 0.6
STABLE_D1 = np.radians(30.0)
STABLE_D2 = np.radians(45.0)

_CACHE: dict = {}


def _key(match) -> str:
    h = hashlib.sha1()
    h.update(match.match_id.encode())
    h.update(np.ascontiguousarray(match.players).tobytes())
    h.update(np.ascontiguousarray(match.period).tobytes())
    return h.hexdigest()


def stable_frames(match) -> np.ndarray:
    """(n,) float: 1 where the nearest 10 Hz sample is STABLE, 0 where it is known and not, NaN
    where unknown."""
    from src import player_cues

    key = _key(match)
    if _CACHE.get("key") != key:
        idx, vals = player_cues.team_values(match)
        pair = player_cues.pair_values(vals)
        coh, d1, d2 = pair["coh_min"], pair["dhead1_max"], pair["dhead2_max"]
        known = ~np.isnan(coh) & ~np.isnan(d1) & ~np.isnan(d2)
        with np.errstate(invalid="ignore"):
            flag = np.where(known, ((coh >= STABLE_COH) & (d1 <= STABLE_D1) & (d2 <= STABLE_D2))
                            .astype(np.float64), np.nan)
        if len(idx):
            step = int(idx[1] - idx[0]) if len(idx) > 1 else 1
            nearest = np.minimum((np.arange(match.n_frames) + step // 2) // step, len(idx) - 1)
            frames = flag[nearest]
        else:
            frames = np.full(match.n_frames, np.nan)
        _CACHE.clear()
        _CACHE.update(key=key, frames=frames)
    return _CACHE["frames"]


def effect(match, original, decoder, gain) -> Effect | None:
    gain = float(gain)
    if gain == 0:
        return None
    if gain < 0:
        raise ValueError("heading: gain must be >= 0 (the cue only makes a switch dearer)")
    stable = stable_frames(match)
    mult = np.where(stable == 1.0, 1.0 + gain, 1.0)
    return Effect.switch(mult=mult)
