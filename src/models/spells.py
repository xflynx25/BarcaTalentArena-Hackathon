"""Smarter short-spell filter on top of a base model (backlog B8, findings F51).

Runs a base model with its own short-spell filter and switch delay turned off, then
postprocess.merge_short (separate minimum lengths for team and D segments; short segments go to the
segment before, the longer neighbour or the neighbour the evidence prefers; optionally keep short
segments the evidence is confident about), then optionally postprocess.delay_switches.

Evidence is the `viterbi` decoder's per-frame scores for H, A, D (src/models/decoder.py, B3's
settings `B3_SETTINGS`, times fps so a full-strength cue is 1): reach-based team evidence and the
dead score. Label-free.

With base="deadball", rule="before", min_team_s == min_dead_s == 2.5 and keep_margin=None this is
exactly `deadball` (B6's drop_short); with base="viterbi" and both lengths 1 s, exactly `viterbi`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.deadball import DeadBall
from src.models.decoder import B3_SETTINGS, SwitchDecoder
from src.models.postprocess import RULES, delay_switches, merge_short, play_runs
from src.models.timing import to_frames
from src.schema import Match

BASES = {
    "viterbi": lambda: SwitchDecoder(min_spell_s=None),
    "deadball": lambda: DeadBall(min_spell_s=0, switch_delay_s=0),
}

# One-slot cache of (match, base) -> (codes, evidence): a tuning grid scores many filter settings on
# the same match object in a row, and the base model (~4 s for viterbi) doesn't depend on them.
# Holding the match itself (compared with `is`) keeps its id from being reused.
_cache: dict = {}


def base_outputs(match: Match, base: str) -> tuple[np.ndarray, np.ndarray]:
    """(codes, evidence) of the base model with its own filter off; evidence (n, 3) for H, A, D."""
    hit = _cache.get(base)
    if hit is not None and hit[0] is match:
        return hit[1], hit[2]
    codes = BASES[base]().predict(match)
    evidence = SwitchDecoder(**B3_SETTINGS).scores(match) * match.fps
    _cache.clear()
    _cache[base] = (match, codes, evidence)
    return codes, evidence


@dataclass(kw_only=True)
class SpellFilter:
    """Base model + merge_short + delay_switches. Durations in seconds, scaled per match."""

    base: str = "viterbi"  # "viterbi" or "deadball", run with their own filter and delay off
    min_team_s: float = 1.0  # H/A segments shorter than this are merged away
    min_dead_s: float = 1.0  # D segments shorter than this are merged away
    rule: str = "before"  # where a short segment goes: "before", "longer" or "evidence"
    keep_margin: float | None = None  # keep short segments with mean evidence margin >= this
    switch_delay_s: float = 0.0  # then move H<->A switches this much later (C2, F43); 0: off

    def __post_init__(self):
        if isinstance(self.keep_margin, str) and self.keep_margin.lower() == "none":
            self.keep_margin = None  # so a grid can say --param keep_margin=none,0.5
        if self.base not in BASES:
            raise ValueError("base must be one of {}, not {!r}".format(sorted(BASES), self.base))
        if self.rule not in RULES:
            raise ValueError("rule must be one of {}, not {!r}".format(RULES, self.rule))

    def predict(self, match: Match) -> np.ndarray:
        codes, evidence = base_outputs(match, self.base)
        runs = play_runs(match.period)
        codes = merge_short(codes, runs, to_frames(self.min_team_s, match.fps),
                            to_frames(self.min_dead_s, match.fps), self.rule, evidence,
                            self.keep_margin)
        if self.switch_delay_s > 0:
            codes = delay_switches(codes, runs, to_frames(self.switch_delay_s, match.fps))
        return codes.astype(np.int8)
