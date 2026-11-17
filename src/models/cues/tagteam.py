"""Cue `tagteam` (B40, wave 5): Smart Tagging's phases as a team prior at decode time.

Why: every tagged phase carries a team, and a model that reads them gains a lot when they are good
(F119: +0.171 ceiling with perfect stand-ins). Inside v4's small learned model the tags only tied
(v5t, F126), but the restart rule (B35) shows that tags help v6 when they act at decode time. So
this cue leaves v6's emission models alone and adds, inside each tagged phase with a team, `gain`
evidence per second to that team's H / A score, tapered at the phase's edges.

Reading (src/smart_tags.py's, B32's `tags_convention=auto`, F131): the attack, transition and long
ball families carry the team in possession, so they count for their team; the defending families
(DEFENDING IN ... THIRD, DEFENSIVE TRANSITION) carry the defending team, so they count for the other
team. N/A tags, SET PIECES and codes outside smart_tags.FAMILIES add nothing. Where several tags
cover a frame, each team's evidence is the largest of its tags' tapered weights (no double count);
where both teams get evidence the two cancel in the H vs A choice.

The tags come from decoder.tags(match) (LearnedDecoder.tags_source: Smart Tagging's file on our
match; none on an external match by default). For grids on external matches the options `src` and
`noise` read stand-ins instead (learned.tags_for with the decoder's seed, flip and convention), so
candidates differ only in `cues` and share one emission per match (check_dropout_grid's CUE_ONLY):

    cues=tagteam:1                          our match: the decoder's own tags
    cues=tagteam:1/src=standin/noise=2      external grid: stand-ins at tagger noise 2 (F118 / F130)

Options (fixed before any run, B40's protocol):
    taper   seconds of linear ramp at each end of a phase (default 1.0; capped at half the phase)
    src     a learned.TAG_SOURCES value overriding decoder.tags_source ("" = the decoder's)
    noise   stand-in tagger noise (a smart_tags.NOISE_LEVELS key) when `src` is given

No tags (None or [], or no phase with a team) or gain 0 -> no effect (None): v6's exact bytes.
Label-free on our match (Smart Tagging is a provided input); stand-ins need labels and refuse to
run on BAR-ATH (smart_tags.stand_in).
"""

from __future__ import annotations

import numpy as np

from src import smart_tags
from src.models.cues import Effect
from src.official import A, H

FOR_TEAM = smart_tags.ATTACK + smart_tags.TRANSITION + smart_tags.LONG  # evidence for the tag's team
FOR_OTHER = smart_tags.DEFEND + smart_tags.DEF_TRANSITION  # evidence for the other team
TAPER_S = 1.0
MODES = ("team", "other", "cost")
COST_CLIP = (0.25, 4.0)

_CACHE: list = [None, None, None]  # (match, key, tags): grids decode one match many times


def _tags(match, decoder, src: str, noise: int):
    if not src:
        return decoder.tags(match)
    from src.models.learned import tags_for  # lazy: learned imports the cue machinery

    key = (src, int(noise), decoder.tags_seed, decoder.tags_flip, decoder.tags_convention)
    if _CACHE[0] is not match or _CACHE[1] != key:
        _CACHE[:] = [match, key, tags_for(match, src, int(noise), decoder.tags_seed,
                                          decoder.tags_flip, decoder.tags_convention)]
    return _CACHE[2]


def weights(tags, n: int, fps: float, taper_s: float = TAPER_S) -> np.ndarray:
    """(n, 2) per-frame evidence weight in [0, 1] for (H, A) from the tags' teams, tapered."""
    out = np.zeros((n, 2))
    ramp = max(0.0, float(taper_s)) * fps
    for t in tags or ():
        if t.team not in (H, A):
            continue
        if t.code in FOR_TEAM:
            team = t.team
        elif t.code in FOR_OTHER:
            team = A if t.team == H else H
        else:
            continue
        a, b = max(0, int(t.start)), min(n, int(t.end))
        if b <= a:
            continue
        frames = np.arange(a, b) + 0.5
        # distance (frames) to the nearer edge of the whole phase, ramped over `ramp` frames
        edge = np.minimum(frames - t.start, t.end - frames)
        w = np.ones(b - a) if ramp <= 0 else np.clip(edge / ramp, 0.0, 1.0)
        col = 0 if team == H else 1
        np.maximum(out[a:b, col], w, out=out[a:b, col])
    return out


def effect(match, original, decoder, gain, taper: float = TAPER_S, src: str = "", noise: int = 0,
           mode: str = "team", seen: float = 1.0):
    """B40: gain x tapered evidence per second on the tagged team's H / A score; None without tags."""
    if mode not in MODES:
        raise ValueError("tagteam: mode must be one of {}".format(MODES))
    if not gain:
        return None
    tags = _tags(match, decoder, str(src), noise)
    if not tags:
        return None
    w = weights(tags, match.n_frames, match.fps, taper)
    if seen != 1.0:  # where the ball is seen v6 reads the touch itself: the tags weigh `seen` there
        w = w * np.where(np.isnan(match.ball[:, 0]), 1.0, float(seen))[:, None]
    if not np.any(w):
        return None
    if mode == "cost":  # a prior on the switches: away from the tagged team dearer, towards it cheaper
        lean = w[:, 0] - w[:, 1]  # +1 inside an H phase, -1 inside an A phase
        ha = np.clip(np.exp(float(gain) * lean), *COST_CLIP)
        return Effect(ha=ha, ah=1.0 / ha)
    add = np.zeros((match.n_frames, 3))
    if mode == "team":
        add[:, :2] = float(gain) * w / match.fps
    else:  # "other": take it from the other team instead, so the team-vs-dead margin is untouched
        add[:, :2] = -float(gain) * w[:, ::-1] / match.fps
    return Effect(add=add)
