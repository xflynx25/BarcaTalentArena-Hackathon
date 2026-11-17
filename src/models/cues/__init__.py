"""Decode-time cues: per-frame transition costs and extra evidence for the learned decoder (wave 5).

Why: wave 4's lesson (findings F181-F184) is that decode-time parts (the restart rule, the switch
cue) carry over to sources the model never saw, while refitting the emission models with more
inputs keeps losing IDSSE / Metrica held out by source. So wave 5's ideas plug in here, after the
emission models, without touching them, each in its own file, so parallel tasks never edit the same
lines of LearnedDecoder.

A cue is a module `src/models/cues/<name>.py` with

    def effect(match, original, decoder, gain, **opts) -> Effect

  match     the match the decoder reads (repaired when track_repair / repairs are set, its ball's
            gaps filled from ball2 when the decoder fills them)
  original  the match as delivered (gates that count the tracking as delivered read this one)
  decoder   the LearnedDecoder (or EmissionDecoder) itself: read .fold, .fps-free settings, .tags()
  gain      the number after the colon in the spec (1.0 when omitted); gain 0 must be "no effect"
  opts      the "/k=v" options in the spec, cast like --param values (int, float, bool, else str)

and returns an Effect: any of six per-frame multipliers on the decoder's transition costs (None =
1 everywhere) and an additive (n, 3) term on its scores (None = 0). Multipliers must be finite and
> 0; each cue clips its own range. Several cues multiply their multipliers and add their terms.

Turned on by the decoder parameter `cues` ("" = off, the default: every preset's bytes unchanged):

    --param cues=stop:0.5                     one cue at gain 0.5
    --param "cues=stop:0.5/win=2+flight:1"    two cues; stop gets opts {"win": 2}

The spec has no commas, so it fits check_dropout_grid.py's --extra NAME=MODEL:K=V,... (the grid
shares one emission per match among candidates that differ only in `cues`: it is in DECODE_ONLY).
A test can register a cue without a file: REGISTRY[name] = effect.

Units: costs are in seconds of full evidence, scores in evidence per frame (per-second evidence /
fps), as everywhere in the decoder (src/models/decoder.py). Multipliers apply to the decoder's own
costs: switch_cost_s (after the noise scaling and, when switch_cue_gain is set, the switch cue's
per-frame costs) and dead_cost_s. The path then comes from viterbi_costs() below, which is
decoder.viterbi's recursion with per-frame costs: with every multiplier 1 and no added term it gives
exactly decoder.viterbi's path (tests/test_cues.py).

Runtime: numpy + one Python loop over frames (~0.7 s on a 173k-frame match, like the switch cue).
Deterministic: scores and costs go on decoder.to_grid first (findings F87).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from src.models.decoder import to_grid

# the six transitions, in the order viterbi_costs takes them: from-state then to-state
KINDS = ("ha", "ah", "hd", "ad", "dh", "da")


@dataclass
class Effect:
    """What one cue does on one match. Each multiplier is None or an (n,) array: entry t applies to
    the transition from frame t-1 to frame t (entry 0 is unused). `add` is None or (n, 3) (H, A, D)."""

    ha: np.ndarray | None = None  # H -> A (a switch to the away team)
    ah: np.ndarray | None = None  # A -> H
    hd: np.ndarray | None = None  # H -> D (the ball goes dead)
    ad: np.ndarray | None = None  # A -> D
    dh: np.ndarray | None = None  # D -> H (play restarts with the home team)
    da: np.ndarray | None = None  # D -> A
    add: np.ndarray | None = None  # added to the decoder's scores

    @staticmethod
    def dead(mult: np.ndarray | None = None, live: np.ndarray | None = None,
             add: np.ndarray | None = None) -> "Effect":
        """Shorthand: `mult` on both team -> D transitions, `live` on both D -> team ones."""
        return Effect(hd=mult, ad=mult, dh=live, da=live, add=add)

    @staticmethod
    def switch(mult: np.ndarray | None = None, add: np.ndarray | None = None) -> "Effect":
        """Shorthand: `mult` on both H <-> A transitions."""
        return Effect(ha=mult, ah=mult, add=add)


REGISTRY: dict = {}  # name -> effect function; filled lazily from src/models/cues/<name>.py


def _cast(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse(spec: str) -> list[tuple[str, float, dict]]:
    """"stop:0.5/win=2+flight" -> [("stop", 0.5, {"win": 2}), ("flight", 1.0, {})]. Raises on an
    empty name, a gain that isn't a number, an option without "=", or a name given twice."""
    out, seen = [], set()
    text = str(spec).strip()
    if not text:
        return out
    for item in text.split("+"):
        head, *opts = item.strip().split("/")
        name, _, gain = head.partition(":")
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("bad cue name {!r} in cues={!r}".format(name, spec))
        if name in seen:
            raise ValueError("cue {!r} given twice in cues={!r}".format(name, spec))
        seen.add(name)
        try:
            g = float(gain) if gain.strip() else 1.0
        except ValueError:
            raise ValueError("cue {!r}: gain {!r} is not a number".format(name, gain)) from None
        kwargs = {}
        for opt in opts:
            key, eq, value = opt.partition("=")
            if not eq or not key:
                raise ValueError("cue {!r}: option {!r} needs key=value".format(name, opt))
            kwargs[key.strip()] = _cast(value.strip())
        out.append((name, g, kwargs))
    return out


def lookup(name: str):
    """The effect function of cue `name` (REGISTRY first, then src/models/cues/<name>.py)."""
    if name not in REGISTRY:
        try:
            module = importlib.import_module("{}.{}".format(__name__, name))
        except ModuleNotFoundError as exc:
            if exc.name == "{}.{}".format(__name__, name):
                raise ValueError("unknown cue {!r}: no src/models/cues/{}.py".format(name, name)) from None
            raise
        REGISTRY[name] = module.effect
    return REGISTRY[name]


def check(spec: str) -> None:
    """Raise now (at decoder construction) if the spec is malformed or names an unknown cue."""
    for name, _, _ in parse(spec):
        lookup(name)


def _mult(name: str, kind: str, value, n: int) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,))
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("cue {!r}: its {} multiplier must be finite and > 0".format(name, kind))
    return arr


def combine(spec: str, match, original, decoder, base: dict, n: int):
    """Per-frame costs and the added score term for this match. `base` maps each kind in KINDS to
    the decoder's own cost (a float or an (n,) array). Returns (costs: dict kind -> (n,) array,
    add: (n, 3) array or None)."""
    costs = {k: np.array(np.broadcast_to(np.asarray(base[k], dtype=np.float64), (n,))) for k in KINDS}
    add = None
    for name, gain, opts in parse(spec):
        eff = lookup(name)(match, original, decoder, gain, **opts)
        if eff is None:
            continue
        for kind in KINDS:
            value = getattr(eff, kind)
            if value is not None:
                costs[kind] = costs[kind] * _mult(name, kind, value, n)
        if eff.add is not None:
            term = np.asarray(eff.add, dtype=np.float64)
            if term.shape != (n, 3) or not np.all(np.isfinite(term)):
                raise ValueError("cue {!r}: add must be a finite (n, 3) array".format(name))
            add = term if add is None else add + term
    return costs, add


def viterbi_costs(scores: np.ndarray, ha, ah, hd, ad, dh, da) -> np.ndarray:
    """decoder.viterbi with a per-frame cost for each of the six transitions (each a float or an
    (n,) array; entry t is the cost of the transition from frame t-1 to frame t, entry 0 unused).
    The same recursion, tie rule (a change happens as late as possible: arriving from another state
    wins ties, in the order decoder.viterbi checks them) and backtracking; scores and costs on the
    grid first. With constant costs it is decoder.viterbi(scores, switch, dead) exactly."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    scores = to_grid(scores)

    def grid(c):
        return to_grid(np.broadcast_to(np.asarray(c, dtype=np.float64), (n,))).tolist()

    c_ha, c_ah, c_hd, c_ad, c_dh, c_da = (grid(c) for c in (ha, ah, hd, ad, dh, da))
    sh, sa, sd = (scores[:, k].tolist() for k in range(3))
    back = bytearray(3 * n)
    vh, va, vd = sh[0], sa[0], sd[0]
    for t in range(1, n):
        # H: stay, else from A, else from D
        best, arg = vh, 0
        x = va - c_ah[t]
        if x >= best:
            best, arg = x, 1
        x = vd - c_dh[t]
        if x >= best:
            best, arg = x, 2
        nh, bh = best + sh[t], arg
        # A: stay, else from H, else from D
        best, arg = va, 1
        x = vh - c_ha[t]
        if x >= best:
            best, arg = x, 0
        x = vd - c_da[t]
        if x >= best:
            best, arg = x, 2
        na, ba = best + sa[t], arg
        # D: stay, else from H, else from A
        best, arg = vd, 2
        x = vh - c_hd[t]
        if x >= best:
            best, arg = x, 0
        x = va - c_ad[t]
        if x >= best:
            best, arg = x, 1
        nd, bd = best + sd[t], arg
        vh, va, vd = nh, na, nd
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
    state = int(np.argmax([vh, va, vd]))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path
