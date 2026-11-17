"""Label-free post-processing any model can apply to its codes before returning them.

drop_short: every predicted segment (H, A or D) shorter than a minimum length takes the label of the
last segment before it that is long enough (the first long one at the start of a half). Findings
F32: with the ball hidden like on BAR-ATH, the majority vote leaves short D and flicker segments
that each split a true spell; removing them is worth more than any smoothing window. Tuned in B6
(findings F34).

To use it in a model: add a `min_spell_s` field and end predict() with
`drop_short(codes, play_runs(match.period), to_frames(self.min_spell_s, match.fps))`.

delay_switches: every direct H<->A switch happens a fixed time later (the old team keeps the ball
meanwhile). Findings F13/F31: the truth flips at the receiver's control, ~0.7 s after the nearest
player changes team, so a nearest-player model switches early. Tuned in C2 (findings F43). Apply it
after drop_short, on clean segments.

merge_short (B8, findings F51): drop_short with three changes, each optional. Team (H/A) and D
segments get their own minimum length; a short segment can go to the longer neighbour, or the
neighbour the per-frame evidence prefers, instead of always the one before; and a short segment the
evidence is confident about (mean margin of its label over the best other label >= keep_margin) is
kept. rule="before" with equal lengths and no keep_margin is exactly drop_short.
"""

from __future__ import annotations

import heapq

import numpy as np

from src.official import A, D, H


def play_runs(period: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) spans of constant period > 0 (the halves). Segments never cross
    one, like the scorer's valid_runs(), so the halftime hole can't merge two spells."""
    period = np.asarray(period)
    breaks = np.flatnonzero(period[1:] != period[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(period)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a and period[a] > 0]


def drop_short(codes: np.ndarray, runs, min_frames: int) -> np.ndarray:
    """Copy of codes where, inside each run, segments shorter than min_frames take the label of the
    last segment of at least min_frames before them (the first such segment at the start of a run).
    Runs with no long segment are left as they are. min_frames <= 1 changes nothing."""
    out = np.array(codes, copy=True)
    if min_frames <= 1:
        return out
    for start, stop in runs:
        window = out[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        lengths = np.diff(bounds)
        long = lengths >= min_frames
        if not long.any():
            continue
        source = np.maximum.accumulate(np.where(long, np.arange(len(long)), -1))
        source[source < 0] = np.flatnonzero(long)[0]
        out[start:stop] = np.repeat(window[bounds[:-1]][source], lengths)
    return out


def delay_switches(codes: np.ndarray, runs, frames: int) -> np.ndarray:
    """Copy of codes where, inside each run, every direct H<->A switch happens `frames` later: the
    first `frames` frames of the new team's segment take the old team's label (a new segment
    shorter than that disappears). Switches to or from D are left alone. frames <= 0 changes
    nothing."""
    out = np.array(codes, copy=True)
    if frames <= 0:
        return out
    team = (codes == H) | (codes == A)
    for start, stop in runs:
        window = codes[start:stop]
        flips = np.flatnonzero((window[1:] != window[:-1]) & team[start + 1:stop]
                               & team[start:stop - 1]) + 1
        for t in flips:
            end = min(t + frames, stop - start)
            span = out[start + t:start + end]
            span[window[t:end] == window[t]] = window[t - 1]
    return out


RULES = ("before", "longer", "evidence")


def segments(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, lengths) of the constant segments of a 1-d array."""
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return bounds[:-1], np.diff(bounds)


def margins(evidence: np.ndarray, codes: np.ndarray, starts: np.ndarray,
            lengths: np.ndarray) -> np.ndarray:
    """Per segment: mean over its frames of evidence[label] - max(evidence[other labels]).
    evidence is (n, 3) scores in code order (H, A, D; higher = more likely)."""
    rows = np.arange(len(codes))
    own = evidence[rows, codes]
    other = np.array(evidence, dtype=float, copy=True)
    other[rows, codes] = -np.inf
    total = np.concatenate(([0.0], np.cumsum(own - other.max(axis=1))))
    return (total[starts + lengths] - total[starts]) / lengths


def merge_short(codes: np.ndarray, runs, min_team: int, min_dead: int, rule: str = "before",
                evidence: np.ndarray | None = None, keep_margin: float | None = None) -> np.ndarray:
    """Copy of codes where, inside each run, team segments shorter than min_team frames and D
    segments shorter than min_dead take a neighbour's label. Codes inside runs must be H, A or D.

    rule  "before": the last segment before it that is long enough (or kept), the first such
                    one at the start of a run (drop_short's rule).
          "longer": shortest first, each short segment joins its longer neighbour (both, if they
                    share a label); merged segments count with their new length.
          "evidence": like "longer", but joins the neighbour whose label has the higher mean
                    evidence over the short segment (ties: the longer one). Needs evidence.
    evidence  (n, 3) per-frame scores for H, A, D, used by rule="evidence" and keep_margin.
    keep_margin  short segments whose mean evidence margin (margins()) is at least this are kept
                 as if long. None: off. Needs evidence."""
    if rule not in RULES:
        raise ValueError("rule must be one of {}, not {!r}".format(RULES, rule))
    if (rule == "evidence" or keep_margin is not None) and evidence is None:
        raise ValueError("rule='evidence' and keep_margin need evidence")
    out = np.array(codes, copy=True)
    if max(min_team, min_dead) <= 1:
        return out
    for start, stop in runs:
        window = out[start:stop]
        if not len(window):
            continue
        starts, lengths = segments(window)
        labels = window[starts]
        need = np.where(labels == D, min_dead, min_team)
        kept = np.zeros(len(starts), bool)
        if keep_margin is not None:
            kept = margins(evidence[start:stop], window, starts, lengths) >= keep_margin
        if rule == "before":
            long = (lengths >= need) | kept
            if not long.any():
                continue
            source = np.maximum.accumulate(np.where(long, np.arange(len(long)), -1))
            source[source < 0] = np.flatnonzero(long)[0]
            out[start:stop] = np.repeat(labels[source], lengths)
        else:
            ev = None if evidence is None else evidence[start:stop]
            out[start:stop] = _merge_neighbours(window, starts, lengths, labels, need, kept, rule, ev)
    return out


def _merge_neighbours(window, starts, lengths, labels, need, kept, rule, evidence):
    """rule "longer"/"evidence" of merge_short on one run: a heap of short segments, shortest
    first (ties: earliest), over a doubly linked list of segments."""
    k = len(starts)
    if k < 2:
        return window
    first, lengths, labels = starts.tolist(), lengths.tolist(), labels.tolist()
    need, kept = need.tolist(), kept.tolist()
    prev, nxt = list(range(-1, k - 1)), list(range(1, k + 1))
    nxt[-1] = -1
    alive = [True] * k
    cum = None  # per-label evidence prefix sums, to score a neighbour's label over a segment
    if evidence is not None:
        cum = np.vstack([np.zeros((1, 3)), np.cumsum(evidence, axis=0)])
    heap = [(lengths[i], first[i], i) for i in range(k) if not kept[i] and lengths[i] < need[i]]
    heapq.heapify(heap)
    while heap:
        length, _, i = heapq.heappop(heap)
        if not alive[i] or length != lengths[i]:
            continue
        p, q = prev[i], nxt[i]
        if p < 0 and q < 0:
            continue
        if q < 0 or (p >= 0 and labels[p] == labels[q]):
            target = p
        elif p < 0:
            target = q
        else:
            target = p if lengths[p] >= lengths[q] else q
            if rule == "evidence":
                a, b = first[i], first[i] + lengths[i]
                sp, sq = (cum[b, labels[j]] - cum[a, labels[j]] for j in (p, q))
                if sp != sq:
                    target = p if sp > sq else q
        # absorb i into target, and the far neighbour too when it shares target's label
        far = q if target == p else p
        group = [i] + ([far] if far >= 0 and labels[far] == labels[target] else [])
        members = sorted([target] + group, key=lambda j: first[j])
        lo, hi = prev[members[0]], nxt[members[-1]]
        for j in group:
            alive[j] = False
            lengths[target] += lengths[j]
            first[target] = min(first[target], first[j])
            kept[target] = kept[target] or kept[j]
        prev[target], nxt[target] = lo, hi
        if lo >= 0:
            nxt[lo] = target
        if hi >= 0:
            prev[hi] = target
        if not kept[target] and lengths[target] < need[target]:
            heapq.heappush(heap, (lengths[target], first[target], target))
    out = np.empty_like(window)
    i = next(j for j in range(k) if alive[j] and prev[j] < 0)
    while i >= 0:
        out[first[i]:first[i] + lengths[i]] = labels[i]
        i = nxt[i]
    return out
