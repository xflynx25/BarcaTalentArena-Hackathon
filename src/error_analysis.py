"""Where does a model lose spellF1? The error breakdown behind backlog B1 (findings F31, F32).

Used by scripts/analyse_errors.py (text summary, see its docstring for what each block means) and by
src.report's timeline pages (per-spell verdicts, spell_causes). External matches only. Uses the
official scorer's own spell functions.
"""

from __future__ import annotations

import numpy as np

from src import evaluate
from src.models.postprocess import delay_switches, drop_short
from src.official import A, D, H, X, pm
from src.schema import AWAY, HOME

HIDDEN = {"ball_missing": 0.41, "ball_quiet": 0.75}  # BAR-ATH-like ball (findings F30)
SETTINGS = {
    "idsse": ("idsse", {}),
    "idsse-hidden": ("idsse", HIDDEN),
    "skillcorner": ("skillcorner", {"cuts": "ungraded"}),
    "skillcorner-hidden": ("skillcorner", dict(HIDDEN, cuts="ungraded")),
    "metrica": ("metrica", {}),
    "metrica-hidden": ("metrica", HIDDEN),
}
# Sources newer than the pages above (A17): named with --setting only, so existing pages and their
# rebuilds (which default to SETTINGS) don't change. A14's source registry can fold them in.
OPT_IN_SETTINGS = {
    "soccertrack": ("soccertrack", {}),
    "soccertrack-hidden": ("soccertrack", HIDDEN),
    # V4: the hidden ball plus a simulated second recording, aligned (Match.ball2)
    "idsse-broadcast": ("idsse", dict(HIDDEN, broadcast=1)),
    "skillcorner-broadcast": ("skillcorner", dict(HIDDEN, cuts="ungraded", broadcast=1)),
    "metrica-broadcast": ("metrica", dict(HIDDEN, broadcast=1)),
}
ALL_SETTINGS = {**SETTINGS, **OPT_IN_SETTINGS}
LENGTH_BINS_S = (0, 2, 5, 10, np.inf)
CONTEXTS = ("missing", "in flight", "contested", "controlled")
FLIGHT_M, CONTEST_M = 3.0, 2.0
MISS_CAUSES = ("absent", "split", "merged", "shifted")
SPURIOUS_KINDS = ("flicker", "in dead", "wrong team", "other")


# -- post-processing and oracles ---------------------------------------------------------------------

def team_ffill(codes: np.ndarray, runs) -> np.ndarray:
    """H/A with D replaced by the last team (the next one at the start of a run)."""
    out = codes.copy()
    out[out == D] = X
    return pm.fill_abstentions(out, runs)


def what_if(gt: np.ndarray, pred: np.ndarray, fps: int, runs) -> dict:
    live = (gt == H) | (gt == A)
    team = team_ffill(pred, runs)
    perfect_dead = np.where(gt == D, D, np.where(team == X, H, team)).astype(np.int8)
    perfect_team = np.where(live & ((pred == H) | (pred == A)), gt, pred).astype(np.int8)
    variants = {
        "model": pred,
        "perfect alive/dead": perfect_dead,
        "perfect team": perfect_team,
        "drop < 1 s": drop_short(pred, runs, fps),
        "drop < 2 s": drop_short(pred, runs, 2 * fps),
        "delay switch 0.5 s": delay_switches(pred, runs, round(0.5 * fps)),
    }
    return {name: evaluate.score(gt, codes)["spellF1"] for name, codes in variants.items()}


# -- breakdowns --------------------------------------------------------------------------------------

def spell_causes(gt: np.ndarray, pred: np.ndarray, fps: int, runs) -> tuple:
    """Each true and predicted spell's verdict at IoU 0.5 (the headline threshold).

    Returns (gt_spells, gt_cause, pred_spells, pred_kind), spells as the scorer's (start, stop, code)
    in frames. gt_cause: "found", or why it was missed: team never predicted on it ("absent"), cut
    into pieces ("split"), swallowed by a longer predicted spell ("merged"), boundaries too far off
    ("shifted"). pred_kind: "matched", or a spurious spell: under 1 s ("flicker"), mostly over true
    dead time ("in dead"), mostly over the other team's spell ("wrong team"), "other"."""
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    hit = pm.matched_gt_flags(overlaps, len(gt_spells), pm.SPELL_IOU_HEADLINE)
    matched_pred = set()
    taken = np.zeros(len(gt_spells), dtype=bool)
    for iou, i, j in overlaps:  # same greedy pass as matched_gt_flags, to know which preds matched
        if iou < pm.SPELL_IOU_HEADLINE:
            break
        if not taken[i] and j not in matched_pred:
            taken[i] = True
            matched_pred.add(j)

    by_gt = {}
    for iou, i, j in overlaps:
        by_gt.setdefault(i, []).append((iou, j))
    gt_cause = []
    for i, (a, b, _) in enumerate(gt_spells):
        pairs = by_gt.get(i, [])
        if hit[i]:
            gt_cause.append("found")
            continue
        if not pairs:
            gt_cause.append("absent")
            continue
        best_iou, best = max(pairs)
        pa, pb, _ = pred_spells[best]
        if len(pairs) > 1 and sum(min(b, pred_spells[j][1]) - max(a, pred_spells[j][0])
                                  for _, j in pairs) >= 0.5 * (b - a):
            gt_cause.append("split")
        elif pb - pa > 2 * (b - a):
            gt_cause.append("merged")
        else:
            gt_cause.append("shifted")

    pred_kind = []
    for j, (a, b, code) in enumerate(pred_spells):
        window = gt[a:b]
        if j in matched_pred:
            pred_kind.append("matched")
        elif b - a < fps:
            pred_kind.append("flicker")
        elif np.mean(window == D) > 0.5:
            pred_kind.append("in dead")
        elif np.mean(window == (A if code == H else H)) > 0.5:
            pred_kind.append("wrong team")
        else:
            pred_kind.append("other")
    return gt_spells, gt_cause, pred_spells, pred_kind


def spell_breakdown(gt: np.ndarray, pred: np.ndarray, fps: int, runs) -> dict:
    """Shares and recall by true spell length, miss causes and spurious kinds (spell_causes)."""
    gt_spells, gt_cause, pred_spells, pred_kind = spell_causes(gt, pred, fps, runs)
    hit = np.array([c == "found" for c in gt_cause], dtype=bool)
    out = {}
    lengths = np.array([(b - a) / fps for a, b, _ in gt_spells])
    for low, high in zip(LENGTH_BINS_S[:-1], LENGTH_BINS_S[1:]):
        in_bin = (lengths >= low) & (lengths < high)
        out["len {}-{}".format(low, high)] = (in_bin.mean(), hit[in_bin].mean() if in_bin.any()
                                             else np.nan)

    n_missed = max(1, int((~hit).sum()))
    out.update({"miss " + k: gt_cause.count(k) / n_missed for k in MISS_CAUSES})
    out["miss share"] = (~hit).mean()

    n_unmatched = len(pred_spells) - pred_kind.count("matched")
    out.update({"spur " + k: pred_kind.count(k) / max(1, n_unmatched) for k in SPURIOUS_KINDS})
    out["spur share"] = n_unmatched / max(1, len(pred_spells))
    return out


def switch_times(codes: np.ndarray, runs) -> tuple:
    """(frames, new team) of every H<->A switch in team_ffill(codes), inside runs only."""
    team = team_ffill(codes, runs)
    frames, teams = [], []
    for start, stop in runs:
        window = team[start:stop]
        at = np.flatnonzero((window[1:] != window[:-1]) & (window[1:] != X) & (window[:-1] != X)) + 1
        frames.append(at + start)
        teams.append(window[at])
    return np.concatenate(frames or [[]]).astype(int), np.concatenate(teams or [[]]).astype(np.int8)


def match_switches(gt_at, gt_team, pred_at, pred_team, tol: int) -> int:
    """How many predicted switches pair one-to-one with a true switch to the same team within tol
    frames (greedy, closest pairs first)."""
    pairs = []
    for j, (t, team) in enumerate(zip(pred_at, pred_team)):
        lo, hi = np.searchsorted(gt_at, [t - tol, t + tol + 1])
        pairs += [(abs(int(gt_at[i]) - int(t)), i, j) for i in range(lo, hi) if gt_team[i] == team]
    used_gt, used_pred = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
    return len(used_gt)


def switch_breakdown(gt: np.ndarray, pred: np.ndarray, fps: int, runs, tol_s: float = 1.0) -> dict:
    """H<->A switch precision / recall at +/- tol_s (findings F53, F61), D carried over by
    team_ffill on both sides, and switches per minute of live play. NaN when there is nothing to
    divide by (the means over matches skip it)."""
    gt_at, gt_team = switch_times(gt, runs)
    pred_at, pred_team = switch_times(pred, runs)
    hits = match_switches(gt_at, gt_team, pred_at, pred_team, round(tol_s * fps))
    live_min = np.sum((gt == H) | (gt == A)) / fps / 60
    return {
        "precision": hits / len(pred_at) if len(pred_at) else np.nan,
        "recall": hits / len(gt_at) if len(gt_at) else np.nan,
        "true per min": len(gt_at) / live_min if live_min else np.nan,
        "pred per min": len(pred_at) / live_min if live_min else np.nan,
    }


def frame_breakdown(match, pred: np.ndarray) -> dict:
    gt = match.labels
    live = (gt == H) | (gt == A)
    dx = match.players[:, :, 0] - match.ball[:, None, 0]
    dy = match.players[:, :, 1] - match.ball[:, None, 1]
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore"):
        nearest = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
        home_close = np.any((dist <= CONTEST_M) & (match.slot_team == HOME)[None, :], axis=1)
        away_close = np.any((dist <= CONTEST_M) & (match.slot_team == AWAY)[None, :], axis=1)
    missing = np.isnan(match.ball[:, 0])
    context = np.select([missing, nearest > FLIGHT_M, home_close & away_close],
                        [0, 1, 2], default=3)
    out = {}
    for c, name in enumerate(CONTEXTS):
        mask = live & (context == c)
        share = mask.sum() / max(1, live.sum())
        wrong = np.mean(((pred == H) | (pred == A)) & (pred != gt), where=mask) if mask.any() else np.nan
        dead = np.mean(pred == D, where=mask) if mask.any() else np.nan
        out[name] = (share, wrong, dead)
    return out
