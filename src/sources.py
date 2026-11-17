"""The external source registry (backlog A14): one entry per labelled source, read by every script that
fits or chooses (scripts/fit_learned.py, check_dropout_grid.py, tune_decoder.py, src.evaluate
--sources). A new source is one entry here plus its loader in src/ingest/.

Per source:
  loader      module name in src/ingest/ (list_matches() + load(id, **opts))
  load        load options for scoring and selection: what the grid, the tuner and
              `src.evaluate --sources` load (SkillCorner: cuts=ungraded, F26)
  train_load  extra load options when fitting `learned` (scripts/fit_learned.py), on top of load
              (SkillCorner: onsets=out since v4, B20/B22, F84). Scoring keeps the source's own labels.
  kind        "optical": all 22 players tracked; "broadcast": off-camera players estimated
              (findings F92); "annotated": positions annotated by hand or semi-automatically
  labels      "frame": the vendor's per-frame possession labels; "events": derived from event data
  fps         native frame rate of the tracking
  weight      relative selection weight, set by the principles in docs/decisions.md (C12, findings
              F114-F115): every source counts the same (1.0) however many matches it has, since the
              unit we generalise across is the source (a labeller, league and tracking system) and
              nothing BAR-ATH-free tells which one is most like BAR-ATH; a source whose input is
              partly rebuilt from the annotations its labels come from counts half (0.5). Matches
              are averaged within a source first, which is the cap: a source with many matches
              still counts as one source.
              weights() renormalises over the sources in use.
  select      in the selection set: what check_dropout_grid.py, tune_decoder.py, check_labels.py
              and `src.evaluate --sources all` score when no --sources is given (C12: all five).
  train       in the training set: what fit_learned.py fits on when no --sources is given. Kept
              apart from select (C12) so scoring a source never moves v4's refit; B30 / A18 decide
              which sources join it (tests/test_sources.py pins v4's three).
  test_only   every match is a lockbox match (A19): never trained on, early-stopped on, tuned on or
              selected with; scored only through a tranche (A19)
  lockbox     match ids of this source kept out of everything but a tranche opening (A19)

Lockbox (A19): a lockbox or test-only match is never in training, a fit's early-stopping split, the
dropout-grid selection set, a tuner, or C14. TRANCHES fixes, per source, which lockbox matches form
each tranche, by a hash of the match id (tranche_split), before anything is scored. `explore` is the
exploration set of the A19 gate (diagnostics, already-decided models only); T1, T2, ... are each
opened **once** (open_tranche records it in experiments/lockbox.json and refuses a second opening),
only for a candidate that has passed C12 (and C14), against DEFAULT, held out by source.

The label standard every source is measured against: docs/label-standard.md
(scripts/check_labels.py prints the table). BAR-ATH is never a source here (CLAUDE.md).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field

KINDS = ("optical", "broadcast", "annotated")
LABEL_ORIGINS = ("frame", "events")


@dataclass(frozen=True)
class Source:
    name: str
    loader: str
    kind: str
    labels: str
    fps: int
    weight: float
    load: dict = field(default_factory=dict)
    train_load: dict = field(default_factory=dict)
    select: bool = True
    train: bool = True
    test_only: bool = False
    lockbox: tuple = ()
    note: str = ""


SOURCES: dict[str, Source] = {s.name: s for s in (
    Source("idsse", "idsse", kind="optical", labels="frame", fps=25, weight=1.0,
           note="DFL Bundesliga 1/2 (7 matches): BallStatus + BallPossession per frame, F12"),
    Source("skillcorner", "skillcorner", kind="broadcast", labels="frame", fps=10, weight=1.0,
           load={"cuts": "ungraded"}, train_load={"onsets": "out"},
           note="A-League 2024/25 (20 matches): possession.group per frame; D onsets early, F84"),
    Source("metrica", "metrica", kind="optical", labels="events", fps=25, weight=1.0,
           note="Metrica sample games 1-3: labels derived from events, F21"),
    # wave 1 (A15, A17): scored in selection since C12; trained on only if B30 / A18 say so
    Source("skillcorner2020", "skillcorner2020", kind="broadcast", labels="frame", fps=10,
           weight=1.0, load={"cuts": "ungraded", "onsets": "out"}, train=False,
           note="SkillCorner open data 2019/20 (9 top-league matches), F103-F104. Scored with "
                "onsets=out as F104 measured it: its D already starts at the line (+0.1 s, F103). "
                "B30: training on it is worse on both folds (F125)"),
    # weight 0.5: its ball is rebuilt from the same actions its labels come from (F108), C12.
    # Scored with its ball smoothed (A18, F151-F152: the shipped grid staircase fakes touches);
    # trained, if B30 says so, on the shipped ball with --source-zero soccertrack=touch (A18's recipe)
    Source("soccertrack", "soccertrack", kind="annotated", labels="events", fps=25, weight=0.5,
           load={"ball": "smooth"}, train_load={"ball": "shipped"}, train=False,
           note="SoccerTrack v2 (10 matches, fixed full-pitch camera), F108-F110. labels=actions "
                "(loader default); its ball is reconstructed from the actions on a grid (F108, F150). "
                "Train only with --source-zero soccertrack=touch+impute (A18, F152, B30); scored with "
                "ball=smooth. B30: better overall but worse on SkillCorner 2019/20, not trained (F125)"),
)}

# A19: tranches of lockbox matches, fixed by tranche_split before anything is scored.
# Empty since 17 Nov: the only source that used it was removed (decisions.md). The functions stay
# so a later source can be held out the same way. tests/test_lockbox.py checks the rule on a
# stand-in source.
TRANCHES: dict[str, dict[str, tuple]] = {}
EXPLORE = "explore"  # the A19 gate's set: not a tranche, never "opened"
# filter_lockbox(..., lockbox=LABELS_ONLY) keeps every lockbox match: only for scripts that read
# labels and tracking and score no model
LABELS_ONLY = "labels-only"
LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments",
                      "lockbox.json")

USES = ("select", "train")
# C6's hand-set weights (SkillCorner 1/2 for its coarse "controlled" labels, F46-F47), kept so older
# grid runs reproduce: check_dropout_grid.py --weights c6. Replaced by the registry's weights (C12).
C6_WEIGHTS = {"idsse": 0.25, "skillcorner": 0.5, "metrica": 0.25}


def names(use: str = "select", everything: bool = False, test_only: bool = False) -> list[str]:
    """The sources in the selection set (use="train": the training set; everything=True: every
    registered one except test-only ones, A19, unless test_only=True), in registry order (the order
    fits and tables use)."""
    if use not in USES:
        raise ValueError("use must be one of {}, not {!r}".format(USES, use))
    return [n for n, s in SOURCES.items()
            if (everything and (test_only or not s.test_only)) or getattr(s, use)]


class LockboxError(ValueError):
    """A lockbox or test-only match or source was asked for where A19 forbids it."""


def tranche_split(source: str, ids, sizes) -> dict[str, tuple]:
    """{tranche: ids}: ids ordered by sha256("<source>:<id>"), cut in order into the named sizes
    (each tranche sorted numerically). Depends on the ids alone, so it is fixed before any score."""
    order = sorted((str(i) for i in ids),
                   key=lambda m: hashlib.sha256("{}:{}".format(source, m).encode()).hexdigest())
    if sum(n for _, n in sizes) != len(order):
        raise ValueError("{}: tranche sizes sum to {}, not {}".format(
            source, sum(n for _, n in sizes), len(order)))
    out, start = {}, 0
    for name, n in sizes:
        out[name] = tuple(sorted(order[start:start + n], key=lambda m: (len(m), m)))
        start += n
    return out


def is_lockbox(source: str, match_id: str) -> bool:
    """True if this match is kept out of training, tuning and selection (A19)."""
    entry = SOURCES.get(source)
    return entry is not None and (entry.test_only or str(match_id) in entry.lockbox)


def tranche_of(source: str, match_id: str) -> str | None:
    for name, ids in TRANCHES.get(source, {}).items():
        if str(match_id) in ids:
            return name
    return None


def tranche_names() -> list[str]:
    return sorted({t for tranches in TRANCHES.values() for t in tranches})


def filter_lockbox(keys: list[str], lockbox: str | None = None, explicit: set | None = None
                   ) -> list[str]:
    """Match keys ("<source>:<id>") with lockbox matches removed, except those in tranche
    `lockbox` (LABELS_ONLY keeps them all). A key in `explicit` (named on the command line, not
    expanded from <source>:all) that would be removed is refused."""
    if lockbox == LABELS_ONLY:
        return list(keys)
    if lockbox is not None and lockbox not in tranche_names():
        raise ValueError("unknown tranche {!r}; tranches: {}".format(lockbox, ", ".join(tranche_names())))
    out = []
    for key in keys:
        source, _, match_id = key.partition(":")
        if not is_lockbox(source, match_id) or (lockbox is not None
                                                  and tranche_of(source, match_id) == lockbox):
            out.append(key)
        elif explicit and key in explicit:
            raise LockboxError("{} is a lockbox match (A19, test-only source or lockbox); score it "
                               "only through its tranche: --lockbox {}".format(
                                   key, tranche_of(source, match_id) or "?"))
    return out


def check_trainable(names_: list[str], what: str = "training") -> None:
    """Refuse a test-only source in training, early stopping, tuning or selection (A19)."""
    bad = [n for n in names_ if n in SOURCES and SOURCES[n].test_only]
    if bad:
        raise LockboxError("{} is test-only (A19): never in {}. Score it only through a tranche"
                           .format(", ".join(bad), what))


def drop_lockbox(keys: list[str]) -> list[str]:
    """Training keys without any lockbox match (refuses a test-only source outright)."""
    check_trainable(sorted({k.partition(":")[0] for k in keys}))
    return [k for k in keys if not is_lockbox(*k.split(":", 1))]


def open_tranche(tranche: str, record: dict, ledger: str = LEDGER) -> None:
    """Record the one opening of a tranche (A19 step 4); refuse a second one. `explore` is not a
    tranche and is never recorded."""
    if tranche == EXPLORE:
        return
    if tranche not in tranche_names():
        raise ValueError("unknown tranche {!r}".format(tranche))
    opened = {}
    if os.path.isfile(ledger):
        with open(ledger) as handle:
            opened = json.load(handle)
    if tranche in opened:
        raise LockboxError("tranche {} was already opened ({}): a tranche is opened once (A19)"
                           .format(tranche, opened[tranche].get("date", "?")))
    opened[tranche] = dict(record, date=datetime.date.today().isoformat())
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, "w") as handle:
        json.dump(opened, handle, indent=1, sort_keys=True)
        handle.write("\n")


def get(name: str) -> Source:
    if name not in SOURCES:
        raise ValueError("unknown source {!r}; registered: {}".format(name, ", ".join(SOURCES)))
    return SOURCES[name]


def load_opts(name: str, train: bool = False) -> dict:
    """The source's default load options for scoring, or for fitting (train=True), as a fresh dict."""
    source = get(name)
    return dict(source.load, **(source.train_load if train else {}))


def weights(names_: list[str] | None = None) -> dict[str, float]:
    """Selection weights of these sources (default: the selection set), renormalised to sum to 1."""
    names_ = names_ or names()
    total = sum(get(n).weight for n in names_)
    return {n: get(n).weight / total for n in names_}


def validate() -> None:
    """Raise if an entry is malformed (tests call this)."""
    for name, source in SOURCES.items():
        if name != source.name:
            raise ValueError("{}: key and name differ".format(name))
        if source.kind not in KINDS or source.labels not in LABEL_ORIGINS:
            raise ValueError("{}: bad kind {!r} or labels {!r}".format(name, source.kind, source.labels))
        if name == "bar_ath" or source.loader == "bar_ath":
            raise ValueError("BAR-ATH is never an external source (CLAUDE.md)")
        if not source.weight > 0 or source.fps <= 0:
            raise ValueError("{}: weight and fps must be positive".format(name))
        if source.test_only and (source.select or source.train):
            raise ValueError("{}: a test-only source is never in select or train (A19)".format(name))
    for name, tranches in TRANCHES.items():
        ids = [m for t in tranches.values() for m in t]
        if len(ids) != len(set(ids)):
            raise ValueError("{}: a match is in two tranches".format(name))
        if not all(is_lockbox(name, m) for m in ids):
            raise ValueError("{}: a tranche holds a match that isn't lockboxed".format(name))
    if not names("select") or not names("train"):
        raise ValueError("the selection and training sets must not be empty")
