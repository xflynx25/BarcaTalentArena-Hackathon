"""Dead-ball onset cues and snapping decoded D onsets to them (backlog B12)."""

import numpy as np
import pytest

from src.models import build
from src.models.onset import CUES, cue_starts, snap_onsets, snap_to_cues
from src.models.postprocess import play_runs
from src.official import A, D, H


def codes(text: str) -> np.ndarray:
    return np.array([{"H": H, "A": A, "D": D}[c] for c in text], dtype=np.int8)


def text(values) -> str:
    return "".join({H: "H", A: "A", D: "D"}[int(v)] for v in values)


def test_snap_starts_dead_at_the_earliest_cue_in_the_lookback():
    c = codes("HHHHHHDDDDHH")
    cue = np.zeros(len(c), bool)
    cue[[3, 4]] = True
    assert text(snap_onsets(c, cue, [(0, len(c))], 3, D)) == "HHHDDDDDDDHH"
    assert text(snap_onsets(c, cue, [(0, len(c))], 2, D)) == "HHHHDDDDDDHH"  # 3 is too far back
    assert text(snap_onsets(c, cue, [(0, len(c))], 0, D)) == "HHHHHHDDDDHH"


def test_snap_never_starts_dead_earlier_without_a_cue_or_past_the_segment_before():
    c = codes("HHAADDDD")
    none = np.zeros(len(c), bool)
    assert text(snap_onsets(c, none, [(0, len(c))], 4, D)) == "HHAADDDD"
    cue = np.zeros(len(c), bool)
    cue[[0, 2]] = True  # 0 is in the segment before the previous one; 2 is the previous one's first
    assert text(snap_onsets(c, cue, [(0, len(c))], 8, D)) == "HHAADDDD"
    cue[3] = True
    assert text(snap_onsets(c, cue, [(0, len(c))], 8, D)) == "HHADDDDD"


def test_snap_stays_inside_runs():
    c = codes("HHHDDDHHHDDD")
    cue = np.zeros(len(c), bool)
    cue[[1, 7]] = True
    out = snap_onsets(c, cue, [(0, 6), (6, 12)], 5, D)
    assert text(out) == "HDDDDDHDDDDD"


def test_cue_starts_on_a_synthetic_match(small):
    match = small["match"]
    starts = cue_starts(match)
    assert set(starts) == set(CUES)
    for mask in starts.values():
        assert mask.dtype == bool and len(mask) == match.n_frames
        assert not mask[match.period == 0].any()
    with pytest.raises(ValueError):
        snap_to_cues(match, np.zeros(match.n_frames, np.int8), play_runs(match.period), 1.0, "nope")


@pytest.mark.parametrize("model", ["viterbi", "learned"])
def test_snap_default_and_only_adds_dead(small, model):
    match = small["match"]
    # off by default in viterbi; on (3 s) in learned since v3 (findings F80)
    assert build(model).onset_snap_s == (3.0 if model == "learned" else 0.0)
    snapped = build(model, onset_snap_s=2, min_spell_s=0, onset_cues="out+gap+stop+calm")
    raw = build(model, min_spell_s=0, onset_snap_s=0).predict(match)
    out = snapped.predict(match)
    assert np.all((out == raw) | (out == D))


# --- restart snapping and set-piece shapes (backlog B24) ---

from src.models.onset import RESTART_CUES, restart_events, snap_ends  # noqa: E402
from src.models import setpiece  # noqa: E402
from src.schema import AWAY, HOME, Match  # noqa: E402


def test_snap_ends_dead_at_the_latest_cue_in_the_lookback():
    c = codes("HHDDDDDDAA")
    cue = np.zeros(len(c), bool)
    cue[[4, 6]] = True
    assert text(snap_ends(c, cue, [(0, len(c))], 3, D)) == "HHDDDDAAAA"  # latest cue wins
    assert text(snap_ends(c, cue, [(0, len(c))], 1, D)) == "HHDDDDDDAA"  # 6 is 2 frames back
    assert text(snap_ends(c, cue, [(0, len(c))], 0, D)) == "HHDDDDDDAA"
    cue[:] = False
    cue[2] = True  # the D segment's first frame: never end D before its second
    assert text(snap_ends(c, cue, [(0, len(c))], 8, D)) == "HHDDDDDDAA"
    cue[3] = True
    assert text(snap_ends(c, cue, [(0, len(c))], 8, D)) == "HHDAAAAAAA"


def test_snap_ends_leaves_dead_at_the_end_of_a_run():
    c = codes("HHDDDD")
    cue = np.ones(len(c), bool)
    assert text(snap_ends(c, cue, [(0, len(c))], 4, D)) == "HHDDDD"


def test_restart_events_on_a_synthetic_match(small):
    match = small["match"]
    events = restart_events(match)
    assert set(events) == set(RESTART_CUES)
    for mask in events.values():
        assert mask.dtype == bool and len(mask) == match.n_frames
        assert not mask[match.period == 0].any()
    with pytest.raises(ValueError):
        build("learned", restart_snap_s=1, restart_cues="nope").predict(match)


@pytest.mark.parametrize("model", ["viterbi", "learned"])
def test_restart_snap_is_off_by_default_and_only_removes_dead(small, model):
    match = small["match"]
    assert build(model).restart_snap_s == 0.0
    raw = build(model, min_spell_s=0).predict(match)
    out = build(model, min_spell_s=0, restart_snap_s=2,
                restart_cues="+".join(RESTART_CUES)).predict(match)
    assert np.all((out == raw) | (raw == D))


def _still_match(n=50, fps=10):
    """Two teams of 11 standing still on a 105 x 68 pitch, home on the left. Frames: n."""
    home = np.array([[5, 34]] + [[20 + 3 * i, 10 + 5 * i] for i in range(10)], float)
    away = np.array([[100, 34]] + [[60 + 3 * i, 10 + 5 * i] for i in range(10)], float)
    players = np.repeat(np.concatenate([home, away])[None], n, axis=0)
    return Match("synthetic:shape", fps, (105.0, 68.0), np.ones(n, np.int8),
                 np.full((n, 2), np.nan), players,
                 np.array([HOME] * 11 + [AWAY] * 11, np.int8))


def test_shapes_see_a_wall_a_corner_a_throw_and_everyone_walking():
    m = _still_match()
    masks = setpiece.shape_masks(m)
    assert set(masks) == set(setpiece.SHAPES)
    assert masks["walk"].all() and not masks["wall"].any() and not masks["corner"].any()
    wall = m.players.copy()
    wall[:, 12:16] = [[70, 30], [70, 31.2], [70, 32.4], [70, 33.6]]  # 4 away players in a line
    wall[:, 1] = [79.5, 32]  # a still home player ~9.5 m away
    assert setpiece.shape_masks(Match(**{**m.__dict__, "players": wall}), ["wall"])["wall"].all()
    corner = m.players.copy()
    corner[:, 1] = [104.5, 0.5]  # at the corner flag
    corner[:, 2:10] = [[95 + i, 30 + i] for i in range(8)]  # 8 home players in the away box
    out = setpiece.shape_masks(Match(**{**m.__dict__, "players": corner}), ["corner"])["corner"]
    assert out.all()
    throw = m.players.copy()
    throw[:, 3] = [50, -0.3]  # on the touchline
    assert setpiece.shape_masks(Match(**{**m.__dict__, "players": throw}), ["throw"])["throw"].all()


def test_kickoff_shape_needs_everyone_in_their_own_half_and_someone_on_the_spot():
    m = _still_match()
    assert not setpiece.shape_masks(m, ["kickoff"])["kickoff"].any()
    kick = m.players.copy()
    kick[:, 12:22, 0] = np.clip(kick[:, 12:22, 0], 54, None)  # away all in their half
    kick[:, 11 + 1] = [53, 34]
    kick[:, 1] = [52.0, 34]  # home player on the spot
    assert setpiece.shape_masks(Match(**{**m.__dict__, "players": kick}), ["kickoff"])["kickoff"].all()
