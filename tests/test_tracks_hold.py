"""C16: the track repair's hold (src/tracks.py, track_repair="1/hold=<s>")."""

import dataclasses

import numpy as np

from src import tracks
from src.ingest import synthetic as synthetic_source
from src.schema import AWAY, HOME, Match


def _line_match(n=200, fps=25):
    t = np.arange(n) / fps
    players = np.stack([np.stack([10 + 2 * t, 20 + 0 * t], 1), np.stack([30 + 0 * t, 10 + 1 * t], 1),
                        np.stack([60 - 1 * t, 40 + 0 * t], 1)], axis=1)
    return Match(match_id="test:line", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=np.full((n, 2), np.nan), players=players,
                 slot_team=np.array([HOME, HOME, AWAY], np.int8)).validate()


def _seen(m):
    return ~np.isnan(m.players).any(axis=2)


def test_hold_off_keeps_track_repair_1_bytes():
    m = synthetic_source.load("0")
    a = tracks.apply(m, "1")
    for spec in ("1/hold=0", "hold=0", "10/5/2/bridge/hold=0"):
        b = tracks.apply(m, spec)
        assert np.array_equal(a.players, b.players, equal_nan=True)
        assert np.array_equal(a.slot_team, b.slot_team)


def test_hold_extends_an_unlinked_end_with_the_team_drift():
    m = _line_match()
    m.players[100:, 0] = np.nan  # player 0 vanishes for good at frame 100
    rep = tracks.repair(m, hold_s=1.0)
    seen = _seen(rep.match)
    assert seen[100:125, 0].all() and not seen[125:, 0].any()
    assert rep.filled[100:125, 0].all() and not rep.filled[:100].any()
    # after frame 99 the team's drift is its one seen player's motion; the held player moves with it
    assert np.allclose(rep.match.players[124, 0], m.players[99, 0] + (rep.match.players[124, 1]
                                                                      - m.players[99, 1]))
    off = tracks.repair(m)
    assert not _seen(off.match)[100:, 0].any()


def test_hold_extends_an_unlinked_start_backwards_and_stays_in_its_period():
    m = _line_match()
    m.players[:150, 2] = np.nan  # the away player appears at 150 ...
    m.period[140:] = 2  # ... 10 frames into the second period
    rep = tracks.repair(m, hold_s=2.0)
    seen = _seen(rep.match)
    assert seen[140:, 2].all() and not seen[:140, 2].any()
    assert rep.filled[140:150, 2].all()


def test_hold_never_overwrites_a_busy_slot_and_linked_ends_are_bridged_not_held():
    m = _line_match()
    players = np.concatenate([m.players, np.full((len(m.players), 1, 2), np.nan)], axis=1)
    players[80:, 3] = players[80:, 0]
    players[70:, 0] = np.nan
    broken = dataclasses.replace(m, players=players,
                                 slot_team=np.array([HOME, HOME, AWAY, HOME], np.int8)).validate()
    a = tracks.repair(broken)
    b = tracks.repair(broken, hold_s=2.0)
    assert np.array_equal(a.match.players, b.match.players, equal_nan=True)
    assert np.array_equal(a.filled, b.filled)


def test_apply_parses_hold_options():
    m = _line_match()
    m.players[100:, 0] = np.nan
    assert _seen(tracks.apply(m, "1/hold=2"))[149, 0]
    assert _seen(tracks.apply(m, "hold=2"))[149, 0]
    assert not _seen(tracks.apply(m, "1"))[149, 0]
    try:
        tracks.apply(m, "1/bogus=2")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown option accepted")
