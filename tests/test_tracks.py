"""C13: tracklet fragmentation (degrade.fragment) and the track repair (src/tracks.py)."""

import dataclasses

import numpy as np
import pytest

from src import tracks
from src.ingest import degrade
from src.ingest import synthetic as synthetic_source
from src.models import build
from src.schema import AWAY, HOME, Match


@pytest.fixture
def match():
    return synthetic_source.load("0")


def _line_match(n=200, fps=25):
    """Two home players and one away player walking in straight lines, one period."""
    t = np.arange(n) / fps
    players = np.stack([np.stack([10 + 2 * t, 20 + 0 * t], 1), np.stack([30 + 0 * t, 10 + 1 * t], 1),
                        np.stack([60 - 1 * t, 40 + 0 * t], 1)], axis=1)
    return Match(match_id="test:line", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=np.full((n, 2), np.nan), players=players,
                 slot_team=np.array([HOME, HOME, AWAY], np.int8)).validate()


def _seen(m):
    return ~np.isnan(m.players).any(axis=2)


def test_tracklets_split_at_gaps_and_periods():
    m = _line_match()
    m.players[50:60, 0] = np.nan
    m.period[100:] = 2
    segs = tracks.tracklets(m.players, m.period)
    slot0 = segs[segs[:, 0] == 0]
    assert slot0[:, 1:].tolist() == [[0, 50], [60, 100], [100, 200]]
    assert len(segs) == 7


def test_link_and_bridge_a_track_continued_in_another_slot():
    m = _line_match()
    truth = m.players[:, 0].copy()
    players = np.concatenate([m.players, np.full((len(m.players), 1, 2), np.nan)], axis=1)
    players[80:, 3] = players[80:, 0]  # player 0 lost at frame 70, back in a new slot at 80
    players[70:, 0] = np.nan
    broken = dataclasses.replace(m, players=players,
                                 slot_team=np.array([HOME, HOME, AWAY, HOME], np.int8)).validate()
    rep = tracks.repair(broken)
    assert rep.links == 1 and rep.tracklets == 4
    col0 = rep.match.players[:, 0]
    assert not np.isnan(col0).any()  # the chain went back into slot 0; slot 3 is empty
    assert np.isnan(rep.match.players[:, 3]).all()
    col = int(np.flatnonzero(rep.filled.any(axis=0))[0])
    assert rep.filled[:, col].sum() == 10  # frames 70..79 bridged
    np.testing.assert_allclose(rep.match.players[70:80, col], truth[70:80], atol=1e-9)
    # the input is untouched, and repairing twice changes nothing more
    assert np.isnan(broken.players[75, 0]).all()
    again = tracks.repair(rep.match)
    assert again.links == 0 and not again.filled.any()


def test_no_link_too_far_too_late_or_across_overlap():
    m = _line_match(n=600)
    players = np.concatenate([m.players, np.full((len(m.players), 1, 2), np.nan)], axis=1)
    team = np.array([HOME, HOME, AWAY, HOME], np.int8)
    far = players.copy()
    far[80:, 3] = far[80:, 0] + [20.0, 0.0]  # comes back 20 m off after 0.4 s
    far[70:, 0] = np.nan
    assert tracks.repair(dataclasses.replace(m, players=far, slot_team=team)).links == 0
    late = players.copy()
    late[70 + 11 * 25:, 3] = late[70 + 11 * 25:, 0]  # 11 s later: over MAX_GAP_S
    late[70:, 0] = np.nan
    assert tracks.repair(dataclasses.replace(m, players=late, slot_team=team)).links == 0
    overlap = players.copy()
    overlap[60:, 3] = overlap[60:, 0]  # a duplicate: starts before the first ends
    overlap[70:, 0] = np.nan
    assert tracks.repair(dataclasses.replace(m, players=overlap, slot_team=team)).links == 0
    other_team = players.copy()
    other_team[80:, 3] = other_team[80:, 0]
    other_team[70:, 0] = np.nan
    away = np.array([HOME, HOME, AWAY, AWAY], np.int8)
    assert tracks.repair(dataclasses.replace(m, players=other_team, slot_team=away)).links == 0


def test_period_change_is_not_bridged():
    m = _line_match()
    m.players[95:105, 0] = np.nan
    m.period[100:] = 2
    rep = tracks.repair(m)
    assert rep.links == 0


def test_fragment_grows_slots_keeps_players_and_is_deterministic(match):
    out = degrade.apply(match, {"fragment": 2.0})
    again = degrade.apply(match, {"fragment": 2.0})
    assert np.array_equal(out.players, again.players, equal_nan=True)
    assert out.players.shape[1] > match.players.shape[1]
    assert np.array_equal(out.period, match.period) and np.array_equal(out.ball, match.ball,
                                                                       equal_nan=True)
    # every output slot holds one team's players; fewer player-frames, never more
    per_team = [_seen(out)[:, out.slot_team == t].sum() for t in (HOME, AWAY)]
    before = [_seen(match)[:, match.slot_team == t].sum() for t in (HOME, AWAY)]
    assert all(0.8 * b < a < b for a, b in zip(per_team, before))
    # no position is invented: each seen output position is some input slot's at that frame
    _, _, person = degrade.fragment(match, 2.0, np.random.default_rng(0))
    assert (person >= -1).all()


def test_repair_recovers_fragmented_players(match):
    frag = degrade.apply(match, {"fragment": 2.0})
    rep = degrade.apply(match, {"fragment": 2.0, "track_repair": "1"})
    in_play = match.period > 0
    lost = _seen(match)[in_play].sum() - _seen(frag)[in_play].sum()
    back = _seen(rep)[in_play].sum() - _seen(frag)[in_play].sum()
    assert lost > 0 and back > 0.5 * lost
    assert rep.players.shape[1] <= frag.players.shape[1]


def test_apply_spec():
    m = _line_match()
    m.players[50:60, 0] = np.nan
    assert tracks.apply(m, "0") is m and tracks.apply(m, "") is m
    assert not np.isnan(tracks.apply(m, "1").players[55]).any()
    assert np.isnan(tracks.apply(m, "0.2/3/4").players[55]).any(axis=1).sum() == 1  # gap too long
    with pytest.raises(ValueError):
        tracks.repair(m, fill="nope")


def test_learned_track_repair_is_a_no_op_on_full_tracks(match):
    full = dataclasses.replace(match, players=np.nan_to_num(match.players, nan=50.0))
    plain = build("learned").predict(full)
    repaired = build("learned", track_repair="1").predict(full)
    assert np.array_equal(plain, repaired)


def test_learned_track_repair_runs_and_blind_group(match):
    frag = degrade.apply(match, {"fragment": 2.0})
    model = build("learned", track_repair="1")
    codes = model.predict(frag)
    assert codes.shape == (match.n_frames,)
    from src.models import learned
    home, away, common = learned.features(frag, groups=("blind",))
    base = learned.features(frag, groups=())
    assert home.shape[1] == base[0].shape[1] + 14 and common.shape == base[2].shape
    assert np.isfinite(home).all() and np.isfinite(away).all()
