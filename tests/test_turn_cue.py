"""Wave 5 B38: the `turn` cue (src/models/cues/turn.py); synthetic only."""

import numpy as np
import pytest

from src.models import switch_cue
from src.models.cues import turn
from src.models.learned import LearnedDecoder
from src.schema import Match


def _match(n=400, seed=0, fps=25):
    """Two 11-player teams, home attacking +x; home runs forward then brakes at the middle."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([np.c_[np.r_[3, rng.uniform(10, 50, 10)], rng.uniform(5, 63, 11)],
                           np.c_[np.r_[102, rng.uniform(55, 95, 10)], rng.uniform(5, 63, 11)]])
    t = np.arange(n) / fps
    speed = np.where(t < t[n // 2], 4.0, 0.5)  # home slows sharply halfway
    pos = base[None] + np.cumsum(rng.normal(0, 0.02, (n, 22, 2)), axis=0)
    pos[:, :11, 0] += (np.cumsum(speed) / fps)[:, None]
    return Match(match_id="t", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, dtype=np.int8),
                 ball=np.full((n, 2), np.nan), players=pos,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_columns_and_symmetry():
    assert len(turn.columns("new")) == 8
    assert turn.columns("full")[:len(switch_cue.COLUMNS)] == [("v1_" + f, w) for f, w in switch_cue.COLUMNS]
    with pytest.raises(ValueError):
        turn.columns("other")
    m = _match()
    u_away, u_home = turn.cue_values(m, "full")
    assert u_away.shape == (m.n_frames, 14)
    assert np.array_equal(u_home, -u_away, equal_nan=True)
    swapped = Match(match_id="t", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
                    players=np.concatenate([m.players[:, 11:], m.players[:, :11]], axis=1),
                    slot_team=m.slot_team).validate()
    u_sw, _ = turn.cue_values(swapped, "new")
    assert np.allclose(u_sw, -turn.cue_values(m, "new")[0], equal_nan=True)


def test_braking_home_team_points_to_the_away_team():
    m = _match()
    u, _ = turn.cue_values(m, "new")
    mid = m.n_frames // 2
    names = turn.NEW_COLUMNS
    acc = names.index(("acc_all", 0.5))
    brake = names.index(("brake", 0.5))
    assert u[mid, acc] > 2.0  # home decelerates by ~3.5 m/s: u toward away is positive
    assert u[mid, brake] > 1.0  # and still runs forward after
    assert abs(u[mid // 2, acc]) < 0.5  # steady running: no signal


def test_effect_gain_zero_set_auto_and_unknown_frames(tmp_path):
    m = _match()
    beta = {w: [0.0] + [0.1] * (turn.design(np.zeros((1, len(turn.columns(w)))), w).shape[1] - 1)
            for w in turn.SETS}
    entry = {w: {"all": switch_cue.fit_entry(np.array(beta[w]), -2.0), "lomo": {}, "loso": {}}
             for w in turn.SETS}
    path = tmp_path / "turn.json"
    path.write_text(__import__("json").dumps({"spec_hash": turn.SPEC_HASH, "sets": entry}))
    dec = LearnedDecoder()
    assert turn.effect(m, m, dec, 0.0, file=str(path)) is None
    assert turn.pick_set("auto", dec) == "full"
    assert turn.pick_set("auto", LearnedDecoder(switch_cue_gain=0.5)) == "new"
    eff = turn.effect(m, m, dec, 1.0, file=str(path))
    assert eff.ha.shape == (m.n_frames,) and np.all((eff.ha >= 0.25) & (eff.ha <= 2.0))
    assert eff.hd is None and eff.add is None
    few = Match(match_id="t", fps=m.fps, pitch=m.pitch, period=m.period, ball=m.ball,
                players=np.where(np.arange(22)[None, :, None] < 6, np.nan, m.players),
                slot_team=m.slot_team).validate()  # home has 5 players: every column unknown
    eff = turn.effect(few, few, dec, 1.0, file=str(path))
    assert np.all(eff.ha == 1.0) and np.all(eff.ah == 1.0)  # unknown -> the flat cost
    bad = tmp_path / "bad.json"
    bad.write_text('{"spec_hash": "x", "sets": {}}')
    with pytest.raises(ValueError, match="another turn-cue spec"):
        turn.load(str(bad))


def test_committed_file_matches_the_spec_and_decodes(small):
    cue = turn.load()
    assert set(cue["sets"]) == set(turn.SETS)
    for w in turn.SETS:
        assert len(cue["sets"][w]["all"]["beta"]) == turn.design(
            np.zeros((1, len(turn.columns(w)))), w).shape[1]
    match = small["match"]
    base = LearnedDecoder(switch_cue_gain=0.5).predict(match)
    same = LearnedDecoder(switch_cue_gain=0.5, cues="turn:0").predict(match)
    assert np.array_equal(base, same)
    a = LearnedDecoder(switch_cue_gain=0.5, cues="turn:0.5").predict(match)
    b = LearnedDecoder(switch_cue_gain=0.5, cues="turn:0.5").predict(match)
    assert np.array_equal(a, b)
    LearnedDecoder(cues="turn:0.5/set=full").predict(match)
