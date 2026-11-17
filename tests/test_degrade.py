import numpy as np
import pytest

from src import evaluate
from src.ingest import degrade
from src.ingest import synthetic as synthetic_source
from src.official import X


@pytest.fixture
def match():
    return synthetic_source.load("0")


def _missing(m):
    graded = m.labels != X
    return np.isnan(m.ball).any(axis=1)[graded].mean()


# The synthetic bundle already lacks the ball on ~42% of graded frames (it mimics BAR-ATH), so
# targets here sit above that.

def test_hits_target_and_is_deterministic(match):
    out = degrade.drop_ball(match, target=0.6)
    assert abs(_missing(out) - 0.6) < 0.03  # short match: one gap is ~1%
    assert np.array_equal(np.isnan(out.ball), np.isnan(degrade.drop_ball(match, target=0.6).ball))
    other = degrade.drop_ball(match, target=0.6, seed=1)
    assert not np.array_equal(np.isnan(out.ball), np.isnan(other.ball))
    # the input is untouched, and only the ball changes
    assert _missing(match) < 0.5
    assert out.players is match.players and np.array_equal(out.labels, match.labels)


def test_gaps_are_runs_and_monotone(match):
    low = np.isnan(degrade.drop_ball(match, target=0.55, gap_s=2.0).ball[:, 0])
    high = np.isnan(degrade.drop_ball(match, target=0.7, gap_s=2.0).ball[:, 0])
    assert (high | ~low).all()  # more hiding only adds gaps
    hidden = high & ~np.isnan(match.ball[:, 0])
    starts = np.flatnonzero(np.diff(hidden.astype(int)) == 1)
    assert hidden.sum() / max(len(starts), 1) > match.fps  # median 2 s gaps: runs, not frames


def test_crowded_frames_lose_the_ball_more(match):
    r = degrade.risk(match)
    hidden = np.isnan(degrade.drop_ball(match, target=0.7).ball[:, 0]) & ~np.isnan(match.ball[:, 0])
    assert r[hidden].mean() > r[~hidden & ~np.isnan(match.ball[:, 0])].mean()


def test_already_missing_enough(match):
    out = degrade.drop_ball(match, target=0.0)
    assert np.array_equal(np.isnan(out.ball), np.isnan(match.ball))
    with pytest.raises(ValueError):
        degrade.drop_ball(match, target=1.0)


def test_evaluate_load_option(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CACHE_DIR", str(tmp_path / "cache"))
    m = evaluate.load_match("synthetic:0", {"ball_missing": 0.6})
    assert abs(_missing(m) - 0.6) < 0.03
    cached = evaluate.load_match("synthetic:0", {"ball_missing": 0.6})
    assert np.array_equal(np.isnan(m.ball), np.isnan(cached.ball))
    with pytest.raises(ValueError):
        evaluate.load_match("synthetic:0", {"ball_seed": 1}, cache=False)
    with pytest.raises(ValueError):
        evaluate.parse_matches("degrade:all")


def test_quiet_stretches_hide_calm_play_first(match):
    plain = degrade.drop_ball(match, target=0.6)
    out = degrade.drop_ball(match, target=0.6, quiet=0.75)
    assert abs(_missing(out) - 0.6) < 0.03
    # deterministic, and the default quiet=0 is the old dropout
    assert np.array_equal(np.isnan(out.ball),
                          np.isnan(degrade.drop_ball(match, target=0.6, quiet=0.75).ball))
    assert np.array_equal(np.isnan(plain.ball),
                          np.isnan(degrade.drop_ball(match, target=0.6, quiet=0.0).ball))
    # the hidden frames are calmer than with plain dropout
    speed = degrade.calm(match)
    had = ~np.isnan(match.ball[:, 0])
    newly = lambda m: np.isnan(m.ball[:, 0]) & had & np.isfinite(speed)
    assert speed[newly(out)].mean() < speed[newly(plain)].mean()
    # quiet stretches are runs (the synthetic ball's own gaps cut them, so only > 1 s here)
    q = degrade.quiet_stretches(match, match.labels != X, had, 0.1)
    starts = np.flatnonzero(np.diff(q.astype(int)) == 1)
    assert q.sum() / max(len(starts), 1) > match.fps
    with pytest.raises(ValueError):
        degrade.drop_ball(match, target=0.6, quiet=1.5)


def test_evaluate_quiet_option(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CACHE_DIR", str(tmp_path / "cache"))
    m = evaluate.load_match("synthetic:0", {"ball_missing": 0.6, "ball_quiet": 0.75}, cache=False)
    assert abs(_missing(m) - 0.6) < 0.03
    with pytest.raises(ValueError):
        evaluate.load_match("synthetic:0", {"ball_quiet": 0.5}, cache=False)


# -- players (C11) -------------------------------------------------------------------------------------

def _present(m):
    return ~np.isnan(m.players).any(axis=2)


def test_clean_and_unknown_options(match):
    assert degrade.apply(match, {"clean": 1, "ball_missing": 0.6, "player_drop": 0.2}) is match
    with pytest.raises(ValueError):
        degrade.apply(match, {"player_dorp": 0.1})
    with pytest.raises(ValueError):
        degrade.apply(match, {"camera_fill": "estimate"})


def test_player_drop_hides_runs_deterministically(match):
    out = degrade.degrade_players(match, player_drop=0.2)
    before, after = _present(match), _present(out)
    assert (before | ~after).all()  # only hides
    lost = (before & ~after).sum() / before.sum()
    assert 0.08 < lost < 0.3
    again = degrade.degrade_players(match, player_drop=0.2)
    assert np.array_equal(np.isnan(out.players), np.isnan(again.players))
    other = degrade.degrade_players(match, player_drop=0.2, seed=3)
    assert not np.array_equal(np.isnan(out.players), np.isnan(other.players))
    assert out.ball is match.ball and np.array_equal(out.labels, match.labels)


def test_camera_hides_off_view_players(match):
    out = degrade.degrade_players(match, camera=0.34)
    view = degrade.camera_view(match, 0.34)
    assert np.array_equal(_present(out), view & _present(match))
    share = view.sum() / _present(match).sum()
    assert 0.15 < share < 0.75
    # a window on the whole pitch, nearly: almost nobody hidden
    assert degrade.camera_view(match, 1.0).sum() > view.sum()


def test_camera_width_matches_share():
    pitch = (105.0, 68.0)
    near = degrade.camera_width(pitch, 0.34)
    assert 15 < near < 25  # SkillCorner's near edge ~19 m (F111)
    assert abs(degrade.camera_width(pitch, 0.17) - near / 2) < 1e-9


def test_camera_estimate_fills_from_last_seen(match):
    out = degrade.degrade_players(match, camera=0.34, camera_fill="estimate")
    view = degrade.camera_view(match, 0.34) & _present(match)
    assert np.array_equal(out.players[view], match.players[view])  # seen players untouched
    filled = _present(out) & ~view
    assert filled.any() and (_present(match) | ~_present(out)).all()  # no ghosts: absent stays absent
    length, width = match.pitch
    assert (out.players[..., 0][filled] >= 0).all() and (out.players[..., 0][filled] <= length).all()
    assert (out.players[..., 1][filled] <= width).all()
    # an estimate stays nearer the truth than a random spot would
    err = np.hypot(*(out.players[filled] - match.players[filled]).T)
    assert np.median(err) < 25


def test_id_swap_keeps_team_positions(match):
    out = degrade.degrade_players(match, id_swap=2.0)
    assert not np.array_equal(np.nan_to_num(out.players), np.nan_to_num(match.players))
    for team in np.unique(match.slot_team):
        slots = match.slot_team == team
        a = np.sort(np.nan_to_num(match.players[:, slots, 0], nan=-1), axis=1)
        b = np.sort(np.nan_to_num(out.players[:, slots, 0], nan=-1), axis=1)
        assert np.array_equal(a, b)  # same positions per team and frame, only the slots change


def test_ball_gaps_do_not_depend_on_player_options(match):
    ball = degrade.apply(match, {"ball_missing": 0.6})
    both = degrade.apply(match, {"ball_missing": 0.6, "camera": 0.34, "player_drop": 0.1})
    assert np.array_equal(np.isnan(ball.ball), np.isnan(both.ball))
    assert np.isnan(both.players).sum() > np.isnan(ball.players).sum()


def test_load_match_takes_player_options(match):
    out = evaluate.load_match("synthetic:0", {"player_drop": 0.2}, cache=False)
    clean = evaluate.load_match("synthetic:0", {"clean": 1}, cache=False)
    assert np.isnan(out.players).sum() > np.isnan(clean.players).sum()


def test_fill_gaps_fills_short_inner_gaps_only(match):
    gappy = degrade.degrade_players(match, player_drop=0.2, player_gap_s=2.0)
    out = degrade.fill_gaps(gappy, 3.0)
    before, after = _present(gappy), _present(out)
    assert (after | ~before).all() and after.sum() > before.sum()
    assert np.array_equal(out.players[before], gappy.players[before])  # seen positions untouched
    # a gap longer than the limit stays open
    long_gap = degrade.fill_gaps(gappy, 0.5)
    assert _present(long_gap).sum() < after.sum()
    via = degrade.apply(match, {"player_drop": 0.2, "player_gap_s": 2.0, "player_fill": 3.0})
    assert np.array_equal(np.isnan(via.players), np.isnan(out.players))
