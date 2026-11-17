"""Learned emissions for the decoder (backlog B10): features, the symmetric logit, coefficient folds."""

import dataclasses
import json

import numpy as np
import pytest

from src import submission
from src.models import build, learned
from src.models.learned import LearnedDecoder
from src.official import A, D, H
from src.schema import AWAY, HOME, Match


def test_hat_basis_interpolates_clamps_and_zeroes_nan():
    out = learned.hat(np.array([0.0, 0.5, 2.0, 9.0, np.nan]), (0.0, 1.0, 2.0))
    assert np.allclose(out, [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]])


def test_window_min_looks_back_and_ahead_ignoring_nan():
    v = np.array([5.0, np.nan, 1.0, 4.0, np.nan, np.nan])
    assert np.allclose(learned.window_min(v, 1, 0), [5, 5, 1, 1, 4, np.nan], equal_nan=True)
    assert np.allclose(learned.window_min(v, 0, 1), [5, 1, 1, 4, np.nan, np.nan], equal_nan=True)


def test_logits_are_home_away_symmetric():
    rng = np.random.default_rng(0)
    th, ta, cm = rng.random((5, 4)), rng.random((5, 4)), rng.random((5, 3))
    theta = rng.normal(size=3 * 4 + 3)
    z, swapped = learned.logits((th, ta, cm), theta), learned.logits((ta, th, cm), theta)
    assert np.allclose(z[:, 0], swapped[:, 1]) and np.allclose(z[:, 2], swapped[:, 2])


def test_fit_recovers_an_easy_rule():
    # home close -> H, away close -> A, neither -> D
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 3000)
    th = np.where(y[:, None] == 0, 1.0, 0.0) + 0.1 * rng.random((3000, 1))
    ta = np.where(y[:, None] == 1, 1.0, 0.0) + 0.1 * rng.random((3000, 1))
    cm = np.ones((3000, 1))
    theta = learned.fit_logit((th, ta, cm), y, np.full(3000, 1 / 3000))
    assert (np.argmax(learned.logits((th, ta, cm), theta), axis=1) == y).mean() > 0.99


def _kick_match(fps=10):
    """Ball still at x=0 for 2 s, then kicked along x at 10 m/s; a home player stands on the
    ball, an away player 5 m away."""
    n = 6 * fps
    t = np.arange(n) / fps
    ball = np.stack([np.where(t < 2, 0.0, 10 * (t - 2)), np.zeros(n)], axis=1)
    players = np.stack([np.zeros((n, 2)), np.tile([0.0, 5.0], (n, 1))], axis=1)
    return Match(match_id="test:kick", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=ball, players=players, slot_team=np.array([HOME, AWAY], np.int8))


def test_ball_impulse_peaks_at_the_kick():
    match = _kick_match()
    impulse = learned.ball_impulse(match)
    assert np.nanargmax(impulse) == 20 and np.isclose(np.nanmax(impulse), 10.0)
    assert np.isnan(impulse[:2]).all() and np.allclose(impulse[2:18], 0)


def test_touches_go_to_the_nearer_team_within_reach():
    match = _kick_match()
    d_home, d_away = learned.team_distances(match)
    t_home, t_away = learned.touches(match, d_home, d_away)
    assert t_home[20] == 1.0 and not t_away.any()
    far = dataclasses.replace(match, players=match.players + [0.0, 3.0])  # home 3 m away
    assert not learned.touches(far, *learned.team_distances(far))[0].any()


def test_since_last_counts_seconds_from_the_last_hit():
    hit = np.array([False, True, False, False, True, False])
    assert np.allclose(learned.since_last(hit, 2), [np.inf, 0, 0.5, 1, 0, 0.5])


def test_feature_groups_add_columns_and_change_the_hash():
    match = _kick_match()
    widths = {g: learned.features(match, g)[0].shape[1]
              for g in [(), ("touch",), ("duel",), ("duel", "touch")]}
    assert widths[("touch",)] > widths[()] and widths[("duel",)] > widths[()]
    assert widths[("duel", "touch")] == widths[("touch",)] + widths[("duel",)] - widths[()]
    assert len({learned.feature_hash(g) for g in widths}) == 4
    with pytest.raises(ValueError):
        learned.features(match, ("nope",))


def test_gap_group_adds_common_columns():
    """C6: group "gap" adds dropout context to the common block only, zero where the ball is seen."""
    match = _kick_match()
    match.ball[10:14] = np.nan
    th, _, cm = learned.features(match, ("touch",))
    th_gap, _, cm_gap = learned.features(match, ("gap", "touch"))
    assert th_gap.shape == th.shape and np.array_equal(cm_gap[:, :cm.shape[1]], cm)
    extra = cm_gap[:, cm.shape[1]:]
    assert extra.shape[1] == len(learned.GAP_KNOTS) + len(learned.SHARE_KNOTS)
    seen = ~np.isnan(match.ball[:, 0])
    assert not extra[seen].any() and np.allclose(extra[~seen].sum(axis=1), 2.0)
    assert learned.feature_hash(("gap", "touch")) != learned.feature_hash(("touch",))


def test_speedspan_group_and_override_change_only_the_speed_columns():
    """C5: "speedspan" (or speed_span_s) changes the two speed hats, no new columns, new hash."""
    match = _kick_match()
    th, ta, cm = learned.features(match, ("touch",))
    th_s, ta_s, cm_s = learned.features(match, ("speedspan", "touch"))
    assert np.array_equal(th, th_s) and np.array_equal(ta, ta_s) and cm.shape == cm_s.shape
    over = learned.features(match, ("touch",), speed_span_s=learned.SPEED_SPAN_S)[2]
    assert np.array_equal(over, cm_s)
    assert np.array_equal(learned.features(match, ("speedspan", "touch"), speed_span_s=0)[2], cm)
    assert learned.feature_hash(("speedspan", "touch")) != learned.feature_hash(("touch",))


def _smeared_kick_match(fps=10, noise=0.0):
    """Like _kick_match, but the kick takes 1 s to reach 4 m/s (a broadcast-smoothed ball), with
    optional seeded jitter on the ball."""
    match = _kick_match(fps)
    t = np.arange(match.n_frames) / fps
    x = np.where(t < 2, 0.0, np.where(t < 3, 2 * (t - 2) ** 2, 2 + 4 * (t - 3)))
    ball = np.stack([x, np.zeros_like(x)], axis=1)
    ball += noise * np.random.default_rng(0).standard_normal(ball.shape)
    return dataclasses.replace(match, ball=ball)


def test_touch_scale_widens_only_on_jittery_balls():
    """B14: s = max(1, ball_jitter / ref) ** exp; 1 on a clean ball, when off, or unmeasurable."""
    assert learned.touch_scale(_kick_match()) == 1.0
    noisy = _smeared_kick_match(noise=0.08)
    jitter = learned.ball_jitter(noisy)
    assert jitter > learned.NTOUCH_REF_M
    assert np.isclose(learned.touch_scale(noisy), (jitter / learned.NTOUCH_REF_M) ** 2)
    assert learned.touch_scale(noisy, noise_ref_m=0) == 1.0
    gone = dataclasses.replace(noisy, ball=np.full_like(noisy.ball, np.nan))
    assert learned.touch_scale(gone) == 1.0


def test_wider_step_sees_a_smeared_kick():
    """A kick spread over 1 s barely moves a 0.2 s impulse; a longer step and a lower threshold
    (what "ntouch" does on a jittery ball) count it as a touch."""
    match = _smeared_kick_match()
    d_home, d_away = learned.team_distances(match)
    assert np.nanmax(learned.ball_impulse(match)) < learned.TOUCH_MPS[0] + 0.5
    narrow = learned.touches(match, d_home, d_away)[0]
    wide = learned.touches(match, d_home, d_away, step_s=0.6, mps=(1.0, 3.0))[0]
    assert narrow.max() < learned.TOUCH_MIN <= wide.max()


def test_ntouch_group_matches_touch_on_a_clean_ball():
    """On a clean ball s = 1, so "ntouch" columns equal "touch" columns (IDSSE/Metrica unchanged)."""
    match = _kick_match()
    base = learned.features(match, ())[0].shape[1]
    touch, ntouch = learned.features(match, ("touch",))[0], learned.features(match, ("ntouch",))[0]
    assert np.array_equal(touch, ntouch)
    both = learned.features(match, ("ntouch", "touch"))[0]
    assert np.array_equal(both[:, base:touch.shape[1]], both[:, touch.shape[1]:])
    assert learned.feature_hash(("ntouch",)) != learned.feature_hash(("touch",))


def test_gap_lengths_cover_each_gap():
    missing = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert np.allclose(learned.gap_lengths(missing, 2), [0, 1, 1, 0, 2, 2, 2, 2, 0])


@pytest.fixture(scope="module")
def coef_file(small, tmp_path_factory):
    """Coefficients fit on the small synthetic match, saved like scripts/fit_learned.py does."""
    match = dataclasses.replace(small["match"], labels=submission.read_codes(small["gt"]))
    rows = {"synthetic:0": learned.training_rows(match)}
    fold = learned.fit(rows)
    path = str(tmp_path_factory.mktemp("learned") / "coefs.json")
    with open(path, "w") as handle:
        json.dump({"feature_hash": learned.FEATURE_HASH, "groups": list(learned.GROUPS),
                   "all": fold, "lomo": {small["match"].match_id: fold}, "loso": {}}, handle)
    return path


def test_model_is_registered_deterministic_and_valid(small, coef_file):
    match = small["match"]
    model = build("learned", coef_file=coef_file)
    assert isinstance(model, LearnedDecoder)
    first = model.predict(match)
    assert first.shape == (match.n_frames,) and first.dtype == np.int8
    assert np.array_equal(first, build("learned", coef_file=coef_file).predict(match))
    assert set(np.unique(first)) <= {H, A, D}


def test_folds_pick_the_held_out_fit(small, coef_file):
    match = small["match"]
    assert np.array_equal(build("learned", coef_file=coef_file, fold="lomo").predict(match),
                          build("learned", coef_file=coef_file).predict(match))
    with pytest.raises(KeyError):
        build("learned", coef_file=coef_file, fold="loso").predict(match)
    with pytest.raises(ValueError):
        build("learned", fold="nope")


def test_coefficients_with_groups_use_their_features(small, tmp_path):
    match = dataclasses.replace(small["match"], labels=submission.read_codes(small["gt"]))
    groups = ("duel", "touch")
    fold = learned.fit({"synthetic:0": learned.training_rows(match, groups=groups)})
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"feature_hash": learned.feature_hash(groups),
                                "groups": list(groups), "all": fold}))
    codes = build("learned", coef_file=str(path)).predict(small["match"])
    assert codes.shape == (match.n_frames,) and set(np.unique(codes)) <= {H, A, D}
    stale = tmp_path / "stale.json"  # base features' hash, but fit with groups
    stale.write_text(json.dumps({"feature_hash": learned.feature_hash(), "groups": list(groups)}))
    with pytest.raises(ValueError, match="fit_learned"):
        learned.load_coefs(str(stale))


def test_noise_knobs_defaults(small, coef_file):
    """C5: speed_span_s=None and hand_pjitter_ref_m=0 are v2's model; the default since v3 (F80)
    is speed_span_s=1.0, which changes the scores, and hand_pjitter_ref_m stays off."""
    match = small["match"]
    base = build("learned", coef_file=coef_file).scores(match)
    assert np.array_equal(build("learned", coef_file=coef_file, speed_span_s=1.0,
                                hand_pjitter_ref_m=0.0).scores(match), base)
    assert not np.array_equal(build("learned", coef_file=coef_file,
                                    speed_span_s=None).scores(match), base)


def test_stale_coefficients_are_refused(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"feature_hash": "other"}))
    with pytest.raises(ValueError, match="fit_learned"):
        learned.load_coefs(str(path))


def test_committed_coefficients_match_the_features():
    coefs = learned.load_coefs()
    assert coefs["groups"] == list(learned.GROUPS)
    shape = sum(len(knots) * len(windows) for knots, windows in learned.SHAPE["features"].values())
    assert shape == 73  # B25's spec: fb_w 3 x 7, goalside 2 x 7, press 5, stretch 5, work and vx 2 x 7
    assert len(coefs["all"]["theta"]) == 3 * (43 + 15 + shape) + 19  # base, touch, shapem
    assert set(coefs["loso"]) == {"idsse", "skillcorner", "metrica"}


def _shape_match(n=120, fps=10, wide_home=True):
    """Two 11-player teams on a pitch; the home outfielders spread wider than the away ones."""
    rng = np.random.default_rng(1)
    home_y = np.linspace(4, 64, 10) if wide_home else np.linspace(24, 44, 10)
    away_y = np.linspace(24, 44, 10) if wide_home else np.linspace(4, 64, 10)
    home = np.c_[np.r_[3, rng.uniform(20, 50, 10)], np.r_[34, home_y]]
    away = np.c_[np.r_[102, rng.uniform(55, 85, 10)], np.r_[34, away_y]]
    walk = np.cumsum(rng.normal(0, 0.05, (n, 22, 2)), axis=0)
    players = np.concatenate([home, away])[None] + walk
    return Match(match_id="test:shape", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=np.full((n, 2), np.nan), players=players,
                 slot_team=np.r_[np.zeros(11), np.ones(11)].astype(np.int8)).validate()


def test_shape_group_is_team_symmetric_and_ball_free():
    """B25: swapping the teams swaps the shape blocks; the ball doesn't change them."""
    match = _shape_match()
    home, away = learned.shape_blocks(match)
    assert home.shape == away.shape and home.shape[0] == match.n_frames
    assert (home.sum(axis=1) > 0).all()  # every frame has evidence (all players seen)
    swapped = dataclasses.replace(match, slot_team=1 - match.slot_team)
    s_home, s_away = learned.shape_blocks(swapped)
    assert np.allclose(home, s_away) and np.allclose(away, s_home)
    with_ball = dataclasses.replace(match, ball=np.tile([52.0, 34.0], (match.n_frames, 1)))
    assert np.allclose(learned.shape_blocks(with_ball)[0], home)


def test_shape_values_see_the_wider_team_and_look_ahead():
    match = _shape_match()
    values = learned.shape_values(match)
    fbw = values[("fb_w", (-2.0, 2.0))]
    assert np.nanmin(fbw) > 30  # home ~60 m wide, away ~20 m
    flipped = _shape_match(wide_home=False)
    assert np.nanmax(learned.shape_values(flipped)[("fb_w", (-2.0, 2.0))]) < -30
    assert any(w[0] >= 0 for _, ws in learned.SHAPE["features"].values() for w in ws)


def test_offset_mean_windows():
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.allclose(learned._offset_mean(v, 0, 1), [1.5, 2, 4, 4.5, 5])
    assert np.allclose(learned._offset_mean(v, -1, 1), [1.5, 1.5, 3, 4.5, 4.5])


def test_shape_group_adds_team_columns_and_changes_the_hash():
    match = _shape_match()
    base = learned.features(match, ("touch",))
    both = learned.features(match, ("shape", "touch"))
    width = learned.shape_blocks(match)[0].shape[1]
    assert both[0].shape[1] == base[0].shape[1] + width and both[2].shape == base[2].shape
    assert learned.feature_hash(("shape", "touch")) != learned.feature_hash(("touch",))


def test_gap_near_stop_marks_gaps_next_to_a_ball_off_the_pitch():
    n, fps = 100, 10
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[20:30] = np.nan  # gap 1: the ball is seen off the pitch 1 s after it
    ball[40:45] = (50.0, -5.0)
    ball[80:90] = np.nan  # gap 2: the ball out is 3.5 s before it
    match = Match(match_id="t:stop", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                  ball=ball, players=np.zeros((n, 2, 2)), slot_team=np.array([HOME, AWAY], np.int8))
    missing = np.isnan(ball[:, 0])
    near = learned.gap_near_stop(match, missing, stop_s=3.0)
    assert near[20:30].all() and not near[80:90].any() and not near[~missing].any()
    assert not learned.gap_near_stop(match, missing, stop_s=1.0)[20:30].any()
    long = learned.gap_long(missing, fps, (0.5, 2.0))
    assert np.allclose(long[20:30], 1 / 3) and np.allclose(long[80:90], 1 / 3)
    assert long[~missing].sum() == 0


def test_split_shape_groups_double_the_shapem_columns():
    match = _shape_match()
    match.ball[:60] = (50.0, 30.0)
    base = learned.features(match, groups=("shapem",))[0]
    plain = learned.features(match, groups=())[0]
    k = base.shape[1] - plain.shape[1]
    for group in ("shapeg", "shapes"):
        split = learned.features(match, groups=(group,))[0]
        assert split.shape[1] == plain.shape[1] + 2 * k
        both = split[:, plain.shape[1]:plain.shape[1] + k] + split[:, plain.shape[1] + k:]
        assert np.allclose(both, base[:, plain.shape[1]:])
    assert learned.feature_hash(("shapeg",)) != learned.feature_hash(("shapem",))
