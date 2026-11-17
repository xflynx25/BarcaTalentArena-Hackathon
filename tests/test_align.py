"""V4: aligning a second recording (src/align.py), the simulated one (degrade.second_recording), the
broadcast ball in the learned model, and the synthetic bundle's broadcast files. Synthetic data only."""

import dataclasses

import numpy as np
import pytest

from src import align, broadcast, synthetic
from src.ingest import bar_ath, degrade
from src.models import build, learned
from src.schema import Match

SPEC = synthetic.Spec(n_frames=40000, halves=((1, 1000, 18999), (2, 21000, 38999)),
                      kickoffs=(1000, 21100), n_slots=30, n_tags=None, n_broadcast=40000, seed=3)


def _clean(spec: synthetic.Spec) -> Match:
    sim = synthetic.simulate(spec)
    period = np.zeros(spec.n_frames, dtype=np.int8)
    for half, first, last in spec.halves:
        period[first:last + 1] = half
    return Match(match_id="synthetic:{}".format(spec.seed), fps=spec.fps, pitch=spec.pitch,
                 period=period, ball=sim["ball_m"], players=sim["pos"],
                 slot_team=np.asarray(sim["team_of"], np.int8), labels=sim["labels"]).validate()


@pytest.fixture(scope="module")
def pair():
    clean = _clean(SPEC)
    tactical = degrade.apply(clean, {"ball_missing": 0.41, "ball_quiet": 0.5, "ball_gap_s": 1.0})
    other, truth, kind = degrade.second_recording(clean, drift_ppm=80.0)
    return clean, tactical, other, truth, kind


def _errors(alignment, other, truth):
    data = align.Recording.of(other).has_data()
    m = alignment.matched & data & ~np.isnan(truth)
    return np.abs(alignment.frame[m] - truth[m]), data


def test_second_recording_has_its_own_timeline(pair):
    clean, _, other, truth, kind = pair
    assert other.n_frames != clean.n_frames
    assert set(np.unique(kind)) <= set(range(len(degrade.SECOND_KINDS)))
    assert (kind == 0).any() and (kind == 1).any() and (kind == 2).any()
    assert np.isnan(truth[kind == 0]).all() and not np.isnan(truth[kind == 1]).any()
    tracked = (kind == 1) | (kind == 3)
    assert np.isnan(other.ball[~tracked]).all() and np.isnan(other.players[~tracked]).all()
    again, truth2, kind2 = degrade.second_recording(clean, drift_ppm=80.0)
    assert np.array_equal(again.ball, other.ball, equal_nan=True) and np.array_equal(kind, kind2)


def test_align_recovers_the_map(pair):
    _, tactical, other, truth, _ = pair
    alignment = align.align(tactical, other)
    err, data = _errors(alignment, other, truth)
    assert (alignment.matched & data).sum() > 0.8 * data.sum()
    assert np.median(err) <= 1.0 and np.mean(err <= 2) > 0.97
    assert abs(alignment.drift_ppm - 80) <= 40
    again = align.align(tactical, other)
    assert np.array_equal(again.frame, alignment.frame)


def test_align_handles_a_flip_and_another_frame_rate(pair):
    clean, tactical, _, _, _ = pair
    other, truth, _ = degrade.second_recording(clean, fps=30, drift_ppm=0.0, seed=1)
    size = np.array(clean.pitch)
    flipped = dataclasses.replace(other, ball=size - other.ball, players=size - other.players)
    alignment = align.align(tactical, flipped)
    assert alignment.flip
    err, _ = _errors(alignment, flipped, truth)
    assert np.mean(err <= 2) > 0.95


def test_a_recording_of_another_match_stays_unmatched(pair):
    _, tactical, _, _, _ = pair
    other = degrade.second_recording(_clean(dataclasses.replace(SPEC, seed=4)))[0]
    alignment = align.align(tactical, other)
    fused = align.attach(tactical, other, alignment)
    health = align.health(tactical, other, alignment, fused)
    assert health["matched_share_of_data"] < 0.2
    assert broadcast.decide(health)[0] == "STAY"


def test_map_frames_is_integer_arithmetic():
    frames = np.arange(5)
    assert np.array_equal(align.map_frames(frames, 1500, 0, 25, 25), frames + 2)  # 1.5 rounds up
    assert np.array_equal(align.map_frames(frames, 0, 0, 25, 50), [0, 1, 1, 2, 2])
    assert align.map_frames(np.array([1_000_000]), 0, 100, 25, 25)[0] == 1_000_100


def test_attach_and_health(pair):
    clean, tactical, other, truth, kind = pair
    exact = align.from_truth(tactical, other, truth, (kind == 1) | (kind == 2))
    fused = align.attach(tactical, other, exact, bias=False)
    assert fused.ball2.shape == tactical.ball.shape
    assert np.isnan(fused.ball2[exact.frame[exact.frame >= 0].max() + 1:]).all()
    hidden = np.isnan(tactical.ball[:, 0]) & ~np.isnan(clean.ball[:, 0]) & (tactical.period > 0)
    filled = hidden & ~np.isnan(fused.ball2[:, 0])
    assert filled.sum() > 0.2 * hidden.sum()
    off = np.hypot(*(fused.ball2[filled] - clean.ball[filled]).T)
    assert np.median(off) < 1.5
    health = align.health(tactical, other, exact, fused)
    assert broadcast.decide(health)[0] == "USE"


def test_degrade_broadcast_option(pair):
    clean = pair[0]
    opts = {"ball_missing": 0.41, "broadcast": 1}
    fused = degrade.apply(clean, opts)
    assert fused.ball2 is not None and fused.ball2.shape == (clean.n_frames, 2)
    oracle = degrade.apply(clean, dict(opts, broadcast="oracle"))
    assert (~np.isnan(oracle.ball2[:, 0])).sum() >= 0.9 * (~np.isnan(fused.ball2[:, 0])).sum()
    with pytest.raises(ValueError):
        degrade.apply(clean, {"ball_missing": 0.41, "bc_fps": 30})
    assert degrade.apply(clean, {"clean": 1}).ball2 is None


def test_learned_reads_ball2_only_when_asked(pair):
    _, tactical, other, truth, kind = pair
    fused = align.attach(tactical, other, align.from_truth(tactical, other, truth), bias=False)
    base = learned.features(fused)
    with_b = learned.features(fused, learned.GROUPS + ("bball",))
    assert with_b[0].shape[1] == base[0].shape[1] + len(learned.DIST_KNOTS) + len(learned.MARGIN_KNOTS)
    assert with_b[2].shape[1] == base[2].shape[1] + 2
    filled = learned.features(fused, learned.GROUPS + ("bfill",))
    assert not np.array_equal(filled[0], base[0])
    plain = learned.features(tactical, learned.GROUPS + ("bfill",))  # no ball2: no change
    assert np.array_equal(plain[0], learned.features(tactical)[0])
    v4, v4b = build("learned"), build("learned", ball2_fill=True)
    assert not v4.reads_ball2() and v4b.reads_ball2()
    assert np.array_equal(v4b.predict(tactical), v4.predict(tactical))  # nothing to fill
    assert not np.array_equal(v4b.predict(fused), v4.predict(fused))


def test_synthetic_bundle_has_a_broadcast_recording(small):
    spec = small["spec"]
    other = bar_ath.load(small["dir"], source="broadcast", n_frames=spec.n_broadcast, tags=False)
    assert other.n_frames == spec.n_broadcast and other.ball2 is None
    assert (~np.isnan(other.ball[:, 0])).any() and not other.period.any()
    fused = broadcast.attach(small["match"], other=other)
    again = broadcast.attach(small["match"], other=other)
    assert np.array_equal(fused.ball2, again.ball2, equal_nan=True)
    assert (~np.isnan(fused.ball2[:, 0])).any()
