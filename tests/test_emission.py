"""Bigger emission models (backlog B31): exact arithmetic, symmetry, training, coefficient files."""

import dataclasses

import numpy as np
import pytest

from src import emission_train as train
from src import submission
from src.models import build, emission
from src.official import D, H


def test_det_exp_and_log_are_accurate():
    x = np.concatenate([np.linspace(-40, 5, 4001), [0.0, -1e-12, 1e-12]])
    assert np.max(np.abs(emission.det_exp(x) / np.exp(x) - 1)) < 1e-14
    y = np.concatenate([np.linspace(1e-6, 10, 4001), [1.0, 2.0, 3.0, 0.5]])
    assert np.max(np.abs(emission.det_log(y) - np.log(y))) < 1e-14
    z = np.random.default_rng(0).normal(size=(500, 3)) * 5
    ref = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    assert np.max(np.abs(emission.det_log_softmax(z) - ref)) < 1e-13


def _net(rng, n_in, embed=4, hidden=5, k=3):
    model = {"w1": rng.normal(size=(n_in, embed)), "b1": rng.normal(size=embed),
             "w2": rng.normal(size=(k * embed, hidden)), "b2": rng.normal(size=hidden),
             "vf": rng.normal(size=hidden), "vg": rng.normal(size=hidden), "c_d": np.array([0.3])}
    return train.quantize_net(model)


def test_net_inference_is_exact_in_any_summation_order():
    rng = np.random.default_rng(1)
    p_team, p_common, n = 3, 2, 50
    u = emission.snap(rng.random((n, 2 * p_team + p_common)), emission.IN_BITS)
    model = _net(rng, u.shape[1])
    taps = emission.tap_index(np.zeros(n, int), np.full(n, n - 1), [-1, 0, 2])
    h = emission.net_hidden(u, taps, model)
    # the same sums, accumulated term by term in reverse order: bit-identical, not just close
    a = np.zeros((n, model["w1"].shape[1]))
    for f in range(u.shape[1] - 1, -1, -1):
        a = a + u[:, [f]] * model["w1"][f]
    a = np.clip(emission.snap(np.maximum(a + model["b1"], 0), emission.IN_BITS), 0, emission.ACT_MAX)
    stacked = a[taps].reshape(n, -1)
    s = np.zeros((n, model["w2"].shape[1]))
    for j in range(stacked.shape[1] - 1, -1, -1):
        s = s + stacked[:, [j]] * model["w2"][j]
    h2 = np.clip(emission.snap(np.maximum(s + model["b2"], 0), emission.IN_BITS), 0, emission.ACT_MAX)
    assert np.array_equal(h, h2)


def test_logits_swap_exactly_with_the_teams():
    rng = np.random.default_rng(2)
    p_team, p_common, n = 3, 2, 40
    th, ta, cm = rng.random((n, p_team)), rng.random((n, p_team)), rng.random((n, p_common))
    u_h, u_a = emission.views((th, ta, cm))
    taps = emission.tap_index(np.zeros(n, int), np.full(n, n - 1), [-1, 0, 2])
    net = _net(rng, u_h.shape[1])
    z, zs = emission.temporal_logits(u_h, u_a, taps, net), emission.temporal_logits(u_a, u_h, taps, net)
    assert np.array_equal(z[:, [1, 0, 2]], zs)
    trees = {"c_d": 0.25, "f": {"feat": np.array([[0, 4], [3, 7]]),
                                "thr": np.array([[0.5 + 2 ** -13, np.inf], [0.25 + 2 ** -13, 0.75 + 2 ** -13]]),
                                "leaf": emission.snap(rng.normal(size=(2, 4)), emission.LEAF_BITS)},
             "g": {"feat": np.array([[1, 6]]), "thr": np.array([[0.5 + 2 ** -13, 0.3 + 2 ** -13]]),
                   "leaf": emission.snap(rng.normal(size=(1, 4)), emission.LEAF_BITS)}}
    z, zs = emission.boosted_logits(u_h, u_a, trees), emission.boosted_logits(u_a, u_h, trees)
    assert np.array_equal(z[:, [1, 0, 2]], zs)


def test_tap_index_stays_inside_the_run():
    first, last = emission.run_bounds(np.array([1, 1, 1, 0, 0, 2, 2, 2, 2]))
    assert first.tolist() == [0, 0, 0, 3, 3, 5, 5, 5, 5] and last.tolist() == [2, 2, 2, 4, 4, 8, 8, 8, 8]
    taps = emission.tap_index(first, last, [-2, 0, 2])
    assert taps[1].tolist() == [0, 1, 2] and taps[5].tolist() == [5, 5, 7]


def _rows(rng, n=4000, p_team=3, p_common=2):
    """An easy synthetic problem: home column 0 high -> H, away column 0 high -> A, common 0 -> D."""
    y = rng.integers(0, 3, n)
    th = rng.random((n, p_team)) * 0.3
    ta = rng.random((n, p_team)) * 0.3
    cm = rng.random((n, p_common)) * 0.3
    th[y == 0, 0] += 0.6
    ta[y == 1, 0] += 0.6
    cm[y == 2, 0] += 0.6
    u_h, _ = emission.views((th, ta, cm))
    first = np.zeros(n, np.int32)
    return {"u": np.rint(u_h * 2 ** emission.IN_BITS).astype(np.uint16), "y": y.astype(np.int8),
            "first": first, "last": np.full(n, n - 1, np.int32), "p": np.array([p_team, p_common])}


@pytest.mark.parametrize("kind", emission.KINDS)
def test_fitters_learn_an_easy_rule_and_round_trip(kind, tmp_path):
    rng = np.random.default_rng(3)
    rows = {"a:1|g0": _rows(rng), "a:2|g0": _rows(rng)}
    data, valid = train.stack(rows, ["a:1|g0"]), train.stack(rows, ["a:2|g0"])
    spec = dict(train.SPECS[kind])
    spec.update(rounds=30, min_rounds=5, depth=3) if kind == "boosted" else \
        spec.update(taps=[0], embed=4, hidden=6, epochs=12, lr=1e-2, batch=128)
    fit = train.FITTERS[kind](data, valid, spec, seed=0)
    assert fit["valid_loss"] < 0.5 and fit["best"] >= 1
    p_team, p_common = 3, 2
    payload = {"kind": kind, "spec": dict(spec, n_in=2 * p_team + p_common, speed_span_s=1.0),
               "groups": list(emission.GROUPS), "feature_hash": emission.learned.feature_hash(emission.GROUPS),
               "all": {"model": emission.encode_model(kind, fit["model"]),
                       "log_prior": emission.to_int(fit["log_prior"], emission.LEAF_BITS)}}
    path = str(tmp_path / "coefs.json")
    payload.update(folds={"0": payload["all"]}, fold_of={"a:2": 0}, loso={})
    emission.write_coefs(path, payload)
    first = [open(p, "rb").read() for p in (path, emission.heldout_path(path))]
    emission.write_coefs(path, payload)  # fixed gzip header: same bytes every time
    assert [open(p, "rb").read() for p in (path, emission.heldout_path(path))] == first
    assert '"folds"' not in first[0].decode()  # the run-time file holds the "all" fit only
    loaded = emission.load_coefs(path)["all"]["model"]
    u_h = valid["u"].astype(np.float64) / 2 ** emission.IN_BITS
    u_a = u_h[:, emission.mirror_index(p_team, p_common)]
    if kind == "boosted":
        z = emission.boosted_logits(u_h, u_a, loaded)
        assert np.array_equal(z, emission.boosted_logits(u_h, u_a, fit["model"]))
    else:
        taps = emission.tap_index(valid["first"], valid["last"], [0])
        z = emission.temporal_logits(u_h, u_a, taps, loaded)
        assert np.array_equal(z, emission.temporal_logits(u_h, u_a, taps, fit["model"]))
    assert (np.argmax(z, axis=1) == valid["y"]).mean() > 0.9


def test_inner_split_is_by_match_and_per_source():
    keys = ["s:{}|g{}".format(m, g) for m in range(10) for g in range(2)] + ["t:1|g0", "t:2|g0"]
    fit_keys, valid_keys = train.inner_split(keys)
    assert set(fit_keys) | set(valid_keys) == set(keys) and not set(fit_keys) & set(valid_keys)
    valid_matches = {k.split("|")[0] for k in valid_keys}
    assert len(valid_matches) == 2 and all(m.startswith("s:") for m in valid_matches)
    assert not any(k.split("|")[0] in valid_matches for k in fit_keys)


@pytest.fixture(scope="module")
def boosted_file(small, tmp_path_factory):
    """A tiny boosted fit on the small synthetic match, saved like scripts/fit_emission.py does."""
    match = dataclasses.replace(small["match"], labels=submission.read_codes(small["gt"]))
    spec = dict(train.BOOSTED_SPEC, rounds=8, min_rounds=8, depth=3)
    rows = {"synthetic:0|g0": train.grid_rows(match, emission.GROUPS, {"speed_span_s": 1.0})}
    data = train.stack(rows, list(rows))
    fit = train.fit_boosted(data, data, spec, seed=0)
    p_team, p_common = (int(v) for v in rows["synthetic:0|g0"]["p"])
    entry = {"model": emission.encode_model("boosted", fit["model"]),
             "log_prior": emission.to_int(fit["log_prior"], emission.LEAF_BITS)}
    payload = {"kind": "boosted", "spec": dict(spec, n_in=2 * p_team + p_common, speed_span_s=1.0),
               "groups": list(emission.GROUPS), "feature_hash": emission.learned.feature_hash(emission.GROUPS),
               "all": entry, "folds": {"0": entry}, "fold_of": {match.match_id: 0}, "loso": {}}
    path = str(tmp_path_factory.mktemp("boosted") / "coefs.json")
    emission.write_coefs(path, payload)
    return path


def test_boosted_decoder_is_registered_deterministic_and_valid(small, boosted_file):
    match = small["match"]
    model = build("boosted", coef_file=boosted_file)
    first = model.predict(match)
    assert first.shape == (match.n_frames,) and first.dtype == np.int8
    assert np.array_equal(first, build("boosted", coef_file=boosted_file).predict(match))
    assert np.array_equal(first, build("boosted", coef_file=boosted_file, fold="lomo").predict(match))
    assert set(np.unique(first)) <= {H, D, 1}
    with pytest.raises(KeyError):
        build("boosted", coef_file=boosted_file, fold="loso").predict(match)
    lp = model.emission(match)
    assert np.isfinite(lp).all() and lp.shape == (match.n_frames, 3)


def test_boosting_from_the_logit_round_trips_exactly_and_stays_symmetric(tmp_path):
    rng = np.random.default_rng(4)
    rows = {"a:1|g0": _rows(rng), "a:2|g0": _rows(rng)}
    data, valid = train.stack(rows, ["a:1|g0"]), train.stack(rows, ["a:2|g0"])
    spec = dict(train.BOOSTED_SPEC, rounds=10, min_rounds=10, depth=3, base="logit")
    fit = train.fit_boosted(data, valid, spec, seed=0)
    assert "lin" in fit["model"] and fit["valid_loss"] < 0.5
    decoded = emission.decode_model("boosted", emission.encode_model("boosted", fit["model"]), spec)
    emission.check_bounds("boosted", decoded, 8)
    u_h = valid["u"].astype(np.float64) / 2 ** emission.IN_BITS
    u_a = u_h[:, emission.mirror_index(3, 2)]
    z = emission.boosted_logits(u_h, u_a, decoded)
    assert np.array_equal(z, emission.boosted_logits(u_h, u_a, fit["model"]))
    assert np.array_equal(z[:, [1, 0, 2]], emission.boosted_logits(u_a, u_h, decoded))


def test_chunked_inference_gives_the_same_bits(small, boosted_file, monkeypatch):
    match = small["match"]
    coefs = emission.load_coefs(boosted_file)
    whole = emission.logits(match, coefs, coefs["all"])
    monkeypatch.setattr(emission, "CHUNK", 777)
    assert np.array_equal(whole, emission.logits(match, coefs, coefs["all"]))
    rng = np.random.default_rng(5)
    net = _net(rng, coefs["spec"]["n_in"], k=3)
    spec = dict(coefs["spec"], taps=[-2, 0, 3])
    t = {"kind": "temporal", "spec": spec, "groups": coefs["groups"]}
    monkeypatch.setattr(emission, "CHUNK", 10 ** 9)
    whole = emission.logits(match, t, {"model": net})
    monkeypatch.setattr(emission, "CHUNK", 333)
    assert np.array_equal(whole, emission.logits(match, t, {"model": net}))


def test_an_ensemble_of_one_file_twice_is_that_file(small, boosted_file):
    match = small["match"]
    one = build("boosted", coef_file=boosted_file).emission(match)
    two = build("boosted", coef_file=boosted_file + "+" + boosted_file).emission(match)
    assert np.allclose(one, two, atol=1e-12)


def test_emission_models_take_the_run_time_hooks_like_learned(small, boosted_file):
    """Wave-2 integration: an ensemble spec ("a+b") doesn't trip V4's ball2 checks (src.run asks every
    model reads_ball2()), and V4's fill reaches the emissions as it reaches learned's features."""
    match = small["match"]
    both = boosted_file + "+" + boosted_file
    plain = build("boosted", coef_file=both)
    assert not plain.reads_ball2() and not plain.fills() and plain.read(match) is match
    hidden = match.ball.copy()
    hidden[::3] = np.nan
    gappy = dataclasses.replace(match, ball=hidden, ball2=match.ball.copy())
    filler = build("boosted", coef_file=both, ball2_fill=True)
    assert filler.reads_ball2()
    assert np.array_equal(filler.read(gappy).ball, match.ball, equal_nan=True)
    assert np.array_equal(filler.emission(gappy), plain.emission(match))
    assert not np.array_equal(plain.emission(gappy), plain.emission(match))
