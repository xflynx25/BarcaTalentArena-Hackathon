"""A18: the per-source knobs of scripts/fit_learned.py (synthetic data only)."""

import importlib.util
import os

import numpy as np

from src.models import learned
from src.schema import AWAY, HOME, Match

_spec = importlib.util.spec_from_file_location(
    "fit_learned", os.path.join(os.path.dirname(__file__), "..", "scripts", "fit_learned.py"))
fit_learned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_learned)


def _match(fps=10):
    n = 8 * fps
    t = np.arange(n) / fps
    ball = np.stack([np.where(t < 2, 0.0, 10 * (t - 2)), np.zeros(n)], axis=1)
    ball[5 * fps:6 * fps] = np.nan
    players = np.stack([np.zeros((n, 2)), np.tile([0.0, 5.0], (n, 1))], axis=1)
    return Match(match_id="test:kick", fps=fps, pitch=(105.0, 68.0), period=np.ones(n, np.int8),
                 ball=ball, players=players, slot_team=np.array([HOME, AWAY], np.int8))


def test_touch_columns_are_the_touch_group():
    match = _match()
    base = learned.features(match, ("shapem",))
    full = learned.features(match, ("shapem", "touch"))
    cols = fit_learned.touch_columns()
    keep = np.ones(full[0].shape[1], bool)
    keep[cols] = False
    assert cols.stop - cols.start == full[0].shape[1] - base[0].shape[1]
    np.testing.assert_array_equal(full[0][:, keep], base[0])
    np.testing.assert_array_equal(full[1][:, keep], base[1])


def _rows(rng, n, shift=0.0):
    p = fit_learned.touch_columns().stop + 2
    y = rng.integers(0, 3, n)
    th = np.where(y[:, None] == 0, 1.0, 0.0) + shift + 0.3 * rng.random((n, p))
    ta = np.where(y[:, None] == 1, 1.0, 0.0) + 0.3 * rng.random((n, p))
    cm = np.hstack([np.ones((n, 1)), (rng.random((n, 1)) < 0.5).astype(float)])
    return (th, ta, cm), y


def test_weights_of_one_equal_the_plain_fit_and_rows_adjust():
    rng = np.random.default_rng(0)
    rows = {"a:1|g0": _rows(rng, 300), "a:2|g0": _rows(rng, 200), "b:1|g0": _rows(rng, 400, 0.5)}
    plain = learned.fit(rows)
    same = fit_learned.weighted_fit(rows, {"b": 1.0})
    np.testing.assert_allclose(same["theta"], plain["theta"], atol=2e-6)
    low = fit_learned.weighted_fit(rows, {"b": 0.01})
    only_a = learned.fit({k: v for k, v in rows.items() if k.startswith("a:")})
    assert np.abs(np.subtract(low["theta"], only_a["theta"])).max() < np.abs(
        np.subtract(plain["theta"], only_a["theta"])).max()

    out = fit_learned.adjust_rows(rows, {"b": "missing"}, {"b": "touch"}, ("touch",))
    assert np.all(out["b:1|g0"][0][2][:, 1] == 1) and len(out["b:1|g0"][1]) < 400
    cols = fit_learned.touch_columns()
    assert not out["b:1|g0"][0][0][:, cols].any() and out["a:1|g0"][0][0][:, cols].any()
    assert out["a:1|g0"][0][0] is rows["a:1|g0"][0][0]
