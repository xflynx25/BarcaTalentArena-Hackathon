"""Model durations in seconds (backlog A5): one config means the same thing at any frame rate."""

import dataclasses

import numpy as np
import pytest

from src.models import build
from src.models.nearest import NearestPlayer, majority_smooth, nearest_team
from src.models.timing import to_frames
from src.resample import to_fps
from src.run import parse_params


@pytest.mark.parametrize("seconds, fps, frames", [
    (2.0, 25, 50), (2.0, 10, 20), (0, 25, 0), (0.25, 10, 3), (0.1, 30, 3), (1 / 3, 30, 10)])
def test_to_frames(seconds, fps, frames):
    assert to_frames(seconds, fps) == frames


def test_to_frames_rejects_negative():
    with pytest.raises(ValueError):
        to_frames(-1, 25)


def test_default_is_two_seconds_and_the_published_floor_at_25_fps(small):
    model = NearestPlayer()
    assert model.window_s == 2.0 and model.window is None
    assert model.frames(25) == 50
    match = small["match"]
    assert match.fps == 25
    assert np.array_equal(model.predict(match), NearestPlayer(window=50).predict(match))


def test_window_s_scales_with_fps(small):
    match10 = to_fps(small["match"], 10)
    model = NearestPlayer(window_s=2)
    assert model.frames(match10.fps) == 20
    assert np.array_equal(model.predict(match10), majority_smooth(nearest_team(match10), 20))


def test_window_in_frames_ignores_fps():
    assert NearestPlayer(window=50).frames(10) == 50


def test_both_or_positional_rejected():
    with pytest.raises(ValueError):
        NearestPlayer(window=50, window_s=2.0)
    with pytest.raises(TypeError):
        NearestPlayer(50)  # keyword-only, so an old positional frame count can't become seconds


def test_cli_params_build_and_record():
    model = build("nearest", **parse_params(["window_s=1.5"]))
    assert dataclasses.asdict(model) == {"window_s": 1.5, "window": None,
                                          "min_spell_s": None, "switch_delay_s": None}
    assert dataclasses.asdict(build("nearest", **parse_params(["window=50"]))) == {
        "window_s": None, "window": 50, "min_spell_s": None, "switch_delay_s": None}


def test_parse_params_booleans():
    """C4 / F50: "False" must become False, not the truthy string "False"."""
    parsed = parse_params(["a=False", "b=true", "c=TRUE", "d=false", "e=1", "f=0.5", "g=abc"])
    assert parsed == {"a": False, "b": True, "c": True, "d": False, "e": 1, "f": 0.5, "g": "abc"}
    assert parsed["a"] is False and parsed["e"] is not True
