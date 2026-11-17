"""src.models.nearest must reproduce official/starter/baseline.py exactly."""

import numpy as np
import pytest

from src.models.nearest import majority_smooth, nearest_team
from src.official import games
from src.submission import LETTERS


@pytest.fixture(scope="module")
def official_codes(small):
    from starter import baseline

    paths = small["paths"]
    return baseline, baseline.nearest_player_labels(
        paths["atd_raw"], baseline.team_of_each_track(paths["atd_meta"]),
        small["spec"].n_frames, games.GAMES["BAR-ATH"]["pitch"])


def test_nearest_team_identical(small, official_codes):
    _, codes = official_codes
    ours = nearest_team(small["match"])
    assert np.array_equal(ours, codes)
    assert (codes >= 0).mean() > 0.3  # the comparison is not vacuous


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_smoothing_identical(small, official_codes, window):
    baseline, codes = official_codes
    assert np.array_equal(LETTERS[majority_smooth(codes, window)], baseline.smooth(codes, window))
