"""The C6 ball-dropout grid: its points and the seeded per-match draws."""

from src import dropout


def test_grid_has_no_duplicate_settings():
    assert len(dropout.GRID) == 15  # 3 x 3 x 2, minus the gap length at quiet 1 (no random gaps)
    assert len({dropout.name(p) for p in dropout.GRID}) == len(dropout.GRID)
    assert {"ball_missing": 0.41, "ball_quiet": 0.5, "ball_gap_s": 1.0} in dropout.GRID


def test_draws_are_seeded_by_the_match_and_distinct():
    a = dropout.draws("idsse:x", 5)
    assert a == dropout.draws("idsse:x", 5) and len(set(a)) == 5
    assert all(0 <= i < len(dropout.GRID) for i in a)
    assert dropout.draws("idsse:y", 5) != a or dropout.draws("idsse:z", 5) != a
    assert len(dropout.draws("k", 99)) == len(dropout.GRID)


def test_player_grid_and_mix():
    assert dropout.GRID == dropout.MIXED[:len(dropout.GRID)]  # GRID unchanged, first in the mix
    assert dropout.MIXED[-1] == dropout.CLEAN and len(dropout.MIXED) == 15 * 6 + 1
    assert abs(dropout.MIXED_WEIGHTS.sum() - 1) < 1e-12
    assert abs(dropout.MIXED_WEIGHTS[-1] - 1 / 6) < 1e-12
    assert abs(dropout.MIXED_WEIGHTS[:15].sum() - 5 / 12) < 1e-12
    names = [dropout.name(p) for p in dropout.MIXED]
    assert len(set(names)) == len(names)
    assert [dropout.name(p) for p in dropout.GRID] == [dropout.name(p) for p in dropout.MIXED[:15]]
    assert len(dropout.PLAYER_EVAL) == 15 and all(len(p) > 3 for p in dropout.PLAYER_EVAL)
    assert set(dropout.EVAL) == {"ball", "player", "clean", "frag"}
    assert len(dropout.FRAG_EVAL) == 9 and all(p.get("fragment") for p in dropout.FRAG_EVAL)
    assert not any(p.get("fragment") for p in dropout.MIXED)  # scoring only (C13)


def test_mixed_draws_are_seeded_and_include_clean_copies():
    a = dropout.mixed_draws("idsse:x", 5)
    assert a == dropout.mixed_draws("idsse:x", 5) and len(set(a)) == 5
    assert all(0 <= i < len(dropout.MIXED) for i in a)
    picks = [dropout.mixed_draws("m{}".format(i), 1)[0] for i in range(600)]
    clean = sum(i == len(dropout.MIXED) - 1 for i in picks) / len(picks)
    degraded = sum(i >= 15 and i < len(dropout.MIXED) - 1 for i in picks) / len(picks)
    assert 0.12 < clean < 0.22 and 0.35 < degraded < 0.5
