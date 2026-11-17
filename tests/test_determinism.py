"""D1: the submission pipeline gives byte-identical CSVs across fresh processes, hash seeds, thread
counts and machines. Uses the full-length synthetic bundle (src.run only reads full-length ones)."""

import pytest

from src import determinism

pytestmark = pytest.mark.slow

# sha256 of `src.run --model nearest --param window=50` on the seed-0 synthetic bundle. Computed in a
# cloud session (Linux, numpy 2.4.6); CI and the owner's machine must get the same bytes. If you
# change src/synthetic.py or the official-baseline path on purpose, update it and say so in the PR.
GOLDEN_BASELINE = "cbb6670a8630eb9174cb3e7b790c153f3b11e024d0e055c880dbdb5b8a66f12a"
BASELINE_ARGS = ["--model", "nearest", "--param", "window=50"]


def test_fresh_runs_identical_and_match_golden(full, tmp_path):
    result = determinism.check(BASELINE_ARGS, runs=2, data_dir=full["dir"],
                               expect=GOLDEN_BASELINE, workdir=str(tmp_path))
    assert result["identical"], result
    assert result["expect_ok"], "hash {} != golden {}".format(result["hashes"][0], GOLDEN_BASELINE)


def test_cli_exit_codes(full, capsys):
    assert determinism.main(["--data-dir", full["dir"], *BASELINE_ARGS,
                             "--expect", GOLDEN_BASELINE]) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert determinism.main(["--data-dir", full["dir"], *BASELINE_ARGS, "--expect", "0" * 64]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_failed_run_raises(full):
    with pytest.raises(RuntimeError, match="run 0 failed"):
        determinism.check(["--model", "nearest", "--param", "window=50", "--param", "window_s=2"],
                          data_dir=full["dir"])


def test_needs_two_runs():
    with pytest.raises(ValueError):
        determinism.check([], runs=1)


def test_short_spell_filter_identical(full, tmp_path):
    """B6's setting (findings F34) goes through drop_short; it must be deterministic too."""
    result = determinism.check(["--model", "nearest", "--param", "window_s=1.5",
                                "--param", "min_spell_s=2.5"], data_dir=full["dir"],
                               workdir=str(tmp_path))
    assert result["identical"], result


# sha256 of `src.run` with the default model (v6f since 17 Nov: v7 without the whistle. The synthetic
# bundle has no video, so this hash was already v7's;
# v6 (W4 close, F182) was 3b7ece5c...8edd; v5 (B30, F126) was 865ef84e...9eec;
# v4's was 4bb28afb...b244) on the seed-0 synthetic bundle, computed on Linux (OpenBLAS); v3's 1c3a9f85...85bd2 was computed on macOS
# and matched Linux after the decoder's score grid (findings F87). Before that grid Linux gave
# bf961767...2508 and macOS 1c3a9f85...: 1-ulp BLAS/SIMD differences flipped Viterbi near-ties.
# CI runs on Linux, so this test passing there is the cross-machine check. If you change the
# default model on purpose, update it (and docs/run-locally.md step d) and say so in the PR.
GOLDEN_DEFAULT = "94d462351b25678ff2d767bf74f852af3b413d860d33b5d1fd0d340f60561ce7"


def test_default_model_matches_golden_across_machines(full, tmp_path):
    result = determinism.check([], runs=2, data_dir=full["dir"], expect=GOLDEN_DEFAULT,
                               workdir=str(tmp_path))
    assert result["identical"], result
    assert result["expect_ok"], "hash {} != golden {}".format(result["hashes"][0], GOLDEN_DEFAULT)
