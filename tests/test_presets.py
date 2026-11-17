"""Presets (src/presets.py): every candidate builds, and each pinned one still makes its exact bytes."""

import os

import pytest

from src import determinism, evaluate, run
from src.models import MODELS, build
from src.presets import DEFAULT, PRESETS, cli_args, model_and_params


def test_every_preset_builds():
    for name in PRESETS:
        model, params = model_and_params(name)
        assert model in MODELS
        build(model, **params)
        if "coef_file" in params:  # bigger: two files joined with "+"
            for path in params["coef_file"].split("+"):
                assert os.path.exists(path), name
        assert PRESETS[name]["what"] and PRESETS[name]["when"]


def test_default_preset_is_srcrun_default():
    import argparse

    args = argparse.Namespace(preset=None, model=None)
    assert run.resolve_model(args) == model_and_params(DEFAULT)  # `src.run` with no flags
    assert PRESETS[DEFAULT]["synthetic_sha256"]  # the default is always pinned


def test_eval_params_are_held_out():
    for name, preset in PRESETS.items():
        if preset["model"] in ("learned", "bigger", "boosted", "temporal"):
            assert model_and_params(name, for_eval=True)[1]["fold"] == "lomo"


def test_cli_args_expand():
    assert cli_args("baseline") == ["--model", "nearest", "--param", "window=50"]


def test_ensemble_coef_files_resolve_each_part():
    """A `bigger` preset joins two coefficient files with "+": both must resolve from the repo root."""
    _, params = model_and_params("v5.1c")
    parts = params["coef_file"].split("+")
    assert len(parts) == 2 and all(os.path.isabs(f) and os.path.isfile(f) for f in parts)


def test_preset_and_model_conflict():
    with pytest.raises(SystemExit):
        run.main(["--preset", "v3", "--model", "nearest"])
    with pytest.raises(SystemExit):
        evaluate.main(["--preset", "v3", "--model", "nearest", "--matches", "synthetic:all"])


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, p in PRESETS.items() if p["synthetic_sha256"]])
def test_pinned_presets_match_their_hash(full, tmp_path, name):
    result = determinism.check(["--preset", name], runs=2, data_dir=full["dir"],
                               expect=PRESETS[name]["synthetic_sha256"], workdir=str(tmp_path))
    assert result["expect_ok"], "{}: {} != {}".format(name, result["hashes"][0],
                                                      PRESETS[name]["synthetic_sha256"])


def test_expand_groups_and_comma_lists():
    from src import presets

    assert presets.expand(["history"]) == presets.GROUPS["history"]
    assert presets.expand(["v6,v4", "v4", " v3 "]) == ["v6", "v4", "v3"]
    assert presets.expand(None) == []
    assert all(name in presets.PRESETS for group in presets.GROUPS.values() for name in group)
    assert presets.GROUPS["history"][0] == presets.DEFAULT
    with pytest.raises(SystemExit):
        presets.expand(["v6,nope"])
