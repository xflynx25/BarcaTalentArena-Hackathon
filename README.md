# Ball possession: More than a Hack 2027, Phase 1

Our code for **More than a Hack 2027, Phase 1: Ball Possession** (Metrica Sports × FC Barcelona).
For every frame of the competition match (Barcelona vs Athletic, 199,319 frames at 25 fps) it says
who has the ball: `H` home, `A` away, `D` dead ball or `X` abstain, and writes the `frame,label`
CSV the competition scores.

The technical note we submitted is [Tracking Back.pdf](Tracking%20Back.pdf).

This repo holds **code only**. It contains no match data and no data derived from it. The
competition data is gated by Metrica Sports: get it through the official process
([official/DATA.md](official/DATA.md)).

## Setup

Python 3.11, dependencies pinned so the CSV regenerates byte for byte.

```bash
git clone --recurse-submodules <this repo>
cd <this repo>
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The submitted preset (`v6f`) does not read the video. `ffmpeg` is not required to regenerate the file.

`official/` is a git submodule: Metrica's official repo (rules, scorer, loaders), pinned to the
commit we used. If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## Where the data goes

Unpack the official bundle with the names it arrives with, either in `official/data/` (see
[official/data/README.md](official/data/README.md)) or anywhere else, and point `MTAH_DATA_DIR`
at it:

```bash
export MTAH_DATA_DIR=/path/to/mtah2027      # holds tactical/ (and optionally broadcast/)
```

Paths are derived by `official/games.py`; nothing in `src/` spells out a file name.

## Regenerate the submission

The submitted file is preset `v6f`. It is the model without the referee's whistle. That cue is
read from the video's sound with ffmpeg, and a different ffmpeg can change the CSV. On external
matches it moved only SoccerTrack, by +0.0002 spellF1. The ball-flight cue, which is in v6f, is
the part that raised the score. Point `MTAH_DATA_DIR` at the official bundle, then:

```bash
.venv/bin/python -m src.run --preset v6f --out "Tracking Back.csv"
.venv/bin/python official/evaluation/validate_submission.py "Tracking Back.csv"
.venv/bin/python -m src.determinism --preset v6f --expect "Tracking Back.csv"
```

`src.run` loads the match, runs the model, writes and validates the CSV, and prints counts only.
`src.determinism` reruns the pipeline in fresh processes (different hash seeds, one run
single-threaded) and compares sha256 hashes with each other and with the file you pass to
`--expect`. The sha256 of the submitted `Tracking Back.csv` is
`255db13b7c82b2ba72b9c60ab8270dd625715d1af034354fc5b8600a08eb8845`. A run with no match data, on the
synthetic bundle below, is pinned at
`94d462351b25678ff2d767bf74f852af3b413d860d33b5d1fd0d340f60561ce7`.

The pipeline is deterministic: no training on the competition match, no random numbers in the
submission path, no hand labels. The JSON files under `src/models/` are frozen fits on external
matches. Regenerating the CSV does not refit them. The scripts that write them are
`scripts/fit_learned.py`, `scripts/fit_emission.py`, `scripts/fit_switch_cue.py`,
`scripts/fit_stop_cue.py`, `scripts/fit_turn_cue.py`, `scripts/fit_ball_impute.py`, and
`python -m src.models.cues.flight fit`. `scripts/check_dropout_grid.py` is the selection check
those fits were ranked with.

## Without the data

A synthetic match in the official formats (fake football, same layout and frame count) runs the
whole pipeline with no gated data:

```bash
.venv/bin/python scripts/make_synthetic_bar_ath.py          # writes data/synthetic/mtah2027
.venv/bin/python -m src.run --data-dir data/synthetic/mtah2027 \
    --gt data/synthetic/mtah2027/synthetic_gt.csv --out outputs/synthetic.csv
.venv/bin/python -m src.determinism --synthetic
```

Tests run on synthetic data only:

```bash
.venv/bin/python -m pytest -q -m "not slow"     # about 30 s
.venv/bin/python -m pytest -q                   # adds full-length runs, a few minutes
```

## Scoring on external matches

Models are chosen on openly licensed external matches with known possession, never on the
competition match. Fetch them, then score any model with the official scorer:

```bash
scripts/fetch_external_data.sh skillcorner       # also: idsse, metrica
.venv/bin/python -m src.evaluate --model <model> --matches skillcorner:all --note "..."
```

## Layout

```
src/
  schema.py           one internal match schema every loader converts into
  ingest/             loaders: the competition match, SkillCorner, IDSSE, Metrica, SoccerTrack, synthetic
  models/             possession models (predict(match) -> codes) and their frozen coefficient files
  presets.py          named candidates; v6f is the submission
  submission.py       writes and validates the frame,label CSV
  run.py              the submission command
  determinism.py      reruns the pipeline and compares hashes
  evaluate.py         scores models on external matches with the official scorer
  dropout.py          the ball-hiding grid used to train and to choose, on external matches
  check_data.py       sanity checks on the data bundle (prints pass/fail only)
  synthetic.py        synthetic match in the official formats
scripts/              synthetic bundle, external data download, the fits, the selection check
tests/                pytest, synthetic data only
official/             submodule: Metrica's official repo
```

## Data credits

The competition bundle is not in this repo. These external datasets were used to train or to
choose the model. They are not included either. Fetch them with `scripts/fetch_external_data.sh`.

- SkillCorner Open Data, MIT. Credit SkillCorner. https://github.com/SkillCorner/opendata
  The 2019/20 release is the tree at commit `1b99d401ddc82e366c5c20d6860878668fd5858f`.
- IDSSE, CC BY 4.0. Credit DFL. Bassek, M. et al. (2025), *Scientific Data* 12:195.
  https://doi.org/10.6084/m9.figshare.28196177
- Metrica Sports sample data, research use. https://github.com/metrica-sports/sample-data
- SoccerTrack v2, CC BY 4.0. Credit the dataset. Hugging Face `atomscott/soccertrack-v2`.
- Wyscout public events, CC BY 4.0. Pappalardo, L. et al. (2019), *Scientific Data* 6:236.
  Event statistics only, not possession labels.
- StatsBomb open data. Credit StatsBomb under the StatsBomb public data licence.
  Event statistics only, not possession labels.

## Licence

Our code: MIT, see [LICENSE](LICENSE). `official/` keeps its own licence. External datasets keep
theirs and are not included.
