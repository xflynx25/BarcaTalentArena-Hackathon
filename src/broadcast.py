"""BAR-ATH's broadcast tracking at run time, and the owner's local check (backlog V4).

src.run calls attach() when the model reads the broadcast ball (a preset with ball2_fill, or a
coefficient file fit with group "bball" / "bfill"): it loads the broadcast recording from the same
bundle (official games.asset_paths(source="broadcast")), aligns it with the tactical timeline
(src/align.py, label-free, deterministic) and returns the tactical match with Match.ball2.

The owner's check, on the owner's machine only (docs/run-locally.md), in their own terminal:

    export MTAH_DATA_DIR=$PWD/data/mtah2027
    .venv/bin/python -m src.broadcast

It prints the alignment's health (aggregates of the two recordings: how much of the broadcast
matched, how well the positions and the balls agree, how much of the tactical ball's gaps it fills)
and ends in one decision word, USE or STAY, with the preset that goes with it. **These numbers
describe the match: they stay on the owner's screen.** Never paste them into a chat with an AI
assistant, a PR or the repo (CLAUDE.md); report only which preset you uploaded.

The decision rule (decide) was fixed in advance on simulated second recordings of external matches
(scripts/check_broadcast.py measure / mismatch, findings F142-F143): a correct alignment there
matches ~95% of the broadcast's frames with data at a median cost under 1 m and its balls agree
within 2 m on ~97% of the frames both see; a recording of another match matches almost nothing.
"""

from __future__ import annotations

import argparse
import sys

from src import align
from src.schema import Match

# USE needs all of these (decide); thresholds well inside the simulated good alignments and far
# from the wrong pairings (scripts/check_broadcast.py mismatch)
MIN_MATCHED = 0.5  # share of the broadcast's frames with data that matched
MAX_COST_M = 1.5  # median pair cost on matched frames (players + ball, m)
MIN_AGREE = 0.8  # share of frames both balls see where they agree within 2 m
MIN_FILL = 0.02  # share of the tactical ball's in-play gaps the broadcast ball fills
USE_PRESET = "v4b"  # the preset that reads the broadcast ball (src/presets.py)
STAY_PRESET = "v4"


def load_broadcast(data_dir: str | None = None) -> Match:
    from src.ingest import bar_ath

    return bar_ath.load(data_dir, source="broadcast", tags=False)


def attach(match: Match, data_dir: str | None = None, other: Match | None = None) -> Match:
    """The tactical match with ball2 from the broadcast recording of the same bundle."""
    other = load_broadcast(data_dir) if other is None else other
    return align.attach(match, other, align.align(match, other))


def decide(health: dict) -> tuple[str, list[str]]:
    """("USE" | "STAY", the failed conditions) from align.health()."""
    failed = []
    if not health.get("matched_share_of_data", 0) >= MIN_MATCHED:
        failed.append("matched share below {}".format(MIN_MATCHED))
    cost = health.get("median_cost_m")
    if cost is None or not cost <= MAX_COST_M:
        failed.append("median cost above {} m".format(MAX_COST_M))
    agree = health.get("ball_agree_share_2m")
    if agree is None or not agree >= MIN_AGREE:
        failed.append("balls agree within 2 m on less than {}".format(MIN_AGREE))
    if not health.get("fill_share", 0) >= MIN_FILL:
        failed.append("fills less than {} of the tactical ball's gaps".format(MIN_FILL))
    return ("USE" if not failed else "STAY"), failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Owner check: align BAR-ATH's broadcast tracking "
                                     "with the tactical timeline (prints to your screen only)")
    parser.add_argument("--data-dir", help="bundle root (default: MTAH_DATA_DIR or official/data)")
    args = parser.parse_args(argv)
    from src.ingest import bar_ath

    tactical = bar_ath.load(args.data_dir, tags=False)
    other = load_broadcast(args.data_dir)
    alignment = align.align(tactical, other)
    fused = align.attach(tactical, other, alignment)
    health = align.health(tactical, other, alignment, fused)
    print("ON YOUR SCREEN ONLY: these numbers describe the match. Don't paste them anywhere.")
    for key, value in health.items():
        print("  {:<24}{}".format(key, value))
    word, failed = decide(health)
    for reason in failed:
        print("  not met: " + reason)
    preset = USE_PRESET if word == "USE" else STAY_PRESET
    print("decision   {}: submit preset {} (report only the preset name)".format(word, preset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
