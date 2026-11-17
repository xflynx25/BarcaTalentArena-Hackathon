"""Check that a data bundle has the layout our loaders expect. Prints PASS/FAIL lines only.

Run it once on the real bundle, in your own terminal:

    MTAH_DATA_DIR=$PWD/data/mtah2027 .venv/bin/python -m src.check_data

Every check compares against a number already published in official/DATA.md, so the output says
nothing new about the data. It only confirms the synthetic bundle (which cloud agents build on)
matches the real format. If anything FAILs, tell the next agent which line failed, not the values.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.ingest import bar_ath
from src.official import games

# From official/DATA.md, tactical source.
EXPECTED_SLOTS = 46
EXPECTED_BALL_SEEN = (0.55, 0.59)  # 56.9% of all frames
EXPECTED_TRACKED_GRADED = (19.0, 22.5)  # mean 20.5 tracked objects on graded frames
EXPECTED_TAGS = 638


def run_checks(data_dir: str | None = None) -> list[tuple[str, bool]]:
    from starter import load_atd, load_homography, load_smart_tagging

    results = []

    def check(name, fn):
        try:
            ok = bool(fn())
        except Exception as error:  # report the failure type, not file contents
            ok = False
            name = "{}  [{}]".format(name, type(error).__name__)
        results.append((name, ok))

    check("ATD: one record per video frame, ids 1..199319 (official check)",
          lambda: load_atd.check_alignment_raw(data_dir=data_dir)["agrees"])

    holder = {}

    def load_match():
        holder["match"] = bar_ath.load(data_dir)
        return True

    check("ATD: parses with src.ingest (consistent line layout, 2 teams)", load_match)
    match = holder.get("match")
    if match is not None:
        graded = match.period > 0
        seen = float(np.mean(~np.isnan(match.ball[:, 0])))
        tracked = float(np.mean(np.sum(~np.isnan(match.players[graded, :, 0]), axis=1)))
        check("ATD: {} track slots".format(EXPECTED_SLOTS),
              lambda: len(match.slot_team) == EXPECTED_SLOTS)
        check("ATD: both teams own slots", lambda: set(np.unique(match.slot_team)) == {0, 1})
        check("ATD: ball seen on 55-59% of frames (published 56.9%)",
              lambda: EXPECTED_BALL_SEEN[0] <= seen <= EXPECTED_BALL_SEEN[1])
        check("ATD: 19-22.5 tracked objects per graded frame (published 20.5)",
              lambda: EXPECTED_TRACKED_GRADED[0] <= tracked <= EXPECTED_TRACKED_GRADED[1])
        check("ATD: >=99% of ball and of player positions inside the pitch (+/-10%)",
              lambda: _inside(match))

    check("homography: one row per video frame (official check)",
          lambda: load_homography.check_alignment(data_dir=data_dir)["agrees"])

    def tags():
        events = load_smart_tagging.load(data_dir=data_dir)
        holder["tags"] = events
        return len(events) == EXPECTED_TAGS

    check("smart tagging: {} instances".format(EXPECTED_TAGS), tags)
    check("smart tagging: kickoffs line up with the video (official check)",
          lambda: load_smart_tagging.check_time_base(holder["tags"])["agrees"])
    check("smart tagging: Team values are Barcelona / Atletic Club / N/A",
          lambda: {e["tags"].get("Team") for e in holder["tags"]}
          <= {"Barcelona", "Atletic Club", "N/A", None})
    return results


def _inside(match, share=0.99) -> bool:
    """Catches wrong units or scale, not outliers: warming-up subs, balls in the stands and tracking
    glitches legitimately put some points off the pitch, so "all points inside" is too strict."""
    length, width = match.pitch

    def ok(xy):
        xy = xy[~np.isnan(xy[:, 0])]
        inside = ((xy[:, 0] > -0.1 * length) & (xy[:, 0] < 1.1 * length)
                  & (xy[:, 1] > -0.1 * width) & (xy[:, 1] < 1.1 * width))
        return len(xy) > 0 and float(np.mean(inside)) >= share

    return ok(match.ball) and ok(match.players.reshape(-1, 2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="bundle root (default: MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    print("bundle: {}".format(args.data_dir or games.DATA_DIR))
    results = run_checks(args.data_dir)
    for name, ok in results:
        print("  {}  {}".format("PASS" if ok else "FAIL", name))
    failed = sum(not ok for _, ok in results)
    print("{} of {} checks passed".format(len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
