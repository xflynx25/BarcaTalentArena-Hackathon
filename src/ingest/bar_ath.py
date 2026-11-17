"""BAR-ATH (the competition match) -> Match.

Paths come from official games.asset_paths(), so MTAH_DATA_DIR or data_dir decides which bundle is
read: the real one (owner's machine only) or the synthetic one (src/synthetic.py).

Alignment, per official/starter/load_atd.py: ATD frame_id counts from 1, submission rows from 0,
so row = frame_id - 1. The loader checks this rather than assuming it.

Smart Tagging (B28): the tactical bundle's tags go in Match.tags as src.smart_tags.Tag intervals
(round(seconds * fps), the starter loader's frames), Team mapped to H/A through the tracking's team
names. tags=False, the broadcast source, or a bundle without the file leaves Match.tags None.
"""

from __future__ import annotations

import sys

import numpy as np

from src import smart_tags
from src.ingest import epts
from src.official import games
from src.schema import Match

GAME = "BAR-ATH"


def load(data_dir: str | None = None, source: str = "tactical", n_frames: int | None = None,
         halves: list[tuple[int, int, int]] | None = None, tags: bool = True) -> Match:
    """The match as a Match. n_frames and halves default to the published BAR-ATH values and
    exist so tests can use a short synthetic bundle."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], GAME, data_dir, source)
    n_frames = n_frames or games.source_frames(source, GAME)
    halves = halves or games.half_video_blocks(GAME)
    length, width = games.GAMES[GAME]["pitch"]

    meta = epts.read_metadata(meta_path)
    raw = epts.read_raw(raw_path, len(meta["slot_team"]))
    if not np.array_equal(raw["frame_id"], np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame ids are not 1..{} in order ({} records); the row = frame_id - 1 "
                         "alignment does not hold".format(n_frames, len(raw["frame_id"])))

    period = np.zeros(n_frames, dtype=np.int8)
    if source == "tactical":  # the published half boundaries are tactical video frames
        for half, first, last in halves:
            period[first:last + 1] = half

    scale = np.array([length, width])
    fps = meta["frame_rate"] or games.GAMES[GAME]["fps"]
    tagged = None
    if tags and source == "tactical" and smart_tags.smart_tagging_path(data_dir, GAME):
        try:
            tagged = [t._replace(start=max(0, t.start), end=min(n_frames, t.end))
                      for t in smart_tags.read(meta["team_names"], data_dir, GAME, fps)]
            tagged = [t for t in tagged if t.end > t.start]
        except Exception as error:  # noqa: BLE001  no default model reads tags: warn, never stop
            print("warning: Smart Tagging not read ({}); Match.tags is None".format(
                type(error).__name__), file=sys.stderr)
    return Match(
        match_id="{}:{}".format(GAME, source),
        fps=fps,
        pitch=(length, width),
        period=period,
        ball=raw["ball"] * scale,
        players=raw["players"] * scale,
        slot_team=meta["slot_team"],
        tags=tagged,
    ).validate()
