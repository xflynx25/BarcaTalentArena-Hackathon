"""Metrica FIFA/EPTS tracking: metadata XML + raw text, read with numpy/pandas only.

Raw line: "<frame_id>:<slot>;<slot>;...;<slot>:<ball>", each group comma-separated fields with the
position first ("x,y,..."), normalised 0-1, "NaN" where not observed. The player groups are in the
order the <Player> elements appear in the metadata, which is also how official/starter/baseline.py
reads it.

The fast path turns every separator into a comma and lets pandas' C parser read the file as one
table, which is about 10x faster than splitting lines in Python. It is strict: every line must have
the same layout, or it raises. Error messages describe structure only, never values.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.schema import AWAY, HOME

_SEPARATORS_TO_COMMA = bytes.maketrans(b":;", b",,")


def read_metadata(meta_path: str) -> dict:
    """{slot_team, team_names, frame_rate}. The first <Team> is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 <Team> elements, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]

    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no <Player> elements".format(meta_path))
    unknown = set(slots) - {team_id for team_id, _ in teams}
    if unknown:
        raise ValueError("{}: players reference unknown team ids".format(meta_path))

    rate = root.findtext(".//GlobalConfig/FrameRate")
    rate = float(rate) if rate else None   # written as "25" or "25.0" depending on the exporter
    return {
        "slot_team": np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8),
        "team_names": (teams[0][1], teams[1][1]),
        "frame_rate": int(rate) if rate is not None and rate.is_integer() else rate,
    }


def read_raw(raw_path: str, n_slots: int) -> dict:
    """{frame_id (n,), players (n, n_slots, 2), ball (n, 2)}, normalised 0-1, NaN = missing."""
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    lines = blob.count(b"\n") + (0 if blob.endswith(b"\n") else 1)

    first = blob[:blob.index(b"\n")] if b"\n" in blob else blob
    parts = first.strip().split(b":")
    if len(parts) != 3:
        raise ValueError("{}: expected 3 ':'-separated parts per line, got {}".format(
            raw_path, len(parts)))
    groups = parts[1].split(b";")
    if len(groups) != n_slots:
        raise ValueError("{}: {} player groups per line but the metadata lists {} slots".format(
            raw_path, len(groups), n_slots))
    per_slot = len(groups[0].split(b","))
    per_ball = len(parts[2].split(b","))
    if per_slot < 2 or per_ball < 2:
        raise ValueError("{}: groups need at least x,y".format(raw_path))

    # Totals catch ragged lines that pandas would otherwise pad with NaN silently.
    if blob.count(b":") != 2 * lines or blob.count(b";") != (n_slots - 1) * lines:
        raise ValueError("{}: lines do not all share the first line's layout".format(raw_path))

    n_columns = 1 + n_slots * per_slot + per_ball
    # pandas 3's default parser rounds exactly like Python's float() (checked up to 17 digits), so
    # results match the official baseline bit for bit. tests/test_baseline_equivalence.py guards it.
    table = pd.read_csv(io.BytesIO(blob.translate(_SEPARATORS_TO_COMMA)), header=None,
                        dtype=np.float64).values
    if table.shape != (lines, n_columns):
        raise ValueError("{}: parsed shape {} != expected {}".format(
            raw_path, table.shape, (lines, n_columns)))

    players = table[:, 1:1 + n_slots * per_slot].reshape(lines, n_slots, per_slot)[:, :, :2]
    ball = table[:, 1 + n_slots * per_slot:1 + n_slots * per_slot + 2]
    return {
        "frame_id": table[:, 0].astype(np.int64),
        "players": np.ascontiguousarray(players),
        "ball": np.ascontiguousarray(ball),
    }
