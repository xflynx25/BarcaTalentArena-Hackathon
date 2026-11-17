import shutil

import numpy as np
import pytest

from src.ingest import bar_ath, epts


def test_shapes_and_teams(small):
    match, spec = small["match"], small["spec"]
    assert match.players.shape == (spec.n_frames, spec.n_slots, 2)
    assert match.ball.shape == (spec.n_frames, 2)
    assert sorted(np.bincount(match.slot_team)) == [spec.n_slots // 2] * 2
    assert match.period[spec.halves[0][1]] == 1 and match.period[spec.halves[1][2]] == 2
    assert match.period[0] == 0


def test_coordinates_in_metres(small):
    match = small["match"]
    length, width = match.pitch
    x = match.players[:, :, 0][~np.isnan(match.players[:, :, 0])]
    assert 0 <= x.min() and x.max() <= length and x.max() > length / 2


def test_agrees_with_kloppy(small):
    """Our fast parser and kloppy's regex parser read the same numbers from the same files."""
    from kloppy import metrica

    dataset = metrica.load_tracking_epts(meta_data=small["paths"]["atd_meta"],
                                         raw_data=small["paths"]["atd_raw"])
    frame = dataset.to_df(engine="pandas")
    match = small["match"]
    length, width = match.pitch
    assert len(frame) == match.n_frames
    np.testing.assert_allclose(frame["ball_x"].values, match.ball[:, 0] / length, atol=1e-9)
    players = [p for team in dataset.metadata.teams for p in team.players]
    slot = {p.player_id: int(p.player_id[1:]) - 1 for p in players}
    for player_id in list(slot)[:5]:
        np.testing.assert_allclose(frame[player_id + "_y"].values,
                                   match.players[:, slot[player_id], 1] / width, atol=1e-9)


def test_frame_rate_written_as_float(small, tmp_path):
    meta = tmp_path / "meta.xml"
    text = open(small["paths"]["atd_meta"]).read()
    fps = small["spec"].fps
    meta.write_text(text.replace("<FrameRate>{}<".format(fps), "<FrameRate>{}.0<".format(fps)))
    assert ".0</FrameRate>" in meta.read_text()
    assert epts.read_metadata(str(meta))["frame_rate"] == fps


def test_rejects_ragged_lines(small, tmp_path):
    raw = tmp_path / "ragged.txt"
    lines = open(small["paths"]["atd_raw"]).read().splitlines()
    head, groups, ball = lines[10].split(":")
    lines[10] = ":".join([head, ";".join(groups.split(";")[:-1]), ball])
    raw.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="layout"):
        epts.read_raw(str(raw), small["spec"].n_slots)


def test_rejects_wrong_frame_count(small):
    with pytest.raises(ValueError, match="frame ids"):
        bar_ath.load(small["dir"], n_frames=small["spec"].n_frames + 1, halves=small["spec"].halves)


def test_missing_bundle_is_a_readable_error(tmp_path):
    with pytest.raises(IOError, match="match data not found"):
        bar_ath.load(str(tmp_path))
