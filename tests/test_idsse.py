import numpy as np
import pytest

from src.ingest import idsse
from src.official import A, D, H, X
from src.schema import AWAY, HOME

KEY = "DFL-MAT-TEST01"
HOME_TEAM, AWAY_TEAM = "DFL-CLU-HOME", "DFL-CLU-AWAY"

META = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="{key}" HomeTeamId="{home}" GuestTeamId="{away}"/>
<Environment PitchX="100.00" PitchY="60.00"/>
<Teams>
 <Team TeamId="{away}" TeamName="Away" Role="guest"><Players>
  <Player PersonId="P-A1"/><Player PersonId="P-A9"/></Players></Team>
 <Team TeamId="{home}" TeamName="Home" Role="home"><Players>
  <Player PersonId="P-H2"/><Player PersonId="P-H1"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>
""".format(key=KEY, home=HOME_TEAM, away=AWAY_TEAM)


def _frameset(section, team, person, frames):
    rows = []
    for f in frames:
        n, x, y = f[:3]
        extra = ' BallPossession="{}" BallStatus="{}"'.format(*f[3:5]) if team == "BALL" else ""
        t = f[5] if len(f) > 5 else "2023-01-01T15:00:00.000+00:00"
        rows.append('<Frame N="{}" T="{}" X="{}" Y="{}" S="0.00"{}/>'.format(n, t, x, y, extra))
    return ('<FrameSet GameSection="{}" MatchId="{}" TeamId="{}" PersonId="{}">\n{}\n</FrameSet>'
            .format(section, KEY, team, person, "\n".join(rows)))


@pytest.fixture
def tiny(tmp_path):
    """First half: frames 10000-10003, frame 10002 missing from the ball. Second half: 100000-100001,
    starting 0.2 s (5 frames) after the first half ends."""
    t = "2023-01-01T15:00:{}+00:00"
    framesets = [
        _frameset("firstHalf", "referee", "REF", [(10000, 0, 0)]),
        _frameset("firstHalf", "BALL", "BALL1", [
            (10000, 0.0, 0.0, 1, 0, t.format("00.000")),   # dead: D
            (10001, 1.0, 1.0, 1, 1, t.format("00.040")),   # home
            (10003, -50.0, -30.0, 2, 1, t.format("00.120")),  # away, 10002 missing -> ffill H
        ]),
        _frameset("firstHalf", HOME_TEAM, "P-H2", [(10000, -1.0, 2.0), (10001, -1.0, 2.5)]),
        _frameset("firstHalf", AWAY_TEAM, "P-A1", [(10003, -49.0, -29.0)]),
        _frameset("secondHalf", "BALL", "BALL1", [
            (100000, 0.0, 0.0, 2, 1, t.format("00.320")),
            (100001, 0.0, 0.0, 2, 0, t.format("00.360")),
        ]),
        _frameset("secondHalf", HOME_TEAM, "P-H1", [(100000, 5.0, 5.0), (100001, 6.0, 5.0)]),
        _frameset("secondHalf", AWAY_TEAM, "P-A1", [(100001, 1.0, 1.0)]),
    ]
    positions = ('<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest><Positions>\n'
                 '<MetaData MatchId="{}" Type="pitch-size"><PitchSize X="100.00" Y="60.00"/></MetaData>\n'
                 '{}\n</Positions></PutDataRequest>\n').format(KEY, "\n".join(framesets))
    (tmp_path / "DFL_02_01_matchinformation_DFL-COM-000001_{}.xml".format(KEY)).write_text(META)
    (tmp_path / "DFL_04_03_positions_raw_observed_DFL-COM-000001_{}.xml".format(KEY)).write_text(positions)
    # an unfinished download must be skipped by list_matches
    (tmp_path / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-PART01.xml.part").write_text("")
    return tmp_path


def test_rows_periods_and_labels(tiny):
    match = idsse.load("TEST01", root=str(tiny))
    assert match.match_id == "idsse:" + KEY and match.fps == 25 and match.pitch == (100.0, 60.0)
    # 4 first-half rows, a 4-row break (0.2 s gap = 5 frames apart), 2 second-half rows
    assert match.period.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 2, 2]
    assert match.labels.tolist() == [D, H, H, A, X, X, X, X, A, D]


def test_slots_and_coordinates(tiny):
    match = idsse.load(KEY, root=str(tiny))
    # slots ordered (team, person id): home P-H1, home P-H2, away P-A1; referee and P-A9 dropped
    assert match.slot_team.tolist() == [HOME, HOME, AWAY]
    np.testing.assert_allclose(match.players[1, 1], [49.0, 32.5])
    np.testing.assert_allclose(match.players[3, 2], [1.0, 1.0])
    np.testing.assert_allclose(match.ball[3], [0.0, 0.0])  # corner of the pitch
    np.testing.assert_allclose(match.players[8, 0], [55.0, 35.0])
    assert np.isnan(match.ball[2]).all()  # missing ball frame
    assert np.isnan(match.players[5]).all()  # half-time break
    assert np.isnan(match.players[8, 2]).all()


def test_list_matches_and_ids(tiny):
    assert idsse.list_matches(str(tiny)) == [KEY]
    assert idsse.list_matches(str(tiny / "nowhere")) == []
    assert idsse.match_key("idsse:" + KEY) == idsse.match_key("TEST01") == KEY


def test_bad_possession_rejected(tiny):
    path = tiny / "DFL_04_03_positions_raw_observed_DFL-COM-000001_{}.xml".format(KEY)
    path.write_text(path.read_text().replace('BallPossession="2" BallStatus="1"', 'BallPossession="0" BallStatus="1"', 1))
    with pytest.raises(ValueError, match="BallPossession"):
        idsse.load(KEY, root=str(tiny))


def test_missing_files(tiny):
    with pytest.raises(FileNotFoundError):
        idsse.load("NOPE01", root=str(tiny))


@pytest.mark.skipif(not idsse.list_matches(), reason="IDSSE tracking not downloaded")
def test_real_match():
    """On real data: shapes, label mix, and that H/A agree with the nearest player's team."""
    from src.models.nearest import nearest_team

    match = idsse.load(idsse.list_matches()[0])
    in_play = match.period > 0
    assert set(np.unique(match.period)) >= {0, 1, 2}
    assert (match.labels[~in_play] == X).all() and (match.labels[in_play] != X).all()
    share = np.bincount(match.labels[in_play], minlength=3) / in_play.sum()
    assert all(0.15 < s < 0.6 for s in share[:3])
    assert 22 <= len(match.slot_team) <= 40 and set(match.slot_team.tolist()) == {HOME, AWAY}
    ball = match.ball[np.isfinite(match.ball[:, 0])]
    assert np.median(ball[:, 0]) == pytest.approx(match.pitch[0] / 2, abs=15)
    # half-time break is real time: several minutes
    assert (~in_play).sum() > 5 * 60 * match.fps
    # If home/guest were swapped, agreement would be well under 50%.
    teams = nearest_team(match)
    live = np.isin(match.labels, [H, A]) & (teams >= 0)
    agree = (np.where(match.labels[live] == H, HOME, AWAY) == teams[live]).mean()
    assert agree > 0.7
