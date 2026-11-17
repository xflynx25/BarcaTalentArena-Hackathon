import json

import numpy as np
import pandas as pd
import pytest

from src.ingest import metrica
from src.official import A, D, H, X
from src.schema import AWAY, HOME


def _events(rows):
    """rows: (team, type, subtype, period, start, end) with team 'H'/'A'/None."""
    side = {"H": HOME, "A": AWAY, None: -1}
    return pd.DataFrame([{"team": side[t], "type": k, "subtype": s, "period": p, "start": a,
                          "end": b} for t, k, s, p, a, b in rows]).astype(
        {"team": np.int8, "period": np.int8, "start": np.int64, "end": np.int64})


def _derive(rows, n=30, halves=((1, 1, 30),)):
    frame = np.arange(1, n + 1)
    period = np.zeros(n, dtype=np.int8)
    for half, first, last in halves:
        period[first - 1:last] = half
    return metrica.derive_labels(_events(rows), frame, period)


def test_pass_keeps_team_recovery_switches():
    labels = _derive([
        ("H", "SET PIECE", "KICK OFF", 1, 3, 0),
        ("H", "PASS", "", 1, 3, 8),
        ("H", "BALL LOST", "INTERCEPTION", 1, 8, 12),
        ("A", "CHALLENGE", "GROUND-WON", 1, 10, 10),  # a duel alone does not switch
        ("A", "RECOVERY", "INTERCEPTION", 1, 12, 12),
        ("A", "PASS", "", 1, 12, 20),
    ], n=20)
    # frames 1-2 before kick-off are dead; H from 3, A from the recovery at 12; last event ends at 20
    assert labels.tolist() == [D] * 2 + [H] * 9 + [A] * 9


def test_dead_until_restart_even_with_same_frame_events_listed_first():
    labels = _derive([
        ("H", "SET PIECE", "KICK OFF", 1, 1, 0),
        ("H", "PASS", "", 1, 1, 5),
        ("H", "BALL OUT", "", 1, 5, 8),         # ball leaves at frame 8
        ("A", "CARD", "YELLOW", 1, 10, 10),     # cards don't extend live time
        ("A", "RECOVERY", "", 1, 14, 14),       # logged on the restart frame, before it
        ("A", "SET PIECE", "THROW IN", 1, 14, 14),
        ("A", "PASS", "", 1, 14, 20),
    ], n=20)
    assert labels.tolist() == [H] * 8 + [D] * 5 + [A] * 7


def test_goal_kick_pass_is_a_restart_and_advantage_stays_live():
    labels = _derive([
        ("H", "SET PIECE", "KICK OFF", 1, 1, 0),
        ("H", "PASS", "", 1, 1, 4),
        ("A", "FAULT RECEIVED", "", 1, 4, 4),   # played on: no restart follows -> stays live
        ("H", "SHOT", "OFF TARGET-OUT", 1, 6, 9),
        ("A", "PASS", "GOAL KICK", 1, 15, 20),
    ], n=20)
    assert labels.tolist() == [H] * 9 + [D] * 5 + [A] * 6


def test_periods_and_tail():
    labels = _derive([
        ("H", "SET PIECE", "KICK OFF", 1, 2, 0),
        ("H", "PASS", "", 1, 2, 6),
        ("A", "SET PIECE", "KICK OFF", 2, 14, 0),
        ("A", "PASS", "", 2, 14, 16),
    ], n=20, halves=((1, 1, 8), (2, 12, 20)))
    assert labels.tolist() == ([D] + [H] * 5 + [D] * 2 + [X] * 3 + [D] * 2 + [A] * 3 + [D] * 4)


# -- file readers on tiny fixtures -------------------------------------------------------------------

def _tracking_csv(path, side, players, rows):
    """Metrica CSV layout: 2 header rows (team, jersey), then Period,Frame,Time,PlayerN,,...,Ball,."""
    jerseys = [p.replace("Player", "") for p in players]
    head1 = ",,," + "".join("{},,".format(side) for _ in players) + ","
    head2 = ",,," + "".join("{},,".format(j) for j in jerseys) + ","
    head3 = "Period,Frame,Time [s]," + "".join("{},,".format(p) for p in players) + "Ball,"
    body = ["{},{},{}," .format(p, f, f * 0.04) + ",".join(str(v) for v in xy) for p, f, xy in rows]
    path.write_text("\n".join([head1, head2, head3] + body) + "\n")


@pytest.fixture
def tiny(tmp_path):
    g1 = tmp_path / "data" / "Sample_Game_1"
    g1.mkdir(parents=True)
    _tracking_csv(g1 / "Sample_Game_1_RawTrackingData_Home_Team.csv", "Home", ["Player1", "Player2"],
                  [(1, 1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.5]), (1, 2, [0.1, 0.2, "NaN", "NaN", 0.6, 0.5]),
                   (2, 3, [0.9, 0.8, 0.3, 0.4, "NaN", "NaN"]), (2, 4, [0.9, 0.8, 0.3, 0.4, 0.5, 0.5])])
    _tracking_csv(g1 / "Sample_Game_1_RawTrackingData_Away_Team.csv", "Away", ["Player9"],
                  [(1, 1, [0.7, 0.7, 0.5, 0.5]), (1, 2, [0.7, 0.7, 0.6, 0.5]),
                   (2, 3, [0.2, 0.2, "NaN", "NaN"]), (2, 4, [0.2, 0.2, 0.5, 0.5])])
    (g1 / "Sample_Game_1_RawEventsData.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,"
        "Start X,Start Y,End X,End Y\n"
        "Home,SET PIECE,KICK OFF,1,1,0.04,0,0,Player1,,NaN,NaN,NaN,NaN\n"
        "Home,PASS,,1,1,0.04,2,0.08,Player1,Player2,0.5,0.5,0.6,0.5\n"
        "Away,SET PIECE,KICK OFF,2,4,0.16,0,0,Player9,,NaN,NaN,NaN,NaN\n")

    g3 = tmp_path / "data" / "Sample_Game_3"
    g3.mkdir(parents=True)
    channels = "".join('<PlayerChannel channelId="{c}" id="{p}_{c}" playerId="{p}"/>'.format(p=p, c=c)
                       for p in ("PA1", "PA2", "PB1") for c in "xy")

    def spec(first, last, pids):
        groups = "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="{0}_x"/>'
                         '<PlayerChannelRef playerChannelId="{0}_y"/></SplitRegister>'.format(p)
                         for p in pids)
        return ('<DataFormatSpecification endFrame="{}" separator=":" startFrame="{}">'
                '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
                '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
                '<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>'
                .format(last, first, groups))

    params = "".join("<ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>"
                     .format(k, v) for k, v in [("first_half_start", 1), ("first_half_end", 2),
                                                ("second_half_start", 3), ("second_half_end", 4),
                                                ("first_extra_half_start", "")])
    (g3 / "Sample_Game_3_metadata.xml").write_text(
        '<?xml version="1.0"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate>'
        "<ProviderGlobalParameters>{}</ProviderGlobalParameters></GlobalConfig>"
        "<Sessions><Session><FieldSize><Width>100</Width><Height>50</Height></FieldSize></Session>"
        '</Sessions><Teams><Team id="TA"><Name>A</Name></Team><Team id="TB"><Name>B</Name></Team>'
        '</Teams><Players><Player id="PA1" teamId="TA"/><Player id="PB1" teamId="TB"/>'
        '<Player id="PA2" teamId="TA"/></Players><Devices><Device><Sensors><Sensor><Channels>{}'
        "</Channels></Sensor></Sensors></Device></Devices></Metadata>"
        "<DataFormatSpecifications>{}{}</DataFormatSpecifications></main>"
        .format(params, channels, spec(1, 2, ["PA1", "PB1"]), spec(3, 4, ["PA2", "PB1"])))
    (g3 / "Sample_Game_3_tracking.txt").write_text(
        "1:0.1,0.1;0.9,0.9:0.5,0.5\n2:0.2,0.1;0.8,0.9:NaN,NaN\n"
        "3:0.3,0.3;0.7,0.7:0.5,0.5\n4:0.4,0.3;0.6,0.7:0.5,0.6\n")
    events = [
        {"team": {"id": "TA"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"},
         "start": {"frame": 1}, "end": {"frame": 1}, "period": 1},
        {"team": {"id": "TA"}, "type": {"name": "CARRY"}, "subtypes": None,
         "start": {"frame": 1}, "end": {"frame": 2}, "period": 1},
        {"team": {"id": "TB"}, "type": {"name": "SET PIECE"}, "subtypes": [{"name": "KICK OFF"}],
         "start": {"frame": 3}, "end": {"frame": 3}, "period": 2},
        {"team": {"id": "TB"}, "type": {"name": "PASS"}, "subtypes": None,
         "start": {"frame": 3}, "end": {"frame": 4}, "period": 2},
    ]
    (g3 / "Sample_Game_3_events.json").write_text(json.dumps({"metadata": [], "data": events}))
    return tmp_path


def test_list_matches(tiny):
    assert metrica.list_matches(str(tiny)) == ["1", "3"]
    assert metrica.list_matches(str(tiny / "nowhere")) == []


def test_csv_game(tiny):
    match = metrica.load(1, root=str(tiny))
    assert match.match_id == "metrica:1" and match.fps == 25 and match.pitch == (105.0, 68.0)
    assert match.period.tolist() == [1, 1, 2, 2]
    assert match.slot_team.tolist() == [HOME, HOME, AWAY]
    np.testing.assert_allclose(match.players[0, 1], [0.3 * 105, 0.4 * 68])
    np.testing.assert_allclose(match.players[2, 2], [0.2 * 105, 0.2 * 68])
    assert np.isnan(match.players[1, 1]).all() and np.isnan(match.ball[2]).all()
    np.testing.assert_allclose(match.ball[1], [0.6 * 105, 0.5 * 68])
    assert match.labels.tolist() == [H, H, D, A]


def test_epts_game_remaps_substitutions(tiny):
    match = metrica.load("3", root=str(tiny))
    assert match.pitch == (100.0, 50.0) and match.period.tolist() == [1, 1, 2, 2]
    # slots: home PA1, PA2 (metadata order within team), then away PB1
    assert match.slot_team.tolist() == [HOME, HOME, AWAY]
    np.testing.assert_allclose(match.players[:, 0, 0], [10, 20, np.nan, np.nan])
    np.testing.assert_allclose(match.players[:, 1, 0], [np.nan, np.nan, 30, 40])
    np.testing.assert_allclose(match.players[:, 2, 0], [90, 80, 70, 60])
    assert match.labels.tolist() == [H, H, A, A]


@pytest.mark.skipif(not metrica.list_matches(), reason="Metrica sample data not fetched")
def test_real_games_load():
    for game in metrica.list_matches():
        match = metrica.load(game)
        in_play = match.period > 0
        assert match.fps == 25 and (match.labels[in_play] != X).all()
        assert 0.15 < (match.labels[in_play] == D).mean() < 0.6
