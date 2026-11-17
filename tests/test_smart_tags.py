"""Smart Tagging (backlog B28): reader, stand-in tagger, SkillCorner mapping, feature group, subtitles.
Synthetic data and tiny fixtures only."""

import json

import numpy as np
import pytest

from src import smart_tags as st
from src.ingest import synthetic as synth_source
from src.models import build, learned
from src.official import A, D, H, X
from src.schema import Match


def test_bar_ath_loader_fills_tags_from_the_synthetic_bundle(small):
    match, spec = small["match"], small["spec"]
    assert match.tags, "the synthetic bundle has a Smart Tagging file"
    assert all(t.code == t.code.upper() and t.team in (H, A, st.NA) for t in match.tags)
    assert all(0 <= t.start < t.end <= match.n_frames for t in match.tags)
    throw = [t for t in match.tags if t.kind == "Throw-in"]
    assert all(t.team == st.NA for t in throw)
    kickoffs = [t for t in match.tags if t.kind == "Kick Off (Start)"]
    assert len(kickoffs) == 2
    for t, frame in zip(sorted(kickoffs), spec.kickoffs):
        assert t.start <= frame < t.end
    assert kickoffs[0].team == H  # the synthetic home team kicks off the first half


def test_loader_can_skip_tags(small):
    from src.ingest import bar_ath
    spec = small["spec"]
    match = bar_ath.load(small["dir"], n_frames=spec.n_frames, halves=spec.halves, tags=False)
    assert match.tags is None


def test_team_side_matches_spellings_and_falls_back_to_published_names():
    side = st.team_side({"Barcelona", "Atletic Club", "N/A"}, ("FC Barcelona", "Athletic Club"))
    assert side == {"Barcelona": H, "Atletic Club": A, "N/A": st.NA}
    assert st.team_side({"Barcelona", "Atletic Club"}, ("Home", "Away")) == {
        "Barcelona": H, "Atletic Club": A}
    with pytest.raises(ValueError):
        st.team_side({"Real Madrid"}, ("Home", "Away"))


def toy_match(seconds=600, fps=10):
    """A labelled toy match: H 20 s, A 20 s, D 10 s repeating; the ball walks the length of the
    pitch; home players on the left half, away on the right (home attacks +x)."""
    n = seconds * fps
    t = np.arange(n)
    cycle = (t // fps) % 50
    labels = np.where(cycle < 20, H, np.where(cycle < 40, A, D)).astype(np.int8)
    period = np.ones(n, dtype=np.int8)
    period[:fps] = 0
    labels[period == 0] = X
    ball = np.stack([52.5 + 50 * np.sin(t / (7.0 * fps)), 34 + 20 * np.cos(t / (11.0 * fps))], axis=1)
    rng = np.random.default_rng(0)
    players = np.concatenate([rng.uniform([5, 5], [50, 63], (n, 11, 2)),
                              rng.uniform([55, 5], [100, 63], (n, 11, 2))], axis=1)
    return Match("toy:1", fps, (105.0, 68.0), period, ball, players,
                 np.array([0] * 11 + [1] * 11, dtype=np.int8), labels).validate()


def test_stand_in_is_deterministic_and_agrees_with_the_labels():
    match = toy_match()
    a, b = st.stand_in(match), st.stand_in(match)
    assert a == b and a
    assert st.stand_in(match, seed=1) != a
    agree = st.team_agreement(a, match.labels)
    assert agree["agree"] > 0.97 and 0 < agree["coverage"] < 1
    oracle = st.stand_in(match, thin=False)
    assert len(oracle) >= len(a)
    restarts = [t for t in oracle if t.code == st.SET_PIECES and t.kind != "Kick Off (Start)"]
    assert len(restarts) == 11  # one per D -> team transition (the cycle restarts with H)
    codes = {t.code for t in oracle}
    assert {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "ATTACKING TRANSITION",
            "DEFENSIVE TRANSITION", "DEFENDING IN MIDDLE THIRD"} <= codes


def test_stand_in_needs_labels():
    match = toy_match()
    match.labels = None
    with pytest.raises(ValueError):
        st.stand_in(match)


def test_thinning_follows_the_published_rates():
    match = toy_match(seconds=3000)
    minutes = float((match.period > 0).sum()) / match.fps / 60
    thinned = st.counts(st.stand_in(match), minutes)
    for code, rate in st.PUBLISHED.items():
        assert thinned.get(code, 0) <= rate * 1.05 + st.PUBLISHED_MINUTES / minutes


def test_noise_degrades_team_agreement():
    match = toy_match(seconds=3000)
    agree = [st.team_agreement(st.stand_in(match, level), match.labels)["agree"]
             for level in (0, 2, 4)]
    assert agree[0] > agree[1] > agree[2]


def test_columns_are_zero_without_tags_and_swap_with_the_teams():
    n = 200
    home, away, common = st.columns(None, n)
    assert not home.any() and not away.any() and not common.any()
    tags = [st.Tag(10, 50, "BUILD UP", H), st.Tag(60, 110, st.SET_PIECES, A, "Free Kick"),
            st.Tag(120, 170, st.SET_PIECES, st.NA, "Throw-in"),
            st.Tag(30, 80, "DEFENDING IN MIDDLE THIRD", A)]
    home, away, common = st.columns(tags, n)
    swapped = [t._replace(team={H: A, A: H}.get(t.team, t.team)) for t in tags]
    h2, a2, c2 = st.columns(swapped, n)
    assert np.array_equal(home, a2) and np.array_equal(away, h2) and np.array_equal(common, c2)
    assert home[20, 0] == 1 and away[40, 3] == 1 and common[140, :3].sum() == pytest.approx(1)
    assert common[:, -1].sum() == 100 + 50  # inside any tag: [10, 110) and [120, 170)


def test_tags_group_is_off_by_default_and_changes_the_hash(small):
    assert "tags" not in learned.GROUPS
    assert learned.feature_hash(learned.GROUPS + ("tags",)) != learned.FEATURE_HASH
    match = small["match"]
    base = learned.features(match, ("touch",))
    with_tags = learned.features(match, ("touch", "tags"))
    assert with_tags[0].shape[1] == base[0].shape[1] + len(st.FAMILIES) + len(st.SET_KNOTS)
    assert with_tags[2].shape[1] == base[2].shape[1] + len(st.SET_KNOTS) + 1
    assert with_tags[0][:, base[0].shape[1]:].any()  # the bundle's tags speak
    none = learned.features(match, ("touch", "tags"), tags=[])
    assert not none[0][:, base[0].shape[1]:].any()


def test_decoder_tag_sources():
    match = synth_source.load("0")
    assert learned.tags_for(match, "none") == []
    assert learned.tags_for(match, "standin") == st.stand_in(match)
    with pytest.raises(ValueError):
        build("learned", tags_source="video")
    unlabelled = synth_source.load("0")
    unlabelled.labels = None
    with pytest.raises(ValueError):
        learned.tags_for(unlabelled, "standin")


def test_skillcorner_phases_map_to_codes(tmp_path):
    folder = tmp_path / "data" / "matches" / "7"
    folder.mkdir(parents=True)
    (folder / "7_match.json").write_text(json.dumps({"home_team": {"id": 1}, "away_team": {"id": 2}}))
    (folder / "7_tracking_extrapolated.jsonl").write_text(json.dumps({"frame": 100}) + "\n")
    rows = [("110", "150", "1", "build_up", "high_block", "False"),
            ("150", "190", "2", "finish", "low_block", "True"),
            ("190", "200", "2", "chaotic", "chaotic", "False")]
    with open(folder / "7_phases_of_play.csv", "w") as handle:
        handle.write("frame_start,frame_end,team_in_possession_id,team_in_possession_phase_type,"
                     "team_out_of_possession_phase_type,penalty_area_end\n")
        handle.writelines(",".join(r) + "\n" for r in rows)
    n = 120
    match = Match("skillcorner:7", 10, (105.0, 68.0), np.ones(n, np.int8), np.zeros((n, 2)),
                  np.zeros((n, 2, 2)), np.array([0, 1], np.int8))
    tags = st.skillcorner(match, root=str(tmp_path))
    assert tags == sorted([st.Tag(10, 50, "BUILD UP", H),
                           st.Tag(10, 50, "DEFENDING IN ATTACKING THIRD", A),
                           st.Tag(50, 90, "BALL IN THE BOX", A),
                           st.Tag(50, 90, "DEFENDING IN DEFENSIVE THIRD", H)])


def test_resample_moves_tags_to_the_new_frame_rate(small):
    from src.resample import to_fps

    match = small["match"]
    match10 = to_fps(match, 10)
    assert len(match10.tags) == len(match.tags)
    for old, new in zip(match.tags, match10.tags):
        assert abs(new.start - old.start * 10 / 25) <= 1 and new.code == old.code


def test_a_broken_tag_file_never_stops_the_loader(small, tmp_path, capsys):
    import shutil

    from src.ingest import bar_ath

    spec = small["spec"]
    folder = tmp_path / "bundle"
    shutil.copytree(small["dir"], folder)
    path = st.smart_tagging_path(str(folder))
    with open(path, "w") as handle:
        handle.write("<file><ALL_INSTANCES><instance><start>x")
    match = bar_ath.load(str(folder), n_frames=spec.n_frames, halves=spec.halves)
    assert match.tags is None and "Smart Tagging not read" in capsys.readouterr().err


# -- B32: box candidates, the defending codes' reading, the line-up check, the mixed fit -------------

def box_match(seconds=600, fps=10):
    """toy_match with three home attackers standing in the box home attacks (+x)."""
    m = toy_match(seconds, fps)
    players = m.players.copy()
    players[:, :3] = [[100.0, 30.0], [98.0, 34.0], [99.0, 38.0]]
    return Match(m.match_id, fps, m.pitch, m.period, m.ball, players, m.slot_team, m.labels).validate()


def test_box_v2_adds_box_candidates():
    match = box_match()
    v1, v2 = st.candidates(match, box_v2=False), st.candidates(match)
    for code in ("BALL IN THE BOX", "PLAYERS IN THE BOX"):
        assert sum(t.code == code for t in v2) > sum(t.code == code for t in v1)
    players = [t for t in v2 if t.code == "PLAYERS IN THE BOX"]
    assert all(t.team == H for t in players)  # only home attacks the box its players stand in
    assert all(dict(t.labels)["Max Players in the box"] == "3" for t in players)


def test_convention_finds_either_reading_and_canonical_undoes_a_flip():
    match = toy_match()
    for level in (0, 1, 2, 3):
        tags = st.stand_in(match, level)
        assert st.convention(tags, match) == "defender"
        flipped = st.flip_defending(tags)
        assert flipped != tags and st.convention(flipped, match) == "attacker"
        assert st.canonical(flipped, match) == tags
    assert st.flip_defending(st.flip_defending(tags)) == tags
    assert st.convention([], match) == "defender" and st.canonical(None, match) is None


def test_decoder_reads_a_flipped_tagger_like_the_original():
    match = toy_match()
    plain = learned.tags_for(match, "standin", 1)
    assert learned.tags_for(match, "standin", 1, flip=True) == plain
    assert learned.tags_for(match, "standin", 1, flip=True, convention="as_is") != plain
    with pytest.raises(ValueError):
        build("learned", tags_convention="guess")


def test_line_up_passes_good_tags_and_fails_shifted_or_missing_ones():
    match = toy_match()
    tags = st.stand_in(match, 0)
    assert st.line_up(tags, match) == {"parse": True, "teams": True, "kickoff": True, "zones": True}
    late = [t._replace(start=t.start + 1500, end=min(match.n_frames, t.end + 1500)) for t in tags]
    late = [t for t in late if t.end > t.start]
    checks = st.line_up(late, match)
    assert not checks["kickoff"] and not checks["zones"]
    assert not any(st.line_up([], match).values())
    assert st.line_up([t._replace(team=H) for t in tags], match)["teams"] is False


def test_line_up_checks_published_kickoffs_by_containment():
    match = toy_match()
    tags = st.stand_in(match, 0)
    kick = [t for t in tags if t.kind == "Kick Off (Start)"]
    inside = (kick[0].start + kick[0].end) // 2
    assert st.line_up(tags, match, [inside])["kickoff"]
    assert not st.line_up(tags, match, [kick[0].end + 5])["kickoff"]
