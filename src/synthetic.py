"""A fake BAR-ATH bundle in the exact official file formats, built only from the public spec.

Cloud agents never get the real match data, so this is what they run the pipeline on end to end:
the official loaders, our loaders, the models, the validator and the scorer. The file layout follows
official/DATA.md, official/starter/*.py and the public Metrica EPTS sample (Sample_Game_3). kloppy
can read it too.

    python scripts/make_synthetic_bar_ath.py              # full length, data/synthetic/mtah2027
    python scripts/make_synthetic_bar_ath.py --small      # 4,000 frames, for quick tests

The broadcast/ folder (V4) holds a simulated TV recording of the same fake match on its own
timeline (src/ingest/degrade.py: second_recording: camera window, cuts, replays, drift).

What gets simulated: possessions of random length that alternate between the teams with dead-ball
stoppages in between, 11 v 11 in a formation that pushes up with the ball, passes as ball flights
between team-mates, a nearby opponent pressing the carrier, the ball missing about 40% of the time,
player dropouts, and players switching track slots. synthetic_gt.csv holds the labels the simulation
used.

The football is crude. Scores on this bundle check that the code works. They say nothing about how
a model will do on the real match. Use external matches for that.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np

from src import submission
from src.official import A, D, H, X, games

GAME = "BAR-ATH"
HOME_NAME, AWAY_NAME = "Barcelona", "Athletic Club"
TAG_TEAM = {H: "Barcelona", A: "Atletic Club"}  # Smart Tagging spelling, per DATA.md
HALF_TAG = {1: "1st Half", 2: "2nd Half"}
# Example matrix from DATA.md (normalised pitch -> pixels).
HOMOGRAPHY = np.array([1822.663, -511.2005, 44.48168, 1.698323, 169.2971, 255.7303,
                       -0.01907188, -0.5269586, 1.0])
# Formation, home attacking +x: (fraction of pitch length, fraction of width). Player 0 is the GK.
FORMATION = np.array([(0.04, 0.50), (0.20, 0.15), (0.18, 0.38), (0.18, 0.62), (0.20, 0.85),
                      (0.34, 0.20), (0.32, 0.42), (0.32, 0.58), (0.34, 0.80),
                      (0.45, 0.38), (0.45, 0.62)])


@dataclass
class Spec:
    n_frames: int = games.GAMES[GAME]["n_video_frames"]
    halves: tuple = tuple(games.half_video_blocks(GAME))
    kickoffs: tuple = (11185, 113742)  # official/starter/load_smart_tagging.py
    n_slots: int = 46
    n_tags: int | None = 638
    n_broadcast: int = games.source_frames("broadcast", GAME)  # broadcast ATD records (V4)
    seed: int = 0
    fps: int = games.GAMES[GAME]["fps"]
    pitch: tuple = games.GAMES[GAME]["pitch"]

    @classmethod
    def small(cls, seed: int = 0) -> "Spec":
        return cls(n_frames=4000, halves=((1, 200, 1799), (2, 2200, 3799)), kickoffs=(200, 2300),
                   n_slots=30, n_tags=None, n_broadcast=3600, seed=seed)


# -- simulation ------------------------------------------------------------------------------------
def _runs(rng, m, mean_on, mean_off, start_on=True):
    """Boolean mask of alternating on/off runs with geometric lengths."""
    mask = np.zeros(m, dtype=bool)
    t, on = 0, start_on
    while t < m:
        length = int(rng.geometric(1.0 / (mean_on if on else mean_off)))
        if on:
            mask[t:t + length] = True
        t += length
        on = not on
    return mask


def _smooth_noise(rng, m, shape, width, scale):
    """Smooth random wander with standard deviation `scale`: white noise through two boxcars."""
    white = rng.standard_normal((m + 2 * width,) + shape)
    for _ in range(2):
        cum = np.cumsum(white, axis=0)
        white = cum[width:] - cum[:-width]
    return white[:m] / white[:m].std() * scale


def _segments(rng, spec, half, first, last, kickoff):
    """[(start, end_exclusive, code)] covering one half: possessions and dead-ball stoppages."""
    fps = spec.fps
    out = []
    t = first
    if kickoff > first:
        out.append((first, kickoff, D))
        t = kickoff
    team = H if half == 1 else A
    while t <= last:
        end = min(t + int(max(2 * fps, rng.exponential(12 * fps))), last + 1)
        out.append((t, end, team))
        t = end
        if t > last:
            break
        if rng.random() < 0.3:
            end = min(t + int(rng.uniform(4, 30) * fps), last + 1)
            out.append((t, end, D))
            t = end
            team = H if rng.random() < 0.5 else A
        else:
            team = A if team == H else H
    return out


def simulate(spec: Spec) -> dict:
    rng = np.random.default_rng(spec.seed)
    n, k, fps = spec.n_frames, spec.n_slots, spec.fps
    length, width = spec.pitch
    team_of = np.array([0] * 11 + [1] * 11)

    labels = np.full(n, X, dtype=np.int8)
    segments = []
    for (half, first, last), kickoff in zip(spec.halves, spec.kickoffs):
        for seg in _segments(rng, spec, half, first, last, kickoff):
            labels[seg[0]:seg[1]] = seg[2]
            segments.append((half,) + seg)

    # Attack direction of the home team per frame: +1 in half 1, -1 in half 2.
    direction = np.ones(n)
    for half, first, last in spec.halves:
        if half == 2:
            direction[first:] = -1.0

    # Team push towards the attack: +8 m with the ball, -5 m without, smoothed over 2 s.
    push = np.zeros((n, 2))
    for team, code in ((0, H), (1, A)):
        alive = (labels == H) | (labels == A)
        push[:, team] = np.where(labels == code, 8.0, np.where(alive, -5.0, 0.0))
    cum = np.cumsum(np.vstack([np.zeros((1, 2)), push]), axis=0)
    w = 2 * fps
    push = (cum[np.minimum(np.arange(n) + w, n)] - cum[np.maximum(np.arange(n) - w, 0)]) / (2 * w)

    sign = np.stack([direction, -direction], axis=1)  # per team: +1 means attacking +x
    base_x = np.concatenate([FORMATION[:, 0], FORMATION[:, 0]]) * length  # own-goal distance
    base_y = np.concatenate([FORMATION[:, 1], 1 - FORMATION[:, 1]]) * width
    pos = np.empty((n, 22, 2))
    team_sign = sign[:, team_of]  # (n, 22)
    own_goal_x = np.where(team_sign > 0, 0.0, length)
    pos[:, :, 0] = own_goal_x + team_sign * (base_x + push[:, team_of])
    pos[:, :, 1] = base_y
    pos += _smooth_noise(rng, n, (22, 2), 3 * fps, 4.0)
    pos[:, :, 0] = np.clip(pos[:, :, 0], 0.3, length - 0.3)
    pos[:, :, 1] = np.clip(pos[:, :, 1], 0.3, width - 0.3)

    ball = np.full((n, 2), np.nan)
    _play_ball(rng, spec, segments, pos, team_of, team_sign, ball)

    # Outside the halves: a ball and a few players wander about.
    in_play = np.zeros(n, dtype=bool)
    for _, first, last in spec.halves:
        in_play[first:last + 1] = True
    idle = ~in_play
    ball[idle] = (np.array([length / 2, width / 2])
                  + _smooth_noise(rng, n, (2,), 4 * fps, 8.0))[idle]

    ball_seen = np.where(in_play, _runs(rng, n, 3.0 * fps, 1.9 * fps),
                         _runs(rng, n, 4 * fps, 5 * fps))
    player_seen = np.empty((n, 22), dtype=bool)
    for p in range(22):
        player_seen[:, p] = np.where(in_play, _runs(rng, n, 15 * fps, 1 * fps),
                                     _runs(rng, n, 2 * fps, 8 * fps))

    slot_team = np.array([0] * (k // 2) + [1] * (k - k // 2), dtype=np.int8)
    rng.shuffle(slot_team)
    slot_of = _slot_assignment(rng, n, slot_team, team_of, per_team_per_minute=1.0, fps=fps)

    speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=-1) * fps
    ball_speed = np.nan_to_num(np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=-1)) * fps

    slots = np.full((n, k, 3), np.nan)
    scale = np.array([length, width])
    for p in range(22):
        ts = np.flatnonzero(player_seen[:, p])
        slots[ts, slot_of[ts, p], :2] = pos[ts, p] / scale
        slots[ts, slot_of[ts, p], 2] = speed[ts, p]
    ball_out = np.full((n, 3), np.nan)
    ball_out[ball_seen, :2] = ball[ball_seen] / scale
    ball_out[ball_seen, 2] = ball_speed[ball_seen]

    return {"slots": slots, "ball": ball_out, "slot_team": slot_team, "labels": labels,
            "segments": segments, "pos": pos, "team_of": team_of, "ball_m": ball}


def _play_ball(rng, spec, segments, pos, team_of, team_sign, ball):
    """Fill ball positions: held by a carrier, flying between team-mates, still when dead."""
    fps = spec.fps
    length, width = spec.pitch
    centre = np.array([length / 2, width / 2])
    chain = []  # touches (frame, player) of consecutive live segments

    def flush(chain_end):
        for i, (t, carrier) in enumerate(chain):
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            stop = nxt[0] if nxt else chain_end
            flight = min(int(rng.uniform(0.4, 1.2) * fps), (stop - t) // 2) if nxt else 0
            hold = slice(t, stop - flight)
            ahead = np.stack([team_sign[hold, carrier] * 0.7, np.zeros(hold.stop - hold.start)], 1)
            ball[hold] = pos[hold, carrier] + ahead
            # The nearest opponent presses the carrier from a random side.
            opponents = np.flatnonzero(team_of != team_of[carrier])
            gaps = np.linalg.norm(pos[t, opponents] - pos[t, carrier], axis=1)
            presser = opponents[np.argmin(gaps)]
            angle = rng.uniform(0, 2 * np.pi)
            offset = rng.uniform(0.8, 3.0) * np.array([np.cos(angle), np.sin(angle)])
            pos[hold, presser] = pos[hold, carrier] + offset
            if flight:
                start = ball[stop - flight - 1]
                end = pos[stop, nxt[1]]
                frac = (np.arange(1, flight + 1) / (flight + 1))[:, None]
                ball[stop - flight:stop] = start + frac * (end - start)
        chain.clear()

    previous_live = False
    for half, start, end, code in segments:
        if code == D:
            if previous_live:
                flush(start)
            last = ball[start - 1] if start > 0 and not np.isnan(ball[start - 1, 0]) else centre
            spot = last.copy()
            if rng.random() < 0.5:  # throw-in
                spot[1] = 0.3 if spot[1] < width / 2 else width - 0.3
            ball[start:end] = spot
            previous_live = False
            continue
        team = 0 if code == H else 1
        players = np.flatnonzero(team_of == team)
        t = start
        while t < end:
            carrier = players[0] if rng.random() < 0.08 else rng.choice(players[1:])
            chain.append((t, carrier))
            t += int(rng.uniform(0.8, 3.5) * fps)
        previous_live = True
        if end in {last + 1 for _, _, last in spec.halves}:
            flush(end)
            previous_live = False
    if chain:
        flush(spec.n_frames)


def _slot_assignment(rng, n, slot_team, team_of, per_team_per_minute, fps):
    """(n, 22) slot index per frame. Players start in their team's first slots and now and then
    jump to a free slot of the same team, like a tracker losing and re-acquiring someone."""
    team_slots = [list(np.flatnonzero(slot_team == team)) for team in (0, 1)]
    current = {}
    for team in (0, 1):
        for i, p in enumerate(np.flatnonzero(team_of == team)):
            current[p] = team_slots[team][i]
    slot_of = np.tile(np.array([current[p] for p in range(22)], dtype=np.int64), (n, 1))
    n_events = int(2 * per_team_per_minute * n / (60 * fps))
    for t in np.sort(rng.choice(n, size=n_events, replace=False)):
        team = int(rng.integers(2))
        p = int(rng.choice(np.flatnonzero(team_of == team)))
        taken = {current[q] for q in range(22) if team_of[q] == team}
        free = [s for s in team_slots[team] if s not in taken]
        if free:
            current[p] = int(rng.choice(free))
            slot_of[t:, p] = current[p]
    return slot_of


# -- writers ---------------------------------------------------------------------------------------
def _metadata_xml(spec, slot_team):
    n, k = spec.n_frames, spec.n_slots
    team_ids = ("FIFATMA", "FIFATMB")
    first, second = spec.halves[0], spec.halves[-1]
    params = {"first_half_start": first[1] + 1, "first_half_end": first[2] + 1,
              "second_half_start": second[1] + 1, "second_half_end": second[2] + 1}
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
             "    <GlobalConfig>", "      <ProviderName>Metrica Sports</ProviderName>",
             "      <FrameRate>{}</FrameRate>".format(spec.fps),
             "      <ProviderGlobalParameters>"]
    lines += ["        <ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>"
              .format(key, value) for key, value in params.items()]
    lines += ["      </ProviderGlobalParameters>", "    </GlobalConfig>",
              '    <Sessions><Session id="session1"><MatchParameters>'
              '<Score idLocalTeam="{}" idVisitingTeam="{}"><LocalTeamScore>0</LocalTeamScore>'
              "<VisitingTeamScore>0</VisitingTeamScore></Score></MatchParameters>"
              "<FieldSize><Width>105</Width><Height>68</Height></FieldSize>"
              "</Session></Sessions>".format(*team_ids),
              "    <Teams>"]
    for team_id, name, attack in zip(team_ids, (HOME_NAME, AWAY_NAME),
                                     ("left_to_right", "right_to_left")):
        lines.append('      <Team id="{}"><Name>{}</Name><ProviderTeamsParameters><ProviderParameter>'
                     "<Name>attack_direction_first_half</Name><Value>{}</Value>"
                     "</ProviderParameter></ProviderTeamsParameters></Team>"
                     .format(team_id, name, attack))
    lines += ["    </Teams>", "    <Players>"]
    lines += ['      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
              "<ShirtNumber>{0}</ShirtNumber></Player>".format(s + 1, team_ids[slot_team[s]])
              for s in range(k)]
    lines += ["    </Players>", '    <Devices><Device id="device1"><Sensors>',
              '      <Sensor id="position"><Name>Position</Name><Channels>'
              '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
              '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>'
              "</Channels></Sensor>",
              '      <Sensor id="speed"><Name>Speed</Name><Channels>'
              '<Channel id="s"><Name>speed</Name><Unit>meters/second</Unit></Channel>'
              "</Channels></Sensor>",
              "    </Sensors></Device></Devices>", "    <PlayerChannels>"]
    for s in range(k):
        for channel in ("x", "y", "s"):
            lines.append('      <PlayerChannel channelId="{0}" id="player{1}_{0}" playerId="P{1}"/>'
                         .format(channel, s + 1))
    lines += ["    </PlayerChannels>", "  </Metadata>", "  <DataFormatSpecifications>",
              '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">'.format(n),
              '      <StringRegister name="frameCount"/>', '      <SplitRegister separator=";">']
    for s in range(k):
        lines.append('        <SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(s + 1, c)
            for c in ("x", "y", "s")) + "</SplitRegister>")
    lines += ["      </SplitRegister>",
              '      <SplitRegister separator=","><BallChannelRef channelId="x"/>'
              '<BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>',
              "    </DataFormatSpecification>", "  </DataFormatSpecifications>", "</main>", ""]
    return "\n".join(lines)


def _raw_text(slots, ball):
    n, k, _ = slots.shape
    table = np.column_stack([np.arange(1, n + 1), slots.reshape(n, -1), ball])
    fmt = "%d:" + ";".join(["%.5f,%.5f,%.2f"] * k) + ":%.5f,%.5f,%.2f"
    buffer = io.StringIO()
    np.savetxt(buffer, table, fmt=fmt)
    return buffer.getvalue().replace("nan", "NaN")


def _smart_tagging_xml(rng, spec, segments):
    fps = spec.fps
    tags = []  # (start_frame, end_frame, code, {group: text}), kickoffs are never dropped

    def add(start, end, code, keep=False, **labels):
        tags.append((max(start, 0), min(end, spec.n_frames - 1), code, labels, keep))

    for (half, _, _), kickoff in zip(spec.halves, spec.kickoffs):
        team = H if half == 1 else A
        add(kickoff - 2 * fps, kickoff + 3 * fps, "SET PIECES", keep=True, Team=TAG_TEAM[team],
            Half=HALF_TAG[half], Type="Kick Off (Start)")
    for i, (half, start, end, code) in enumerate(segments):
        if code == D:
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if nxt and nxt[3] != D and nxt[0] == half:
                kind = rng.choice(["Throw-in", "Free Kick", "Corner Kick", "Goal Kick"])
                team = "N/A" if kind == "Throw-in" else TAG_TEAM[nxt[3]]
                add(end - 3 * fps, end + 2 * fps, "SET PIECES", Team=team, Half=HALF_TAG[half],
                    Type=str(kind))
            continue
        if end - start > 5 * fps:
            middle = (start + end) // 2
            add(start + fps, middle, str(rng.choice(["BUILD UP", "PROGRESSION"])),
                Team=TAG_TEAM[code], Half=HALF_TAG[half])
            add(middle, end - fps, str(rng.choice(["BALL IN FINAL THIRD", "CREATING CHANCES"])),
                Team=TAG_TEAM[code], Half=HALF_TAG[half])
            if end - start > 10 * fps:  # the defending side gets its own phase tag
                add(start + 2 * fps, end - 2 * fps,
                    str(rng.choice(["DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD"])),
                    Team=TAG_TEAM[A if code == H else H], Half=HALF_TAG[half])

    if spec.n_tags is not None and len(tags) > spec.n_tags:
        keep = [t for t in tags if t[4]]
        rest = [t for t in tags if not t[4]]
        chosen = rng.choice(len(rest), size=spec.n_tags - len(keep), replace=False)
        tags = keep + [rest[i] for i in sorted(chosen)]
    tags.sort(key=lambda t: t[0])

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for i, (start, end, code, labels, _) in enumerate(tags):
        lines.append("    <instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                     .format(i + 1, start / fps, end / fps, code)
                     + "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                               for g, t in labels.items())
                     + "</instance>")
    lines += ["  </ALL_INSTANCES>", "</file>", ""]
    return "\n".join(lines)


def _homography_csv(rng, n, valid=None):
    matrices = HOMOGRAPHY * (1 + rng.normal(0, 1e-4, (n, 9)))
    error = np.abs(rng.normal(0.24, 0.05, n))
    if valid is not None:  # the broadcast file: NaN matrices where no pitch transform exists
        matrices[~valid] = np.nan
        error[~valid] = np.nan
    table = np.column_stack([np.arange(n), matrices, error,
                             np.ones(n) if valid is None else valid.astype(float)])
    buffer = io.StringIO()
    header = "frame," + ",".join("h{}{}".format(r, c) for r in range(3) for c in range(3))
    np.savetxt(buffer, table, fmt=["%d"] + ["%.7g"] * 10 + ["%d"], delimiter=",",
               header=header + ",reprojection_error,valid", comments="")
    return buffer.getvalue().replace("nan", "NaN")


def broadcast_recording(spec: Spec, sim: dict):
    """(players, ball) normalised, (n_broadcast, 22, 2) and (n_broadcast, 2), plus slot_team: a TV
    recording of the simulated match on its own timeline (src/ingest/degrade.py: second_recording,
    the same simulation V4 scores on), cut or padded with empty frames to spec.n_broadcast."""
    import importlib

    from src.schema import Match

    # imported by name: the synthetic *source* (src/ingest/synthetic.py) writes bundles with
    # broadcast=False, so its cached matches must not depend on degrade.py (evaluate.loader_files)
    degrade = importlib.import_module("src.ingest.degrade")

    period = np.zeros(spec.n_frames, dtype=np.int8)
    for half, first, last in spec.halves:
        period[first:last + 1] = half
    clean = Match(match_id=GAME + ":synthetic", fps=spec.fps, pitch=spec.pitch, period=period,
                  ball=sim["ball_m"], players=sim["pos"], slot_team=np.asarray(sim["team_of"], np.int8),
                  labels=sim["labels"]).validate()
    other, _, _ = degrade.second_recording(clean, fps=spec.fps, seed=spec.seed)
    m, scale = spec.n_broadcast, np.array(spec.pitch, dtype=np.float64)
    players = np.full((m, other.players.shape[1], 2), np.nan)
    ball = np.full((m, 2), np.nan)
    keep = min(m, other.n_frames)
    players[:keep], ball[:keep] = other.players[:keep] / scale, other.ball[:keep] / scale
    return players, ball, other.slot_team


def _write_broadcast(out_dir: str, spec: Spec, sim: dict) -> dict:
    """broadcast/ ATD xml+txt and homography (valid where the recording has tracking)."""
    players, ball, slot_team = broadcast_recording(spec, sim)
    m = len(ball)
    paths = games.asset_paths(GAME, data_dir=out_dir, source="broadcast")
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    meta = Spec(n_frames=m, halves=spec.halves, kickoffs=spec.kickoffs, n_slots=len(slot_team),
                n_tags=None, n_broadcast=m, seed=spec.seed, fps=spec.fps, pitch=spec.pitch)
    with open(paths["atd_meta"], "w") as handle:
        handle.write(_metadata_xml(meta, slot_team))
    slots = np.concatenate([players, np.zeros(players.shape[:2] + (1,))], axis=2)
    slots[np.isnan(players[..., 0])] = np.nan
    ball3 = np.column_stack([ball, np.where(np.isnan(ball[:, 0]), np.nan, 0.0)])
    with open(paths["atd_raw"], "w", newline="\n") as handle:
        handle.write(_raw_text(slots, ball3))
    tracked = ~np.isnan(players[..., 0]).all(axis=1) | ~np.isnan(ball[:, 0])
    rng = np.random.default_rng(spec.seed + 7)
    with open(paths["homography"], "w", newline="\n") as handle:
        handle.write(_homography_csv(rng, m, valid=tracked))
    return paths


def write_bundle(out_dir: str, spec: Spec | None = None, broadcast: bool = True) -> dict:
    """Write tactical/ (ATD xml+txt, homography, Smart Tagging), broadcast/ (ATD xml+txt and
    homography of a simulated TV recording, V4; broadcast=False skips it) and synthetic_gt.csv."""
    spec = spec or Spec()
    sim = simulate(spec)
    rng = np.random.default_rng(spec.seed + 1)
    paths = games.asset_paths(GAME, data_dir=out_dir, source="tactical")
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)

    with open(paths["atd_meta"], "w") as handle:
        handle.write(_metadata_xml(spec, sim["slot_team"]))
    with open(paths["atd_raw"], "w", newline="\n") as handle:
        handle.write(_raw_text(sim["slots"], sim["ball"]))
    with open(paths["homography"], "w", newline="\n") as handle:
        handle.write(_homography_csv(rng, spec.n_frames))
    with open(paths["smart_tagging"], "w") as handle:
        handle.write(_smart_tagging_xml(rng, spec, sim["segments"]))
    if broadcast:
        _write_broadcast(out_dir, spec, sim)
    gt_path = submission.write(os.path.join(out_dir, "synthetic_gt.csv"), sim["labels"])
    with open(os.path.join(out_dir, "SYNTHETIC.txt"), "w") as handle:
        handle.write("Synthetic BAR-ATH bundle from src/synthetic.py (seed {}). Not match data.\n"
                     .format(spec.seed))
    return {"paths": paths, "gt": gt_path, "spec": spec}
