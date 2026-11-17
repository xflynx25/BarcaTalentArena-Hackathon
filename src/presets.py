"""Named model presets: the candidates we could actually submit, each with a plain-language note.

A *model* in src/models/ (nearest, deadball, viterbi, spells, learned) is a building block with many
settings. A *preset* is one fixed choice of block + settings that we treat as a candidate file, e.g.
"v4" = `learned` with the v4 coefficient file. Every CLI that runs a model takes `--preset <name>`:

    .venv/bin/python -m src.run --preset v3 --out submissions/v3.csv
    .venv/bin/python -m src.determinism --preset v3 --expect submissions/v3.csv
    .venv/bin/python -m src.evaluate --preset v3 --matches skillcorner:all

`--param key=value` still works on top of a preset and overrides its settings. The easy front end
is `scripts/models.py` (list / submit / compare). Adding a candidate = one entry in PRESETS below
(plus a line in docs/models.md). Old learned versions keep their own frozen coefficient file, so
an old preset keeps producing the same bytes after the default is refit.
"""

from __future__ import annotations

import os

from src.official import ROOT

DEFAULT = "v6f"  # what `src.run` makes when no --model/--preset is given (submission, 17 Nov)

# name -> model, params, eval (extra params for honest external scoring), load (optional: extra load
# options for external scoring, e.g. V4's simulated second recording), synthetic hash, notes.
# Keep "what" to one line a person can read; "when" says when you'd pick it.
PRESETS: dict[str, dict] = {
    "v7": {
        "model": "bigger",
        # W5 close (F334-F337): the wave-5 stack on v6's files, decode-time only. B39's ball-flight cue
        # (gated off on a smoothed or drawn ball: SkillCorner, SoccerTrack) + V3b's whistle (read from
        # the tactical video's sound at run time; no usable sound -> off, v6f's bytes). B35b's restart
        # windows were STACK-ELIGIBLE but C14's stricter bar dropped them (3 of 5 selections, not
        # "better" alone). Passed C14 (trusted) and C12's rule against v6 on both folds.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3+whistle:3"},
        "eval": {"fold": "lomo"},
        "shape_check": True,  # step e (team shape needs the players, owner only): LOW -> v3
        "tags_check": True,  # step g: v6's restart rule reads Smart Tagging; FAIL -> fallback
        "fallback": "v7c",  # v7 without the restart rule (the only tag-reading part)
        # = v6f's: the synthetic bundle has no video, so the whistle is off
        "synthetic_sha256": "94d462351b25678ff2d767bf74f852af3b413d860d33b5d1fd0d340f60561ce7",
        "date": "2026-11-14",
        "what": "Default since the W5 close. v6 + two decode-time cues: a ball that only flies past a "
                "player is no touch (B39), and going dead is cheaper right after the referee's "
                "whistle, heard on the video (V3b).",
        "when": "v6f plus the whistle. Not the submission: the whistle depends on the machine's ffmpeg.",
    },
    "v7a": {
        "model": "bigger",
        # 17 Nov, anchor cue (src/models/cues/anchor.py, F345-F348): v7 + evidence for the team of two
        # same-team clear moments (ball within 1.5 m of one team, 2 m clear, 0.3 s) on the frames
        # between them, gap < 8 s, no break. DROP: lomo grid points won 2/15, IDSSE -0.0015 (F347).
        # Pinned for comparison only, at the gentlest gain tried.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3+whistle:3+anchor:0.25/gap=8"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v7c",
        "synthetic_sha256": "ed2f8f78d70c0cc0d51402fd497786a806e10981c8efb726fb2a6852223ea531",
        "date": "2026-11-17",
        "what": "v7 + the anchor cue: between two clear moments of the same team less than 8 s apart, "
                "the frames lean to that team (DROP, comparison only).",
        "when": "Never as the submission: it lost to v7 (F347). For comparison only.",
    },
    "v7c": {
        "model": "bigger",
        # v7 without the restart rule, i.e. without any part that reads Smart Tagging: what to upload
        # when step g says FAIL (as v5c was for v6). Same cues as v7.
        "params": {"track_repair": "1", "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3+whistle:3"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "e7c6b8ca06dc3084b11bc33c2d774f3130bfc6ca3ef47dcca66fcf8eff58f2c6",
        "date": "2026-11-14",
        "what": "v7 without the restart rule (whose windows come from Smart Tagging): the file to "
                "upload when step g says FAIL.",
        "when": "Only when `submit v7` step g says FAIL (step e LOW still means v3).",
    },
    "v7g": {
        "model": "bigger",
        # v7 + C15's goalkeeper repair (src/repairs/gkteam.py) at run time. Not the default: the W5
        # close's per-source reading of the repair rule fails it (F333); C15's own reading passes it.
        # Pinned so the owner can make and compare it (group "finalists"). On tracking with no flipped
        # keeper it is exactly v7.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3+whistle:3", "repairs": "gkteam"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v7c",  # step g FAIL: as v7 (the repair reads no tags; v7c + gkteam isn't pinned)
        "synthetic_sha256": "94d462351b25678ff2d767bf74f852af3b413d860d33b5d1fd0d340f60561ce7",  # = v7 (nothing to repair)
        "date": "2026-11-14",
        "what": "v7 + the goalkeeper repair: a keeper the tracker put in the other team (keepers wear "
                "their own kit) is handed back to his team (C15).",
        "when": "A finalist to compare with v7: pick it if you read C15's rule as 'better on all "
                "sources together' (decisions 14 Nov).",
    },
    "v7h": {
        "model": "bigger",
        # 16 Nov (F341-F342): v7 + cue `heading` at gain 0.25 (src/models/cues/heading.py): both H<->A
        # costs x1.25 while both teams keep a coherent, unchanged heading (players only; SkillCorner
        # estimates masked; never cheaper). STACK-ELIGIBLE by the session's rule at 0.25 / 0.5 / 1;
        # 0.25 is the gentlest. Pinned for comparison; DEFAULT unchanged.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3+whistle:3+heading:0.25"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v7c",  # step g FAIL: as v7 (the heading cue reads no tags; v7c + heading isn't pinned)
        "synthetic_sha256": "a60bacc8411d55d2ee8f6d46948506cfc24fd23a88b6e73929907f85a472853b",
        "date": "2026-11-16",
        "what": "v7 + a heading veto: a switch of team costs a little more while both teams keep "
                "running the same way (nobody has turned around, so it was probably no turnover).",
        "when": "A comparison only: STACK-ELIGIBLE but small (+0.0007 held out by match), not the "
                "default; a wave close decides.",
    },
    "v6": {
        "model": "bigger",
        # W4 close (F180-F185): the wave-4 stack, every STACK-ELIGIBLE part on v5's files (no B34
        # refit was eligible): B35's restart rule at gain 2 (windows from Smart Tagging's SET
        # PIECES; without the file it adds nothing) + B36's switch-timing cue at 0.5. Decode-time
        # only: v5's emission models are unchanged. Passed C14 (trusted) and C12's rule against v5.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5},
        "eval": {"fold": "lomo"},
        "shape_check": True,  # step e (team shape needs the players, owner only): LOW -> v3
        "tags_check": True,  # step g: the restart windows come from Smart Tagging; FAIL -> fallback
        "fallback": "v5c",  # v6 without the restart rule: what to upload when step g says FAIL
        "synthetic_sha256": "3b7ece5ce0003f6c543193f7bb65dc347774f15454fce6bd0714bc147b868edd",
        "date": "2026-11-13",
        "what": "Default from the W4 close to the W5 close. v5 + two decode-time parts: the few seconds "
                "before each restart kick count as dead (B35, derek-chui) and switches get cheaper where "
                "both teams' movement says a turnover is coming (B36).",
        "when": "Previous default; v7's base (step g FAIL with v6 -> v5c).",
    },
    "v6g": {
        "model": "bigger",
        # C15 (F316-F319): v6 + the run-time repair `gkteam` (src/repairs/gkteam.py): a keeper the
        # kit-colour team classification delivered as the other team is handed back to his team and
        # his ghost dropped. No refit, v6's files; on tracking with nothing to repair it returns the
        # match unchanged (v6's bytes, so the synthetic hash is v6's)
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "repairs": "gkteam"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v5c",  # step g FAIL: v6's fallback (the repair only matters on flipped keepers)
        "synthetic_sha256": "3b7ece5ce0003f6c543193f7bb65dc347774f15454fce6bd0714bc147b868edd",
        "date": "2026-11-14",
        "what": "v6 + a tracking repair: a goalkeeper the tracker put in the other team (keepers wear "
                "their own kit) is handed back, and the stray detection left in his place is dropped "
                "(C15).",
        "when": "Candidate for the W5 close; byte-identical to v6 on tracking without keeper flips.",
    },
    "v6h": {
        "model": "bigger",
        # C16 (wave 5, findings F320-F323): v6 + the track repair's hold. After C13 links what it
        # can, a track end nothing continues keeps moving with its team for up to 2 s, and a start
        # nothing continues is extended back 2 s, so a team doesn't lose a player the moment the
        # tracker does (off camera, lost in a crowd). Run-time only: v6's files are unchanged.
        # hold 2 s chosen on external data without the model (scripts/check_hold.py, F320).
        "params": {"track_repair": "1/hold=2", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v5c",
        "synthetic_sha256": "ef68d02f9c73fd5781302310a8fac301266a9be240a296936cdaa18912876dc3",
        "date": "2026-11-14",
        "what": "v6 + a 2 s hold: when the tracker loses a player and C13 can't link him back, he "
                "keeps moving with his team for 2 s instead of vanishing (C16).",
        "when": "Wave-5 candidate (C16); the W5 close decides whether it stacks.",
    },
    "v6s": {
        "model": "bigger",
        # W5 B37 (F300-F303): v6 + the stop cue at gain 0.5 (src/models/cues/stop.py, coefficients
        # src/models/cues/stop_coefs.json), go cue off: going dead gets cheaper where the players brake
        # and slow down. Decode-time only. DROP: ties v6 held out by match, loses by source (F300).
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "stop:0.5"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v6",
        "synthetic_sha256": "91303f5cfe27a93b2a44a94fbdd5a37bc8890d4141c1f5c3665075b145d0af09",
        "date": "2026-11-14",
        "what": "v6 + a stop cue: where the players brake and slow down (the one at the ball first), the "
                "decoder lets the ball go dead there more easily (B37).",
        "when": "Comparison only: DROP in wave 5 (ties v6 held out by match, loses by source, F300).",
    },
    "v6a": {
        "model": "bigger",
        # W5 B38 (F304-F307): v6 + the `turn` cue on top of v6's switch cue, decode-time only:
        # switches get cheaper where the players' forward acceleration, braking-while-still-going-
        # forward and the delayed turn say a turnover is coming (src/models/cues/turn.py, the "new"
        # set of src/models/cues/turn_coefs.json). Gain 0.25, the gentlest of the grid; DROP in
        # wave 5 (12/15 lomo points, 13 needed), pinned for comparison and the close
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "turn:0.25"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v5c",
        "synthetic_sha256": "9811e67778214c83f83928e97dec0d786f02d2125a9a1214da279c7a0e126281",
        "date": "2026-11-14",
        "what": "v6 + a second switch-timing cue: the team losing the ball slows while still running "
                "forward and turns back a little later; switches get cheaper where that shows (B38).",
        "when": "Comparison only: DROP in wave 5 (a tie with v6, 12 of 15 grid points, F305).",
    },
    "v6f": {
        "model": "bigger",
        # W5 B39 (F308-F311): v6 + the `flight` cue (src/models/cues/flight.py, hazard
        # flight_coefs_nodx.json), decode-time only: switches get dearer while the seen ball flies
        # smoothly and cheaper at a bend next to the other team. Off (v6's bytes on that match)
        # when the ball's moving jitter > 0.045 m or straight share > 0.3 (smoothed or drawn ball)
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "flight:0.5/fit=nodx/jitter=0.045/lin=0.3"},
        "eval": {"fold": "lomo"},
        "synthetic_sha256": "94d462351b25678ff2d767bf74f852af3b413d860d33b5d1fd0d340f60561ce7",
        "date": "2026-11-17",
        "what": "The submission. v7 without the whistle: a ball that only flies past a player is no "
                "touch, and the file does not depend on ffmpeg decoding the video's sound (B39).",
        "when": "The file to upload. Chosen because the whistle's gain is +0.0002 on one external "
                "source and a different ffmpeg can change the bytes.",
    },
    "v6t": {
        "model": "bigger",
        # B40 (W5, F324-F327): v6 + cue `tagteam` at gain 0.25 (src/models/cues/tagteam.py): inside
        # each Smart Tagging phase with a team, 0.25 evidence per second for that team (defending
        # codes for the other team), tapered 1 s at the edges. Decode-time only. External scoring
        # reads stand-ins at tagger noise 2 (the pick level); without tags it is v6's bytes. DROP
        # by B40's rule (Metrica held out by source -0.0024 at noise 2); pinned for the W5 close.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "tagteam:0.25"},
        "eval": {"fold": "lomo", "cues": "tagteam:0.25/src=standin/noise=2"},
        "shape_check": True,
        "tags_check": True,  # step g: the restart windows and the phases come from Smart Tagging
        "fallback": "v5c",  # without any tag-reading part, as v6's
        "synthetic_sha256": "87a7643f88ebcf1928390d0843fb1319b697c8c1cfcffad72343dadef882c1ee",
        "date": "2026-11-14",
        "what": "v6 + Smart Tagging's phases as a gentle team prior at decode time (B40): inside a "
                "tagged phase the tagged team gets a little extra evidence.",
        "when": "Not on its own: B40's rule says DROP (Metrica held out by source). For the W5 close.",
    },
    "v6w": {
        "model": "bigger",
        # W5 V3b (F328-F331): v6 + the `whistle` cue (src/models/cues/whistle.py), decode-time
        # only: team->D gets cheaper for -1.5..+2 s around a whistle heard in the tactical video's
        # sound (3.45-3.75 kHz line), more when the players brake within 1 s. src.run reads the
        # audio (ffmpeg); no video, no audio, silence or a detection rate outside SoccerTrack's
        # range -> off with a printed note (v6's bytes; the synthetic bundle has no video)
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "whistle:3"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,
        "fallback": "v5c",
        "synthetic_sha256": "3b7ece5ce0003f6c543193f7bb65dc347774f15454fce6bd0714bc147b868edd",
        "date": "2026-11-14",
        "what": "v6 + the referee's whistle from the video's sound: going dead is cheaper just "
                "around a whistle, more so when the players brake; off without sound (V3b).",
        "when": "Wave-5 candidate; the W5 close decides whether it joins the stack.",
    },
    "v5": {
        "model": "bigger",
        # B30's pick (F126): B31's emission models (src/models/boosted_coefs.json +
        # temporal_coefs.json, unchanged) with C13's track repair at run time
        "params": {"track_repair": "1"},
        "eval": {"fold": "lomo"},
        "shape_check": True,  # submit runs step e (team shape needs the players, owner only)
        "synthetic_sha256": "865ef84e2a2aa4c67ab1bad99c8acbf69ffaa7c2aa9d8194e3e02af57f529eec",
        "date": "2026-11-13",
        "what": "Default from B30 to the W4 close (13 Nov). v4's decoder fed by two bigger models over "
                "v4's clues (decision trees + a small network that looks +-4 s around each moment, "
                "B31), broken player tracks repaired first (C13).",
        "when": "Previous default; v6 is v5 plus two decode-time parts (F182).",
    },
    "v5.1a": {
        "model": "bigger",
        # B34a (F170-F171): v5's recipe refit with the local groups shapeo + deadm (F204's pair)
        "params": {"coef_file": "src/models/boosted_coefs_v51a.json+src/models/temporal_coefs_v51a.json",
                   "track_repair": "1"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "7fc7f53f30e8a6df3c2cbcd9619bad02ae7b0340477893e44330c4e7e68c8905",
        "date": "2026-11-13",
        "what": "v5 refit with two extra clues: team shape while the ball is loose, and whether the "
                "players are moving while the ball is lost (local L1/L3).",
        "when": "Comparison only: DROP in wave 4 (loses IDSSE and Metrica held out by source).",
    },
    "v5.1b": {
        "model": "bigger",
        # B34b (wave 4): v5's recipe with the local L3 group `deadm` added to both emission
        # models (`fit_emission.py --kind boosted|temporal --groups shapem,touch,deadm`, findings F172)
        "params": {"coef_file": "src/models/boosted_coefs_v51b.json+src/models/temporal_coefs_v51b.json",
                   "track_repair": "1"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "9edf679a3c4f17ee31d3bba6b3eae456cfede7d031cf4bdefa4a13582b590552",
        "date": "2026-11-13",
        "what": "v5 retrained with one more clue inside ball gaps: how fast the players move and "
                "whether they stand in a set-piece shape, as live-or-dead evidence (local L3, deadm).",
        "when": "Comparison only: DROP in wave 4 (loses IDSSE held out by source, F172).",
    },
    "v5.1c": {
        "model": "bigger",
        # B34c (wave 4, F173): v5's recipe refit with the `shapeo` group added (team shape on frames
        # where the ball is seen but nobody is within 3 m, F201); v5's run-time track repair
        "params": {"track_repair": "1",
                   "coef_file": "src/models/boosted_coefs_v5_1c.json+src/models/temporal_coefs_v5_1c.json"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "5fe095796a538309d638c4a06e776dbfef74a1bbea3d87be4244e6debc9d2688",
        "date": "2026-11-13",
        "what": "v5 + team shape when the ball is seen but free (`shapeo`), both bigger models refit (B34c).",
        "when": "Comparison only: DROP in wave 4 (loses IDSSE held out by source, F173).",
    },
    "v5.1d": {
        "model": "bigger",
        # B34d (wave 4, F174): v5's recipe refit with three more groups from the local L1/L3 work
        # (shapef, shapeo, deadm; F201, F202), scored as v5 with the track repair
        "params": {"coef_file": "src/models/boosted_coefs_v51d.json+src/models/temporal_coefs_v51d.json",
                   "track_repair": "1"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "d330e8fbe4c4c05424fc3d7d0ad872a60aab5846cb68f4d0be108b336b860f03",
        "date": "2026-11-13",
        "what": "v5 retrained with three more clues: team shape when the ball is loose or flying, "
                "and how players move while the ball is hidden (live or dead play).",
        "when": "Comparison only: DROP in wave 4 (loses IDSSE and Metrica held out by source, F174).",
    },
    "v5r": {
        "model": "bigger",
        # B35 (findings F175-F177): v5 + PR #51's restart rule (derek-chui), decode-time only: the
        # 6 s before a kick found in a restart window count as dead. BAR-ATH's windows are Smart
        # Tagging's SET PIECES (Match.tags); without the file the rule adds nothing (= v5).
        # Gain 2, not the clean-window best 4: 2 stays ahead of v5 with tagger-like window errors
        # (noise 2 and 3), 4 ties at noise 3 (F176).
        "params": {"track_repair": "1", "restart_rule_gain": 2.0},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "9a7be8f74be1b30b75f23e35f86b6a83732aeef04e0e13bb5fe5e44c676bce20",
        "date": "2026-11-13",
        "what": "v5 + the restart rule (derek-chui, PR #51): the few seconds before each restart kick "
                "count as dead; restarts from Smart Tagging's set pieces.",
        "when": "Comparison: v6 without the switch cue (F177, F182).",
    },
    "v6r": {
        "model": "bigger",
        # W5 B35b (F312-F315): v6 + cue `restartw` at gain 0.5 (src/models/cues/restartw.py):
        # restart windows found in the tracking (the ball rests at a restart spot, then is kicked)
        # add dead-ball evidence before the kick, unioned with v6's restart rule (no frame twice).
        # Decode-time only, reads no tags: v6's emission models and files are unchanged.
        "params": {"track_repair": "1", "restart_rule_gain": 2.0, "switch_cue_gain": 0.5,
                   "cues": "restartw:0.5"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "tags_check": True,  # v6's restart rule still reads Smart Tagging
        "fallback": "v5c",
        # = v6's hash: the synthetic bundle has no real restarts (the cue finds 2 windows there and
        # they change no frame), so the pin proves determinism, not the cue (tests/test_restartw.py)
        "synthetic_sha256": "3b7ece5ce0003f6c543193f7bb65dc347774f15454fce6bd0714bc147b868edd",
        "date": "2026-11-14",
        "what": "v6 + restarts found in the tracking: where the ball rests at a restart spot (goal "
                "area, corner, touchline, centre spot, a wall) and is then kicked, the seconds "
                "before the kick count as dead ball, also where Smart Tagging has no set piece (B35b).",
        "when": "Wave-5 candidate; the W5 close decides whether it joins the stack.",
    },
    "v5c": {
        "model": "bigger",
        # B36 (F178-F179): v5 + the L2 switch-timing cue at decode time (no refit): switches get
        # cheaper where both teams' shape changes say a turnover is coming. The cue's hazard is
        # src/models/switch_cue_coefs.json (fit on the five registered sources); gain 0.5 ungated
        # was the wave-4 STACK-ELIGIBLE pick (the coverage gate only switched SkillCorner off)
        "params": {"track_repair": "1", "switch_cue_gain": 0.5},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "fallback": "v5",
        "synthetic_sha256": "f76b3a9b2f4e35bc19ea0d204ba446a486bd3f34aab0d8846df700199ffab14a",
        "date": "2026-11-13",
        "what": "v5 + a switch-timing cue: when both teams' running and shape change the way they do "
                "at a turnover, the decoder lets possession switch there more easily (B36).",
        "when": "v6 without the restart rule: upload it when v6's step g (Smart Tagging) says FAIL "
                "(F179, F182).",
    },
    "v5b": {
        "model": "bigger",
        "params": {"track_repair": "1", "ball2_fill": True},
        "eval": {"fold": "lomo"},
        "load": {"broadcast": 1},  # external scoring: a simulated second recording, aligned (V4)
        "broadcast": True,  # src.run aligns BAR-ATH's broadcast tracking (src/broadcast.py)
        "shape_check": True,
        "fallback": "v5",  # what to upload when step f says STAY
        "synthetic_sha256": "921a974ab1a5dde3f602eb2657d79ba5ece47e119ccd3d7faf43f87eb369ba1a",
        "date": "2026-11-13",
        "what": "v5 + the broadcast ball: BAR-ATH's TV tracking lined up with the tactical timeline, "
                "its ball used where the tactical ball is missing (V4).",
        "when": "Comparison, not recommended: +0.005 overall on simulated TV recordings but worse on "
                "SoccerTrack (all 10) and, held out by source, SkillCorner (F126). Step f still runs.",
    },
    "v5l": {
        "model": "learned",
        # B30's learned refit: v4's sources and groups + the hidden-ball imputer, training and
        # run-time tracks repaired (findings F124)
        "params": {"coef_file": "src/models/learned_coefs_v5.json", "track_repair": "1"},
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "41fa98df3cccd2a7e768204deaa8a7147149416472f24aca2ec4c6a63d9cc8d0",
        "date": "2026-11-12",
        "what": "B30's refit of v4's small model: adds an estimate of where the hidden ball is (B29) "
                "and repairs broken player tracks first (C13).",
        "when": "Comparison: beats v4 by source (+0.0045) but loses SkillCorner 2019/20 (F124); v5 "
                "beats it on every source.",
    },
    "v5t": {
        "model": "learned",
        # v5l + the Smart Tagging group, trained on B32's tagger mix (mixB, F132); reads BAR-ATH's
        # tags file at run time, defending codes put in our reading label-free (tags_convention=auto)
        "params": {"coef_file": "src/models/learned_coefs_v5t.json", "track_repair": "1"},
        "eval": {"fold": "lomo", "tags_source": "none"},  # external matches carry no tags file
        "shape_check": True,
        "tags_check": True,
        "fallback": "v5",  # what to upload when step g says FAIL
        "synthetic_sha256": "7cc0b682d272fc0b8861bd2437e7c01f53a0463a6c6185badb346dcd75d57c4a",
        "date": "2026-11-12",
        "what": "v5l + Smart Tagging: also reads BAR-ATH's tags (B32's recipe), trained so it doesn't "
                "depend on them.",
        "when": "Comparison, not recommended: even with fairly clean tags it only ties v5 and loses "
                "the optical sources (F126). Step g checks the tags if you run it.",
    },
    "v4": {
        "model": "learned",
        "params": {},  # learned's defaults + src/models/learned_coefs.json
        "eval": {"fold": "lomo"},
        "shape_check": True,
        "synthetic_sha256": "4bb28afb85dcdc9f3839c36d30b74649f68c5c49d084fc4d14145c8feb78b244",
        "date": "2026-11-10",
        "what": "Previous default (until B30). Learned Viterbi decoder + team-shape features when "
                "the ball is missing.",
        "when": "Comparison; v5 beats it on every source (F126).",
    },
    "v4r": {
        "model": "learned",
        "params": {"track_repair": "1"},
        "eval": {"fold": "lomo"},
        "synthetic_sha256": "511d40f07a5e3e6651ce73f06603dd43dcf51001a6f52c6ea102462f866f8619",
        "date": "2026-11-12",
        "what": "v4 + track repair: broken player tracks are joined and their gaps filled before the model (C13).",
        "when": "Comparison: v5 folds the repair in; same as v4 where tracks are complete.",
    },
    "b31": {
        "model": "bigger",
        "params": {},  # src/models/boosted_coefs.json + temporal_coefs.json, gain 1.5, switch 0.5
        "eval": {"fold": "lomo"},
        "synthetic_sha256": "324aa8e152ba560aa9c02462a96a690dd5bd1e5423d9b236645e05d91faea8d3",
        "date": "2026-11-12",
        "what": "B31's model without the track repair: v4's decoder fed by boosted trees and a "
                "small temporal network over v4's features, averaged.",
        "when": "Comparison: v5 is this + track repair (same on full tracks, F126).",
    },
    "v4b": {
        "model": "learned",
        "params": {"ball2_fill": True},  # v4's coefficients; the ball's gaps filled from the broadcast
        "eval": {"fold": "lomo"},
        "load": {"broadcast": 1},  # external scoring: a simulated second recording, aligned (V4)
        "broadcast": True,  # src.run aligns BAR-ATH's broadcast tracking (src/broadcast.py)
        "shape_check": True,
        "fallback": "v4",  # what to upload when step f says STAY
        "synthetic_sha256": "6ef859ee1bd24b08ddcfa99ac77d669f5aa21b5acf6fb25f626c0ce1d67d3666",
        "date": "2026-11-12",
        "what": "v4 + the broadcast ball: BAR-ATH's TV tracking aligned with the tactical timeline "
                "(src/align.py), its ball used where the tactical ball is missing.",
        "when": "Comparison: v5b is the same fill on v5.",
    },
    "v3": {
        "model": "learned",
        "params": {"coef_file": "src/models/learned_coefs_v3.json"},
        "eval": {"fold": "lomo"},
        "synthetic_sha256": "1c3a9f85707d0bccfc99d055aae54fdef38627e46a92c4ecc551163defa85bd2",
        "date": "2026-11-04",
        "what": "Previous default. Same learned decoder, no team-shape features (ball/touch/speed only).",
        "when": "Safe fallback if our tracking lacks the players v5's (and v4's) team-shape features need.",
    },
    "viterbi": {
        "model": "viterbi",
        "params": {},
        "eval": {},
        "synthetic_sha256": None,
        "date": "2026-10-31",
        "what": "Hand-set rules (no training): nearest team within reach + calm-players dead-ball cue, "
                "smoothed by a Viterbi decoder.",
        "when": "Comparison: the best model with no learned part.",
    },
    "deadball": {
        "model": "deadball",
        "params": {"switch_delay_s": 1.0},
        "eval": {},
        "synthetic_sha256": None,
        "date": "2026-10-30",
        "what": "Official baseline + dead-ball detector (calm players / ball out) + short-spell "
                "filter + 1 s switch delay.",
        "when": "Comparison: the simplest improvement on the baseline.",
    },
    "baseline": {
        "model": "nearest",
        "params": {"window": 50},
        "eval": {},
        "synthetic_sha256": None,
        "date": "2026-10-29",
        "what": "The organisers' baseline (nearest player + 2 s majority vote), reproduced exactly.",
        "when": "Comparison / wiring check: `src.run --preset baseline --compare-official "
                "baseline_smooth50` should say IDENTICAL.",
    },
}


# Named groups of presets, newest first. `history` is the line of defaults from the organisers'
# baseline (v0) to today's: what the viewer stacks to show what each step changed.
GROUPS: dict[str, list[str]] = {
    "history": ["v6f", "v7", "v6", "v5", "v4", "v3", "viterbi", "deadball", "baseline"],
    # the submission candidates after the W5 close (owner, 14 Nov): `submit finalists` makes all four
    "finalists": ["v7", "v7g", "v6f", "v6"],
}


def expand(names) -> list[str]:
    """Preset names from CLI values: each may be a name, a group (GROUPS) or a comma list of
    either. Order kept, repeats dropped, unknown names refused."""
    out: list[str] = []
    for value in names or []:
        for name in (part.strip() for part in str(value).split(",")):
            if not name:
                continue
            for item in GROUPS.get(name, [name]):
                if item not in PRESETS:
                    raise SystemExit("unknown preset {!r}; have {} (groups: {})".format(
                        item, ", ".join(PRESETS), ", ".join(GROUPS)))
                if item not in out:
                    out.append(item)
    return out


def _resolve(params: dict) -> dict:
    out = dict(params)
    if "coef_file" in out:  # one file, or several joined by "+" (bigger's ensemble, B34a)
        out["coef_file"] = "+".join(f if os.path.isabs(f) else os.path.join(ROOT, f)
                                    for f in out["coef_file"].split("+"))
    return out


def model_and_params(name: str, for_eval: bool = False) -> tuple[str, dict]:
    if name not in PRESETS:
        raise SystemExit("unknown preset {!r}; have {}".format(name, ", ".join(PRESETS)))
    preset = PRESETS[name]
    params = dict(preset["params"], **(preset["eval"] if for_eval else {}))
    return preset["model"], _resolve(params)


def cli_args(name: str, for_eval: bool = False) -> list[str]:
    """The equivalent `--model ... --param ...` flags (what the preset expands to)."""
    model, params = model_and_params(name, for_eval)
    out = ["--model", model]
    for key, value in params.items():
        out += ["--param", "{}={}".format(key, value)]
    return out
