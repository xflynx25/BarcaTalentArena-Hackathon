"""Cue `whistle` (wave 5, V3b): the referee's whistle, heard and seen together.

Why (owner's error review, 14 Nov: "revisit the whistle"; findings F139-F141, F328-F331): in
SoccerTrack's audio the whistle is a narrow lasting line at ~3.6 kHz, heard at 20-30% of fouls,
~1.2 s after the foul. V3 snapped v4's decoded dead-ball edges onto it (a hard rule) and tied. Here
it is a soft decode-time cost instead: for a short window around each detection the team -> D
transitions get cheaper, and more so when the players brake within BRAKE_S of it (B21's measure,
src/models/onset.py: whistle_events), so a lone false alarm does little.

    s(t)       = 1 where a detection at w has t in [w - pre, w + post] and a braking event within
                 +-brake s of w; `alone` (default 0.5) where only the sound is there; 0 elsewhere
                 (overlapping windows take the larger)
    mult_{H->D} = mult_{A->D} = exp(-gain * s(t)), clipped to >= MIN_MULT

Detections (src/video/whistle.py, label-free, one match's audio only): frames whose lasting line
(src/video/stack.py: line_db) is >= thr dB inside the band with at most `others` other lasting
lines, the highest within +-1 s. Band: BAND_HZ (F139/F140's, picked on SoccerTrack train), or with
`band=auto` the match's own pitch (pitch() below: the centre of the BAND_HALF_HZ-wide band holding
the most short isolated lines in SEARCH_HZ). F328: the label-free pitch locks onto a ~4.1-4.2 kHz
line in most matches, not the whistle, so the default is the fixed band.

Audio, per match (audio()): AUDIO[match_id] if set (src.run fills it for BAR-ATH from the local
tactical video, attach_run()), else SoccerTrack's cached features (src/video/soccertrack_audio.py)
for "soccertrack:<id>", else none. **Off (gain 0, no effect, the base model's bytes)** when there is
no audio, the track is digital silence, the feature rows don't match the match's frames, or the
detection rate per in-play minute is outside RATE_RANGE (SoccerTrack's 8 matches with sound span
0.08-0.59 at the defaults; the range is half the lowest to twice the highest). Every other source
has no audio, so the cue is off there.

Options (`cues=whistle:1/pre=1.5/post=2`): thr, others, band ("fixed" | "auto"), pre, post, brake,
alone (see DEFAULTS). Gain 0 = no effect.

Runtime: numpy over the detections (a few hundred per match); BAR-ATH's audio decode in attach_run
takes ~1-2 min (ffmpeg, local file). Deterministic: the features are rounded to 3 decimals (F139),
detections are frame indices, the decoder puts costs on its grid.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

from src.models.cues import Effect

BAND_HZ = (3450.0, 3750.0)
SEARCH_HZ = (2800.0, 4600.0)
BAND_HALF_HZ = 150.0
RATE_RANGE = (0.04, 1.2)  # detections per in-play minute at the defaults (F329)
MIN_MULT = 0.02
DEFAULTS = {"thr": 18.0, "others": 3, "band": "fixed", "pre": 1.5, "post": 2.0, "brake": 1.0,
            "alone": 0.5}

AUDIO: dict = {}  # match_id -> stack features (src.run, tests)
_WARNED: set = set()  # SoccerTrack matches whose missing audio cache was reported (once per process)


def audio(match) -> dict | None:
    """The match's per-frame audio features, or None (no audio known for it)."""
    feats = AUDIO.get(match.match_id)
    if feats is not None:
        return feats
    mid = str(match.match_id)
    if mid.startswith("soccertrack:"):
        from src.video import soccertrack_audio
        key = mid.split(":", 1)[1]
        if all(os.path.isfile(soccertrack_audio.path(key, h)) for h in ("1st", "2nd")):
            return soccertrack_audio.load(key)
        if key not in _WARNED:  # W5 close (F337): without the cache the cue silently gives v6's bytes
            _WARNED.add(key)
            print("whistle: no SoccerTrack audio cache for {} (run `python -m src.video.soccertrack_audio "
                  "fetch`); the cue gives no effect there".format(key), file=sys.stderr)
    return None


def pitch(feats: dict, fps: int, thr: float = 15.0, others: int = 3) -> float:
    """Label-free whistle pitch (Hz): among the lasting lines in SEARCH_HZ (>= thr dB, <= others
    other lines, peaks 1 s apart), keep the short isolated ones (0.25-1.5 s long, the same pitch
    +-47 Hz on <= 5% of the frames 0.5-3 s before and after), then the centre of the
    2 * BAND_HALF_HZ band holding most of them (the lowest on ties), refined to their median
    inside it. BAND_HZ's centre if none."""
    from src.video import whistle as vw
    line_hz, line_db = feats["line_hz"], feats["line_db"]
    ok = (feats["has_audio"] > 0) & (feats["others"] <= others) \
        & (line_hz >= SEARCH_HZ[0]) & (line_hz <= SEARCH_HZ[1])
    ev = vw.peaks(np.where(ok, line_db, 0.0), thr, fps)
    n, keep = len(line_hz), []
    near_lo, near_hi = int(round(0.5 * fps)), int(round(3 * fps))
    for e in ev:
        hz = line_hz[e]
        same = (np.abs(line_hz - hz) <= 47.0)
        run = same & (line_db >= line_db[e] - 8.0)
        a, b = e, e
        while a > 0 and run[a - 1]:
            a -= 1
        while b < n - 1 and run[b + 1]:
            b += 1
        length = (b - a + 4) / fps  # + the line's LINE_T - 1 frames
        if not 0.25 <= length <= 1.5:
            continue
        ctx = np.concatenate([same[max(0, a - near_hi):max(0, a - near_lo)] & (line_db[max(0, a - near_hi):max(0, a - near_lo)] >= 10),
                              same[b + near_lo:b + near_hi] & (line_db[b + near_lo:b + near_hi] >= 10)])
        if len(ctx) and ctx.mean() > 0.05:
            continue
        keep.append(hz)
    if not keep:
        return float(np.mean(BAND_HZ))
    step = 15.625
    bins = np.arange(SEARCH_HZ[0], SEARCH_HZ[1] + step, step)
    hist, _ = np.histogram(np.array(keep), bins)
    width = int(round(2 * BAND_HALF_HZ / step)) + 1
    smooth = np.convolve(hist, np.ones(width), "same")
    c = bins[int(np.argmax(smooth))] + step / 2
    keep = np.array(keep)
    return float(np.round(np.median(keep[np.abs(keep - c) <= BAND_HALF_HZ]), 3))


def detections(match, feats: dict, thr: float, others: int, band: str) -> tuple[np.ndarray, str]:
    """(whistle frames inside play, sorted; "" or why the cue is off)."""
    from src.video import whistle as vw
    if len(feats["has_audio"]) != match.n_frames:
        return np.zeros(0, dtype=np.int64), "off: audio rows != match frames"
    if not np.any(feats["has_audio"] > 0):
        return np.zeros(0, dtype=np.int64), "off: no audio track or digital silence"
    if band == "auto":
        c = pitch(feats, match.fps)
        band_hz = (c - BAND_HALF_HZ, c + BAND_HALF_HZ)
    elif band == "fixed":
        band_hz = BAND_HZ
    else:
        raise ValueError("cue whistle: band must be fixed or auto, not {!r}".format(band))
    ev = vw.events(feats, match.fps, thr_db=float(thr), max_others=int(others), band_hz=band_hz)
    live = match.period > 0
    ev = ev[live[ev]] if len(ev) else ev
    minutes = float(live.sum()) / match.fps / 60.0
    rate = len(ev) / minutes if minutes > 0 else 0.0
    if not RATE_RANGE[0] <= rate <= RATE_RANGE[1]:
        return np.zeros(0, dtype=np.int64), "off: detection rate outside SoccerTrack's range"
    return ev.astype(np.int64), ""


def strength(match, ev: np.ndarray, pre: float, post: float, brake: float, alone: float) -> np.ndarray:
    """(n,) s(t) of the module docstring."""
    from src.models import onset
    n, fps = match.n_frames, match.fps
    s = np.zeros(n)
    if len(ev) == 0:
        return s
    braked = np.flatnonzero(onset.whistle_events(match))
    k = int(round(brake * fps))
    for w in ev:
        hit = len(braked) and np.any(np.abs(braked - w) <= k)
        value = 1.0 if hit else float(alone)
        lo, hi = max(0, w - int(round(pre * fps))), min(n, w + int(round(post * fps)) + 1)
        np.maximum(s[lo:hi], value, out=s[lo:hi])
    return s


def effect(match, original, decoder, gain, **opts):
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise ValueError("cue whistle: unknown options {}".format(sorted(unknown)))
    o = dict(DEFAULTS, **opts)
    if gain == 0:
        return None
    feats = audio(original) or audio(match)
    if feats is None:
        return None
    ev, why = detections(original, feats, o["thr"], o["others"], o["band"])
    if why:
        return None
    s = strength(match, ev, o["pre"], o["post"], o["brake"], o["alone"])
    if not np.any(s > 0):
        return None
    return Effect.dead(mult=np.maximum(np.exp(-float(gain) * s), MIN_MULT))


# -- BAR-ATH at run time (src.run) ------------------------------------------------------------------

def uses_whistle(model) -> bool:
    from src.models import cues
    spec = getattr(model, "cues", "") or ""
    return any(name == "whistle" and gain != 0 for name, gain, _ in cues.parse(spec))


TOOLS_ERROR = ("whistle: {}. This preset reads the tactical video's sound, so it needs ffmpeg and "
               "ffprobe on the PATH (macOS: `brew install ffmpeg`); without them the CSV would differ from "
               "one made on a machine that has them. Install them, or use a preset without the whistle "
               "(v6f is v7 without it). W5 close, F338.")


def needs_tools() -> None:
    """Stop when ffmpeg / ffprobe are missing: whether the cue acts must depend on the match (its sound),
    never on the machine, or a re-run elsewhere would not regenerate the same file (W5 close)."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise SystemExit(TOOLS_ERROR.format("{} not found".format(" and ".join(missing))))


def attach_run(model, match, data_dir: str | None = None) -> str | None:
    """src.run: when the model's cues include `whistle`, read the tactical video's audio (ffmpeg, as
    src/video/ does), put it in AUDIO for this match and print whether the cue is on. None when the
    model doesn't use the cue. Any problem (no video file, no ffmpeg, no audio stream, digital
    silence, a detection rate outside SoccerTrack's range) -> off with a note: the base model's bytes."""
    if not uses_whistle(model):
        return None
    from src.official import games
    from src.video import stack
    note = None
    try:
        video = games.asset_paths(games.GAME, data_dir, "tactical")["video"]
        if not os.path.isfile(video):
            note = "off: no tactical video file"
        else:
            needs_tools()
            try:
                feats = stack.frame_features(video, match.n_frames, match.fps)
            except (subprocess.CalledProcessError, OSError) as error:
                raise SystemExit(TOOLS_ERROR.format("ffmpeg failed on the video ({})".format(
                    type(error).__name__))) from None
            opts = {}
            from src.models import cues
            for name, _, kw in cues.parse(getattr(model, "cues", "")):
                if name == "whistle":
                    opts = dict(DEFAULTS, **kw)
            _, why = detections(match, feats, opts["thr"], opts["others"], opts["band"])
            if why:
                note = why
            else:
                AUDIO[match.match_id] = feats
                note = "on: whistle detections from the tactical video's audio"
    except SystemExit:
        raise  # the machine can't read the sound: stop (another machine would make other bytes)
    except Exception as error:  # noqa: BLE001  the sound itself is unusable: v6f's bytes
        note = "off: audio not read ({})".format(type(error).__name__)
    print("whistle    cue {}".format(note), file=sys.stderr)
    return note
