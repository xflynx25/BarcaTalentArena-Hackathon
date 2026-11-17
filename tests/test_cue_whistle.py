"""V3b's cue `whistle` (src/models/cues/whistle.py); synthetic only."""

import dataclasses
import shutil

import numpy as np
import pytest

from src.models.cues import whistle
from src.models.learned import LearnedDecoder
from src.video import stack


def _feats(n_frames, fps, whistle_s=(), hz=3600.0, seed=0):
    """Stack features of a quiet noise track with a 0.6 s pure tone at each time in whistle_s."""
    rng = np.random.default_rng(seed)
    n = int(n_frames / fps * stack.SR)
    pcm = rng.normal(0, 0.003, n)
    for t0 in whistle_s:
        a, b = int(t0 * stack.SR), int((t0 + 0.6) * stack.SR)
        tt = np.arange(b - a) / stack.SR
        pcm[a:b] += 0.2 * np.sin(2 * np.pi * hz * tt)
    return stack.audio_features(pcm, n_frames, fps)


@pytest.fixture(autouse=True)
def _clean_registry():
    whistle.AUDIO.clear()
    yield
    whistle.AUDIO.clear()


def test_detections_and_pitch(small):
    match = small["match"]
    live = np.flatnonzero(match.period > 0)
    times = [live[k] / match.fps for k in range(250, len(live) - 100, 1500)]
    feats = _feats(match.n_frames, match.fps, times)
    ev, why = whistle.detections(match, feats, 18.0, 3, "fixed")
    assert why == "" and len(ev) == len(times)
    assert np.all(np.abs(ev - np.round(np.array(times) * match.fps)) <= 15)
    assert abs(whistle.pitch(feats, match.fps) - 3600.0) <= 160.0
    # a referee at 4.2 kHz: the fixed band misses him (rate 0 -> off), band=auto finds him
    high = _feats(match.n_frames, match.fps, times, hz=4200.0)
    assert whistle.detections(match, high, 18.0, 3, "fixed")[1].startswith("off")
    ev, why = whistle.detections(match, high, 18.0, 3, "auto")
    assert why == "" and len(ev) == len(times)


def test_off_cases_give_the_base_bytes(small):
    match = small["match"]
    dec = LearnedDecoder()
    base = LearnedDecoder().predict(match)
    assert whistle.effect(match, match, dec, 1.0) is None  # no audio known for this match
    whistle.AUDIO[match.match_id] = stack.empty_features(match.n_frames)  # silence
    assert whistle.effect(match, match, dec, 1.0) is None
    whistle.AUDIO[match.match_id] = _feats(match.n_frames, match.fps, [])  # no whistle at all
    assert whistle.effect(match, match, dec, 1.0) is None
    whistle.AUDIO[match.match_id] = _feats(match.n_frames - 5, match.fps, [20.0])  # wrong length
    assert whistle.effect(match, match, dec, 1.0) is None
    assert np.array_equal(LearnedDecoder(cues="whistle:1").predict(match), base)
    with pytest.raises(ValueError, match="unknown options"):
        whistle.effect(match, match, dec, 1.0, win=2)


def test_effect_window_and_decoder(small):
    match = small["match"]
    live = np.flatnonzero(match.period > 0)
    t0 = (live[0] + 500) / match.fps
    whistle.AUDIO[match.match_id] = _feats(match.n_frames, match.fps, [t0])
    dec = LearnedDecoder()
    assert whistle.effect(match, match, dec, 0.0) is None
    eff = whistle.effect(match, match, dec, 2.0, pre=1, post=2)
    cheap = np.flatnonzero(eff.hd < 1)
    assert len(cheap) == 3 * match.fps + 1 and cheap[-1] - cheap[0] == 3 * match.fps  # [w - 1 s, w + 2 s]
    assert abs(cheap[0] + match.fps - round(t0 * match.fps)) <= 15
    assert np.all(eff.hd[cheap] >= np.exp(-2.0) - 1e-12) and np.array_equal(eff.hd, eff.ad)
    assert eff.dh is None and eff.ha is None
    a = LearnedDecoder(cues="whistle:2").predict(match)
    b = LearnedDecoder(cues="whistle:2").predict(match)
    assert np.array_equal(a, b) and len(a) == match.n_frames
    assert np.array_equal(LearnedDecoder(cues="whistle:0").predict(match), LearnedDecoder().predict(match))


def test_braking_makes_it_stronger(small):
    match = small["match"]
    ev = np.array([np.flatnonzero(match.period > 0)[0] + 400])
    alone = whistle.strength(match, ev, 1.0, 2.0, 1.0, 0.5)
    assert set(np.unique(alone)) <= {0.0, 0.5, 1.0} and alone.max() > 0
    stopped = match.players.copy()
    w = int(ev[0])
    stopped[w:] = stopped[w]  # everyone stands still from the whistle on
    moving = match.players.copy()
    step = np.zeros_like(moving)
    step[..., 0] = 0.3  # 7.5 m/s before the whistle
    moving[:w] = stopped[w] - step[:w] * (w - np.arange(w))[:, None, None]
    braked = dataclasses.replace(match, players=moving)
    assert whistle.strength(braked, ev, 1.0, 2.0, 1.0, 0.5).max() == 1.0


def test_attach_run_without_video_is_off(small, tmp_path):
    match = small["match"]
    assert whistle.attach_run(LearnedDecoder(), match, small["dir"]) is None  # cue not used
    note = whistle.attach_run(LearnedDecoder(cues="whistle:1"), match, str(tmp_path))
    assert note.startswith("off") and match.match_id not in whistle.AUDIO


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_attach_run_reads_a_silent_video(small, tmp_path):
    import os
    import subprocess

    from src.official import games
    match = small["match"]
    video = games.asset_paths(games.GAME, str(tmp_path), "tactical")["video"]
    os.makedirs(os.path.dirname(video), exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=mono", "-f", "lavfi", "-i", "color=c=black:s=32x32:r=25",
                    "-t", "2", "-shortest", "-c:a", "aac", video], check=True)
    note = whistle.attach_run(LearnedDecoder(cues="whistle:1"), match, str(tmp_path))
    assert note.startswith("off") and match.match_id not in whistle.AUDIO


def test_missing_soccertrack_cache_is_reported_once(monkeypatch, capsys):
    """W5 close (F337): a SoccerTrack match without the audio cache gives no effect, and says so."""
    from types import SimpleNamespace

    from src.models.cues import whistle
    from src.video import soccertrack_audio

    monkeypatch.setattr(soccertrack_audio, "path", lambda key, half: "/nonexistent/{}_{}.npz".format(key, half))
    monkeypatch.setattr(whistle, "_WARNED", set())
    m = SimpleNamespace(match_id="soccertrack:999")
    assert whistle.audio(m) is None and whistle.audio(m) is None
    err = capsys.readouterr().err
    assert err.count("no SoccerTrack audio cache for 999") == 1


def test_attach_run_stops_without_ffmpeg(small, tmp_path, monkeypatch):
    """W5 close: with a video but no ffmpeg the run stops instead of silently making other bytes."""
    import os

    from src.official import games
    match = small["match"]
    video = games.asset_paths(games.GAME, str(tmp_path), "tactical")["video"]
    os.makedirs(os.path.dirname(video), exist_ok=True)
    open(video, "wb").close()
    monkeypatch.setattr(whistle.shutil, "which", lambda tool: None)
    with pytest.raises(SystemExit, match="needs ffmpeg"):
        whistle.attach_run(LearnedDecoder(cues="whistle:1"), match, str(tmp_path))
