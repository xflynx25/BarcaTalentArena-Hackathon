"""Video on synthetic data only. V2: decode, calibration, ball candidates, scorer and tracker. V1: the
stack's exactness (fixed-point executor, frames, tiles); both were added as tests/test_video.py."""

import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from src.video import ball, decode, detect, frames, infer, track
from src.video.calib import FrameHomography, SoccerTrackCalib

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_video(path, frames=60, gop=10):
    """A small synthetic H.264 (or MPEG-4) yuv420p video with an audio track, GOP `gop`."""
    enc = ["-c:v", "libx264", "-bf", "0", "-g", str(gop)]
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "libx264" not in probe.stdout:
        enc = ["-c:v", "mpeg4", "-bf", "0", "-g", str(gop), "-q:v", "3"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-itsoffset", "0.5", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=25:duration={}".format(frames / 25),
                    "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", str(frames / 25 + 0.6),
                    "-map", "0:v", "-map", "1:a", "-pix_fmt", "yuv420p"] + enc + ["-c:a", "aac", path],
                   check=True, capture_output=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_decode_seek_matches_sequential(tmp_path):
    path = str(tmp_path / "clip.mp4")
    try:
        _make_video(path)
    except subprocess.CalledProcessError:  # an ffmpeg build without lavfi sources
        pytest.skip("cannot build a test clip with this ffmpeg")
    meta = decode.probe(path)
    assert meta["pix_fmt"] == "yuv420p" and meta["width"] == 96
    seq = decode.frame_md5s(path, 0, 50)
    full = {}
    for i, y, u, v in decode.frames(path):
        h = hashlib.md5()
        for plane in (y, u, v):
            h.update(plane.tobytes())
        full[i] = h.hexdigest()
    assert [full[i] for i in range(50)] == seq
    for start in (1, 9, 10, 11, 37):
        got = [(i, hashlib.md5(y.tobytes() + u.tobytes() + v.tobytes()).hexdigest())
               for i, y, u, v in decode.frames(path, start=start, count=3)]
        assert got == [(start + k, seq[start + k]) for k in range(3)]


def _zoom_homography(x0=40.0, x1=58.0, y0=25.0, y1=37.0, w=480, h=320, pitch=(105.0, 68.0)):
    """Normalised pitch -> pixels showing [x0, x1] x [y0, y1] metres (y down the image)."""
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    return np.array([[sx * pitch[0], 0.0, -sx * x0], [0.0, sy * pitch[1], -sy * y0], [0.0, 0.0, 1.0]])


def test_frame_homography_round_trip():
    c = FrameHomography(_zoom_homography())
    xy = np.array([[41.0, 26.0], [50.0, 30.0], [57.5, 36.5]])
    assert np.allclose(c.to_pitch(c.to_pixel(xy)), xy, atol=1e-9)
    assert np.allclose(c.scale_x(c.to_pixel(xy)), 480 / 18.0, rtol=1e-6)


def test_soccertrack_calib_inverts_its_own_maps():
    h, w = 120, 200
    # identity remap (undistorted == raw) in OpenCV fixed-point form, with a 0.5 px fraction
    gu, gv = np.meshgrid(np.arange(w), np.arange(h))
    map1 = np.stack([gu, gv], axis=-1).astype(np.int16)
    map2 = np.full((h, w), 16, dtype=np.uint16)  # u + 16/32
    hom = np.array([[1.8, 0.0, 5.0], [0.0, 1.6, 4.0], [0.0, 0.0, 1.0]])  # keypoint metres -> px
    c = SoccerTrackCalib(hom, map1, map2)
    xy = np.array([[10.0, 20.0], [60.0, 50.0], [100.0, 5.0]])
    uv = c.to_pixel(xy)
    assert np.allclose(uv[0], [1.8 * 10 + 5 + 0.5, 1.6 * (68 - 20) + 4], atol=1e-9)
    assert np.allclose(c.to_pitch(uv), xy, atol=1e-6)


def _frames(t, rng, ball_at, w=480, h=320):
    """yuv420p planes: noisy grass, a painted line, a static white spot, a white ball at ball_at."""
    y = (90 + rng.integers(-8, 9, size=(h, w))).astype(np.uint8)
    u = np.full((h // 2, w // 2), 100, np.uint8)
    v = np.full((h // 2, w // 2), 110, np.uint8)
    y[200:203, :] = 200  # a line across the image
    gv, gu = np.mgrid[0:h, 0:w]
    spot = (gu - 400) ** 2 + (gv - 60) ** 2 <= 9
    y[spot] = 210
    bu, bv = ball_at
    disk = (gu - bu) ** 2 + (gv - bv) ** 2 <= 9
    y[disk] = 200
    u[(disk[::2, ::2])] = 128
    v[(disk[::2, ::2])] = 128
    return y, u, v


def test_detect_finds_the_moving_ball_not_the_line_and_is_deterministic():
    calib = FrameHomography(_zoom_homography())
    ctx = ball.Context.build(calib, (320, 480))
    rng = np.random.default_rng(0)
    path = [(100 + 6 * t, 120 + 2 * t) for t in range(5)]
    fr = [_frames(t, rng, path[t]) for t in range(5)]
    players = np.array([[45.0, 28.0], [55.0, 35.0]])
    a = ball.detect(fr[0], fr[2], fr[4], ctx, players_xy=players)
    b = ball.detect(fr[0], fr[2], fr[4], ctx, players_xy=players)
    assert len(a) > 0
    assert a["X"].tobytes() == b["X"].tobytes() and a["u"].tobytes() == b["u"].tobytes()
    assert abs(int(a["u"][0]) - path[2][0]) <= 2 and abs(int(a["v"][0]) - path[2][1]) <= 2
    col = ball.FEATURES.index("mot_min")
    assert a["X"][0, col] > 20  # it moves
    on_line = np.abs(a["v"].astype(int) - 201) <= 2
    assert not on_line.any() or a["X"][on_line, 0].max() < a["X"][0, 0]
    expect = calib.to_pitch(np.array([[path[2][0], path[2][1]]], dtype=float))[0]
    assert np.hypot(a["x"][0] - expect[0], a["y"][0] - expect[1]) < 0.15


def _random_scorer(rng, hidden=4):
    d = len(ball.FEATURES) + len(ball.EXTRA)
    return {"features": list(ball.FEATURES) + ball.EXTRA, "mu": rng.normal(size=d).tolist(),
            "sd": (1 + rng.random(d)).tolist(), "W1": rng.normal(size=(hidden, d)).tolist(),
            "b1": rng.normal(size=hidden).tolist(), "w2": rng.normal(size=hidden).tolist(), "b2": 0.3}


def test_scorer_matches_matrix_form_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    params = _random_scorer(rng)
    sc = ball.Scorer(params)
    X = rng.normal(size=(50, len(params["mu"])))
    z = (X - np.array(params["mu"])) / np.array(params["sd"])
    ref = np.maximum(z @ np.array(params["W1"]).T + np.array(params["b1"]), 0) @ np.array(params["w2"]) + 0.3
    assert np.allclose(sc.score(X), ref, atol=1e-10)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(params))
    assert ball.Scorer.load(str(path)).score(X).tobytes() == sc.score(X).tobytes()
    bad = dict(params, features=["x"])
    with pytest.raises(ValueError):
        ball.Scorer(bad)


def test_design_adds_spot_distance():
    X = np.zeros((3, len(ball.FEATURES)))
    D = ball.design(X, np.array([52.5, 11.0, 30.0]), np.array([34.0, 35.0, 10.0]))
    assert D.shape == (3, len(ball.FEATURES) + 1)
    assert np.allclose(D[:, -1], [0.0, 1.0, 5.0])


def test_link_follows_the_ball_through_clutter():
    rng = np.random.default_rng(2)
    T = 30
    truth = np.stack([20 + 1.2 * np.arange(T), 30 + 0.4 * np.arange(T)], axis=1)
    positions, scores = [], []
    for t in range(T):
        clutter = rng.uniform([0, 0], [105, 68], size=(6, 2))
        if t in (10, 11):  # the ball is hidden: clutter only
            positions.append(clutter)
            scores.append(rng.uniform(-3, 0.5, 6))
        else:
            positions.append(np.vstack([clutter, truth[t] + rng.normal(0, 0.1, 2)]))
            scores.append(np.concatenate([rng.uniform(-3, 1.5, 6), [2.0]]))
    tr = track.link(np.arange(0, 5 * T, 5), positions, scores, 0.2)
    seen = np.isfinite(tr["x"])
    err = np.hypot(tr["x"] - truth[:, 0], tr["y"] - truth[:, 1])
    assert seen.sum() >= T - 3
    assert np.all(err[seen & ~np.isin(np.arange(T), (10, 11))] < 0.5)
    b, c = track.to_frames(tr, 5 * T)
    assert np.isfinite(b[7]).all() and np.isnan(b[5 * T - 1]).all() or np.isfinite(b[5 * (T - 1)]).all()
    again = track.link(np.arange(0, 5 * T, 5), positions, scores, 0.2)
    assert again["x"].tobytes() == tr["x"].tobytes()


def test_frame_jobs_and_saved_track_round_trip(tmp_path, monkeypatch):
    from src.video import run

    jobs = run.frame_jobs(1000, [(1, 100, 200), (2, 500, 520)])
    assert jobs[0] == 100 and jobs[-1] == 520 and all(f % 5 == 0 for f in jobs)
    scorer = tmp_path / "s.json"
    scorer.write_text(json.dumps(_random_scorer(np.random.default_rng(3))))
    monkeypatch.setattr(ball, "SCORER_FILE", str(scorer))
    monkeypatch.setattr(run.stack_id, "__defaults__", (str(scorer),))
    res = {"ball": np.full((10, 2), np.nan), "conf": np.full(10, np.nan), "stack": run.stack_id()}
    out = str(tmp_path / "vb.npz")
    run.save(out, res)
    run.save(str(tmp_path / "vb2.npz"), res)
    assert open(out, "rb").read() == open(str(tmp_path / "vb2.npz"), "rb").read()
    assert run.load(out, 10)["stack"] == res["stack"]
    with pytest.raises(ValueError):
        run.load(out, 11)


def _toy_match(n=500, seed=0):
    from src.schema import Match

    rng = np.random.default_rng(seed)
    t = np.arange(n) / 25.0
    ballxy = np.stack([50 + 10 * np.sin(t / 3), 34 + 8 * np.cos(t / 4)], axis=1)
    players = ballxy[:, None, :] + rng.normal(0, 6, size=(1, 10, 2)) + rng.normal(0, 0.05, (n, 10, 2))
    return Match(match_id="toy:{}".format(seed), fps=25, pitch=(105.0, 68.0),
                 period=np.ones(n, np.int8), ball=ballxy, players=players,
                 slot_team=np.array([0] * 5 + [1] * 5, np.int8)).validate()


def test_simulated_video_ball_oracle_and_determinism():
    from src.video import simulate

    clean = _toy_match()
    hidden = np.zeros(clean.n_frames, bool)
    hidden[100:200] = True
    vb, conf = simulate.simulate(clean, hidden, recall={"open": 1, "fast": 1, "crowd": 1},
                                 wrong={"open": 0, "fast": 0, "crowd": 0}, noise_m=0.0)
    assert np.isnan(vb[~hidden]).all()
    seen = np.isfinite(vb[:, 0])
    assert seen[100:196].all()
    assert np.max(np.hypot(*(vb[seen] - clean.ball[seen]).T)) < 0.2  # linear between 5 Hz picks
    a = simulate.simulate(clean, hidden, seed=3)[0]
    b = simulate.simulate(clean, hidden, seed=3)[0]
    assert a.tobytes() == b.tobytes()


def test_vball_group_is_zero_without_video_and_speaks_on_missing_frames():
    import dataclasses

    from src.models import learned
    from src.video import simulate

    clean = _toy_match()
    deg = dataclasses.replace(clean, ball=clean.ball.copy())
    deg.ball[100:200] = np.nan
    th, ta, c = learned.features(deg, ("vball",))
    base = learned.features(deg, ())
    width = th.shape[1] - base[0].shape[1]
    assert width > 0 and not th[:, -width:].any() and not c[:, -2:].any()
    withv = simulate.attach(deg, clean, recall={"open": 1, "fast": 1, "crowd": 1},
                            wrong={"open": 0, "fast": 0, "crowd": 0}, noise_m=0.0)
    th2, ta2, c2 = learned.features(withv, ("vball",))
    assert th2[100:190, -width:].any() and not th2[:100, -width:].any()
    assert c2[100:190, -2].all() and not c2[:100, -2].any()
    # swapping the teams swaps the blocks
    sw = dataclasses.replace(withv, slot_team=1 - withv.slot_team)
    th3, ta3, _ = learned.features(sw, ("vball",))
    assert np.array_equal(th3[:, -width:], ta2[:, -width:])


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_run_end_to_end_on_a_synthetic_video(tmp_path, monkeypatch):
    from src.video import run

    n, w, h = 40, 480, 320
    rng = np.random.default_rng(5)
    path = [(60 + 9 * t, 100 + 3 * t) for t in range(n)]
    raw = b"".join(np.concatenate([p.ravel() for p in _frames(t, rng, path[t])]).tobytes()
                   for t in range(n))
    video = str(tmp_path / "syn.mp4")
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    codec = ["-c:v", "libx264", "-bf", "0", "-g", "10", "-qp", "0"] if "libx264" in enc else \
        ["-c:v", "mpeg4", "-q:v", "1", "-g", "10"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s", "{}x{}".format(w, h), "-r", "25", "-i", "pipe:0"] + codec
                   + ["-pix_fmt", "yuv420p", video], input=raw, check=True, capture_output=True)
    scorer = tmp_path / "s.json"
    params = _random_scorer(np.random.default_rng(3))
    # a scorer that likes moving, contrasted blobs: logit = (mot_min + c_band - 60) / 10
    d = len(params["mu"])
    params.update(mu=[0.0] * d, sd=[1.0] * d, W1=[[0.0] * d], b1=[-60.0], w2=[0.1], b2=0.0)
    params["W1"][0][ball.FEATURES.index("mot_min")] = 1.0
    params["W1"][0][ball.FEATURES.index("c_band")] = 1.0
    params["b1"] = [0.0]
    params["b2"] = -6.0
    scorer.write_text(json.dumps(params))
    hom = np.repeat(_zoom_homography()[None], n, axis=0)
    players = np.full((n, 4, 2), np.nan)
    res = run.run(video, hom, np.ones(n, bool), players, [(1, 5, n - 6)], (105.0, 68.0), jobs=1,
                  scorer_path=str(scorer))
    assert res["ball"].shape == (n, 2) and res["conf"].shape == (n,)
    calib = FrameHomography(_zoom_homography())
    seen = np.flatnonzero(np.isfinite(res["ball"][:, 0]))
    assert len(seen) >= 10
    truth = calib.to_pitch(np.array(path, dtype=float))
    assert np.median(np.hypot(*(res["ball"][seen] - truth[seen]).T)) < 0.3
    again = run.run(video, hom, np.ones(n, bool), players, [(1, 5, n - 6)], (105.0, 68.0), jobs=2,
                    scorer_path=str(scorer))
    assert again["ball"].tobytes() == res["ball"].tobytes()


# --- V1: video stack (frames, fixed-point inference, detector helpers) ---
def test_exp_det_matches_exp():
    x = np.linspace(-60, 60, 200001)
    rel = np.abs(infer.exp_det(x) / np.exp(x) - 1)
    assert rel.max() < 1e-14


def test_sigmoid_table_is_the_direct_formula():
    x = infer.snap(np.random.default_rng(0).normal(0, 8, 200000))
    t = infer.sigmoid_table()
    half = (len(t) - 1) // 2
    k = np.clip(x * 2 ** infer.ACT_BITS, -half, half).astype(np.int64)
    assert np.array_equal(t[k + half], infer.snap(1 / (1 + infer.exp_det(-x))))


def test_fixed_point_sums_do_not_depend_on_order():
    rng = np.random.default_rng(1)
    a = infer.snap(rng.normal(0, 20, (64, 2304)))
    w = rng.normal(0, 0.05, (2304,))
    e = infer._weight_bits(w[None])[0]
    w = np.round(w * 2.0 ** e) / 2.0 ** e
    fwd = a @ w
    loop = np.zeros(64)
    for j in range(2304 - 1, -1, -1):  # the opposite order, one product at a time
        loop = loop + a[:, j] * w[j]
    assert np.array_equal(fwd, loop)
    # plain float64 is not order-independent (the reason for the fixed point)
    a2, w2 = rng.normal(0, 20, (64, 2304)), rng.normal(0, 0.05, 2304)
    loop2 = np.zeros(64)
    for j in range(2304 - 1, -1, -1):
        loop2 = loop2 + a2[:, j] * w2[j]
    assert not np.array_equal(a2 @ w2, loop2)


def _tiny_onnx(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(2)
    init = [numpy_helper.from_array(rng.normal(0, 0.2, (8, 12, 3, 3)).astype(np.float32), "w1"),
            numpy_helper.from_array(rng.normal(0, 0.1, (8,)).astype(np.float32), "b1"),
            numpy_helper.from_array(rng.normal(0, 0.3, (8, 1, 3, 3)).astype(np.float32), "wd"),
            numpy_helper.from_array(rng.normal(0, 0.2, (5, 16, 1, 1)).astype(np.float32), "w2"),
            numpy_helper.from_array(np.array([0, 0], np.int64), "s0"),
            numpy_helper.from_array(np.array([2 ** 31, 2 ** 31], np.int64), "s1"),
            numpy_helper.from_array(np.array([2, 3], np.int64), "ax"),
            numpy_helper.from_array(np.array([2, 2], np.int64), "st"),
            numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "sc"),
            numpy_helper.from_array(np.array([], np.float32), "roi")]
    nodes = [
        helper.make_node("Slice", ["x", "s0", "s1", "ax", "st"], ["xa"]),
        helper.make_node("Concat", ["xa", "xa", "xa", "xa"], ["f"], axis=1),
        helper.make_node("Conv", ["f", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Sigmoid", ["c1"], ["g1"]),
        helper.make_node("Mul", ["c1", "g1"], ["a1"]),
        helper.make_node("Conv", ["a1", "wd"], ["d1"], pads=[1, 1, 1, 1], group=8),
        helper.make_node("MaxPool", ["d1"], ["m1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Concat", ["a1", "m1"], ["cat"], axis=1),
        helper.make_node("Conv", ["cat", "w2"], ["c2"]),
        helper.make_node("Resize", ["c2", "roi", "sc"], ["y"], mode="nearest"),
    ]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], init)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=7), path)


def test_fixed_executor_matches_onnxruntime(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path = str(tmp_path / "tiny.onnx")
    _tiny_onnx(path)
    infer.compile_onnx(path, str(tmp_path / "tiny.npz"))
    x = np.round(np.random.default_rng(3).uniform(0, 255, (1, 3, 32, 32)))
    ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.astype(np.float32)})[0]
    fixed = infer.Net(str(tmp_path / "tiny.npz")).run(x)[0]
    plain = infer.Net(str(tmp_path / "tiny.npz"), exact=False).run(x)[0]
    assert fixed.shape == ref.shape == (1, 5, 32, 32)
    assert np.abs(plain - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())
    assert np.abs(fixed - ref).max() < 2e-2 * max(1.0, np.abs(ref).max())
    assert np.array_equal(fixed * 2 ** infer.ACT_BITS, np.round(fixed * 2 ** infer.ACT_BITS))  # on the grid


def test_to_rgb_integer_conversion():
    y = np.array([[16, 235], [16, 235]], np.uint8)
    u = v = np.array([[128]], np.uint8)
    rgb = frames.to_rgb(frames.Frame(y, u, v), 1)
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 1].tolist() == [255, 255, 255]
    big = frames.Frame(np.full((8, 8), 200, np.uint8), np.full((4, 4), 90, np.uint8), np.full((4, 4), 170, np.uint8))
    assert frames.to_rgb(big, 2).shape == (4, 4, 3)
    assert np.array_equal(frames.to_rgb(frames.crop(big, (1, 1, 7, 7)), 1), frames.to_rgb(big, 1)[0:8, 0:8][:, :])


def test_iter_frames_on_a_synthetic_video(tmp_path):
    av = pytest.importorskip("av")
    path = str(tmp_path / "clip.mp4")
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=25)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            img = np.full((48, 64, 3), i * 20, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    got = [i for i, _ in frames.iter_frames(path, indices=[2, 5, 9])]
    assert got == [2, 5, 9]
    a = [f.y.copy() for _, f in frames.iter_frames(path, threads=1)]
    b = [f.y.copy() for _, f in frames.iter_frames(path, threads=4)]
    assert len(a) == 10 and all(np.array_equal(p, q) for p, q in zip(a, b))
    assert frames.info(path)["width"] == 64


def test_tiles_cover_the_box():
    corners = detect.tiles(1920, 1080, 416, overlap=32, box=(100, 200, 1800, 1000))
    covered = np.zeros((1080, 1920), bool)
    for x, y in corners:
        assert 0 <= x <= 1920 - 416 and 0 <= y <= 1080 - 416
        covered[y:y + 416, x:x + 416] = True
    assert covered[200:1000, 100:1800].all()


def test_nms_is_stable():
    b = np.array([[0, 0, 10, 10, 0.5, 32], [1, 1, 11, 11, 0.5, 32], [50, 50, 60, 60, 0.9, 32],
                  [0, 0, 10, 10, 0.7, 0]], float)
    out = detect.nms(b)
    assert out[:, 4].tolist() == [0.9, 0.7, 0.5] and out[2, 0] == 0
