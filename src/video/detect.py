"""Object detection on video tiles (V1 interface): YOLOX ONNX models, two backends.

  backend="fixed"  src/video/infer.py: numpy, fixed point in float64, the same bits on every machine.
                   The only backend allowed at run time (anything that feeds a submission).
  backend="float64" the same numpy graph in plain float64 (no fixed point): a control for F135 only.
  backend="ort"    ONNX Runtime, float32, ~5-10x faster, kernels differ by CPU. Experiments and
                   training-time evaluation only.

Models are YOLOX (Apache-2.0). The COCO baseline weights come from the YOLOX release 0.1.1rc0
(`yolox_nano.onnx`, `yolox_tiny.onnx`: 416 x 416 input; `yolox_s.onnx`: 640). Input is the tile in BGR
order as float, no normalisation (YOLOX >= 0.1.1 weights). Outputs are decoded here (grid + exp, with
infer.exp_det so the fixed backend stays exact), then greedy NMS with a stable order.

`detect` returns float64 rows (x1, y1, x2, y2, score, class) in tile pixels, snapped to 1/16 px and
scores to 1/1024: the snap matters only for "ort" (the fixed backend is already exact).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from src.video import infer

COCO_PERSON, COCO_BALL = 0, 32
MODELS_DIR = os.path.join("data", "raw", "external", "models")
# The COCO baselines, fetched by sha256 (scripts/video/fetch_models.sh).
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"
YOLOX_SHA256 = {
    "yolox_nano": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    "yolox_tiny": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "yolox_s": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_model(name: str, root: str = MODELS_DIR) -> str:
    """Download a baseline model once and check its sha256."""
    import urllib.request

    path = os.path.join(root, name + ".onnx")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(YOLOX_URL.format(name), path + ".part")
        os.replace(path + ".part", path)
    if sha256(path) != YOLOX_SHA256[name]:
        raise ValueError("{}: sha256 mismatch".format(path))
    return path


def nms(boxes: np.ndarray, iou: float = 0.45) -> np.ndarray:
    """Greedy NMS per class; order by score then index (stable), so ties break the same way."""
    if len(boxes) == 0:
        return boxes
    order = np.lexsort((np.arange(len(boxes)), -boxes[:, 4]))
    b = boxes[order]
    keep = np.ones(len(b), dtype=bool)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(len(b)):
        if not keep[i]:
            continue
        rest = np.arange(i + 1, len(b))
        rest = rest[keep[rest] & (b[rest, 5] == b[i, 5])]
        if len(rest) == 0:
            continue
        w = np.clip(np.minimum(b[rest, 2], b[i, 2]) - np.maximum(b[rest, 0], b[i, 0]), 0, None)
        h = np.clip(np.minimum(b[rest, 3], b[i, 3]) - np.maximum(b[rest, 1], b[i, 1]), 0, None)
        inter = w * h
        keep[rest[inter / (area[i] + area[rest] - inter) > iou]] = False
    return b[keep]


class Detector:
    def __init__(self, model_path: str, backend: str = "fixed", sha: str | None = None, threads: int = 0,
                 classes=(COCO_PERSON, COCO_BALL)):
        if sha is not None and sha256(model_path) != sha:
            raise ValueError("{}: sha256 mismatch".format(model_path))
        self.backend = backend
        self.classes = np.array(classes)
        if backend == "ort":
            import onnxruntime as ort

            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.input_name, shape = inp.name, inp.shape
        elif backend in ("fixed", "float64"):
            compiled = model_path[:-5] + ".fixed.npz" if model_path.endswith(".onnx") else model_path
            if not os.path.exists(compiled):
                infer.compile_onnx(model_path, compiled)
            self.net = infer.Net(compiled, exact=backend == "fixed")
            shape = self.net.meta["inputs"][0]["shape"]
        else:
            raise ValueError(backend)
        self.size = (int(shape[2]), int(shape[3]))
        self._grid = self._make_grid(self.size)

    @staticmethod
    def _make_grid(size):
        grids, strides = [], []
        for s in (8, 16, 32):
            h, w = size[0] // s, size[1] // s
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((h * w, 1), s))
        return np.concatenate(grids).astype(np.float64), np.concatenate(strides).astype(np.float64)

    def raw(self, tile_rgb: np.ndarray) -> np.ndarray:
        """The network's undecoded output (N, 85) for one RGB uint8 tile of the model's size."""
        if tile_rgb.shape[:2] != self.size:
            raise ValueError("tile {} != model input {}".format(tile_rgb.shape[:2], self.size))
        x = np.ascontiguousarray(tile_rgb[..., ::-1].transpose(2, 0, 1))[None]  # BGR, CHW
        if self.backend == "ort":
            return self.sess.run(None, {self.input_name: x.astype(np.float32)})[0][0].astype(np.float64)
        return self.net.run(x.astype(np.float64))[0][0]

    def detect(self, tile_rgb: np.ndarray, score: float = 0.05, iou: float = 0.45) -> np.ndarray:
        return self.decode(self.raw(tile_rgb), score, iou)

    def decode(self, out: np.ndarray, score: float = 0.05, iou: float = 0.45) -> np.ndarray:
        """Raw network output -> snapped, NMS'd boxes (x1, y1, x2, y2, score, class) in tile px."""
        grid, stride = self._grid
        xy = (out[:, :2] + grid) * stride
        wh = infer.exp_det(out[:, 2:4]) * stride
        cls = out[:, 5:][:, self.classes]
        sc = out[:, 4:5] * cls
        rows = []
        for j, c in enumerate(self.classes):
            m = sc[:, j] >= score
            if m.any():
                b = np.c_[xy[m] - wh[m] / 2, xy[m] + wh[m] / 2, sc[m, j], np.full(m.sum(), c)]
                rows.append(b)
        if not rows:
            return np.zeros((0, 6))
        b = np.concatenate(rows)
        b[:, :4] = np.round(b[:, :4] * 16) / 16
        b[:, 4] = np.round(b[:, 4] * 1024) / 1024
        return nms(b, iou)


def tiles(width: int, height: int, size: int, overlap: int = 32, box=None) -> list[tuple[int, int]]:
    """Top-left corners of size x size tiles covering box=(u0, v0, u1, v1) (default: the image),
    overlapping by at least `overlap` px, all inside the image."""
    u0, v0, u1, v1 = box or (0, 0, width, height)
    u0, v0 = max(0, int(u0)), max(0, int(v0))
    u1, v1 = min(width, int(np.ceil(u1))), min(height, int(np.ceil(v1)))

    def starts(a, b, limit):
        if b - a <= size:
            return [min(max(0, a), limit - size)]
        n = int(np.ceil((b - a - overlap) / (size - overlap)))
        return sorted({min(int(round(a + i * (b - a - size) / (n - 1))), limit - size) for i in range(n)})

    return [(x, y) for y in starts(v0, v1, height) for x in starts(u0, u1, width)]


def detect_frame(det: Detector, rgb: np.ndarray, corners, score: float = 0.05) -> np.ndarray:
    """Run the detector on each tile and merge the boxes (frame pixels) with one more NMS."""
    h, w = det.size
    out = []
    for (x, y) in corners:
        b = det.detect(rgb[y:y + h, x:x + w], score=score)
        if len(b):
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            out.append(b)
    return nms(np.concatenate(out)) if out else np.zeros((0, 6))
