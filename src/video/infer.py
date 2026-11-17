"""Run an ONNX convolutional network in numpy, bit-identically on every machine (V1).

Why: float sums depend on their order, and every BLAS / CPU / thread count orders them differently,
so ONNX Runtime or a plain numpy float64 network gives slightly different numbers on the owner's Mac
and on the organisers' machine, and a snapped output still flips now and then. Here the cause is
removed (docs/video-plan.md, "Determinism"):

  * Fixed point inside float64. Activations live on the grid 2^-ACT_BITS. Each conv output channel's
    weights are snapped to a power-of-two grid that keeps WEIGHT_BITS significant bits for its
    largest weight. A product of two grid values is exact in float64, and so is every sum of them
    while it stays under 2^53 grid units, which `run` checks layer by layer (it raises otherwise).
    So matmul gives the same bits under every BLAS, kernel and thread count.
  * Every op that leaves the grid (Conv, Mul, Sigmoid) snaps back with np.round (exact, half to
    even) on a power-of-two scale (exact).
  * Sigmoid uses `exp_det`: range reduction and a fixed polynomial, only + - * / and ldexp, each
    correctly rounded by IEEE 754. np.exp is not used: its last bit varies by CPU. Its inputs are on
    the grid, so it is a table lookup (`sigmoid_table`, built once with exp_det): same bits, ~20x
    faster.

The op set is what YOLOX's ONNX export uses (Conv, Sigmoid, Mul, Add, Concat, Slice, MaxPool,
Resize nearest, Reshape, Transpose). `compile_onnx` turns a .onnx file into a .npz that `Net` runs
with numpy alone (the `onnx` package is only needed to compile). An unsupported op raises.
"""

from __future__ import annotations

import json

import numpy as np

ACT_BITS = 12
WEIGHT_BITS = 16
EXACT_LIMIT = 2.0 ** 52  # stay one bit under float64's 53-bit mantissa

_LN2_HI = 6.93147180369123816490e-01  # fdlibm's split of ln 2: n * _LN2_HI is exact for |n| < 2^11
_LN2_LO = 1.90821492927058770002e-10
_LOG2E = 1.44269504088896338700e00
_EXP_COEF = [1.0 / np.prod(np.arange(1, k + 1, dtype=np.float64)) for k in range(14)][::-1]


def exp_det(x: np.ndarray) -> np.ndarray:
    """exp(x) in float64 from + - * / and ldexp only: the same bits on every IEEE 754 machine.
    |relative error| < 1e-15 on [-60, 60]; inputs are clipped to that range."""
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    n = np.floor(x * _LOG2E + 0.5)
    r = (x - n * _LN2_HI) - n * _LN2_LO
    p = np.full_like(r, _EXP_COEF[0])
    for c in _EXP_COEF[1:]:
        p = p * r + c
    return np.ldexp(p, n.astype(np.int64))


def sigmoid_table(bits: int = ACT_BITS, limit: float = 16.0) -> np.ndarray:
    """snap(sigmoid(x)) for every grid value x in [-limit, limit] (beyond it the snapped sigmoid is
    exactly 0 or 1 for bits <= 20). Built with exp_det, so it is the same table on every machine."""
    k = np.arange(-int(limit * 2 ** bits), int(limit * 2 ** bits) + 1, dtype=np.float64)
    return snap(1.0 / (1.0 + exp_det(-k / 2 ** bits)), bits)


def snap(x: np.ndarray, bits: int = ACT_BITS) -> np.ndarray:
    s = float(2 ** bits)
    return np.round(np.asarray(x, dtype=np.float64) * s) / s


def _weight_bits(w: np.ndarray) -> np.ndarray:
    """Per output channel: the power-of-two scale exponent giving WEIGHT_BITS significant bits."""
    m = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
    m = np.where(m > 0, m, 1.0)
    return (WEIGHT_BITS - 1 - np.ceil(np.log2(m))).astype(np.int64)


def compile_onnx(onnx_path: str, out_path: str) -> dict:
    """Read an ONNX graph into a numpy-only .npz: tensors plus a JSON list of nodes."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    g = model.graph
    tensors = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    nodes = []
    for n in g.node:
        attrs = {}
        for a in n.attribute:
            v = onnx.helper.get_attribute_value(a)
            if isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, onnx.TensorProto):
                name = "{}__{}".format(n.output[0], a.name)
                tensors[name] = numpy_helper.to_array(v)
                v = {"tensor": name}
            elif isinstance(v, (list, tuple)):
                v = list(v)
            attrs[a.name] = v
        nodes.append({"op": n.op_type, "in": list(n.input), "out": list(n.output), "attrs": attrs})
    meta = {"inputs": [{"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]}
                       for i in g.input if i.name not in tensors],
            "outputs": [o.name for o in g.output], "nodes": nodes}
    safe = {"t{}".format(i): v for i, v in enumerate(tensors.values())}
    names = {"t{}".format(i): k for i, k in enumerate(tensors)}
    np.savez_compressed(out_path, __graph__=np.frombuffer(json.dumps({"meta": meta, "names": names}).encode(),
                                                          dtype=np.uint8), **safe)
    return meta


def _windows(x: np.ndarray, kh: int, kw: int, sh: int, sw: int) -> np.ndarray:
    """(C, H, W) -> (C, kh, kw, Ho, Wo) strided view."""
    v = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # (C, H', W', kh, kw)
    v = v[:, ::sh, ::sw]
    return v.transpose(0, 3, 4, 1, 2)


class Net:
    """A compiled graph. `run(x)` takes the input as float64 already on the activation grid (uint8
    images are) and returns the outputs as float64 arrays on the grid. Batch size 1.

    exact=False runs the same graph in plain float64 (raw weights, no snapping, np.exp): only to show
    what the fixed point buys (F135). Never at run time."""

    def __init__(self, npz_path: str, act_bits: int = ACT_BITS, exact: bool = True):
        with np.load(npz_path, allow_pickle=False) as z:
            spec = json.loads(bytes(z["__graph__"]).decode())
            self.tensors = {spec["names"][k]: z[k] for k in z.files if k != "__graph__"}
        self.meta = spec["meta"]
        self.act_bits = act_bits
        self.exact = exact
        self._sig = sigmoid_table(act_bits)
        self._sig_half = (len(self._sig) - 1) // 2
        self._conv = {}
        for n in self.meta["nodes"]:
            if n["op"] == "Conv":
                w = self.tensors[n["in"][1]].astype(np.float64)
                e = _weight_bits(w)
                scale = np.ldexp(1.0, e).reshape(-1, *([1] * (w.ndim - 1)))
                wq = np.round(w * scale) / scale
                b = self.tensors[n["in"][2]].astype(np.float64) if len(n["in"]) > 2 else np.zeros(w.shape[0])
                bq = snap(b, act_bits)
                # Exactness bound: sum |w| in grid units of the finest (weights x activations) grid.
                units = np.abs(wq.reshape(w.shape[0], -1)).sum(axis=1) * np.ldexp(1.0, e + act_bits)
                if not exact:
                    wq, bq, units = w, b, 0.0
                self._conv[n["out"][0]] = (wq, bq, float(units.max()) if exact else 0.0)

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        env = {self.meta["inputs"][0]["name"]: np.asarray(x, dtype=np.float64)}
        get = lambda name: env[name] if name in env else self.tensors[name]
        for n in self.meta["nodes"]:
            op, a, ins = n["op"], n["attrs"], n["in"]
            if op == "Conv":
                y = self._conv2d(get(ins[0]), n, a)
            elif op == "Sigmoid":
                if self.exact:
                    y = self._sigmoid(get(ins[0]))
                else:
                    with np.errstate(over="ignore"):
                        y = 1.0 / (1.0 + np.exp(-get(ins[0])))
            elif op == "Mul":
                y = get(ins[0]) * get(ins[1])
                y = snap(y, self.act_bits) if self.exact else y
            elif op == "Add":
                y = get(ins[0]) + get(ins[1])  # exact on the grid
            elif op == "Concat":
                y = np.concatenate([get(i) for i in ins], axis=a["axis"])
            elif op == "Slice":
                y = self._slice(get(ins[0]), [np.asarray(get(i)).tolist() for i in ins[1:]])
            elif op == "MaxPool":
                y = self._maxpool(get(ins[0]), a)
            elif op == "Resize":
                y = self._resize(get(ins[0]), [get(i) for i in ins[1:] if i], a)
            elif op == "Reshape":
                y = get(ins[0]).reshape([int(v) for v in get(ins[1])])
            elif op == "Transpose":
                y = get(ins[0]).transpose(a["perm"])
            else:
                raise NotImplementedError("op {} is not supported (src/video/infer.py)".format(op))
            env[n["out"][0]] = y
        return [env[o] for o in self.meta["outputs"]]

    def _sigmoid(self, x):
        k = np.clip(x * 2.0 ** self.act_bits, -self._sig_half, self._sig_half).astype(np.int64)
        return self._sig[k + self._sig_half]

    def _conv2d(self, x, n, a):
        wq, bq, units = self._conv[n["out"][0]]
        if float(np.abs(x).max()) * units >= EXACT_LIMIT:
            raise ArithmeticError("conv {}: sums could exceed 2^52 grid units".format(n["out"][0]))
        group = int(a.get("group", 1))
        sh, sw = a.get("strides", [1, 1])
        pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
        kh, kw = wq.shape[2], wq.shape[3]
        x = x[0]
        if pt or pl or pb or pr:
            x = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))
        cout, cin_g = wq.shape[0], wq.shape[1]
        ho, wo = (x.shape[1] - kh) // sh + 1, (x.shape[2] - kw) // sw + 1
        if group == 1:
            if kh == kw == 1:
                cols = x[:, ::sh, ::sw][:, :ho, :wo].reshape(x.shape[0], -1)
            else:
                cols = _windows(x, kh, kw, sh, sw).reshape(cin_g * kh * kw, -1)
            y = wq.reshape(cout, -1) @ cols
        elif group == x.shape[0] and cin_g == 1:  # depthwise: elementwise products and sums, exact
            win = _windows(x, kh, kw, sh, sw)  # (C, kh, kw, Ho, Wo)
            y = np.zeros((cout, ho, wo))
            for i in range(kh):
                for j in range(kw):
                    y += win[:, i, j] * wq[:, 0, i, j][:, None, None]
            y = y.reshape(cout, -1)
        else:
            cg = cout // group
            parts = []
            for g in range(group):
                xs = x[g * cin_g:(g + 1) * cin_g]
                cols = _windows(xs, kh, kw, sh, sw).reshape(cin_g * kh * kw, -1)
                parts.append(wq[g * cg:(g + 1) * cg].reshape(cg, -1) @ cols)
            y = np.concatenate(parts)
        y = (y + bq[:, None]).reshape(1, cout, ho, wo)
        return snap(y, self.act_bits) if self.exact else y

    @staticmethod
    def _slice(x, args):
        starts, ends = args[0], args[1]
        axes = args[2] if len(args) > 2 else list(range(len(starts)))
        steps = args[3] if len(args) > 3 else [1] * len(starts)
        idx = [slice(None)] * x.ndim
        for s, e, ax, st in zip(starts, ends, axes, steps):
            idx[ax] = slice(int(s), int(min(e, 2 ** 62)), int(st))
        return x[tuple(idx)]

    @staticmethod
    def _maxpool(x, a):
        kh, kw = a["kernel_shape"]
        sh, sw = a.get("strides", [1, 1])
        pt, pl, pb, pr = a.get("pads", [0, 0, 0, 0])
        x = np.pad(x[0], ((0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
        return _windows(x, kh, kw, sh, sw).max(axis=(1, 2))[None]

    @staticmethod
    def _resize(x, args, a):
        if a.get("mode", "nearest") != "nearest":
            raise NotImplementedError("Resize mode {}".format(a.get("mode")))
        scales = [np.asarray(v) for v in args if np.asarray(v).size == 4]
        s = scales[-1] if scales else None
        if s is None or not np.all(s == np.round(s)):
            raise NotImplementedError("Resize needs integer scales")
        fy, fx = int(s[2]), int(s[3])
        return np.repeat(np.repeat(x, fy, axis=2), fx, axis=3)
