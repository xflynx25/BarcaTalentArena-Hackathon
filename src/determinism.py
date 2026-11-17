"""Determinism check: run the submission pipeline several times in fresh processes, compare the CSV hashes.

The final repo must regenerate the exact submitted file (official/SUBMISSION.md: "Not
approximately -- the same file"). This runs `python -m src.run` N times (default 2), each in its
own process with a different PYTHONHASHSEED, and every other run with BLAS/OpenMP pinned to one
thread. So set/dict ordering and thread-count effects show up as different hashes.

On the synthetic bundle (anywhere, including cloud agents and CI):

    .venv/bin/python -m src.determinism --synthetic

On the real match (owner, in your own terminal), with the exact model arguments of the submission:

    export MTAH_DATA_DIR=$PWD/data/mtah2027
    .venv/bin/python -m src.determinism --model nearest --param window=50
    .venv/bin/python -m src.determinism --model nearest --param window=50 \
        --expect submissions/<team>.csv        # also the same bytes as the file you submitted

Any argument it doesn't know (--model, --param...) goes to src.run unchanged; --out and --gt are
set here. It prints hashes, sizes and counts only, never rows. Exit code 0 = identical, 1 = not.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

from src import submission
from src.official import ROOT

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_env(index: int) -> dict:
    """Run i gets PYTHONHASHSEED=i; odd runs are single-threaded."""
    env = dict(os.environ, PYTHONHASHSEED=str(index))
    if index % 2:
        env.update({name: "1" for name in THREAD_VARS})
    return env


def check(run_args: list[str], runs: int = 2, data_dir: str | None = None,
          expect: str | None = None, workdir: str | None = None) -> dict:
    """Run src.run `runs` times and compare outputs. expect is a sha256 or a CSV path to match too.
    Returns {"identical", "hashes", "bytes", "frames_differing", "expect_ok", "seconds"}."""
    if runs < 2:
        raise ValueError("need at least 2 runs to compare, got {}".format(runs))
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        hashes, sizes, outs, seconds = [], [], [], []
        for index in range(runs):
            out = os.path.join(tmp, "run{}.csv".format(index))
            command = [sys.executable, "-m", "src.run", *run_args, "--out", out, "--gt", ""]
            if data_dir:
                command += ["--data-dir", data_dir]
            started = time.time()
            result = subprocess.run(command, cwd=ROOT, env=run_env(index),
                                    capture_output=True, text=True)
            seconds.append(round(time.time() - started, 1))
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["(no output)"]
                raise RuntimeError("run {} failed (exit {}): {}".format(
                    index, result.returncode, tail[0]))
            hashes.append(sha256(out))
            sizes.append(os.path.getsize(out))
            outs.append(out)

        # frames that differ from run 0 (None if the lengths differ), only when hashes differ
        differing = []
        if len(set(hashes)) > 1:
            first = submission.read_codes(outs[0])
            for out in outs[1:]:
                codes = submission.read_codes(out)
                differing.append(int(np.sum(codes != first)) if len(codes) == len(first) else None)

    expect_ok = None
    if expect:
        wanted = sha256(expect) if os.path.isfile(expect) else expect.lower()
        expect_ok = all(h == wanted for h in hashes)
    return {
        "identical": len(set(hashes)) == 1,
        "hashes": hashes,
        "bytes": sizes,
        "frames_differing": differing,
        "expect_ok": expect_ok,
        "seconds": seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2, help="how many fresh runs (default 2)")
    parser.add_argument("--data-dir", help="bundle root (default: MTAH_DATA_DIR or official/data)")
    parser.add_argument("--synthetic", action="store_true",
                        help="write a full-length synthetic bundle (seed 0) to a temp dir and use it")
    parser.add_argument("--expect", metavar="SHA256|CSV",
                        help="also require this hash, or the hash of this file (the submitted CSV)")
    args, run_args = parser.parse_known_args(argv)
    if args.synthetic and args.data_dir:
        parser.error("give --synthetic or --data-dir, not both")

    print("python     {}  numpy {}  {}".format(platform.python_version(), np.__version__,
                                               platform.platform(terse=True)))
    print("src.run    {}".format(" ".join(run_args) or "(defaults)"))
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = args.data_dir
        if args.synthetic:
            from src import synthetic
            data_dir = os.path.join(tmp, "mtah2027")
            synthetic.write_bundle(data_dir, synthetic.Spec(seed=0))
            print("bundle     synthetic, seed 0")
        result = check(run_args, args.runs, data_dir, args.expect)

    for index, (digest, size, secs) in enumerate(zip(result["hashes"], result["bytes"],
                                                     result["seconds"])):
        print("run {}      sha256 {}  {} bytes  {} s  (PYTHONHASHSEED={}{})".format(
            index, digest, size, secs, index, ", 1 thread" if index % 2 else ""))
    if not result["identical"]:
        print("frames     differing from run 0: {}".format(result["frames_differing"]))
    print("verdict    {}".format("IDENTICAL" if result["identical"] else "DIFFERENT"))
    if result["expect_ok"] is not None:
        print("expect     {}".format("MATCH" if result["expect_ok"] else "MISMATCH"))
    return 0 if result["identical"] and result["expect_ok"] is not False else 1


if __name__ == "__main__":
    sys.exit(main())
