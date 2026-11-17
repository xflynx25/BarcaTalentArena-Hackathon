"""Write a synthetic BAR-ATH bundle (official formats, fake football). See src/synthetic.py.

    .venv/bin/python scripts/make_synthetic_bar_ath.py                 # data/synthetic/mtah2027
    .venv/bin/python scripts/make_synthetic_bar_ath.py --small --out /tmp/syn
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import synthetic  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("data", "synthetic", "mtah2027"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true", help="4,000 frames instead of 199,319")
    args = parser.parse_args()

    started = time.time()
    spec = synthetic.Spec.small(args.seed) if args.small else synthetic.Spec(seed=args.seed)
    info = synthetic.write_bundle(args.out, spec)
    print("wrote synthetic bundle to {} ({} frames, seed {}) in {:.0f} s".format(
        args.out, spec.n_frames, spec.seed, time.time() - started))
    print("ground truth: {}".format(info["gt"]))
