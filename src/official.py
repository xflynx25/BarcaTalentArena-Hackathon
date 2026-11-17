"""Bridge to the official repo, which is a read-only submodule at official/.

Its modules import each other by bare name (`from games import ...`, `import possession_metrics`),
so both official/ and official/evaluation/ go on sys.path here, once, and the rest of src/ imports
them through this module:

    from src.official import games, pm
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "official")
EVALUATION = os.path.join(OFFICIAL, "evaluation")
RESULTS_DIR = os.path.join(EVALUATION, "results")
PUBLIC_GT = os.path.join(EVALUATION, "gt", "BAR-ATH_possession_gt_public.csv")

if not os.path.isfile(os.path.join(OFFICIAL, "games.py")):
    raise ImportError("official/ is empty. Run: git submodule update --init")

for path in (OFFICIAL, EVALUATION):
    if path not in sys.path:
        sys.path.insert(0, path)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Label codes used everywhere in src/, identical to the scorer's: H=0, A=1, D=2, X=3.
H, A, D, X = (pm.LABELS.index(name) for name in ("H", "A", "D", "X"))
