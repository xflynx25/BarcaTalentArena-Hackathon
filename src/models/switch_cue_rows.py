"""Training rows for the switch-timing cue (src/models/switch_cue.py): scripts/fit_switch_cue.py and
scripts/check_switch_cue.py read the same rows. Kept out of switch_cue so the decoder (and with it
the loaders' cache keys, src.evaluate.loader_files) never imports error_analysis. Labels only:
external matches.
"""

from __future__ import annotations

import numpy as np

from src.error_analysis import switch_times, team_ffill
from src.models import switch_cue
from src.models.postprocess import play_runs
from src.official import A, H


LOOKBACK_S = 1.0  # the current team is the labelled team this long before the row


def cue_rows(match) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(index (m,), u (m, 6), y (m,)) for one labelled match: rows every 1 / CUE_RATE_HZ s on frames
    inside a half labelled H or A; the current team is error_analysis.team_ffill(labels) LOOKBACK_S
    earlier (not before the run start); u is the cue toward the other team; y = 1 if
    error_analysis.switch_times has a switch to the other team within +/- CUE_TOL_S. Rows whose u is
    all NaN are dropped. External matches only (it reads the labels)."""
    fps = match.fps
    step = max(1, int(round(fps / switch_cue.CUE_RATE_HZ)))
    runs = play_runs(match.period)
    team = team_ffill(match.labels, runs)
    run_start = np.zeros(match.n_frames, dtype=np.int64)
    for start, stop in runs:
        run_start[start:stop] = start
    index = np.arange(0, match.n_frames, step)
    index = index[(match.period[index] > 0) & np.isin(match.labels[index], (H, A))]
    current = team[np.maximum(index - int(round(LOOKBACK_S * fps)), run_start[index])]
    keep = np.isin(current, (H, A))
    index, current = index[keep], current[keep]
    u_away, u_home = switch_cue.cue_values(match)
    u = np.where((current == H)[:, None], u_away[index], u_home[index])
    other = np.where(current == H, A, H)
    at, new = switch_times(match.labels, runs)
    tol = int(round(switch_cue.CUE_TOL_S * fps))
    y = np.zeros(len(index), dtype=np.int64)
    for code in (H, A):
        frames = np.sort(at[new == code])
        near = (np.searchsorted(frames, index + tol, side="right")
                - np.searchsorted(frames, index - tol, side="left")) > 0
        y |= (near & (other == code)).astype(np.int64)
    known = ~np.isnan(u).all(axis=1)
    return index[known], u[known], y[known]
