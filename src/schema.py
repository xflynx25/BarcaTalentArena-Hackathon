"""The internal format. Every loader (BAR-ATH, IDSSE, SkillCorner...) produces a Match, so models
never care where the data came from.

Conventions:
  - one row per frame, at the source's frame rate. For BAR-ATH row i is submission row i.
  - coordinates in metres, origin at a pitch corner: x in [0, length], y in [0, width].
  - NaN means "not observed", for the ball and for player slots alike.
  - player slots are tracklets, not people. A slot belongs to one team for the whole match, but
    a player can move between slots (the BAR-ATH tracking has 46 slots for 22 players).
  - label codes are the scorer's: H=0, A=1, D=2, X=3 (see src.official).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOME, AWAY = 0, 1  # values of Match.slot_team


@dataclass
class Match:
    match_id: str
    fps: int
    pitch: tuple[float, float]  # (length, width) in metres
    period: np.ndarray  # (n,) int8: 1, 2, ... inside a half, 0 outside play
    ball: np.ndarray  # (n, 2) float64 metres
    players: np.ndarray  # (n, k, 2) float64 metres
    slot_team: np.ndarray  # (k,) int8, HOME or AWAY
    labels: np.ndarray | None = None  # (n,) int8 label codes, only where a source has them
    # Smart Tagging intervals (src/smart_tags.py: Tag(start, end, code, team, ...)), only where a
    # source has them (BAR-ATH's tactical bundle, B28). Transforms that rebuild a Match drop them.
    tags: list | None = None
    # The ball as the video sees it (src/video, V2): (n, 2) metres, NaN where unseen, and (n,) the
    # linked candidate's logit. Only where a video run (BAR-ATH: src.video.run, local) or the
    # simulator (src/video/simulate.py, external matches) supplied it; transforms drop it like tags.
    video_ball: np.ndarray | None = None
    video_conf: np.ndarray | None = None
    # A second recording's ball on this timeline (src/align.py: attach, V4), NaN where nothing maps
    # or it has no ball; BAR-ATH's broadcast ball on tactical frames. Transforms that rebuild a
    # Match drop it, like tags.
    ball2: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return len(self.period)

    def validate(self) -> "Match":
        """Shape and range checks. Raises with a structural message, never with data values."""
        n, k = self.n_frames, len(self.slot_team)
        checks = [
            (self.ball.shape == (n, 2), "ball shape {} != {}".format(self.ball.shape, (n, 2))),
            (self.players.shape == (n, k, 2),
             "players shape {} != {}".format(self.players.shape, (n, k, 2))),
            (set(np.unique(self.slot_team)) <= {HOME, AWAY}, "slot_team has values other than 0/1"),
            (self.labels is None or self.labels.shape == (n,), "labels shape mismatch"),
            (self.ball2 is None or self.ball2.shape == (n, 2), "ball2 shape mismatch"),
            (self.tags is None or all(0 <= t[0] < t[1] <= n for t in self.tags),
             "a tag interval is empty or outside the frames"),
            (self.video_ball is None or self.video_ball.shape == (n, 2), "video_ball shape mismatch"),
            (self.video_conf is None or self.video_conf.shape == (n,), "video_conf shape mismatch"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError("{}: {}".format(self.match_id, message))
        return self
