"""Models: anything with predict(match) -> (n,) int8 label codes (H=0, A=1, D=2, X=3).

Register new models in MODELS so `python -m src.run --model <name> --param key=value` can build
them. Keep them deterministic: fixed seeds, no wall-clock or ordering dependence.
Give durations in seconds (names ending in `_s`) and convert with
src.models.timing.to_frames(seconds, match.fps), so one config works at every frame rate.
"""

from src.models.deadball import DeadBall
from src.models.emission import BiggerDecoder, BoostedDecoder, TemporalDecoder
from src.models.decoder import SwitchDecoder
from src.models.learned import LearnedDecoder
from src.models.nearest import NearestPlayer
from src.models.spells import SpellFilter

MODELS = {
    "nearest": NearestPlayer,
    "deadball": DeadBall,
    "viterbi": SwitchDecoder,
    "spells": SpellFilter,
    "learned": LearnedDecoder,
    "boosted": BoostedDecoder,  # B31 (a): gradient-boosted trees as learned's emissions
    "temporal": TemporalDecoder,  # B31 (b): a small temporal conv net as learned's emissions
    "bigger": BiggerDecoder,  # B31: both together (the candidate, F128)
}


def build(name: str, **params):
    if name not in MODELS:
        raise KeyError("unknown model {!r}; have {}".format(name, sorted(MODELS)))
    return MODELS[name](**params)
