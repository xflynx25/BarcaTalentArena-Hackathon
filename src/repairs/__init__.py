"""Run-time repairs of the tracking before the model reads it (wave 5), next to C13's track repair.

Why: automatic tracking fails in ways a model trained on clean external tracking never saw
(official DATA.md: slots appear, vanish and swap identity; the tracker sometimes duplicates people).
C13 (src/tracks.py) links broken tracklets. Wave 5 tests more repairs of that kind, each in its own
file so parallel tasks don't edit the same lines.

A repair is a module `src/repairs/<name>.py` with

    def apply(match, **opts) -> Match

label-free and ball-free unless its docstring says otherwise, deterministic, using only the match's
own data (so it runs the same on BAR-ATH as on an external match). It must return the match object
itself (unchanged) when it finds nothing to repair, so clean matches keep their bytes.

Turned on by the decoder parameter `repairs` ("" = off, the default):

    --param repairs=gkteam                 one repair
    --param "repairs=gkteam/reach=3+ghost" two, applied left to right

Order in LearnedDecoder.prepare(): `repairs` first, then `track_repair` (C13), so a repair that
fixes team labels runs before tracklets are linked within a team. A repair that should run after
C13 (e.g. extending the ends C13 couldn't link) belongs in src/tracks.py as a track_repair option.
For fits and checks the same repairs are not (yet) a degrade load option: score them through the
decoder parameter. A test can register a repair without a file: REGISTRY[name] = apply.
"""

from __future__ import annotations

import importlib

from src.models.cues import _cast

REGISTRY: dict = {}  # name -> apply function; filled lazily from src/repairs/<name>.py


def parse(spec: str) -> list[tuple[str, dict]]:
    """"gkteam/reach=3+ghost" -> [("gkteam", {"reach": 3}), ("ghost", {})]."""
    out, seen = [], set()
    text = str(spec).strip()
    if not text:
        return out
    for item in text.split("+"):
        name, *opts = item.strip().split("/")
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("bad repair name {!r} in repairs={!r}".format(name, spec))
        if name in seen:
            raise ValueError("repair {!r} given twice in repairs={!r}".format(name, spec))
        seen.add(name)
        kwargs = {}
        for opt in opts:
            key, eq, value = opt.partition("=")
            if not eq or not key:
                raise ValueError("repair {!r}: option {!r} needs key=value".format(name, opt))
            kwargs[key.strip()] = _cast(value.strip())
        out.append((name, kwargs))
    return out


def lookup(name: str):
    if name not in REGISTRY:
        try:
            module = importlib.import_module("{}.{}".format(__name__, name))
        except ModuleNotFoundError as exc:
            if exc.name == "{}.{}".format(__name__, name):
                raise ValueError("unknown repair {!r}: no src/repairs/{}.py".format(name, name)) from None
            raise
        REGISTRY[name] = module.apply
    return REGISTRY[name]


def check(spec: str) -> None:
    for name, _ in parse(spec):
        lookup(name)


def apply(match, spec: str):
    """Every repair in the spec, left to right."""
    for name, opts in parse(spec):
        match = lookup(name)(match, **opts)
    return match
