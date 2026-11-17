"""B8: merge_short (postprocess) and the `spells` model."""

import numpy as np
import pytest

from src.models import build
from src.models.postprocess import drop_short, merge_short, play_runs
from src.official import A, D, H


def codes(text: str) -> np.ndarray:
    return np.array([{"H": H, "A": A, "D": D}[c] for c in text], dtype=np.int8)


def reference(values, runs, min_team, min_dead, rule, evidence=None):
    """Plain loop version of merge_short's "longer"/"evidence" rules (no keep_margin)."""
    out = values.copy()
    for start, stop in runs:
        segs = []  # [label, length]
        for v in values[start:stop]:
            if segs and segs[-1][0] == v:
                segs[-1][1] += 1
            else:
                segs.append([v, 1])
        while len(segs) > 1:
            short = [(n, sum(s[1] for s in segs[:i]), i) for i, (lab, n) in enumerate(segs)
                     if n < (min_dead if lab == D else min_team)]
            if not short:
                break
            _, a, i = min(short)
            n = segs[i][1]
            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            if right is None or (left is not None and left[0] == right[0]):
                target = i - 1
            elif left is None:
                target = i + 1
            else:
                target = i - 1 if left[1] >= right[1] else i + 1
                if rule == "evidence":
                    col = evidence[start + a:start + a + n]
                    sl, sr = col[:, left[0]].sum(), col[:, right[0]].sum()
                    if sl != sr:
                        target = i - 1 if sl > sr else i + 1
            segs[target][1] += n
            del segs[i]
            # neighbours with the same label join
            j = 0
            while j + 1 < len(segs):
                if segs[j][0] == segs[j + 1][0]:
                    segs[j][1] += segs.pop(j + 1)[1]
                else:
                    j += 1
        out[start:stop] = np.concatenate([np.full(n, lab, np.int8) for lab, n in segs])
    return out


def test_before_with_equal_lengths_is_drop_short():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.integers(0, 3, 300).astype(np.int8), rng.integers(1, 8, 300))
    runs = [(0, 500), (600, len(values))]
    for m in (2, 3, 5):
        assert np.array_equal(merge_short(values, runs, m, m), drop_short(values, runs, m))


def test_separate_lengths_for_team_and_dead():
    # D of 2 survives min_dead=2 while the 2-frame A goes
    assert np.array_equal(merge_short(codes("HHHHDDHHHAAHHHH"), [(0, 15)], 3, 2),
                          codes("HHHHDDHHHHHHHHH"))


def test_longer_neighbour_and_same_label_neighbours():
    # short A between a 2-frame D and a 5-frame H goes to H (the longer); "before" would give D
    values = codes("HHHHHDDAHHHHH")
    assert np.array_equal(merge_short(values, [(0, 13)], 2, 2, "longer"), codes("HHHHHDDHHHHHH"))
    assert np.array_equal(merge_short(values, [(0, 13)], 2, 2, "before"), codes("HHHHHDDDHHHHH"))
    assert np.array_equal(merge_short(codes("AAAHHHHHHDAAAAA"), [(0, 15)], 3, 3, "longer"),
                          codes("AAAHHHHHHHAAAAA"))


def test_evidence_rule_follows_the_evidence():
    values = codes("HHHHHDAAA")
    evidence = np.zeros((9, 3))
    evidence[5, A] = 1.0  # the short D looks like A
    assert np.array_equal(merge_short(values, [(0, 9)], 2, 2, "evidence", evidence),
                          codes("HHHHHAAAA"))
    assert np.array_equal(merge_short(values, [(0, 9)], 2, 2, "longer"), codes("HHHHHHAAA"))


def test_keep_margin_keeps_confident_short_spells():
    values = codes("HHHHHAHHHHH")
    evidence = np.zeros((11, 3))
    evidence[:, H] = 1.0
    evidence[5] = (0.0, 1.0, 0.0)
    for rule in ("before", "longer", "evidence"):
        assert np.array_equal(merge_short(values, [(0, 11)], 3, 3, rule, evidence, 0.5), values)
        assert np.array_equal(merge_short(values, [(0, 11)], 3, 3, rule, evidence, 1.5),
                              codes("H" * 11))


def test_neighbour_rules_match_reference_on_random_codes():
    rng = np.random.default_rng(1)
    for _ in range(20):
        values = np.repeat(rng.integers(0, 3, 150).astype(np.int8), rng.integers(1, 9, 150))
        runs = [(0, len(values) // 2), (len(values) // 2 + 10, len(values))]
        evidence = rng.normal(size=(len(values), 3))
        for rule in ("longer", "evidence"):
            for mt, md in ((3, 3), (2, 5), (6, 2)):
                assert np.array_equal(merge_short(values, runs, mt, md, rule, evidence),
                                      reference(values, runs, mt, md, rule, evidence))


def test_rule_checks():
    with pytest.raises(ValueError):
        merge_short(codes("HHA"), [(0, 3)], 2, 2, "nope")
    with pytest.raises(ValueError):
        merge_short(codes("HHA"), [(0, 3)], 2, 2, "evidence")


def test_spells_reproduces_its_bases(small):
    match = small["match"]
    assert np.array_equal(build("spells", base="deadball", min_team_s=2.5, min_dead_s=2.5).predict(match),
                          build("deadball").predict(match))
    assert np.array_equal(
        build("spells", base="deadball", min_team_s=2.5, min_dead_s=2.5, switch_delay_s=1).predict(match),
        build("deadball", switch_delay_s=1).predict(match))
    assert np.array_equal(build("spells").predict(match), build("viterbi").predict(match))
    out = build("spells", min_team_s=2, min_dead_s=3, rule="evidence", keep_margin=0.5).predict(match)
    assert out.dtype == np.int8 and len(out) == match.n_frames
    runs = play_runs(match.period)
    assert all(set(np.unique(out[a:b])) <= {H, A, D} for a, b in runs)
