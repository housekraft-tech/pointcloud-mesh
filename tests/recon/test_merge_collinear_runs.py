"""merge_collinear_runs: length-weighted, order-independent wall positioning.

The group centre becomes a gridline in wall_plan_dimensioned.png and the
placement of the extruded 3D wall, so an error here moves everything keyed off
it. These tests pin the two properties the previous running pairwise mean
(`c = (c + c_new) / 2`) did not have.
"""
import itertools
import random

import pytest

from scripts.recon.regularize import merge_collinear_runs

MM = 1e-3


def _old_running_mean(segs, tol=0.16):
    """The previous implementation, kept as a differential reference."""
    segs = sorted(segs, key=lambda s: (s[2], s[0]))
    out = []
    for a0, a1, c in segs:
        for w in out:
            if abs(w[2] - c) <= tol and a0 <= w[1] + tol and a1 >= w[0] - tol:
                w[0] = min(w[0], a0); w[1] = max(w[1], a1)
                w[2] = (w[2] + c) / 2
                break
        else:
            out.append([a0, a1, c])
    return [(a, b, c) for a, b, c in out]


class TestLengthWeighting:
    def test_a_stub_cannot_drag_a_long_wall(self):
        """A 0.3m stub 100mm off must barely move a 6m wall."""
        segs = [(0.0, 6.0, 0.000), (2.0, 2.3, 0.100)]
        (a0, a1, c), = merge_collinear_runs(segs)
        assert (a0, a1) == (0.0, 6.0)
        assert c == pytest.approx(0.100 * 0.3 / 6.3, abs=1e-9)
        assert c < 5 * MM                            # ~4.8mm

        (_, _, c_old), = _old_running_mean(segs)
        assert c_old == pytest.approx(0.050)         # old: 50mm, the midpoint
        assert abs(c) < 0.1 * abs(c_old)

    def test_equal_length_segments_give_the_plain_mean(self):
        """Three equal segments: the answer is their mean, not c3-weighted."""
        segs = [(0.0, 4.0, 0.00), (0.0, 4.0, 0.10), (0.0, 4.0, 0.20)]
        (_, _, c), = merge_collinear_runs(segs)
        assert c == pytest.approx(0.10, abs=1e-12)

        (_, _, c_old), = _old_running_mean(segs)
        assert c_old == pytest.approx(0.125)         # c1/4 + c2/4 + c3/2
        assert abs(c_old - 0.10) > 20 * MM

    def test_zero_length_segment_is_safe_and_ignored(self):
        segs = [(0.0, 5.0, 0.0), (2.0, 2.0, 0.15)]
        (_, _, c), = merge_collinear_runs(segs)
        assert c == pytest.approx(0.0, abs=1e-6)


class TestOrderIndependence:
    def test_all_permutations_agree(self):
        segs = [(0.0, 6.0, 0.00), (1.0, 3.0, 0.04), (2.0, 2.4, 0.09),
                (0.5, 5.0, 0.02)]
        results = {merge_collinear_runs(list(p))[0][2]
                   for p in itertools.permutations(segs)}
        assert len(results) == 1

    def test_old_implementation_was_deterministic_but_biased(self):
        """The old form sorted by (c, a0) first, so it was NOT sensitive to
        input order -- the defect was the merge-order WEIGHTING, which pulls
        the result toward whichever member is processed last, i.e. the one with
        the largest offset. Pinned here so the distinction stays on record."""
        segs = [(0.0, 4.0, 0.00), (0.0, 4.0, 0.06), (0.0, 4.0, 0.12)]
        results = {_old_running_mean(list(p))[0][2]
                   for p in itertools.permutations(segs)}
        assert len(results) == 1                       # deterministic
        old = results.pop()
        true_mean = 0.06
        assert old > true_mean                         # but biased high
        assert old == pytest.approx(0.075)             # c1/4 + c2/4 + c3/2
        (_, _, new), = merge_collinear_runs(segs)
        assert new == pytest.approx(true_mean, abs=1e-12)

    def test_shuffled_large_input_is_stable(self):
        rng = random.Random(0)
        segs = [(rng.uniform(0, 2), rng.uniform(4, 8), rng.gauss(0, 0.05))
                for _ in range(40)]
        ref = merge_collinear_runs(segs)
        for _ in range(10):
            s = segs[:]
            rng.shuffle(s)
            assert merge_collinear_runs(s) == pytest.approx(ref)


class TestGrouping:
    def test_distinct_walls_stay_separate(self):
        segs = [(0.0, 5.0, 0.0), (0.0, 5.0, 3.0)]
        assert len(merge_collinear_runs(segs)) == 2

    def test_non_overlapping_runs_stay_separate(self):
        """Same offset but a gap wider than tol: two walls, not one."""
        segs = [(0.0, 2.0, 0.0), (5.0, 7.0, 0.0)]
        assert len(merge_collinear_runs(segs)) == 2

    def test_touching_within_tol_merges(self):
        segs = [(0.0, 2.0, 0.0), (2.1, 4.0, 0.0)]
        merged = merge_collinear_runs(segs)
        assert len(merged) == 1
        assert merged[0][0] == 0.0 and merged[0][1] == 4.0

    def test_extent_is_the_union(self):
        segs = [(1.0, 3.0, 0.0), (2.0, 6.0, 0.01), (0.0, 1.5, -0.01)]
        (a0, a1, _), = merge_collinear_runs(segs)
        assert (a0, a1) == (0.0, 6.0)

    def test_output_is_sorted_by_offset(self):
        segs = [(0.0, 5.0, 3.0), (0.0, 5.0, 0.0), (0.0, 5.0, 1.5)]
        assert [c for _, _, c in merge_collinear_runs(segs)] == [0.0, 1.5, 3.0]

    def test_empty(self):
        assert merge_collinear_runs([]) == []
