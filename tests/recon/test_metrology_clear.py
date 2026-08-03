"""clear_between: inner-face-to-inner-face dimensions without a thickness guess.

The house fixture's wall CENTRELINES are x=0, 3, 6 and y=0, 5, with 0.2m
exterior and 0.1m partition walls, so the true clear dimensions are:
    x: 0 -> 3 = 2.850    3 -> 6 = 2.850    0 -> 6 = 5.800
    y: 0 -> 5 = 4.800
These are the numbers wall_plan_dimensioned.png prints.
"""
import numpy as np
import pytest

from scripts.recon.metrology import clear_between, detect_wall_faces
from tests.fixtures import two_room_house

MM = 1e-3
TOL_MM = 1.0


@pytest.fixture(scope="module")
def house():
    pts, gt = two_room_house(rng=np.random.default_rng(42))
    return pts, gt


def _axis(pts, axis, z_lo=0.2, z_hi=2.6):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    band = (z >= z_lo) & (z <= z_hi)
    return (x[band] if axis == "x" else y[band]), z[band]


class TestClearBetween:
    @pytest.mark.parametrize("axis,c_lo,c_hi,truth", [
        ("x", 0.0, 3.0, 2.850),      # west room width
        ("x", 3.0, 6.0, 2.850),      # east room width
        ("x", 0.0, 6.0, 5.800),      # full span, both exterior walls
        ("y", 0.0, 5.0, 4.800),      # room depth
    ])
    def test_matches_ground_truth(self, house, axis, c_lo, c_hi, truth):
        co, zc = _axis(house[0], axis)
        got = clear_between(co, zc, c_lo, c_hi)
        assert got is not None
        clear, err = got
        assert abs(clear - truth) < TOL_MM * MM, f"{clear*1000:.2f} vs {truth*1000:.0f}mm"
        assert err < 0.5 * MM

    def test_beats_the_centreline_minus_thickness_estimate(self, house):
        """The old path: centreline distance minus two percentile-spread halves.

        NOTE this fixture FLATTERS the old estimator -- it contains wall faces
        and nothing else, so the ±0.30m percentile window sees only the two
        faces it is meant to. Even so it lands ~6mm out, because a 10-90
        percentile spread of two noisy surfaces is not their separation. On a
        real scan the same window also holds floor, ceiling and furniture; see
        test_old_estimator_collapses_on_realistic_clutter.
        """
        pts, _ = house
        x = pts[:, 0]

        def gl_thick_old(coord, vals=None):
            vals = x if vals is None else vals
            sel = np.abs(vals - coord) <= 0.30
            return float(np.clip(np.percentile(vals[sel], 90) - np.percentile(vals[sel], 10),
                                 0.06, 0.35))

        old = 3.0 - gl_thick_old(0.0) / 2 - gl_thick_old(3.0) / 2
        new, _ = clear_between(*_axis(pts, "x"), 0.0, 3.0)
        assert abs(new - 2.850) < TOL_MM * MM
        assert abs(old - 2.850) > 5 * MM                     # old: ~6mm out
        assert abs(new - 2.850) < 0.05 * abs(old - 2.850)    # new: >20x better

    def test_old_estimator_collapses_on_realistic_clutter(self, house):
        """Add the floor, ceiling and a bookcase the fixture omits. The old
        window is z-blind, so they land inside it; the new one is not."""
        pts, _ = house
        rng = np.random.default_rng(11)
        n = 20000
        floor = np.column_stack([rng.uniform(0.1, 2.95, n), rng.uniform(0.1, 4.9, n),
                                 rng.normal(0.0, 0.002, n)])
        ceil = floor + np.array([0, 0, 2.7])
        book = np.column_stack([rng.uniform(0.12, 0.45, 6000),      # 330mm deep
                                rng.uniform(1.0, 2.0, 6000),
                                rng.uniform(0.0, 1.8, 6000)])
        cluttered = np.vstack([pts, floor, ceil, book])
        xc, zc = cluttered[:, 0], cluttered[:, 2]

        sel = np.abs(xc - 0.0) <= 0.30
        t_old = float(np.clip(np.percentile(xc[sel], 90) - np.percentile(xc[sel], 10),
                              0.06, 0.35))
        band = (zc >= 0.2) & (zc <= 2.6)
        new, _ = clear_between(xc[band], zc[band], 0.0, 3.0)

        # 200mm wall read as ~231mm: >30mm of thickness error, half of which
        # lands directly on every clear dimension derived from this gridline.
        assert abs(t_old - 0.200) > 20 * MM
        assert abs(new - 2.850) < TOL_MM * MM            # unaffected

    def test_rejects_parallel_walls_as_phantom_faces(self, house):
        """Walls PARALLEL to the axis are full-height too. Without the density
        gate they seed phantom faces and corrupt the inner-face pick."""
        co, zc = _axis(house[0], "x")
        w = (co >= -0.3) & (co <= 0.3)
        ungated = detect_wall_faces(co[w], zc[w], min_bin_frac=0.0)
        gated = detect_wall_faces(co[w], zc[w], min_bin_frac=0.15)
        assert len(gated) == 2                          # exactly -0.1 and +0.1
        assert len(ungated) > len(gated)                # the gate is doing work
        assert abs(gated[0].value + 0.1) < TOL_MM * MM
        assert abs(gated[1].value - 0.1) < TOL_MM * MM

    def test_windows_cannot_overlap_across_a_narrow_gap(self, house):
        """For gridlines closer than 2*half_window the windows must clip, not
        meet -- otherwise each side measures the other side's face."""
        co, zc = _axis(house[0], "x")
        got = clear_between(co, zc, 2.95, 3.05, half_window=0.30)
        assert got is None or got[0] < 0.11

    def test_returns_none_when_an_inner_face_is_unseen(self):
        """A wall whose inner face was never scanned must fail loudly."""
        rng = np.random.default_rng(0)
        co = np.concatenate([0.1 + rng.normal(0, 0.002, 3000)])
        zc = rng.uniform(0, 2.6, co.size)
        assert clear_between(co, zc, 0.0, 3.0) is None

    def test_degenerate(self):
        assert clear_between(np.array([]), np.array([]), 0.0, 3.0) is None
        assert clear_between(np.zeros(100), np.zeros(100), 3.0, 3.0) is None
