"""Integration tests: metrology wired into the isolidarflow flow.

The estimator itself is covered by test_metrology*.py. These tests pin the
*wiring* -- that regularize.pair_thickness measures thickness through the
M-estimator (with an uncertainty), and that the pipeline emits per-room clear
spans measured inner-face to inner-face.
"""
import numpy as np
import pytest
from shapely.geometry import box

from scripts.recon import floorplan2d, regularize
from tests.fixtures import two_room_house

MM = 1e-3


def _slab(x_faces, y_range, z_range, per_face=4000, noise=0.002, rng=None):
    """Parallel vertical faces (a wall) at the given x coordinates."""
    rng = rng or np.random.default_rng(0)
    chunks = []
    for xf in x_faces:
        n = per_face
        y = rng.uniform(*y_range, n)
        z = rng.uniform(*z_range, n)
        x = xf + rng.normal(0.0, noise, n)
        chunks.append(np.column_stack([x, y, z]))
    return np.vstack(chunks)


def _steps(back_offset_m, length=5.0, z=(0.0, 2.7)):
    """A minimal two-step run: main face at 0, back face at back_offset_m --
    the topology prior structure.group_wall_runs supplies on real data."""
    from types import SimpleNamespace
    kw = dict(u_min_m=0.0, u_max_m=length, z_min_m=z[0], z_max_m=z[1])
    return [SimpleNamespace(offset_m=0.0, **kw),
            SimpleNamespace(offset_m=back_offset_m, **kw)]


class TestThicknessSeam:
    def test_pair_thickness_measures_via_metrology_with_uncertainty(self):
        # Faces at x=0 and x=+0.20 => 0.20 m thick; steps carry the prior.
        rng = np.random.default_rng(7)
        pts = _slab((0.0, 0.20), (0.0, 5.0), (0.0, 2.7), rng=rng)
        # A skirting board hugging the inner face -- must not drag the estimate.
        skirt = _slab((0.02,), (0.0, 5.0), (0.0, 0.12), per_face=800, rng=rng)
        pts = np.vstack([pts, skirt])

        run = {"p0": (0.0, 0.0), "p1": (0.0, 5.0), "normal": (1.0, 0.0, 0.0),
               "steps": _steps(0.20)}
        out = regularize.pair_thickness([run], pts, default_m=0.10)
        w = out[0]

        assert w["thickness_source"] == "measured"
        assert w["thickness_m"] == pytest.approx(0.20, abs=1.0 * MM)
        # The M-estimator reports its own error bar; the old median path did not.
        assert "thickness_stderr_m" in w
        assert w["thickness_stderr_m"] < 1.0 * MM


class TestThicknessPairSelection:
    def test_relief_step_does_not_win_over_the_back_face(self):
        # A wall truly 0.20 m thick with a shallow relief pilaster at 0.069 m.
        # The steps prior (0.20) is what tells the real back face from the
        # relief, which the raw points alone cannot.
        rng = np.random.default_rng(11)
        front = _slab((0.0,), (0.0, 5.0), (0.0, 2.7), per_face=4000, rng=rng)
        back = _slab((0.20,), (0.0, 5.0), (0.0, 2.7), per_face=4000, rng=rng)
        relief = _slab((0.069,), (2.0, 2.6), (0.0, 2.7), per_face=400, rng=rng)
        pts = np.vstack([front, back, relief])

        run = {"p0": (0.0, 0.0), "p1": (0.0, 5.0), "normal": (1.0, 0.0, 0.0),
               "steps": _steps(0.20)}
        res = regularize._measure_thickness_metrology(
            run, pts, min_thickness_m=0.03, max_thickness_m=0.6)

        assert res is not None
        thickness, _stderr = res
        assert thickness == pytest.approx(0.20, abs=1.0 * MM)

    def test_neighbouring_wall_face_does_not_win_over_own_back_face(self):
        # This wall is 0.20 m thick; a parallel neighbour wall's face sits
        # 0.45 m away (full height, high support). Support-only ranking would
        # grab it; the steps prior keeps the measurement on this wall.
        rng = np.random.default_rng(5)
        front = _slab((0.0,), (0.0, 5.0), (0.0, 2.7), per_face=4000, rng=rng)
        back = _slab((0.20,), (0.0, 5.0), (0.0, 2.7), per_face=4000, rng=rng)
        neighbour = _slab((0.45,), (0.0, 5.0), (0.0, 2.7), per_face=6000, rng=rng)
        pts = np.vstack([front, back, neighbour])

        run = {"p0": (0.0, 0.0), "p1": (0.0, 5.0), "normal": (1.0, 0.0, 0.0),
               "steps": _steps(0.20)}
        res = regularize._measure_thickness_metrology(
            run, pts, min_thickness_m=0.03, max_thickness_m=0.6)

        assert res is not None
        thickness, _stderr = res
        assert thickness == pytest.approx(0.20, abs=1.0 * MM)


class TestClearSpanSeam:
    def test_room_clear_dims_measured_inner_face_to_inner_face(self):
        pts, _ = two_room_house(rng=np.random.default_rng(42))
        # West room: centrelines x=0..3, y=0..5; inner faces 0.1/2.95 & 0.1/4.9
        # => 2.850 x 4.800 m clear.
        west = box(0.0, 0.0, 3.0, 5.0)

        dims = floorplan2d.measure_room_clear_dims(west, pts, z_lo=0.2, z_hi=2.6)

        assert dims["clear_x_m"] == pytest.approx(2.850, abs=1.0 * MM)
        assert dims["clear_y_m"] == pytest.approx(4.800, abs=1.0 * MM)
        assert dims["clear_x_stderr_m"] < 1.0 * MM
        assert dims["clear_y_stderr_m"] < 1.0 * MM
