"""End-to-end metrology accuracy against the synthetic house ground truth.

This exercises the estimator on the same shape of data the pipeline sees --
two faces per wall, a doorway, a window, and 500 stray SLAM-drift points --
and checks the clear spans the floorplan actually prints.

Ground-truth face coordinates (metres):
    x: -0.1, 0.1 | 2.95, 3.05 | 5.9, 6.1      (0.2 exterior, 0.1 partition)
    y: -0.1, 0.1 | 4.9, 5.1
so the west room measures 2.850 x 4.800 m clear, inner face to inner face.
"""
import numpy as np
import pytest

from scripts.recon.metrology import detect_wall_faces, face_gap
from tests.fixtures import two_room_house

MM = 1e-3
TOL_MM = 1.0        # the mm-level claim this pipeline is allowed to make


@pytest.fixture(scope="module")
def house():
    pts, gt = two_room_house(rng=np.random.default_rng(42))
    return pts, gt


def _strip(pts, axis, perp_c, half=0.12, z_lo=0.2, z_hi=2.6):
    """The measurement strip the pipeline takes through a room centre."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    band = (z >= z_lo) & (z <= z_hi)
    if axis == "x":
        m = band & (np.abs(y - perp_c) <= half)
        return x[m], z[m]
    m = band & (np.abs(x - perp_c) <= half)
    return y[m], z[m]


class TestHouseFaces:
    def test_all_x_faces_recovered_sub_mm(self, house):
        pts, _ = house
        co, zc = _strip(pts, "x", perp_c=1.0)     # y=1.0 clears the doorway
        faces = detect_wall_faces(co, zc)
        got = np.array([f.value for f in faces])
        expect = np.array([-0.1, 0.1, 2.95, 3.05, 5.9, 6.1])
        assert len(got) == len(expect), f"got {got}"
        assert np.max(np.abs(got - expect)) < TOL_MM * MM

    def test_all_y_faces_recovered_sub_mm(self, house):
        pts, _ = house
        co, zc = _strip(pts, "y", perp_c=1.5)
        faces = detect_wall_faces(co, zc)
        got = np.array([f.value for f in faces])
        expect = np.array([-0.1, 0.1, 4.9, 5.1])
        assert len(got) == len(expect), f"got {got}"
        assert np.max(np.abs(got - expect)) < TOL_MM * MM

    @pytest.mark.parametrize("perp_c,axis,lo,hi,truth", [
        (1.0, "x", 0.1, 2.95, 2.850),      # west room width
        (4.0, "x", 3.05, 5.9, 2.850),      # east room width
        (1.5, "y", 0.1, 4.9, 4.800),       # west room depth
        (4.5, "y", 0.1, 4.9, 4.800),       # east room depth
    ])
    def test_clear_spans_match_ground_truth(self, house, perp_c, axis, lo, hi, truth):
        pts, _ = house
        co, zc = _strip(pts, axis, perp_c)
        faces = {round(f.value, 3): f for f in detect_wall_faces(co, zc)}
        a = min(faces.values(), key=lambda f: abs(f.value - lo))
        b = min(faces.values(), key=lambda f: abs(f.value - hi))
        span, err = face_gap(a, b)
        assert abs(span - truth) < TOL_MM * MM, f"{span*1000:.2f}mm vs {truth*1000:.0f}mm"
        assert err < 0.5 * MM

    def test_wall_thickness_from_measured_faces(self, house):
        """Thickness = two fitted faces, replacing the percentile-spread guess."""
        pts, gt = house
        co, zc = _strip(pts, "x", perp_c=1.0)
        faces = detect_wall_faces(co, zc)
        v = sorted(f.value for f in faces)
        partition = v[3] - v[2]
        exterior = v[1] - v[0]
        assert abs(partition - gt["partition_walls"][0]["thickness_m"]) < TOL_MM * MM
        assert abs(exterior - gt["exterior_walls"][0]["thickness_m"]) < TOL_MM * MM

    def test_strip_through_a_doorway_drops_the_partition(self, house):
        """At y=2.5 the door voids the partition below 2.1m -> not a full-height
        face. It must be omitted rather than measured from the header alone."""
        pts, _ = house
        co, zc = _strip(pts, "x", perp_c=2.5)
        got = [f.value for f in detect_wall_faces(co, zc)]
        assert not any(2.9 < v < 3.1 for v in got), f"got {got}"

    def test_stray_drift_points_are_rejected(self, house):
        """The 500 uniform stray points must not manufacture a face."""
        pts, _ = house
        co, zc = _strip(pts, "x", perp_c=1.0)
        for f in detect_wall_faces(co, zc):
            assert f.n > 200, "a face was built from stray points"
            assert f.sigma < 0.006


class TestRepeatability:
    """Precision across independent scans -- the number that separates
    'repeatable to a millimetre' from 'accurate to a millimetre'."""

    def test_span_repeatability_under_independent_noise(self):
        spans = []
        for seed in range(12):
            pts, _ = two_room_house(rng=np.random.default_rng(seed))
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            m = (z >= 0.2) & (z <= 2.6) & (np.abs(y - 1.0) <= 0.12)
            faces = detect_wall_faces(x[m], z[m])
            v = sorted(f.value for f in faces)
            spans.append(v[2] - v[1])          # 0.1 -> 2.95
        spans = np.array(spans)
        assert abs(spans.mean() - 2.850) < 0.5 * MM     # unbiased
        assert spans.std() < 0.3 * MM                   # repeatable
