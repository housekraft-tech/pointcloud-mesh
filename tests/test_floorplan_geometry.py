import numpy as np
import pytest
from scripts.floorplan_geometry import (
    plane_normal, signed_plane_distance, refine_plane_model,
    wall_uv_basis, project_to_plane, points_to_wall_uv,
    crop_to_percentile_bounds,
)
from tests.fixtures import two_room_house


def test_plane_normal_normalizes():
    n = plane_normal([3.0, 4.0, 0.0, -1.0])
    assert np.allclose(n, [0.6, 0.8, 0.0])


def test_signed_plane_distance_sign_and_magnitude():
    plane = [1.0, 0.0, 0.0, -2.0]  # x = 2 plane
    pts = np.array([[2.0, 0, 0], [2.1, 0, 0], [1.9, 0, 0]])
    dist = signed_plane_distance(pts, plane)
    assert np.allclose(dist, [0.0, 0.1, -0.1], atol=1e-9)


def test_refine_plane_model_recovers_noisy_plane():
    rng = np.random.default_rng(0)
    y = 0.1
    n = 2000
    pts = np.column_stack([
        rng.uniform(0, 5, n),
        np.full(n, y) + rng.normal(0, 0.002, n),
        rng.uniform(0, 2.7, n),
    ])
    coarse = [0.0, 1.0, 0.0, -(y - 0.02)]  # deliberately off by 20mm
    refined = refine_plane_model(pts, coarse)
    assert abs(refined[3] - (-y)) < 0.001  # within 1mm of true offset


def test_wall_uv_basis_orthonormal_and_v_is_up_projected():
    normal = np.array([1.0, 0.0, 0.0])
    u, v = wall_uv_basis(normal)
    assert abs(np.dot(u, normal)) < 1e-9
    assert abs(np.dot(v, normal)) < 1e-9
    assert abs(np.linalg.norm(u) - 1.0) < 1e-9
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert v[2] > 0.99  # for a vertical wall, v should be ~world-up


def test_project_to_plane_puts_points_exactly_on_plane():
    plane = [0.0, 1.0, 0.0, -0.1]  # y = 0.1
    pts = np.array([[1.0, 0.5, 2.0], [3.0, -0.3, 1.0]])
    projected = project_to_plane(pts, plane)
    assert np.allclose(projected[:, 1], 0.1)


def test_points_to_wall_uv_shape_and_v_is_height():
    plane = [1.0, 0.0, 0.0, -3.0]  # x = 3
    pts = np.array([[3.0, 1.0, 2.0], [3.0, 2.0, 2.5]])
    origin = np.array([3.0, 0.0, 0.0])
    u_axis = np.array([0.0, 1.0, 0.0])
    uv = points_to_wall_uv(pts, plane, origin, u_axis)
    assert uv.shape == (2, 2)
    assert np.allclose(uv[:, 0], [1.0, 2.0])  # u = along wall (y here)
    assert np.allclose(uv[:, 1], [2.0, 2.5])  # v = world-up height


# ---------- Phase 0: bounding-box auto-crop ----------

def test_crop_to_percentile_bounds_drops_stray_tail_not_real_room():
    pts, _gt = two_room_house()
    lo, hi, keep_mask, stats = crop_to_percentile_bounds(pts, low_pct=1.0, high_pct=99.0, margin_m=0.5)
    assert stats["dropped_fraction"] < 0.01
    assert lo[0] < 0.0 and hi[0] > 6.0  # room x-extent [0,6] preserved with margin
    assert lo[1] < 0.0 and hi[1] > 5.0  # room y-extent [0,5] preserved with margin
    assert lo[2] < 0.0 and hi[2] > 2.7  # room z-extent [0,2.7] preserved with margin


def test_crop_to_percentile_bounds_raises_on_empty_input():
    with pytest.raises(ValueError):
        crop_to_percentile_bounds(np.empty((0, 3)))
