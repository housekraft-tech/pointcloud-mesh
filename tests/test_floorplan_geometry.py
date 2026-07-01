import numpy as np
import pytest
import cv2
from scripts.floorplan_geometry import (
    plane_normal, signed_plane_distance, refine_plane_model,
    wall_uv_basis, project_to_plane, points_to_wall_uv,
    crop_to_percentile_bounds,
    points_to_density_image, threshold_density_image,
    extract_wall_segments,
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


# ---------- Phase 1: density image ----------

def test_points_to_density_image_counts_correctly():
    xy = np.array([[0.0, 0.0], [0.01, 0.01], [0.5, 0.5]])
    image, origin = points_to_density_image(xy, cell_size_m=0.02, bounds_min=[0, 0], bounds_max=[1, 1])
    assert image[0, 0] == 2  # first two points land in the same cell
    assert np.allclose(origin, [0.0, 0.0])


def test_threshold_density_image_drops_sparse_cells():
    image = np.zeros((5, 5), dtype=np.uint16)
    image[2, 2] = 5  # dense
    image[0, 0] = 1  # sparse, below threshold
    binary = threshold_density_image(image, min_count=2, morph_kernel=1)
    assert binary[2, 2] == 255
    assert binary[0, 0] == 0


# ---------- Phase 2: wall segment extraction ----------

def test_extract_wall_segments_recovers_rectangle():
    # a filled 3m x 2m rectangle at 20mm cells = 150x100 px, 1px border thickness would
    # be too thin to test area-filter regression; use a 3px-thick rectangle outline
    image = np.zeros((100, 150), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (140, 90), 255, thickness=3)
    segments = extract_wall_segments(image, origin=np.array([0.0, 0.0]), cell_size_m=0.02, epsilon_cells=2.0)
    assert len(segments) >= 4
    lengths = sorted(s["length"] for s in segments)
    # long sides ~ (140-10)*0.02=2.6m, short sides ~ (90-10)*0.02=1.6m
    assert any(abs(l - 2.6) < 0.1 for l in lengths)
    assert any(abs(l - 1.6) < 0.1 for l in lengths)


def test_extract_wall_segments_keeps_single_sided_thin_line():
    """Regression test for the bug found during design validation: a
    single-sided wall face (no opposing face) produces a near-zero-area
    contour that an area-based filter would incorrectly discard."""
    image = np.zeros((50, 300), dtype=np.uint8)
    cv2.line(image, (10, 25), (290, 25), 255, thickness=1)
    segments = extract_wall_segments(image, origin=np.array([0.0, 0.0]), cell_size_m=0.02, epsilon_cells=2.0)
    assert len(segments) >= 1
    assert any(s["length"] > 5.0 for s in segments)  # the ~280px*0.02m=5.6m line survives
