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


# ---------- Phase 3: wall pairing (mutual nearest-neighbor) ----------

from scripts.floorplan_geometry import pair_wall_surfaces, apply_modal_thickness_fallback


def _seg(p0, p1):
    p0, p1 = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
    return {"p0": p0, "p1": p1, "length": float(np.linalg.norm(p1 - p0))}


def test_pair_wall_surfaces_pairs_parallel_segments_within_thickness_envelope():
    # two parallel 3m segments 200mm apart (a real wall)
    segs = [_seg((0, 0), (3, 0)), _seg((0, 0.2), (3, 0.2))]
    walls = pair_wall_surfaces(segs)
    assert len(walls) == 1
    assert walls[0]["thickness_source"] == "measured"
    assert abs(walls[0]["thickness_m"] - 0.2) < 0.01


def test_pair_wall_surfaces_rejects_gap_outside_thickness_envelope():
    # two parallel segments 3m apart (a room width, not a wall) must NOT pair
    segs = [_seg((0, 0), (3, 0)), _seg((0, 3.0), (3, 3.0))]
    walls = pair_wall_surfaces(segs)
    assert all(w["thickness_source"] == "assumed" for w in walls)


def test_apply_modal_thickness_fallback_fills_assumed_walls():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([1.0, 0.0]), "thickness_m": 0.2, "thickness_source": "measured"},
        {"p0": np.array([0.0, 1.0]), "p1": np.array([1.0, 1.0]), "thickness_m": None, "thickness_source": "assumed"},
    ]
    walls = apply_modal_thickness_fallback(walls)
    assert walls[1]["thickness_m"] == 0.2


# ---------- Phase 4: endpoint snapping + short-stub filter ----------

from scripts.floorplan_geometry import snap_wall_endpoints, drop_short_walls


def test_snap_wall_endpoints_merges_nearby_corners():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([3.0, 0.02])},
        {"p0": np.array([3.01, 0.0]), "p1": np.array([3.0, 3.0])},
    ]
    walls, clusters = snap_wall_endpoints(walls, tolerance_m=0.05)
    assert np.allclose(walls[0]["p1"], walls[1]["p0"], atol=1e-9)
    assert "length_m" in walls[0]


def test_snap_wall_endpoints_does_not_collapse_short_walls_own_endpoints():
    # Regression test: a short wall whose own p0/p1 are within tolerance_m of
    # each other must NOT have both endpoints pulled into the same cluster
    # (which would collapse length_m to 0.0). Endpoints from the SAME wall
    # index must never merge with each other, only with other walls' endpoints.
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([0.03, 0.0])},
    ]
    walls, clusters = snap_wall_endpoints(walls, tolerance_m=0.05)
    assert walls[0]["length_m"] > 0.0
    assert np.isclose(walls[0]["length_m"], 0.03, atol=1e-9)


def test_snap_wall_endpoints_does_not_collapse_via_third_party_bridge():
    # Regression test: a wall whose two endpoints are each within tolerance_m
    # of a THIRD wall's shared endpoint (but not within tolerance of each
    # other directly) must not be transitively merged into a single cluster
    # via that bridging point, which would also collapse length_m to 0.0.
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([10.0, 10.0])},  # bridge wall
        {"p0": np.array([-0.04, 0.0]), "p1": np.array([0.04, 0.0])},  # wall under test
    ]
    walls, clusters = snap_wall_endpoints(walls, tolerance_m=0.05)
    assert walls[1]["length_m"] > 0.0


def test_drop_short_walls_removes_corner_stubs_keeps_real_walls():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.0]), "length_m": 5.0},
        {"p0": np.array([5.0, 0.0]), "p1": np.array([5.1, 0.05]), "length_m": 0.11},
    ]
    kept = drop_short_walls(walls, min_length_m=0.3)
    assert len(kept) == 1
    assert kept[0]["length_m"] == 5.0


# ---------- Phase 5: wall dedup/merge ----------

from scripts.floorplan_geometry import merge_duplicate_walls


def test_merge_duplicate_walls_collapses_collinear_overlapping_entries():
    # two near-identical entries for the "same" 5m wall run, one measured one assumed,
    # plus one genuinely different wall
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.02]),
         "thickness_m": 0.2, "thickness_source": "measured", "length_m": 5.0},
        {"p0": np.array([0.02, 0.01]), "p1": np.array([4.98, 0.0]),
         "thickness_m": 0.197, "thickness_source": "assumed", "length_m": 4.96},
        {"p0": np.array([0.0, 5.0]), "p1": np.array([6.0, 5.0]),
         "thickness_m": 0.2, "thickness_source": "measured", "length_m": 6.0},
    ]
    merged = merge_duplicate_walls(walls)
    assert len(merged) == 2
    kept = [w for w in merged if w["length_m"] < 5.5][0]
    assert kept["thickness_source"] == "measured"  # prefers the measured duplicate


def test_merge_duplicate_walls_full_pipeline_recovers_five_walls():
    """End-to-end regression: confirmed this fixture produces 20 raw entries
    without dedup; after merge_duplicate_walls only the 5 real walls (4
    exterior + 1 partition) should remain."""
    from scripts.floorplan_geometry import (
        crop_to_percentile_bounds, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls,
    )
    from tests.fixtures import two_room_house

    pts, _gt = two_room_house()
    _lo, _hi, keep_mask, _stats = crop_to_percentile_bounds(pts)
    cropped = pts[keep_mask]
    ceiling = cropped[(cropped[:, 2] >= 2.4) & (cropped[:, 2] <= 2.6)]
    xy = ceiling[:, :2]
    cell_size = 0.02
    bmin, bmax = xy.min(axis=0) - 0.1, xy.max(axis=0) + 0.1
    image, origin = points_to_density_image(xy, cell_size, bmin, bmax)
    binary = threshold_density_image(image, min_count=2, morph_kernel=3)
    segments = extract_wall_segments(binary, origin, cell_size, epsilon_cells=2.0, min_span_cells=3.0)
    segments = [s for s in segments if s["length"] >= 5 * cell_size]
    walls = pair_wall_surfaces(segments)
    walls = apply_modal_thickness_fallback(walls)
    walls, _clusters = snap_wall_endpoints(walls, tolerance_m=0.08)
    walls = drop_short_walls(walls, min_length_m=0.3)
    assert len(walls) == 20  # confirmed count before dedup -- documents the gap
    merged = merge_duplicate_walls(walls)
    assert len(merged) == 5  # 4 exterior + 1 partition


# ---------- Phase 6: corner-aware point selection + two-pass plane refit ----------

from scripts.floorplan_geometry import (
    select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
)


def test_refine_wall_plane_two_pass_recovers_known_plane():
    rng = np.random.default_rng(1)
    n = 5000
    pts = np.column_stack([
        rng.uniform(0, 6, n),
        np.full(n, 0.1) + rng.normal(0, 0.002, n),
        rng.uniform(0, 2.7, n),
    ])
    coarse = [0.0, 1.0, 0.0, -0.08]  # off by 20mm
    refined = refine_wall_plane_two_pass(pts, coarse)
    assert abs(refined[3] - (-0.1)) < 0.001
    resid = np.abs(signed_plane_distance(pts, refined))
    assert resid.mean() < 0.005


def test_select_wall_band_points_and_refit_corrects_t_junction_contamination():
    """Full-scale regression for the corner-contamination bug found during
    design validation: without corner exclusion this measured ~96.8mm
    (residual mean ~23mm); with select_wall_band_points's corner margin it
    should land within 1mm of the true 100mm partition thickness."""
    pts, _gt = two_room_house()
    lo = np.percentile(pts, 1, axis=0) - 0.5
    hi = np.percentile(pts, 99, axis=0) + 0.5
    cropped = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]

    partition_wall = {
        "p0": np.array([3.0, 0.0]), "p1": np.array([3.0, 5.0]), "length_m": 5.0,
    }
    full_height = cropped[(cropped[:, 0] > 1) & (cropped[:, 0] < 5) &
                           (cropped[:, 1] >= 0) & (cropped[:, 1] <= 5)]
    band_pts = select_wall_band_points(full_height, partition_wall, corner_margin_m=0.5, band_m=0.06)
    assert len(band_pts) > 1000

    d = partition_wall["p1"] - partition_wall["p0"]
    d = d / np.linalg.norm(d)
    normal2d = np.array([-d[1], d[0]])
    mid = band_pts[:, 0] * normal2d[0] + band_pts[:, 1] * normal2d[1]
    med = np.median(mid)
    side_a, side_b = band_pts[mid < med], band_pts[mid >= med]

    coarse_a = [normal2d[0], normal2d[1], 0.0, -np.dot(normal2d, side_a[:, :2].mean(axis=0))]
    coarse_b = [normal2d[0], normal2d[1], 0.0, -np.dot(normal2d, side_b[:, :2].mean(axis=0))]
    refined_a = refine_wall_plane_two_pass(side_a, coarse_a)
    refined_b = refine_wall_plane_two_pass(side_b, coarse_b)
    thickness = abs(refined_a[3] - refined_b[3])
    assert abs(thickness - 0.1) < 0.005  # within 5mm of the true 100mm


# ---------- Phase 7: opening detection (void flood-fill + classification) ----------

from scripts.floorplan_geometry import (
    merge_grid_cells, classify_opening, detect_openings_on_wall_face,
    cross_check_opening_both_faces,
)


def test_classify_opening_thresholds():
    assert classify_opening(1.2, 1.2, 0.9) == "window"
    assert classify_opening(0.9, 2.1, 0.0) == "door"
    assert classify_opening(1.5, 2.2, 0.1) == "balcony_door"


def test_merge_grid_cells_merges_rectangle():
    occupied = {(0, 0), (1, 0), (0, 1), (1, 1)}
    rects = merge_grid_cells(occupied)
    assert rects == [(0, 1, 0, 1)]


def test_detect_openings_on_wall_face_window_case():
    rng = np.random.default_rng(2)
    n_per_cell = 5
    u = np.arange(0, 6, 0.05)
    v = np.arange(0, 2.7, 0.05)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    keep = ~((uu >= 4.0) & (uu <= 5.2) & (vv >= 0.9) & (vv <= 2.1))
    uv = np.column_stack([uu[keep], vv[keep]])
    uv = np.repeat(uv, n_per_cell, axis=0)
    openings = detect_openings_on_wall_face(uv, wall_length_m=6.0, cell_m=0.05)
    assert len(openings) == 1
    op = openings[0]
    assert abs(op["width_m"] - 1.2) < 0.06
    assert abs(op["height_m"] - 1.2) < 0.06
    assert abs(op["sill_m"] - 0.9) < 0.06
    assert op["type"] == "window"


def test_detect_openings_on_wall_face_floor_level_door_case():
    """Regression test for the floor-boundary flood-fill bug: a full-height
    door (sill=0) must still be detected as an enclosed opening."""
    n_per_cell = 5
    u = np.arange(0, 5, 0.05)
    v = np.arange(0, 2.7, 0.05)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    keep = ~((uu >= 2.0) & (uu <= 2.9) & (vv >= 0.0) & (vv <= 2.1))
    uv = np.column_stack([uu[keep], vv[keep]])
    uv = np.repeat(uv, n_per_cell, axis=0)
    openings = detect_openings_on_wall_face(uv, wall_length_m=5.0, cell_m=0.05)
    assert len(openings) == 1
    op = openings[0]
    assert abs(op["sill_m"] - 0.0) < 1e-9
    assert op["type"] == "door"


def test_cross_check_opening_both_faces_rejects_one_sided_occlusion():
    opening = {"u_min": 1.0, "u_max": 2.0, "v_min": 0.5, "v_max": 1.5}
    # other face is fully occupied in that rect => furniture occlusion, not a real opening
    u = np.arange(1.0, 2.0, 0.05)
    v = np.arange(0.5, 1.5, 0.05)
    uu, vv = np.meshgrid(u, v)
    other_face = np.repeat(np.column_stack([uu.ravel(), vv.ravel()]), 5, axis=0)
    assert cross_check_opening_both_faces(opening, other_face) is False
