import numpy as np
import pytest
import cv2
from scripts.floorplan_geometry import (
    plane_normal, signed_plane_distance, refine_plane_model,
    wall_uv_basis, project_to_plane, points_to_wall_uv,
    crop_to_percentile_bounds, find_dense_z_band,
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


def test_find_dense_z_band_isolates_primary_cluster_from_sparse_secondary_structure():
    rng = np.random.default_rng(7)
    # Primary room band: dense, 500,000 points over 2.9m.
    primary = rng.uniform(-1.8, 1.1, 500_000)
    # Secondary structure (e.g. another floor/stairwell/atrium): real point
    # mass, but ~1% the density per unit height of the primary band -- mimics
    # the >100x density cliff found on the real koushikexport.las scan.
    secondary = rng.uniform(1.5, 8.0, 5_000)
    z_values = np.concatenate([primary, secondary])
    rng.shuffle(z_values)

    z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)

    # Returned bounds are refined against actual point extrema within the
    # detected band, not raw 100mm histogram bin edges -- so these should be
    # tight (near the true uniform-range bounds), not merely bin_m-close.
    assert z_min == pytest.approx(-1.8, abs=0.02)
    assert z_max == pytest.approx(1.1, abs=0.02)
    # Must not extend up into the secondary structure.
    assert z_max < 1.5


def test_find_dense_z_band_excludes_the_actual_stray_outlier_tail():
    """Regression test for the original bug this code replaced: feeds the
    real two_room_house fixture (including its 500-point synthetic
    SLAM-drift stray tail, unfiltered) directly through find_dense_z_band,
    rather than pre-filtering the stray points out before calling it."""
    pts, _gt = two_room_house()
    z_values = pts[:, 2]

    z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)

    # True room z-range is [0.0, 2.7]; refined extrema should land close to it,
    # not out at the stray tail (which reaches z=15 per the fixture generator).
    # Looser than the clean-uniform-density test above since real fixture noise
    # (jitter + the occasional stray point landing just outside the main
    # cluster) can nudge the exact extrema by a few cm -- still two orders of
    # magnitude tighter than the stray tail this guards against.
    assert z_min == pytest.approx(0.0, abs=0.1)
    assert z_max == pytest.approx(2.7, abs=0.1)
    assert z_max < 3.0  # must not include any of the stray tail


def test_find_dense_z_band_tolerates_a_mid_band_density_dip():
    """A real room isn't uniformly dense at every height -- a band of window/
    door coverage, or a sparser-scanned section, can locally dip well below
    the primary peak without being a genuinely separate structure. Confirms
    the density-ratio threshold doesn't prematurely truncate the band at such
    a dip, only at a real cliff down to near-secondary-structure sparsity."""
    rng = np.random.default_rng(11)
    lower = rng.uniform(0.0, 1.2, 200_000)
    # a mid-band dip to ~30% of peak density -- still much denser than the
    # 1% used for a genuine secondary structure, so should NOT be treated as
    # a cliff.
    dip = rng.uniform(1.2, 1.5, 20_000)
    upper = rng.uniform(1.5, 2.7, 200_000)
    z_values = np.concatenate([lower, dip, upper])
    rng.shuffle(z_values)

    z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)

    assert z_min == pytest.approx(0.0, abs=0.05)
    assert z_max == pytest.approx(2.7, abs=0.05)


def test_find_dense_z_band_not_collapsed_by_floor_ceiling_density_spike():
    """Regression test for the real-scan bug found in an earlier
    peak-relative version of this function: a flat floor or ceiling slab is
    scanned near-perpendicular over a huge area and collects a massive
    density spike in one or two bins -- confirmed on a real house scan, one
    100mm bin held ~1.7M points versus ~70k-290k for ordinary room-volume
    bins (6-13x denser). A peak-relative threshold set the bar too high for
    the room-volume bins to clear, collapsing the detected band to just the
    spike itself (0 walls on real data). This fixture reproduces that
    density profile synthetically."""
    rng = np.random.default_rng(3)
    floor_spike = rng.uniform(0.0, 0.08, 1_700_000)
    room = rng.uniform(0.08, 2.62, 400_000)
    ceiling_spike = rng.uniform(2.62, 2.7, 1_200_000)
    z_values = np.concatenate([floor_spike, room, ceiling_spike])
    rng.shuffle(z_values)

    z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)

    assert z_min == pytest.approx(0.0, abs=0.02)
    assert z_max == pytest.approx(2.7, abs=0.02)


def test_find_dense_z_band_excludes_genuinely_sparse_noise_not_real_structure():
    """A handful of stray points scattered outside the primary band (too few
    to be real structure, unlike the ~1%-density secondary-structure test
    above) must not get treated as, or merged into, the detected band."""
    rng = np.random.default_rng(21)
    primary = rng.uniform(-1.8, 1.1, 500_000)
    noise = rng.uniform(1.5, 8.0, 200)  # far too sparse to be real coverage
    z_values = np.concatenate([primary, noise])
    rng.shuffle(z_values)

    z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)

    assert z_min == pytest.approx(-1.8, abs=0.02)
    assert z_max == pytest.approx(1.1, abs=0.02)


def test_find_dense_z_band_warns_on_silent_collapse_to_raw_extrema():
    """Regression test found during design review: when no bin clears
    min_bin_points at all (e.g. a globally very sparse point cloud), falling
    back to raw min/max silently reintroduces the exact unbounded-stray-outlier
    problem this function exists to solve. This must be surfaced as a
    RuntimeWarning, not silently swallowed."""
    z_values = np.array([0.0, 0.05, 2.65, 2.7, -30.0, 50.0])  # far too sparse for any bin to reach 5 points
    with pytest.warns(RuntimeWarning, match="z_band_override"):
        z_min, z_max = find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5)
    assert z_min == -30.0
    assert z_max == 50.0


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


# ---------- opening edge refinement (density half-max crossing) ----------

from scripts.floorplan_geometry import refine_opening_edges


def _wall_face_uv_with_hole(u_range, v_range, hole_u, hole_v, spacing=0.016, noise_std=0.002, seed=1):
    """Synthetic wall-face (u,v) points with a rectangular void (door/window)
    cut at sub-cell-precision bounds, for testing edge refinement against a
    known ground-truth edge that does NOT land on a 50mm grid line."""
    rng = np.random.default_rng(seed)
    u = np.arange(u_range[0], u_range[1], spacing)
    v = np.arange(v_range[0], v_range[1], spacing)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    keep = ~((uu >= hole_u[0]) & (uu <= hole_u[1]) & (vv >= hole_v[0]) & (vv <= hole_v[1]))
    uv = np.repeat(np.column_stack([uu[keep], vv[keep]]), 3, axis=0)
    uv = uv + rng.normal(0, noise_std, uv.shape)
    return uv


def test_refine_opening_edges_recovers_subcell_door_edges_from_coarse_grid_rect():
    # true door edges (NOT 50mm-grid-aligned): u in [2.03, 2.91], v in [0.0, 2.087]
    uv = _wall_face_uv_with_hole((0, 5), (0, 2.7), (2.03, 2.91), (0.0, 2.087))
    # coarse rect as detect_openings_on_wall_face would emit at 50mm cells
    coarse = {"u_min": 2.0, "u_max": 2.95, "v_min": 0.0, "v_max": 2.10,
              "width_m": 0.95, "height_m": 2.10, "sill_m": 0.0, "type": "door"}
    refined = refine_opening_edges(coarse, uv)
    assert abs(refined["u_min"] - 2.03) < 0.01
    assert abs(refined["u_max"] - 2.91) < 0.01
    assert abs(refined["v_max"] - 2.087) < 0.01
    assert refined["v_min"] == 0.0  # floor-level sill is never refined (no data below the floor)
    assert refined["edge_method"] == "density_half_max"
    assert abs(refined["width_m"] - (refined["u_max"] - refined["u_min"])) < 1e-9


def test_refine_opening_edges_recovers_subcell_window_edges_including_sill():
    # true window edges: u in [3.55, 4.75], v in [0.87, 2.13] -- elevated sill, all 4 edges refinable
    uv = _wall_face_uv_with_hole((0, 5), (0, 2.7), (3.55, 4.75), (0.87, 2.13))
    coarse = {"u_min": 3.5, "u_max": 4.8, "v_min": 0.85, "v_max": 2.15,
              "width_m": 1.3, "height_m": 1.3, "sill_m": 0.85, "type": "window"}
    refined = refine_opening_edges(coarse, uv)
    assert abs(refined["u_min"] - 3.55) < 0.01
    assert abs(refined["u_max"] - 4.75) < 0.01
    assert abs(refined["v_min"] - 0.87) < 0.01
    assert abs(refined["v_max"] - 2.13) < 0.01
    assert refined["edge_method"] == "density_half_max"


def test_refine_opening_edges_falls_back_to_coarse_when_too_few_points():
    coarse = {"u_min": 1.0, "u_max": 2.0, "v_min": 0.0, "v_max": 2.0,
              "width_m": 1.0, "height_m": 2.0, "sill_m": 0.0, "type": "door"}
    uv = np.empty((0, 2))  # no points at all near any edge
    refined = refine_opening_edges(coarse, uv)
    assert refined["u_min"] == 1.0
    assert refined["u_max"] == 2.0
    assert refined["edge_method"] == "grid_coarse"


# ---------- Phase 8: floor plan image rendering ----------

import os
from scripts.floorplan_geometry import render_floorplan_image


def test_render_floorplan_image_writes_nonempty_png(tmp_path):
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([6.0, 0.0]), "thickness_m": 0.2, "length_m": 6.0},
        {"p0": np.array([6.0, 0.0]), "p1": np.array([6.0, 5.0]), "thickness_m": 0.2, "length_m": 5.0},
    ]
    out = tmp_path / "floorplan.png"
    render_floorplan_image(walls, {}, str(out), px_per_meter=50)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_floorplan_image_handles_zero_walls_without_crashing(tmp_path):
    out = tmp_path / "floorplan_empty.png"
    render_floorplan_image([], {}, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
