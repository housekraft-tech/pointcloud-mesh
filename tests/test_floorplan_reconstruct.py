import numpy as np
from scripts.floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG, _refine_wall
from tests.fixtures import two_room_house


def test_build_floorplan_outputs_recovers_five_walls_and_two_openings():
    pts, _gt = two_room_house()
    manifest, walls = build_floorplan_outputs(pts, DEFAULT_CONFIG)
    assert len(walls) == 5
    assert manifest["wall_count"] == 5
    all_openings = [op for w in walls for op in w.openings]
    assert len(all_openings) == 2
    types = sorted(op["type"] if isinstance(op, dict) else op.type for op in all_openings)
    assert types == ["door", "window"]

    thicknesses_mm = sorted(w.thickness_m * 1000 for w in walls)
    # 4 exterior (~200mm) + 1 partition (~100mm), refined to within a few mm
    assert abs(thicknesses_mm[0] - 100) < 5
    assert all(abs(t - 200) < 5 for t in thicknesses_mm[1:])

    # Clean, well-separated synthetic walls should all reach the highest-confidence
    # two-plane-verified state, each with a real plane_back -- the schema invariant
    # a real-scan bug found violated (thickness_source=="measured" co-occurring
    # with plane_back is None).
    for w in walls:
        assert w.thickness_source == "measured_3d"
        assert w.plane_back is not None


def test_build_floorplan_outputs_handles_zero_walls_gracefully():
    empty_room_pts = np.random.default_rng(0).normal(0, 0.01, size=(50, 3))  # too few/sparse to form any wall
    manifest, walls = build_floorplan_outputs(empty_room_pts, DEFAULT_CONFIG)
    assert manifest["wall_count"] == 0
    assert walls == []


def test_build_floorplan_outputs_honors_z_band_override():
    """find_dense_z_band's auto-detection has been confirmed on real scans to
    not reliably generalize across point-cloud densities/profiles -- z_band_override
    lets a caller bypass it entirely with a known-good manually-determined band.
    Confirms the override is actually used (not silently ignored) by passing an
    obviously-wrong band that should detect nothing, contrasted with the correct
    default-detected result."""
    pts, _gt = two_room_house()

    config_bad_override = dict(DEFAULT_CONFIG)
    config_bad_override["z_band_override"] = (10.0, 12.0)  # nowhere near the real 0.0-2.7m room
    manifest_bad, walls_bad = build_floorplan_outputs(pts, config_bad_override)
    assert manifest_bad["wall_count"] == 0
    assert manifest_bad["floor_z_m"] == 10.0
    assert manifest_bad["ceiling_z_m"] == 12.0

    config_good_override = dict(DEFAULT_CONFIG)
    config_good_override["z_band_override"] = (0.0, 2.7)
    manifest_good, walls_good = build_floorplan_outputs(pts, config_good_override)
    assert manifest_good["wall_count"] == 5


def test_refine_wall_rejects_implausible_thickness_from_non_parallel_refit():
    """Regression test for a real-scan bug: on messy/scattered input (real
    clutter within the search band -- furniture, unrelated structure -- not
    modeled by the clean synthetic fixtures elsewhere in this suite), each
    side of a wall can independently refit to a meaningfully different plane
    orientation, and comparing raw offsets between two non-parallel planes can
    produce a physically-impossible thickness (observed on real data: up to
    ~9.5m). Side A here is a normal wall face (normal ~[0,1,0]); side B is a
    vertical strip clustered at a near-constant X (a planar cluster with
    normal ~[1,0,0], NOT parallel to side A) rather than a real opposite wall
    face -- this must be rejected, falling back to the wall's original
    (already envelope-bounded) coarse thickness rather than reporting the
    implausible refit result."""
    rng = np.random.default_rng(5)
    wall_raw = {
        "p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.0]), "length_m": 5.0,
        "thickness_m": 0.1, "thickness_source": "assumed",
    }

    # side A: a clean wall face near y=-0.05 (normal ~[0,1,0])
    side_a_x = rng.uniform(1.0, 4.0, 3000)
    side_a_z = rng.uniform(0.0, 2.7, 3000)
    side_a = np.column_stack([side_a_x, np.full(3000, -0.05) + rng.normal(0, 0.002, 3000), side_a_z])

    # side B: a vertical strip near a near-constant X (normal ~[1,0,0], NOT
    # parallel to side A's normal) -- not a real opposite wall face.
    side_b_y = rng.uniform(0.1, 0.3, 3000)
    side_b_z = rng.uniform(0.0, 2.7, 3000)
    side_b = np.column_stack([np.full(3000, 2.5) + rng.normal(0, 0.002, 3000), side_b_y, side_b_z])

    full_height = np.vstack([side_a, side_b])
    config = dict(DEFAULT_CONFIG)

    result = _refine_wall(wall_raw, full_height, floor_z=0.0, ceiling_z=2.7, config=config, index=0)
    assert result is not None
    wall, _side_a_pts, _side_b_pts = result

    # Must NOT accept the implausible non-parallel-plane "thickness"; must fall
    # back to the wall's original coarse thickness/source instead.
    assert wall.thickness_source != "measured_3d"
    assert wall.thickness_m == wall_raw["thickness_m"]
    assert wall.plane_back is None


def test_refine_wall_rejects_horizontal_surface_masquerading_as_wall():
    """Regression test for a real-scan bug found via adversarial cross-check:
    a single-sided ('assumed') wall's search band can be dominated by
    horizontal clutter (a floor, furniture top, or ledge) rather than a real
    vertical wall face -- select_wall_band_points only filters by 2D
    perpendicular distance, not by whether the points actually form a
    vertical surface. Confirmed on real data: a wall passed through with
    plane_front normal (0.013, 0.020, -0.9997) -- i.e. essentially
    horizontal, not a wall at all. This must be rejected outright (the wall
    dropped), not reported as a wall with a bogus orientation."""
    rng = np.random.default_rng(9)
    wall_raw = {
        "p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.0]), "length_m": 5.0,
        "thickness_m": 0.1, "thickness_source": "assumed",
    }
    # A horizontal surface: points spread across X (along the "wall") and a
    # narrow Y band (within the search band), but with Z nearly constant --
    # a floor/furniture-top plane, normal ~[0,0,1], not a vertical wall.
    x = rng.uniform(1.0, 4.0, 3000)
    y = rng.uniform(-0.05, 0.05, 3000)
    z = np.full(3000, 1.2) + rng.normal(0, 0.002, 3000)
    full_height = np.column_stack([x, y, z])
    config = dict(DEFAULT_CONFIG)

    result = _refine_wall(wall_raw, full_height, floor_z=0.0, ceiling_z=2.7, config=config, index=0)
    assert result is None
