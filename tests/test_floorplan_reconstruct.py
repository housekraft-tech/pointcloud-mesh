import numpy as np
from scripts.floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG
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


def test_build_floorplan_outputs_handles_zero_walls_gracefully():
    empty_room_pts = np.random.default_rng(0).normal(0, 0.01, size=(50, 3))  # too few/sparse to form any wall
    manifest, walls = build_floorplan_outputs(empty_room_pts, DEFAULT_CONFIG)
    assert manifest["wall_count"] == 0
    assert walls == []
