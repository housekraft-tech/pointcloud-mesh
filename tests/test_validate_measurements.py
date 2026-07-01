import json
from scripts.validate_measurements import compare_measurements

TOLERANCES_MM = {
    "wall_thickness": 3.0,
    "opening_width": 5.0,
    "opening_height": 5.0,
    "wall_length": None,  # computed as 5 + 1*meters
}


def test_compare_measurements_flags_pass_warn_fail():
    manifest = {
        "walls": [
            {"wall_id": "wall_000", "thickness_m": 0.1003, "length_m": 5.001,
             "openings": [{"opening_id": "wall_000_op_00", "width_m": 0.905, "height_m": 2.108}]},
        ]
    }
    ground_truth = {
        "measurements": [
            {"id": "wall_000", "type": "wall_thickness", "expected_mm": 100.0},
            {"id": "wall_000", "type": "wall_length", "expected_mm": 5000.0},
            {"id": "wall_000_op_00", "type": "opening_width", "expected_mm": 900.0},
            {"id": "wall_000_op_00", "type": "opening_height", "expected_mm": 2100.0},
        ]
    }
    results = compare_measurements(manifest, ground_truth)
    by_id_type = {(r["id"], r["type"]): r for r in results}
    assert by_id_type[("wall_000", "wall_thickness")]["status"] == "pass"  # 0.3mm error, tol 3mm
    assert by_id_type[("wall_000_op_00", "opening_width")]["status"] == "pass"  # 5mm error, tol 5mm -> exactly at tol, still pass
    assert all(r["error_mm"] >= 0 for r in results)


def test_compare_measurements_hard_fails_over_double_tolerance():
    manifest = {"walls": [{"wall_id": "wall_000", "thickness_m": 0.115, "length_m": 5.0, "openings": []}]}
    ground_truth = {"measurements": [{"id": "wall_000", "type": "wall_thickness", "expected_mm": 100.0}]}
    results = compare_measurements(manifest, ground_truth)
    assert results[0]["status"] == "fail"  # 15mm error > 2x3mm tolerance
