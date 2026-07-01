"""Diff hand tape-measured ground truth against a floorplan_reconstruct.py manifest.json.

Usage:
    python scripts/validate_measurements.py manifest.json ground_truth.json
"""
import sys
import json
from pathlib import Path

TOLERANCE_MM = {
    "wall_thickness": 3.0,
    "opening_width": 5.0,
    "opening_height": 5.0,
    "groove_position": 3.0,
    "groove_depth": 3.0,
}


def _wall_length_tolerance_mm(expected_mm):
    meters = expected_mm / 1000.0
    return 5.0 + 1.0 * meters


def _tolerance_for(measurement_type, expected_mm):
    if measurement_type == "wall_length":
        return _wall_length_tolerance_mm(expected_mm)
    return TOLERANCE_MM[measurement_type]


def load_ground_truth(path):
    return json.loads(Path(path).read_text())


def _find_actual_mm(manifest, measurement_id, measurement_type):
    for wall in manifest.get("walls", []):
        if measurement_type == "wall_thickness" and wall.get("wall_id") == measurement_id:
            return wall["thickness_m"] * 1000.0
        if measurement_type == "wall_length" and wall.get("wall_id") == measurement_id:
            return wall["length_m"] * 1000.0
        for op in wall.get("openings", []):
            op_id = op["opening_id"] if isinstance(op, dict) else op.opening_id
            if op_id != measurement_id:
                continue
            if measurement_type == "opening_width":
                return (op["width_m"] if isinstance(op, dict) else op.width_m) * 1000.0
            if measurement_type == "opening_height":
                return (op["height_m"] if isinstance(op, dict) else op.height_m) * 1000.0
    return None


def compare_measurements(manifest, ground_truth):
    results = []
    for m in ground_truth["measurements"]:
        actual_mm = _find_actual_mm(manifest, m["id"], m["type"])
        tolerance_mm = _tolerance_for(m["type"], m["expected_mm"])
        if actual_mm is None:
            results.append({
                "id": m["id"], "type": m["type"], "expected_mm": m["expected_mm"],
                "actual_mm": None, "error_mm": None, "tolerance_mm": tolerance_mm,
                "status": "missing",
            })
            continue
        error_mm = abs(actual_mm - m["expected_mm"])
        if error_mm > 2 * tolerance_mm:
            status = "fail"
        elif error_mm > tolerance_mm:
            status = "warn"
        else:
            status = "pass"
        results.append({
            "id": m["id"], "type": m["type"], "expected_mm": m["expected_mm"],
            "actual_mm": actual_mm, "error_mm": error_mm, "tolerance_mm": tolerance_mm,
            "status": status,
        })
    return results


def main(manifest_path, ground_truth_path):
    manifest = json.loads(Path(manifest_path).read_text())
    ground_truth = load_ground_truth(ground_truth_path)
    results = compare_measurements(manifest, ground_truth)

    print(f"{'id':<20} {'type':<16} {'expected':>10} {'actual':>10} {'error':>8} {'tol':>6}  status")
    any_fail = False
    any_missing = False
    for r in results:
        actual_str = f"{r['actual_mm']:.1f}" if r["actual_mm"] is not None else "N/A"
        error_str = f"{r['error_mm']:.1f}" if r["error_mm"] is not None else "N/A"
        print(f"{r['id']:<20} {r['type']:<16} {r['expected_mm']:>10.1f} {actual_str:>10} "
              f"{error_str:>8} {r['tolerance_mm']:>6.1f}  {r['status']}")
        if r["status"] == "fail":
            any_fail = True
        if r["status"] == "missing":
            any_missing = True

    if any_fail or any_missing:
        print("\nRESULT: FAIL -- do not trust this manifest for a production cutlist yet.")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
