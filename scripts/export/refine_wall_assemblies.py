"""Refine persistent wall-assembly candidates to dual-walk metric planes.

Columns remain part of their parent wall: a passing column is a bounded interval
where one wall face moves outward.  Beams/soffits and niches use the same face-
profile representation at bounded height ranges.  Coarse 50 mm detections are
never emitted; all release coordinates come from fitted surface/return planes.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from snap_to_planes import load_frame  # noqa: E402
from validate_openings_dual_scan import robust_peak  # noqa: E402


# ``proud_relief_review`` is a bounded local change in the same wall face.  It
# used to stop at coarse detection, which made genuine low/mid-height wall
# profile steps disappear even when both walks saw them.  Refine it like every
# other parent-wall face modifier; the build stage still decides release/review.
STRUCTURAL_KINDS = {
    "embedded_column", "beam_or_soffit", "niche", "proud_relief_review",
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def span(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.quantile(values, 0.95) - np.quantile(values, 0.05))


def supported_peak(
    values: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    seed: float,
    min_a: float,
    min_b: float,
    search_m: float = 0.11,
) -> dict | None:
    fit = robust_peak(values, seed, search_m=search_m, min_points=16)
    if fit is None:
        return None
    selected = np.abs(values - fit["value_m"]) <= 0.024
    a_span, b_span = span(support_a[selected]), span(support_b[selected])
    fit["support_a_mm"] = round(a_span * 1000, 1)
    fit["support_b_mm"] = round(b_span * 1000, 1)
    if a_span < min_a or b_span < min_b:
        return None
    return fit


def reference_face(wall: dict, face_index: int) -> float:
    face = float(wall["faces_m"][face_index])
    deltas = wall.get("face_repeatability_mm", [])
    if len(deltas) > face_index:
        return face - float(deltas[face_index]) / 1000
    if len(deltas) == 1:
        measured = wall.get("measured_face_m")
        if measured is not None and abs(face - float(measured)) < 1e-5:
            return face - float(deltas[0]) / 1000
    return face


def refine_one(
    xyz: np.ndarray,
    candidate: dict,
    wall: dict,
    floor_z: float,
    ceiling_z: float,
    use_reference_face: bool,
) -> dict:
    axis = 0 if wall["axis"] == "x" else 1
    along_axis = 1 - axis
    face_index = int(candidate["face_index"])
    base = reference_face(wall, face_index) if use_reference_face else float(wall["faces_m"][face_index])
    outward = -1.0 if face_index == 0 else 1.0
    direction = -1.0 if candidate["kind"] == "niche" else 1.0
    depth_seed = float(candidate["depth_mm"]) / 1000
    surface_seed = base + outward * direction * depth_seed
    a0, a1 = map(float, candidate["along_m"])
    z0, z1 = map(float, candidate["z_m"])
    if candidate["kind"] == "embedded_column":
        z0, z1 = floor_z, ceiling_z

    cross, along, z = xyz[:, axis], xyz[:, along_axis], xyz[:, 2]
    along_margin = min(0.10, max(0.025, 0.12 * (a1 - a0)))
    z_margin = min(0.12, max(0.025, 0.12 * (z1 - z0)))
    surface_band = ((along >= a0 + along_margin) & (along <= a1 - along_margin)
                    & (z >= z0 + z_margin) & (z <= z1 - z_margin))
    surface = supported_peak(
        cross[surface_band], along[surface_band], z[surface_band], surface_seed,
        min_a=max(0.08, 0.42 * (a1-a0)), min_b=max(0.12, 0.35 * (z1-z0)),
        search_m=0.10,
    )
    if surface is None:
        return {"complete": False, "reason": "feature_surface_not_supported"}
    surface_value = float(surface["value_m"])
    exact_depth = abs(surface_value - base)
    cross_lo, cross_hi = sorted((base, surface_value))
    return_band = ((cross >= cross_lo - 0.035) & (cross <= cross_hi + 0.035)
                   & (z >= z0 + z_margin) & (z <= z1 - z_margin))
    left = supported_peak(
        along[return_band], cross[return_band], z[return_band], a0,
        min_a=max(0.025, 0.35 * exact_depth), min_b=max(0.12, 0.30 * (z1-z0)),
    )
    right = supported_peak(
        along[return_band], cross[return_band], z[return_band], a1,
        min_a=max(0.025, 0.35 * exact_depth), min_b=max(0.12, 0.30 * (z1-z0)),
    )

    horizontal_band = ((cross >= cross_lo - 0.035) & (cross <= cross_hi + 0.035)
                       & (along >= a0 + along_margin) & (along <= a1 - along_margin))
    if candidate["kind"] == "embedded_column":
        bottom = {"value_m": float(floor_z), "source": "floor_plane"}
        top = {"value_m": float(ceiling_z), "source": "ceiling_plane"}
    else:
        bottom = supported_peak(
            z[horizontal_band], cross[horizontal_band], along[horizontal_band], z0,
            min_a=max(0.025, 0.35 * exact_depth), min_b=max(0.08, 0.35 * (a1-a0)),
        )
        top = ({"value_m": float(ceiling_z), "source": "ceiling_plane"}
               if z1 >= ceiling_z - 0.12 else supported_peak(
                   z[horizontal_band], cross[horizontal_band], along[horizontal_band], z1,
                   min_a=max(0.025, 0.35 * exact_depth), min_b=max(0.08, 0.35 * (a1-a0)),
               ))

    # A wall-thickness run may legitimately terminate at the parent wall end;
    # that exact wall corner supplies the missing return.
    wall_start, wall_stop = map(float, wall["span_m"])
    if left is None and abs(a0 - wall_start) <= 0.08:
        left = {"value_m": wall_start, "source": "parent_wall_end"}
    if right is None and abs(a1 - wall_stop) <= 0.08:
        right = {"value_m": wall_stop, "source": "parent_wall_end"}
    complete = all(item is not None for item in (left, right, bottom, top))
    result = {
        "complete": complete,
        "base_face_m": base,
        "surface_face_m": surface_value,
        "depth_mm": round(exact_depth * 1000, 2),
        "surface_fit": surface,
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
    }
    if complete:
        result["along_m"] = [left["value_m"], right["value_m"]]
        result["z_m"] = [bottom["value_m"], top["value_m"]]
        result["width_mm"] = round((right["value_m"] - left["value_m"]) * 1000, 2)
        result["height_mm"] = round((top["value_m"] - bottom["value_m"]) * 1000, 2)
    else:
        result["reason"] = "one_or_more_return_planes_not_supported"
    return result


def combine(candidate: dict, primary: dict, reference: dict, wall: dict,
            axis_bias_mm: dict[str, float], z_bias_mm: float, tolerance_mm: float) -> dict:
    item = {key: value for key, value in candidate.items() if key != "reference"}
    item["primary_fit"], item["reference_fit"] = primary, reference
    if not primary["complete"] or not reference["complete"]:
        item["metric_status"] = "review_incomplete_precise_returns"
        return item
    along_axis = "y" if wall["axis"] == "x" else "x"
    raw = {
        "left": 1000 * (primary["along_m"][0] - reference["along_m"][0]),
        "right": 1000 * (primary["along_m"][1] - reference["along_m"][1]),
        "bottom": 1000 * (primary["z_m"][0] - reference["z_m"][0]),
        "top": 1000 * (primary["z_m"][1] - reference["z_m"][1]),
    }
    residual = {
        "left": raw["left"] - axis_bias_mm[along_axis],
        "right": raw["right"] - axis_bias_mm[along_axis],
        "bottom": raw["bottom"] - z_bias_mm,
        "top": raw["top"] - z_bias_mm,
    }
    dimensions = {
        "width": primary["width_mm"] - reference["width_mm"],
        "height": primary["height_mm"] - reference["height_mm"],
        "depth": primary["depth_mm"] - reference["depth_mm"],
    }
    item["repeat_boundary_delta_mm"] = {key: round(value, 2) for key, value in raw.items()}
    item["repeat_boundary_residual_mm"] = {key: round(value, 2) for key, value in residual.items()}
    item["repeat_dimension_delta_mm"] = {key: round(value, 2) for key, value in dimensions.items()}
    item["measured_along_m"] = [float(np.mean(pair)) for pair in zip(primary["along_m"], reference["along_m"])]
    item["measured_z_m"] = [float(np.mean(pair)) for pair in zip(primary["z_m"], reference["z_m"])]
    item["measured_depth_mm"] = round(float(np.mean([primary["depth_mm"], reference["depth_mm"]])), 2)
    item["measured_surface_face_m"] = float(np.mean([primary["surface_face_m"], reference["surface_face_m"]]))
    errors = list(residual.values()) + list(dimensions.values())
    item["metric_status"] = ("pass_two_walk_10mm" if max(abs(value) for value in errors) <= tolerance_mm
                             else "review_two_walk_precise_disagreement")
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemblies", default="cage_output/metrology/mujammel_wall_assemblies.json")
    parser.add_argument("--walls", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--repeat-graph", default="cage_output/metrology/mujammel_persistent_wall_graph_8m.json")
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument("--primary-transform", default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json")
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--max-points", type=int, default=6_000_000)
    parser.add_argument("--tolerance-mm", type=float, default=10.0)
    parser.add_argument("--out", default="cage_output/metrology/mujammel_wall_assemblies_refined.json")
    args = parser.parse_args()

    coarse = json.loads(resolve(args.assemblies).read_text())
    graph = json.loads(resolve(args.walls).read_text())
    repeat = json.loads(resolve(args.repeat_graph).read_text())
    wall_map = {wall["wall_id"]: wall for wall in graph["walls"]}
    # Coarse 50 mm gating is only candidate generation.  Re-test matched review
    # candidates with the precise return-plane fitter; a grid-edge disagreement
    # must not make a visible wall profile disappear from the final model.
    candidates = [
        item for item in (coarse["features"] + coarse.get("review", []))
        if item["kind"] in STRUCTURAL_KINDS
        and item.get("reference") is not None
        and np.isfinite(float(item.get("depth_mm", np.nan)))
    ]

    print(f"refining {len(candidates)} structural candidates in primary", flush=True)
    primary_xyz, _, _, _ = load_frame(args.primary_las, args.max_points,
                                      args.primary_transform, args.frame_meta)
    primary = [refine_one(primary_xyz, item, wall_map[item["wall_id"]],
                          repeat["primary"]["floor_z_m"], repeat["primary"]["ceiling_z_m"], False)
               for item in candidates]
    del primary_xyz; gc.collect()
    print("refining independent reference returns", flush=True)
    reference_xyz, _, _, _ = load_frame(args.reference_las, args.max_points,
                                        args.reference_transform, args.frame_meta)
    reference = [refine_one(reference_xyz, item["reference"], wall_map[item["wall_id"]],
                            repeat["reference"]["floor_z_m"], repeat["reference"]["ceiling_z_m"], True)
                 for item in candidates]
    del reference_xyz; gc.collect()

    axis_bias_mm = {}
    for axis in ("x", "y"):
        values = [value for wall in graph["walls"] if wall["axis"] == axis
                  for value in wall.get("face_repeatability_mm", [])]
        axis_bias_mm[axis] = float(np.median(values)) if values else 0.0
    z_bias_mm = 1000 * float(np.mean([
        repeat["primary"]["floor_z_m"] - repeat["reference"]["floor_z_m"],
        repeat["primary"]["ceiling_z_m"] - repeat["reference"]["ceiling_z_m"],
    ]))
    items = [combine(item, p, r, wall_map[item["wall_id"]], axis_bias_mm, z_bias_mm,
                     args.tolerance_mm) for item, p, r in zip(candidates, primary, reference)]
    passed = [item for item in items if item["metric_status"].startswith("pass")]
    result = {
        "schema": "dual-walk-wall-assembly-metrology-v2",
        "role": "columns are merged face-profile steps of parent walls",
        "tolerance_mm": args.tolerance_mm,
        "registration_bias_mm": {**axis_bias_mm, "z": z_bias_mm},
        "summary": {"coarse_candidates": len(items), "pass": len(passed),
                    "review": len(items)-len(passed)},
        "features": items,
    }
    destination = resolve(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
