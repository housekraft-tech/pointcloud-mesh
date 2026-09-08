"""Measure the architect's shared wall network in two registered LiDAR walks.

The drawing is authoritative only about topology: which spans are walls and
which room labels own them.  Metric face positions come from local raw-point
fits.  Mujammel is the primary measurement; Koushik is an independent repeat
used to disambiguate furniture from structure and expose SLAM disagreement.

Unlike the older room-edge snapper, this operates on ``global_walls`` so a
partition shared by two rooms is measured once and becomes one modular wall.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))

from recon.metrology import detect_wall_faces, refine_face  # noqa: E402
from snap_to_planes import load_frame, WALL_BAND  # noqa: E402


THICKNESS_MODES_M = np.array([0.195, 0.253], dtype=float)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _rot(points: np.ndarray, quarter_turns: int) -> np.ndarray:
    theta = np.deg2rad(90 * quarter_turns)
    matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return np.asarray(points, dtype=float) @ matrix.T


def load_plan_walls(plan_path: Path, registration_path: Path) -> list[dict]:
    """Transform plan wall rectangles into the stable Koushik/CAGE frame."""
    plan = json.loads(plan_path.read_text())
    placed = json.loads(registration_path.read_text())
    registration = placed["registration"]
    scale = float(plan["dimensions"][0]["scale"])
    shift = np.asarray(registration["t"], dtype=float)
    quarter_turns = int(registration["rot90"])

    walls = []
    for index, wall in enumerate(plan["global_walls"]):
        corners_plan = np.array(
            [
                [wall["x1"] * scale, -wall["y1"] * scale],
                [wall["x2"] * scale, -wall["y1"] * scale],
                [wall["x2"] * scale, -wall["y2"] * scale],
                [wall["x1"] * scale, -wall["y2"] * scale],
            ]
        )
        corners = _rot(corners_plan - shift, quarter_turns)
        lo = corners.min(axis=0)
        hi = corners.max(axis=0)
        extent = hi - lo
        across_index = int(np.argmin(extent))
        along_index = 1 - across_index
        area_names = sorted(
            {
                item.get("area_name", "Unassigned")
                for item in wall.get("areas", [])
                if item.get("area_name")
            }
        )
        walls.append(
            {
                "wall_id": f"wall_{index + 1:02d}",
                "axis": "x" if across_index == 0 else "y",
                "nominal_center_m": float(0.5 * (lo[across_index] + hi[across_index])),
                "nominal_faces_m": [float(lo[across_index]), float(hi[across_index])],
                "nominal_thickness_mm": float(extent[across_index] * 1000),
                "along_m": [float(lo[along_index]), float(hi[along_index])],
                "nominal_length_mm": float(extent[along_index] * 1000),
                "area_names": area_names,
            }
        )
    return walls


def _candidate_dict(face) -> dict:
    return {
        "value_m": float(face.value),
        "sigma_mm": float(face.sigma * 1000),
        "stderr_mm": float(face.stderr * 1000),
        "n_points": int(face.n),
    }


def local_candidates(
    band: np.ndarray,
    wall: dict,
    search_m: float = 0.42,
    along_margin_m: float = 0.18,
) -> tuple[list[dict], int]:
    """Find full-height face candidates only inside one plan wall's footprint."""
    across = 0 if wall["axis"] == "x" else 1
    along = 1 - across
    s0, s1 = wall["along_m"]
    centre = wall["nominal_center_m"]
    mask = (
        (band[:, along] >= s0 - along_margin_m)
        & (band[:, along] <= s1 + along_margin_m)
        & (np.abs(band[:, across] - centre) <= search_m)
    )
    points = band[mask]
    if len(points) < 200:
        return [], int(len(points))

    faces = detect_wall_faces(
        points[:, across],
        points[:, 2],
        bin_m=0.025,
        min_points=80,
        min_span_m=0.75,
        merge_tol=0.010,
        min_bin_frac=0.012,
        windows=(0.028, 0.016, 0.009),
    )
    if not faces:
        return [], int(len(points))

    # Refit against exactly this wall span.  ``detect_wall_faces`` already
    # does this, but the explicit pass makes the final 9 mm inlier window and
    # support count part of the persisted evidence contract.
    output = []
    max_support = max(face.n for face in faces)
    for face in faces:
        refined = refine_face(
            points[:, across], face.value, windows=(0.020, 0.012, 0.008), min_points=70
        )
        if refined is None:
            continue
        if refined.n < max(70, int(0.025 * max_support)):
            continue
        output.append(_candidate_dict(refined))
    output.sort(key=lambda item: item["value_m"])
    return output, int(len(points))


def candidate_pairs(candidates: list[dict], nominal_center: float) -> list[dict]:
    """All plausible two-sided wall measurements, ordered by evidence score."""
    if len(candidates) < 2:
        return []
    max_support = max(item["n_points"] for item in candidates)
    pairs = []
    for i, low in enumerate(candidates):
        for high in candidates[i + 1 :]:
            thickness = high["value_m"] - low["value_m"]
            # Below 80 mm is far more likely to be a wardrobe back, wall
            # panelling, or a niche return than a masonry/partition wall in
            # this survey.  Do not let repeatable built-in joinery become a
            # passing wall-thickness observation.
            if not 0.080 <= thickness <= 0.400:
                continue
            centre = 0.5 * (low["value_m"] + high["value_m"])
            mode_error = float(np.min(np.abs(THICKNESS_MODES_M - thickness)))
            support = (low["n_points"] + high["n_points"]) / max(max_support, 1)
            # Drawing distance is deliberately soft; it locates the wall but
            # may not dictate either finished face.  Thickness modes are also
            # priors, not snapping targets.
            score = (
                abs(centre - nominal_center) / 0.16
                + mode_error / 0.060
                - 0.12 * np.log1p(support)
            )
            pairs.append(
                {
                    "faces_m": [low["value_m"], high["value_m"]],
                    "center_m": float(centre),
                    "thickness_mm": float(thickness * 1000),
                    "score": float(score),
                    "support_points": int(low["n_points"] + high["n_points"]),
                    "face_stderr_mm": [low["stderr_mm"], high["stderr_mm"]],
                }
            )
    pairs.sort(key=lambda item: item["score"])
    return pairs[:12]


def measure_one_scan(
    walls: list[dict],
    las: str,
    max_points: int,
    frame_meta: str,
    align_transform: str | None,
) -> dict:
    xyz, z_band, _centre, stats = load_frame(
        las_name=las,
        max_points=max_points,
        align_transform=align_transform,
        frame_meta=frame_meta,
    )
    floor_seed, ceiling_seed = z_band
    floor = refine_face(
        xyz[xyz[:, 2] < floor_seed + 0.45, 2],
        floor_seed,
        windows=(0.12, 0.05, 0.022, 0.010),
        min_points=200,
    )
    ceiling = refine_face(
        xyz[xyz[:, 2] > ceiling_seed - 0.55, 2],
        ceiling_seed,
        windows=(0.12, 0.05, 0.022, 0.010),
        min_points=200,
    )
    z_floor = floor.value if floor is not None else floor_seed
    band = xyz[
        (xyz[:, 2] >= z_floor + WALL_BAND[0])
        & (xyz[:, 2] <= z_floor + WALL_BAND[1])
    ]
    measured = []
    for wall in walls:
        candidates, local_points = local_candidates(band, wall)
        measured.append(
            {
                "wall_id": wall["wall_id"],
                "local_points": local_points,
                "candidates": candidates,
                "pairs": candidate_pairs(candidates, wall["nominal_center_m"]),
            }
        )
    result = {
        "las": las,
        "align_transform": align_transform,
        "loaded_points": int(len(xyz)),
        "isolation": stats,
        "floor_z_m": None if floor is None else float(floor.value),
        "ceiling_z_m": None if ceiling is None else float(ceiling.value),
        "clear_height_mm": None
        if floor is None or ceiling is None
        else float((ceiling.value - floor.value) * 1000),
        "walls": measured,
    }
    del band, xyz
    gc.collect()
    return result


def estimate_axis_bias(
    walls: list[dict], primary: dict, reference: dict, gate_m: float = 0.080
) -> dict[str, float]:
    """Robust residual translation between already-registered scan frames."""
    values = {"x": [], "y": []}
    for wall, p, r in zip(walls, primary["walls"], reference["walls"]):
        for face in p["candidates"]:
            if not r["candidates"]:
                continue
            nearest = min(r["candidates"], key=lambda q: abs(q["value_m"] - face["value_m"]))
            delta = face["value_m"] - nearest["value_m"]
            if abs(delta) <= gate_m:
                values[wall["axis"]].append(delta)
    return {
        axis: float(np.median(delta)) if delta else 0.0
        for axis, delta in values.items()
    }


def choose_joint(wall: dict, primary: dict, reference: dict, bias_m: float) -> dict:
    """Choose the primary wall pair using agreement with the independent walk."""
    choices = []
    for p in primary["pairs"]:
        for r in reference["pairs"]:
            centre_error = (p["center_m"] - r["center_m"]) - bias_m
            thickness_error = p["thickness_mm"] - r["thickness_mm"]
            score = (
                p["score"]
                + r["score"]
                + abs(centre_error) / 0.030
                + abs(thickness_error) / 18.0
            )
            choices.append((score, p, r, centre_error, thickness_error))

    if choices:
        score, p, r, centre_error, thickness_error = min(choices, key=lambda item: item[0])
        face_delta = [
            1000 * ((p["faces_m"][index] - r["faces_m"][index]) - bias_m)
            for index in (0, 1)
        ]
        repeatability = max(abs(value) for value in face_delta)
        status = "pass" if repeatability <= 10.0 and abs(thickness_error) <= 10.0 else "review"
        return {
            "status": status,
            "evidence": "two_scans_two_faces",
            "faces_m": p["faces_m"],
            "center_m": p["center_m"],
            "thickness_mm": p["thickness_mm"],
            "reference_faces_m": r["faces_m"],
            "reference_thickness_mm": r["thickness_mm"],
            "face_repeatability_mm": face_delta,
            "thickness_repeatability_mm": float(thickness_error),
            "joint_score": float(score),
        }

    if primary["pairs"]:
        p = primary["pairs"][0]
        return {
            "status": "review",
            "evidence": "primary_two_faces",
            "faces_m": p["faces_m"],
            "center_m": p["center_m"],
            "thickness_mm": p["thickness_mm"],
        }

    if primary["candidates"]:
        face = min(
            primary["candidates"],
            key=lambda item: abs(item["value_m"] - wall["nominal_center_m"]),
        )
        return {
            "status": "review",
            "evidence": "primary_single_face",
            "faces_m": [face["value_m"]],
            "center_m": None,
            "thickness_mm": None,
        }

    return {
        "status": "fail",
        "evidence": "no_measured_face",
        "faces_m": [],
        "center_m": None,
        "thickness_mm": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="data/bbox_data.json")
    parser.add_argument("--registration", default="cage_output/polygons/plan_rooms.json")
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument(
        "--primary-transform",
        default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json",
    )
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--max-points", type=int, default=8_000_000)
    parser.add_argument(
        "--out", default="cage_output/metrology/mujammel_plan_walls.json"
    )
    args = parser.parse_args()

    walls = load_plan_walls(_resolve(args.plan), _resolve(args.registration))
    print(f"architect topology: {len(walls)} shared walls", flush=True)
    print(f"measuring primary: {args.primary_las}", flush=True)
    primary = measure_one_scan(
        walls, args.primary_las, args.max_points, args.frame_meta, args.primary_transform
    )
    print(f"measuring reference: {args.reference_las}", flush=True)
    reference = measure_one_scan(
        walls,
        args.reference_las,
        args.max_points,
        args.frame_meta,
        args.reference_transform,
    )

    bias = estimate_axis_bias(walls, primary, reference)
    selected = []
    for wall, p, r in zip(walls, primary["walls"], reference["walls"]):
        chosen = choose_joint(wall, p, r, bias[wall["axis"]])
        selected.append({**wall, **chosen})

    counts = {
        status: sum(item["status"] == status for item in selected)
        for status in ("pass", "review", "fail")
    }
    output = {
        "schema": "shared-plan-wall-metrology-v1",
        "primary": primary,
        "reference": reference,
        "axis_bias_m": bias,
        "summary": {"n_walls": len(walls), **counts},
        "walls": selected,
    }
    destination = _resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"], indent=2))
    print(f"axis bias primary-reference: x={1000*bias['x']:+.1f} mm, "
          f"y={1000*bias['y']:+.1f} mm")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
