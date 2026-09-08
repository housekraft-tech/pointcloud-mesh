"""Extract a physical wall graph from two registered LiDAR walks.

The architect drawing contributes topology/location evidence only.  Finished
face coordinates are fitted from Mujammel and independently repeated in
Koushik.  Full-height persistence suppresses ordinary furniture; the drawing
distance suppresses fixed joinery which happens to repeat in both walks.

The result deliberately distinguishes measured two-face wall bodies from
single-face walls whose hidden thickness still has to be inferred or reviewed.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))

from recon.metrology import detect_wall_faces, refine_face  # noqa: E402
from snap_to_planes import load_frame  # noqa: E402


THICKNESS_MODES_M = np.array([0.195, 0.253], dtype=float)
ACROSS_WINDOW_M = 0.011
U_BIN_M = 0.040
Z_BIN_M = 0.100


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _drawing_wall_points(
    image_path: Path, transform_path: Path, frame_meta_path: Path | None = None
) -> np.ndarray:
    """Return long dark drawing strokes transformed to registered scan XY."""
    grey = np.asarray(Image.open(image_path).convert("L"))
    # The apartment drawing occupies the left side; excluding the tower/key
    # plan and text panel prevents irrelevant long strokes becoming priors.
    crop = grey[:, : min(1520, grey.shape[1])]
    dark = (crop < 105).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    )
    vertical = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))
    )
    wall_mask = cv2.bitwise_or(horizontal, vertical)
    yy, xx = np.nonzero(wall_mask)
    # A 2 px stride keeps the KD tree compact without weakening a 250 mm gate.
    keep = np.arange(len(xx)) % 2 == 0
    pixels = np.column_stack([xx[keep], -yy[keep]]).astype(float)

    transform = json.loads(transform_path.read_text())
    centre = np.asarray(transform["drawing_centre_px"], dtype=float)
    local = pixels - centre
    local[:, 0] *= float(transform.get("mirror", 1))
    local *= float(transform["scale_m_per_px"])
    theta = np.deg2rad(float(transform["rotation_deg"]))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    points = local @ rotation.T + np.array([transform["tx"], transform["ty"]])
    if frame_meta_path is not None:
        # drawing_transform.json targets the raw Koushik XY frame, whereas all
        # metrology in this pipeline is in the centred/yaw-corrected CAGE
        # frame.  Apply the identical final frame transform used by load_frame.
        frame = json.loads(frame_meta_path.read_text())
        points -= np.asarray(frame["centre_xy"], dtype=float)
        theta = np.deg2rad(-float(frame["yaw_deg"]))
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        points = points @ rotation.T
    return points


def _runs(active: np.ndarray, origin: float, step: float, min_length: float) -> list[list[float]]:
    padded = np.r_[False, active, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    output = []
    for start, stop in changes.reshape(-1, 2):
        lo = origin + start * step
        hi = origin + stop * step
        if hi - lo >= min_length:
            output.append([float(lo), float(hi)])
    return output


def _support_spans(
    xyz: np.ndarray,
    axis: int,
    face_m: float,
    floor_z: float,
    ceiling_z: float,
    u_limits: tuple[float, float],
) -> tuple[list[list[float]], int, int]:
    """Measure along-wall intervals supported by vertical or header returns."""
    along = 1 - axis
    mask = (
        (np.abs(xyz[:, axis] - face_m) <= ACROSS_WINDOW_M)
        & (xyz[:, 2] >= floor_z + 0.12)
        # Keep the ceiling plane itself out: otherwise a horizontal ceiling
        # return can join disconnected collinear walls across an open room.
        & (xyz[:, 2] <= ceiling_z - 0.08)
    )
    points = xyz[mask]
    if len(points) < 100:
        return [], int(len(points)), 0

    u0, u1 = u_limits
    n_u = int(np.ceil((u1 - u0) / U_BIN_M))
    u_index = np.floor((points[:, along] - u0) / U_BIN_M).astype(int)
    z_index = np.floor((points[:, 2] - floor_z) / Z_BIN_M).astype(int)
    valid = (u_index >= 0) & (u_index < n_u)
    u_index = u_index[valid]
    z_index = z_index[valid]
    points = points[valid]
    if len(points) == 0:
        return [], 0, 0

    pairs = np.unique(np.column_stack([u_index, z_index]), axis=0)
    vertical_bins = np.bincount(pairs[:, 0], minlength=n_u)
    raw_counts = np.bincount(u_index, minlength=n_u)
    near_head = points[:, 2] >= ceiling_z - 0.42
    head_counts = np.bincount(u_index[near_head], minlength=n_u)

    # Five 100 mm vertical cells represents a persistent surface.  A lintel
    # above a door is allowed to keep the physical wall run continuous.
    active = ((vertical_bins >= 5) & (raw_counts >= 8)) | (head_counts >= 6)
    active = binary_closing(active, structure=np.ones(5, dtype=bool))
    spans = _runs(active, u0, U_BIN_M, min_length=0.32)
    return spans, int(len(points)), int(active.sum())


def _fit_scan(
    las: str,
    transform: str | None,
    frame_meta: str,
    max_points: int,
) -> dict:
    xyz, z_seed, _centre, stats = load_frame(
        las_name=las,
        max_points=max_points,
        align_transform=transform,
        frame_meta=frame_meta,
    )
    floor = refine_face(
        xyz[xyz[:, 2] < z_seed[0] + 0.45, 2], z_seed[0],
        windows=(0.12, 0.05, 0.022, 0.010), min_points=200,
    )
    ceiling = refine_face(
        xyz[xyz[:, 2] > z_seed[1] - 0.55, 2], z_seed[1],
        windows=(0.12, 0.05, 0.022, 0.010), min_points=200,
    )
    floor_z = float(floor.value if floor is not None else z_seed[0])
    ceiling_z = float(ceiling.value if ceiling is not None else z_seed[1])
    wall_band = xyz[
        (xyz[:, 2] >= floor_z + 0.18) & (xyz[:, 2] <= ceiling_z - 0.12)
    ]
    frame = json.loads(_resolve(frame_meta).read_text())["variants"]["robust"]
    limits = [tuple(frame["min_coords"]), tuple(frame["max_coords"])]
    inventory: dict[str, list[dict]] = {"x": [], "y": []}
    for axis, label in ((0, "x"), (1, "y")):
        faces = detect_wall_faces(
            wall_band[:, axis], wall_band[:, 2], bin_m=0.050,
            min_points=100, min_span_m=0.90, merge_tol=0.020,
            min_bin_frac=0.012, windows=(0.028, 0.016, 0.009),
        )
        along = 1 - axis
        u_limits = (limits[0][along], limits[1][along])
        for face in faces:
            refined = refine_face(
                wall_band[:, axis], face.value,
                windows=(0.020, 0.012, 0.008), min_points=80,
            )
            if refined is None:
                continue
            spans, support, active_bins = _support_spans(
                xyz, axis, float(refined.value), floor_z, ceiling_z, u_limits
            )
            if not spans:
                continue
            inventory[label].append(
                {
                    "value_m": float(refined.value),
                    "sigma_mm": float(refined.sigma * 1000),
                    "stderr_mm": float(refined.stderr * 1000),
                    "fit_points": int(refined.n),
                    "support_points": support,
                    "active_bins": active_bins,
                    "spans_m": spans,
                }
            )
        inventory[label].sort(key=lambda item: item["value_m"])
    result = {
        "las": las,
        "align_transform": transform,
        "loaded_points": int(len(xyz)),
        "isolation": stats,
        "floor_z_m": floor_z,
        "ceiling_z_m": ceiling_z,
        "clear_height_mm": float((ceiling_z - floor_z) * 1000),
        "faces": inventory,
    }
    del wall_band, xyz
    gc.collect()
    return result


def _overlap(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _plan_distance(
    tree: cKDTree, axis: str, across: float, span: list[float]
) -> float:
    along = np.linspace(span[0], span[1], max(3, int((span[1] - span[0]) / 0.25)))
    if axis == "x":
        sample = np.column_stack([np.full(len(along), across), along])
    else:
        sample = np.column_stack([along, np.full(len(along), across)])
    distance, _ = tree.query(sample, k=1)
    return float(np.median(distance))


def _nearest_reference(face: dict, references: list[dict]) -> dict | None:
    viable = []
    for reference in references:
        delta = face["value_m"] - reference["value_m"]
        if abs(delta) > 0.035:
            continue
        overlap = max(
            (_overlap(a, b) for a in face["spans_m"] for b in reference["spans_m"]),
            default=0.0,
        )
        if overlap >= 0.25:
            viable.append((abs(delta), -overlap, reference))
    return min(viable, key=lambda item: item[:2])[2] if viable else None


def _joint_wall_segments(primary: dict, reference: dict, plan_tree: cKDTree) -> list[dict]:
    output = []
    for axis in ("x", "y"):
        pfaces = primary["faces"][axis]
        rfaces = reference["faces"][axis]
        reference_match = {
            id(face): _nearest_reference(face, rfaces) for face in pfaces
        }
        candidates = []
        for low_index, low in enumerate(pfaces):
            for high_index in range(low_index + 1, len(pfaces)):
                high = pfaces[high_index]
                thickness = high["value_m"] - low["value_m"]
                if not 0.080 <= thickness <= 0.320:
                    continue
                low_ref = reference_match[id(low)]
                high_ref = reference_match[id(high)]
                if low_ref is None or high_ref is None:
                    continue
                repeat = [
                    1000 * (low["value_m"] - low_ref["value_m"]),
                    1000 * (high["value_m"] - high_ref["value_m"]),
                ]
                thickness_repeat = 1000 * (
                    thickness - (high_ref["value_m"] - low_ref["value_m"])
                )
                for low_span in low["spans_m"]:
                    for high_span in high["spans_m"]:
                        overlap = _overlap(low_span, high_span)
                        if overlap < 0.34:
                            continue
                        span = [max(low_span[0], high_span[0]), min(low_span[1], high_span[1])]
                        # A reference face at the same coordinate but in a
                        # different room is not repeatability evidence for
                        # this physical run.  Both repeated faces must cover
                        # this exact along-wall interval.
                        low_reference_overlap = max(
                            (_overlap(span, item) for item in low_ref["spans_m"]),
                            default=0.0,
                        )
                        high_reference_overlap = max(
                            (_overlap(span, item) for item in high_ref["spans_m"]),
                            default=0.0,
                        )
                        if min(low_reference_overlap, high_reference_overlap) < min(0.25, 0.55 * overlap):
                            continue
                        centre = 0.5 * (low["value_m"] + high["value_m"])
                        plan_distance = _plan_distance(plan_tree, axis, centre, span)
                        mode_error = float(np.min(np.abs(THICKNESS_MODES_M - thickness)))
                        score = (
                            plan_distance / 0.16 + mode_error / 0.040
                            + max(abs(repeat[0]), abs(repeat[1])) / 12.0
                            - min(overlap, 5.0) / 8.0
                        )
                        candidates.append(
                            {
                                "axis": axis,
                                "faces_m": [low["value_m"], high["value_m"]],
                                "center_m": centre,
                                "thickness_mm": float(thickness * 1000),
                                "span_m": span,
                                "length_mm": float(overlap * 1000),
                                "plan_distance_mm": float(plan_distance * 1000),
                                "face_repeatability_mm": repeat,
                                "thickness_repeatability_mm": float(thickness_repeat),
                                "mode_error_mm": float(mode_error * 1000),
                                "score": float(score),
                                "face_indices": [low_index, high_index],
                            }
                        )

        # Keep the strongest observation where candidate wall bodies overlap
        # substantially.  The same long face may still support disjoint walls.
        for candidate in sorted(candidates, key=lambda item: item["score"]):
            duplicate = False
            for accepted in output:
                if accepted["axis"] != axis:
                    continue
                centre_delta = abs(candidate["center_m"] - accepted["center_m"])
                common = _overlap(candidate["span_m"], accepted["span_m"])
                short = min(candidate["length_mm"], accepted["length_mm"]) / 1000
                if centre_delta < 0.12 and common > 0.55 * short:
                    duplicate = True
                    break
            if duplicate:
                continue
            repeatability = max(abs(value) for value in candidate["face_repeatability_mm"])
            candidate["status"] = (
                "pass"
                if repeatability <= 10.0
                and abs(candidate["thickness_repeatability_mm"]) <= 10.0
                and candidate["plan_distance_mm"] <= 250.0
                and candidate["mode_error_mm"] <= 35.0
                else "review"
            )
            candidate["evidence"] = "two_scans_two_faces_persistent"
            output.append(candidate)
    output.sort(key=lambda item: (item["axis"], item["center_m"], item["span_m"][0]))
    for index, wall in enumerate(output, 1):
        wall["wall_id"] = f"wall_{index:03d}"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument(
        "--primary-transform",
        default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json",
    )
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--drawing", default="floorplan_original.png")
    parser.add_argument(
        "--drawing-transform",
        default="output2/koushik_all/skeleton_3d/annotated/drawing_transform.json",
    )
    parser.add_argument("--max-points", type=int, default=8_000_000)
    parser.add_argument(
        "--out", default="cage_output/metrology/mujammel_persistent_wall_graph.json"
    )
    args = parser.parse_args()

    print("loading drawing topology prior", flush=True)
    drawing_points = _drawing_wall_points(
        _resolve(args.drawing), _resolve(args.drawing_transform), _resolve(args.frame_meta)
    )
    plan_tree = cKDTree(drawing_points)
    print(f"drawing prior: {len(drawing_points):,} transformed stroke samples", flush=True)
    print(f"measuring primary {args.primary_las}", flush=True)
    primary = _fit_scan(
        args.primary_las, args.primary_transform, args.frame_meta, args.max_points
    )
    print(f"measuring reference {args.reference_las}", flush=True)
    reference = _fit_scan(
        args.reference_las, args.reference_transform, args.frame_meta, args.max_points
    )
    walls = _joint_wall_segments(primary, reference, plan_tree)
    summary = {
        "n_walls": len(walls),
        "pass": sum(wall["status"] == "pass" for wall in walls),
        "review": sum(wall["status"] == "review" for wall in walls),
        "primary_faces": {axis: len(primary["faces"][axis]) for axis in ("x", "y")},
        "reference_faces": {axis: len(reference["faces"][axis]) for axis in ("x", "y")},
    }
    result = {
        "schema": "persistent-dual-scan-wall-graph-v1",
        "drawing_role": "topology/location prior only; no metric snapping",
        "primary": primary,
        "reference": reference,
        "summary": summary,
        "walls": walls,
    }
    destination = _resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
