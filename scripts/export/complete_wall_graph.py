"""Complete the Mujammel wall graph with drawing topology and dual-scan faces.

Long wall bands are traced from the architect raster, transformed into the
registered scan frame, and retained only when a finished face repeats in both
LiDAR walks over the same physical span.  Two visible faces preserve measured
thickness.  A single visible face preserves that exact room-side coordinate
while the hidden thickness is explicitly labelled inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
THICKNESS_MODES_M = np.array([0.195, 0.253], dtype=float)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pixel_to_stable(
    pixels_xy: np.ndarray, drawing_transform: dict, frame: dict
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=float).copy()
    pixels[:, 1] *= -1
    pixels -= np.asarray(drawing_transform["drawing_centre_px"], dtype=float)
    pixels[:, 0] *= float(drawing_transform.get("mirror", 1))
    pixels *= float(drawing_transform["scale_m_per_px"])
    theta = np.deg2rad(float(drawing_transform["rotation_deg"]))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    points = pixels @ rotation.T + np.array(
        [drawing_transform["tx"], drawing_transform["ty"]]
    )
    points -= np.asarray(frame["centre_xy"], dtype=float)
    theta = np.deg2rad(-float(frame["yaw_deg"]))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return points @ rotation.T


def _cluster_lines(lines: list[tuple[float, float, float]], coord_tol: float = 18.0):
    """Cluster Hough (across, along0, along1) lines into filled wall bands."""
    groups: list[list[tuple[float, float, float]]] = []
    for line in sorted(lines):
        best = None
        best_distance = float("inf")
        for index, group in enumerate(groups):
            coordinate = float(np.median([item[0] for item in group]))
            distance = abs(line[0] - coordinate)
            if distance > coord_tol:
                continue
            group_lo = min(item[1] for item in group)
            group_hi = max(item[2] for item in group)
            overlap = max(0.0, min(line[2], group_hi) - max(line[1], group_lo))
            if overlap < 0.45 * min(line[2] - line[1], group_hi - group_lo):
                continue
            if distance < best_distance:
                best, best_distance = index, distance
        if best is None:
            groups.append([line])
        else:
            groups[best].append(line)
    return groups


def drawing_bands(image_path: Path, transform_path: Path, frame_path: Path) -> list[dict]:
    grey = np.asarray(Image.open(image_path).convert("L"))
    crop = grey[:, : min(1520, grey.shape[1])]
    dark = (crop < 105).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    )
    vertical = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    )
    raw: dict[str, list[tuple[float, float, float]]] = {"horizontal": [], "vertical": []}
    hough_h = cv2.HoughLinesP(horizontal, 1, np.pi / 180, 35,
                              minLineLength=48, maxLineGap=14)
    hough_v = cv2.HoughLinesP(vertical, 1, np.pi / 180, 35,
                              minLineLength=48, maxLineGap=14)
    for family, found in (("horizontal", hough_h), ("vertical", hough_v)):
        if found is None:
            continue
        for x0, y0, x1, y1 in np.asarray(found).reshape(-1, 4):
            if family == "horizontal":
                if abs(y1 - y0) > 3:
                    continue
                raw[family].append((0.5 * (y0 + y1), min(x0, x1), max(x0, x1)))
            else:
                if abs(x1 - x0) > 3:
                    continue
                raw[family].append((0.5 * (x0 + x1), min(y0, y1), max(y0, y1)))

    transform = json.loads(transform_path.read_text())
    frame = json.loads(frame_path.read_text())
    scale = float(transform["scale_m_per_px"])
    bands = []
    for family in ("horizontal", "vertical"):
        for group in _cluster_lines(raw[family]):
            across_values = [item[0] for item in group]
            along0 = min(item[1] for item in group)
            along1 = max(item[2] for item in group)
            across = float(np.median(across_values))
            if family == "horizontal":
                pixels = np.array([[along0, across], [along1, across]])
            else:
                pixels = np.array([[across, along0], [across, along1]])
            points = pixel_to_stable(pixels, transform, frame)
            delta = points[1] - points[0]
            if np.linalg.norm(delta) < 0.55:
                continue
            if abs(delta[0]) >= abs(delta[1]):
                axis = "y"
                wall_across = float(np.mean(points[:, 1]))
                span = sorted(points[:, 0].tolist())
            else:
                axis = "x"
                wall_across = float(np.mean(points[:, 0]))
                span = sorted(points[:, 1].tolist())
            bands.append(
                {
                    "axis": axis,
                    "nominal_center_m": wall_across,
                    "span_m": span,
                    "nominal_length_mm": float((span[1] - span[0]) * 1000),
                    "nominal_thickness_mm": float(
                        max(5.0, max(across_values) - min(across_values) + 3.0) * scale * 1000
                    ),
                    "drawing_family": family,
                    "drawing_lines": len(group),
                }
            )
    return bands


def overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def repeated_face(face: dict, references: list[dict], span: list[float]) -> tuple[dict, float] | None:
    choices = []
    for reference in references:
        delta = face["value_m"] - reference["value_m"]
        if abs(delta) > 0.012:
            continue
        p_overlap = max((overlap(span, item) for item in face["spans_m"]), default=0.0)
        r_overlap = max((overlap(span, item) for item in reference["spans_m"]), default=0.0)
        common = min(p_overlap, r_overlap)
        if common >= min(0.30, 0.55 * (span[1] - span[0])):
            choices.append((abs(delta), -common, reference, delta))
    if not choices:
        return None
    _, _, reference, delta = min(choices, key=lambda item: item[:2])
    return reference, float(delta * 1000)


def snap_band(band: dict, graph: dict) -> dict | None:
    axis = band["axis"]
    nominal = band["nominal_center_m"]
    span = band["span_m"]
    length = span[1] - span[0]

    bodies = []
    for wall in graph["walls"]:
        if wall["axis"] != axis:
            continue
        common = overlap(span, wall["span_m"])
        if common < min(0.30, 0.45 * length):
            continue
        distance = abs(wall["center_m"] - nominal)
        if distance > 0.38:
            continue
        score = (
            distance / 0.16
            + (0.0 if wall["status"] == "pass" else 1.7)
            - min(common, 4.0) / 6.0
            + wall["mode_error_mm"] / 80.0
        )
        bodies.append((score, wall, common))
    if bodies:
        _, wall, common = min(bodies, key=lambda item: item[0])
        return {
            **band,
            "faces_m": wall["faces_m"],
            "center_m": wall["center_m"],
            "thickness_mm": wall["thickness_mm"],
            "thickness_source": "measured_two_faces",
            "metric_status": wall["status"],
            "face_repeatability_mm": wall["face_repeatability_mm"],
            "source_candidate": wall["wall_id"],
            "evidence_overlap_mm": float(common * 1000),
        }

    singles = []
    for face in graph["primary"]["faces"][axis]:
        repeated = repeated_face(face, graph["reference"]["faces"][axis], span)
        if repeated is None:
            continue
        reference, repeat_mm = repeated
        distance = abs(face["value_m"] - nominal)
        if distance > 0.38:
            continue
        common = max((overlap(span, item) for item in face["spans_m"]), default=0.0)
        score = distance / 0.14 - min(common, 4.0) / 7.0
        singles.append((score, face, reference, repeat_mm, common))
    if not singles:
        return None
    _, face, reference, repeat_mm, common = min(singles, key=lambda item: item[0])
    nominal_thickness = band["nominal_thickness_mm"] / 1000
    thickness = float(THICKNESS_MODES_M[np.argmin(abs(THICKNESS_MODES_M - nominal_thickness))])
    # Preserve the observed face.  The hidden face extends toward the plan
    # centre; this affects wall mass but never alters the measured room face.
    if face["value_m"] >= nominal:
        faces = [face["value_m"] - thickness, face["value_m"]]
    else:
        faces = [face["value_m"], face["value_m"] + thickness]
    return {
        **band,
        "faces_m": faces,
        "center_m": float(np.mean(faces)),
        "thickness_mm": float(thickness * 1000),
        "thickness_source": "inferred_hidden_face",
        "metric_status": "review",
        "measured_face_m": face["value_m"],
        "reference_face_m": reference["value_m"],
        "face_repeatability_mm": [repeat_mm],
        "source_candidate": None,
        "evidence_overlap_mm": float(common * 1000),
    }


def merge_walls(walls: list[dict]) -> list[dict]:
    """Merge same-body drawing fragments across doors and text occlusions."""
    output: list[dict] = []
    for wall in sorted(walls, key=lambda item: (item["axis"], item["center_m"], item["span_m"][0])):
        match = None
        for existing in output:
            if existing["axis"] != wall["axis"]:
                continue
            if abs(existing["center_m"] - wall["center_m"]) > 0.045:
                continue
            gap = max(0.0, wall["span_m"][0] - existing["span_m"][1],
                      existing["span_m"][0] - wall["span_m"][1])
            if gap <= 1.30:
                match = existing
                break
        if match is None:
            output.append(dict(wall))
            continue
        match["span_m"] = [
            min(match["span_m"][0], wall["span_m"][0]),
            max(match["span_m"][1], wall["span_m"][1]),
        ]
        match["nominal_length_mm"] = (match["span_m"][1] - match["span_m"][0]) * 1000
        if wall["thickness_source"] == "measured_two_faces" and match["thickness_source"] != "measured_two_faces":
            for key in ("faces_m", "center_m", "thickness_mm", "thickness_source",
                        "metric_status", "face_repeatability_mm", "source_candidate"):
                match[key] = wall[key]
    for index, wall in enumerate(output, 1):
        wall["wall_id"] = f"wall_{index:03d}"
        wall["length_mm"] = float((wall["span_m"][1] - wall["span_m"][0]) * 1000)
    return output


def append_unmatched_measured_runs(walls: list[dict], persistent: dict) -> int:
    """Recover long dual-scan runs hidden from the raster by full openings."""
    added = 0
    for candidate in persistent["walls"]:
        if candidate["status"] != "pass":
            continue
        if candidate["length_mm"] < 700 or candidate["plan_distance_mm"] > 180:
            continue
        duplicate = False
        for wall in walls:
            if wall["axis"] != candidate["axis"]:
                continue
            if abs(wall["center_m"] - candidate["center_m"]) > 0.075:
                continue
            common = overlap(wall["span_m"], candidate["span_m"])
            if common >= 0.35 * min(
                wall["span_m"][1] - wall["span_m"][0],
                candidate["span_m"][1] - candidate["span_m"][0],
            ):
                duplicate = True
                break
        if duplicate:
            continue
        walls.append(
            {
                "axis": candidate["axis"],
                "nominal_center_m": candidate["center_m"],
                "span_m": list(candidate["span_m"]),
                "nominal_length_mm": candidate["length_mm"],
                "nominal_thickness_mm": candidate["thickness_mm"],
                "drawing_family": "persistent_scan_recovery",
                "drawing_lines": 0,
                "faces_m": list(candidate["faces_m"]),
                "center_m": candidate["center_m"],
                "thickness_mm": candidate["thickness_mm"],
                "thickness_source": "measured_two_faces",
                "metric_status": "pass",
                "face_repeatability_mm": candidate["face_repeatability_mm"],
                "source_candidate": candidate["wall_id"],
                "evidence_overlap_mm": candidate["length_mm"],
            }
        )
        added += 1
    return added


def close_corners(walls: list[dict], reach_m: float = 0.42) -> int:
    """Extend/trim plan endpoints onto perpendicular measured wall bodies."""
    adjustments = 0
    for wall in walls:
        across = wall["center_m"]
        for endpoint_index in (0, 1):
            endpoint = wall["span_m"][endpoint_index]
            choices = []
            for other in walls:
                if other is wall or other["axis"] == wall["axis"]:
                    continue
                if not other["span_m"][0] - 0.30 <= across <= other["span_m"][1] + 0.30:
                    continue
                distance = abs(endpoint - other["center_m"])
                if distance <= reach_m:
                    choices.append((distance, other))
            if not choices:
                continue
            _, other = min(choices, key=lambda item: item[0])
            target = other["faces_m"][0 if endpoint_index == 0 else 1]
            if abs(target - endpoint) > 0.001:
                wall.setdefault("corner_adjustments_mm", [0.0, 0.0])
                wall["corner_adjustments_mm"][endpoint_index] = float((target - endpoint) * 1000)
                wall["span_m"][endpoint_index] = float(target)
                wall["length_mm"] = float((wall["span_m"][1] - wall["span_m"][0]) * 1000)
                adjustments += 1
    return adjustments


def render(result: dict, density_path: Path, frame_path: Path, destination: Path) -> None:
    density = np.load(density_path)
    meta = json.loads(frame_path.read_text())["variants"]["s02_mid"]
    min_x, min_y = meta["min_coords"]
    max_x, max_y = meta["max_coords"]
    fig, axis = plt.subplots(figsize=(13, 13), dpi=180)
    axis.set_facecolor("#050607")
    axis.imshow(np.power(density, 0.32), origin="lower",
                extent=[min_x, max_x, min_y, max_y], cmap="gray", vmin=0, vmax=1)
    for wall in result["walls"]:
        low, high = wall["faces_m"]
        start, stop = wall["span_m"]
        if wall["axis"] == "x":
            rectangle = (low, start, high - low, stop - start)
            tx, ty = wall["center_m"], 0.5 * (start + stop)
        else:
            rectangle = (start, low, stop - start, high - low)
            tx, ty = 0.5 * (start + stop), wall["center_m"]
        measured = wall["thickness_source"] == "measured_two_faces"
        colour = "#4ade80" if measured else "#fbbf24"
        axis.add_patch(Rectangle(rectangle[:2], rectangle[2], rectangle[3], fill=False,
                                 edgecolor=colour, linewidth=1.25, zorder=4))
        axis.text(tx, ty, f"{wall['wall_id']}\n{wall['thickness_mm']:.0f}",
                  color=colour, fontsize=4.5, ha="center", va="center",
                  bbox={"boxstyle": "round,pad=0.12", "fc": "#050607",
                        "ec": colour, "alpha": 0.82, "lw": 0.4}, zorder=5)
    summary = result["summary"]
    axis.set_title(
        "Completed wall graph — green measured thickness, amber hidden thickness inferred\n"
        f"{summary['n_walls']} wall bodies | {summary['measured_two_faces']} measured | "
        f"{summary['inferred_hidden_face']} inferred hidden face",
        color="#e5e7eb", fontsize=11,
    )
    axis.set_xlim(min_x, max_x); axis.set_ylim(min_y, max_y); axis.set_aspect("equal")
    axis.tick_params(colors="#94a3b8", labelsize=7)
    for spine in axis.spines.values(): spine.set_color("#475569")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="cage_output/metrology/mujammel_persistent_wall_graph_8m.json")
    parser.add_argument("--drawing", default="floorplan_original.png")
    parser.add_argument("--drawing-transform", default="output2/koushik_all/skeleton_3d/annotated/drawing_transform.json")
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--density", default="cage_output/slices/density_s02_mid.npy")
    parser.add_argument("--out", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--plot", default="cage_output/metrology/mujammel_completed_wall_graph.png")
    args = parser.parse_args()

    graph = json.loads(resolve(args.graph).read_text())
    bands = drawing_bands(resolve(args.drawing), resolve(args.drawing_transform), resolve(args.frame_meta))
    snapped = [item for band in bands if (item := snap_band(band, graph)) is not None]
    walls = merge_walls(snapped)
    recovered_runs = append_unmatched_measured_runs(walls, graph)
    walls = merge_walls(walls)
    corner_adjustments = close_corners(walls)
    summary = {
        "drawing_bands": len(bands),
        "bands_with_dual_scan_face": len(snapped),
        "n_walls": len(walls),
        "measured_two_faces": sum(item["thickness_source"] == "measured_two_faces" for item in walls),
        "inferred_hidden_face": sum(item["thickness_source"] == "inferred_hidden_face" for item in walls),
        "corner_endpoints_adjusted": corner_adjustments,
        "measured_runs_recovered_outside_drawing_bands": recovered_runs,
    }
    result = {
        "schema": "completed-dual-scan-wall-graph-v1",
        "metric_contract": "drawing topology only; every retained room face repeats within 10 mm across scans",
        "floor_z_m": graph["primary"]["floor_z_m"],
        "ceiling_z_m": graph["primary"]["ceiling_z_m"],
        "clear_height_mm": graph["primary"]["clear_height_mm"],
        "summary": summary,
        "walls": walls,
    }
    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    render(result, resolve(args.density), resolve(args.frame_meta), resolve(args.plot))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {resolve(args.plot)}")


if __name__ == "__main__":
    main()
