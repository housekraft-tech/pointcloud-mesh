"""Build grouped Mujammel as-built solids and a native SketchUp file.

Inputs are the completed dual-scan wall graph plus independently refitted
opening and wall-assembly planes.  Walls are one group each; a wall profile
step is unioned into its parent wall, never exported as a separate column.
Openings are boolean cuts in the resulting profiled wall.  The staged JSON is
consumed directly by SketchUp's Ruby API, avoiding importer flattening and
preserving editable named groups.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from skp_client import rb  # noqa: E402


COLOURS = {
    "wall_measured": [205, 197, 183],
    "wall_inferred": [221, 190, 110],
    "wall_profile_review": [192, 172, 146],
    "floor": [168, 168, 160],
    "ceiling": [196, 196, 190],
    "column_review": [184, 140, 120],
    "beam_review": [170, 150, 135],
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stable_xy(points: np.ndarray, frame: dict) -> np.ndarray:
    output = np.asarray(points, dtype=float).copy()
    output[:, :2] -= np.asarray(frame["centre_xy"], dtype=float)
    theta = np.deg2rad(-float(frame["yaw_deg"]))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    output[:, :2] = output[:, :2] @ rotation.T
    return output


def tail_objects(path: Path, tail_bytes: int = 24_000_000) -> dict[str, np.ndarray]:
    """Parse late OBJ objects (columns and opening boxes) without reading 305 MB."""
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - tail_bytes))
        if stream.tell() > 0:
            stream.readline()
        text = stream.read().decode("utf-8", errors="ignore")
    objects: dict[str, list[list[float]]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("o "):
            name = line[2:].strip()
            current = name if (
                name.startswith("column_")
                or name.startswith("window_")
                or name.startswith("door_")
                or name.startswith("balcony_door_")
                or name.startswith("opening_")
                or name.startswith("archway_")
            ) else None
            if current is not None:
                objects.setdefault(current, [])
        elif current is not None and line.startswith("v "):
            values = line.split()
            objects[current].append([float(values[1]), float(values[2]), float(values[3])])
    return {
        name: np.asarray(vertices, dtype=float)
        for name, vertices in objects.items() if vertices
    }


def box_mesh(lo: list[float] | np.ndarray, hi: list[float] | np.ndarray) -> trimesh.Trimesh:
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    mesh = trimesh.creation.box(extents=np.maximum(hi - lo, 1e-5))
    mesh.apply_translation(0.5 * (lo + hi))
    return mesh


def opening_axis(vertices: np.ndarray) -> tuple[str, float]:
    xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    covariance = np.cov(xy.T)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, np.argmax(values)]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)
    deviation = min(abs(angle), abs(angle - 90), abs(angle - 180))
    return ("y" if abs(direction[0]) >= abs(direction[1]) else "x"), deviation


def map_openings(
    objects: dict[str, np.ndarray], frame: dict, walls: list[dict], manifest: dict
) -> tuple[list[dict], list[dict]]:
    metadata = {item["name"]: item for item in manifest["openings"]}
    mapped, review = [], []
    for name, raw in objects.items():
        if name.startswith("column_"):
            continue
        vertices = stable_xy(raw, frame)
        axis, angle_deviation = opening_axis(vertices)
        centre = vertices[:, :2].mean(axis=0)
        along_axis = 1 if axis == "x" else 0
        along_lo = float(vertices[:, along_axis].min())
        along_hi = float(vertices[:, along_axis].max())
        z_lo, z_hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        choices = []
        nearest = []
        for wall in walls:
            if wall["axis"] != axis:
                continue
            if axis == "x":
                perpendicular = abs(centre[0] - wall["center_m"])
                along = centre[1]
            else:
                perpendicular = abs(centre[1] - wall["center_m"])
                along = centre[0]
            along_gap = max(0.0, wall["span_m"][0] - along, along - wall["span_m"][1])
            nearest.append((perpendicular + 1.8 * along_gap, wall, perpendicular, along_gap))
            bridge_cap = max(0.45, 0.55 * (along_hi - along_lo) + 0.15)
            if perpendicular <= 0.60 and along_gap <= bridge_cap:
                choices.append((perpendicular + 1.8 * along_gap, wall, perpendicular, along_gap))
        item = dict(metadata.get(name, {"name": name}))
        item["legacy_vertices_raw"] = int(len(raw))
        item["axis_deviation_deg"] = angle_deviation
        item["stable_center_xy"] = centre.tolist()
        item["stable_z_m"] = [z_lo, z_hi]
        item["opening_axis"] = axis
        item["stable_along_m"] = [along_lo, along_hi]
        if not choices or angle_deviation > 15.0:
            item["mapping_status"] = "review_unmapped"
            if nearest:
                _, near_wall, perpendicular, along_gap = min(nearest, key=lambda value: value[0])
                item["nearest_wall_id"] = near_wall["wall_id"]
                item["nearest_perpendicular_mm"] = float(perpendicular * 1000)
                item["nearest_along_gap_mm"] = float(along_gap * 1000)
            review.append(item)
            continue
        _, wall, perpendicular, along_gap = min(choices, key=lambda value: value[0])
        face_lo, face_hi = wall["faces_m"]
        if axis == "x":
            cut_lo = [face_lo - 0.025, along_lo, z_lo]
            cut_hi = [face_hi + 0.025, along_hi, z_hi]
        else:
            cut_lo = [along_lo, face_lo - 0.025, z_lo]
            cut_hi = [along_hi, face_hi + 0.025, z_hi]
        item.update(
            {
                "mapping_status": "mapped_legacy_lidar",
                "wall_id": wall["wall_id"],
                "perpendicular_offset_mm": float(perpendicular * 1000),
                "along_gap_mm": float(along_gap * 1000),
                "cut_lo": cut_lo,
                "cut_hi": cut_hi,
            }
        )
        mapped.append(item)
    return mapped, review


def dedupe_openings(openings: list[dict]) -> list[dict]:
    """Merge the same through-opening observed from opposite wall faces."""
    output: list[dict] = []
    for opening in openings:
        match = None
        for existing in output:
            if existing["wall_id"] != opening["wall_id"]:
                continue
            common = max(
                0.0,
                min(existing["stable_along_m"][1], opening["stable_along_m"][1])
                - max(existing["stable_along_m"][0], opening["stable_along_m"][0]),
            )
            short = min(
                existing["stable_along_m"][1] - existing["stable_along_m"][0],
                opening["stable_along_m"][1] - opening["stable_along_m"][0],
            )
            if short > 0 and common >= 0.65 * short:
                match = existing
                break
        if match is None:
            item = dict(opening)
            item["aliases"] = [opening["name"]]
            output.append(item)
            continue
        match["aliases"].append(opening["name"])
        match["name"] = "combined__" + "__".join(match["aliases"])
        match["mapping_status"] = "mapped_combined_opposite_faces"
        match["stable_along_m"] = [
            min(match["stable_along_m"][0], opening["stable_along_m"][0]),
            max(match["stable_along_m"][1], opening["stable_along_m"][1]),
        ]
        match["stable_z_m"] = [
            min(match["stable_z_m"][0], opening["stable_z_m"][0]),
            max(match["stable_z_m"][1], opening["stable_z_m"][1]),
        ]
        match["cut_lo"] = np.minimum(match["cut_lo"], opening["cut_lo"]).tolist()
        match["cut_hi"] = np.maximum(match["cut_hi"], opening["cut_hi"]).tolist()
        match["width_mm"] = round(
            1000 * (match["stable_along_m"][1] - match["stable_along_m"][0]), 1
        )
        match["height_mm"] = round(
            1000 * (match["stable_z_m"][1] - match["stable_z_m"][0]), 1
        )
    return output


def validated_openings(path: Path, walls: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert only two-walk pass opening planes into through-wall cuts."""
    evidence = json.loads(path.read_text())
    wall_map = {wall["wall_id"]: wall for wall in walls}
    passed, review = [], []
    for item in evidence["openings"]:
        if not item.get("metric_status", "").startswith("pass") \
                or item.get("wall_id") not in wall_map:
            review.append(item)
            continue
        wall = wall_map[item["wall_id"]]
        fit = item["primary_fit"]["fits"]
        along_lo = float(fit["left"]["value_m"])
        along_hi = float(fit["right"]["value_m"])
        z_lo = float(fit["sill"]["value_m"])
        z_hi = float(fit["header"]["value_m"])
        face_lo, face_hi = map(float, wall["faces_m"])
        # The cut traverses the largest permitted wall profile step so an
        # opening remains through-cut if it overlaps a thickened wall interval.
        if wall["axis"] == "x":
            cut_lo = [face_lo - 0.45, along_lo, z_lo]
            cut_hi = [face_hi + 0.45, along_hi, z_hi]
            centre = [wall["center_m"], 0.5 * (along_lo + along_hi)]
        else:
            cut_lo = [along_lo, face_lo - 0.45, z_lo]
            cut_hi = [along_hi, face_hi + 0.45, z_hi]
            centre = [0.5 * (along_lo + along_hi), wall["center_m"]]
        passed.append({
            **item,
            "stable_along_m": [along_lo, along_hi],
            "stable_z_m": [z_lo, z_hi],
            "stable_center_xy": centre,
            "cut_lo": cut_lo,
            "cut_hi": cut_hi,
            "width_mm": round((along_hi - along_lo) * 1000, 1),
            "height_mm": round((z_hi - z_lo) * 1000, 1),
            "geometry_source": "primary precise returns; reference is repeatability gate",
        })
    return passed, review


def validated_wall_profiles(
    path: Path, evidence_source: str = "50mm_primary",
    max_coarse_edge_delta_mm: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return scan-supported parent-wall modifiers with the naming contract.

    A profile is release-grade only when all of its return planes repeat within
    tolerance. A real broad face can still lose one end return because it
    terminates at another wall or because that end was occluded. Do not erase
    those visible forms: retain structural steps/soffits when both walks see
    the same offset face to 10 mm, but label their boundary geometry for review.
    Niches remain excluded until their cutting boundary is closed.
    """
    evidence = json.loads(path.read_text())
    included, review = [], []
    for item in evidence["features"]:
        status = item.get("metric_status", "")
        primary = item.get("primary_fit") or {}
        reference = item.get("reference_fit") or {}
        is_pass = status.startswith("pass")
        edge_deltas = item.get("repeat_edge_delta_mm") or []
        if max_coarse_edge_delta_mm is not None and edge_deltas and max(
            abs(float(value)) for value in edge_deltas if np.isfinite(float(value))
        ) > max_coarse_edge_delta_mm:
            review.append({
                **item,
                "metric_status": "review_sensitive_edge_repeat_exceeds_gate",
            })
            continue
        structural = item.get("kind") in {
            "embedded_column", "beam_or_soffit", "proud_relief_review", "niche",
        }
        depth_repeat_limit_mm = 20.0 if item.get("kind") == "niche" else 10.0
        two_walk_surface = (
            primary.get("surface_fit") is not None
            and reference.get("surface_fit") is not None
            and primary.get("depth_mm") is not None
            and reference.get("depth_mm") is not None
            and abs(float(primary["depth_mm"]) - float(reference["depth_mm"]))
            <= depth_repeat_limit_mm
            and 15.0 <= float(primary["depth_mm"]) <= 350.0
        )
        include_review = not is_pass and structural and two_walk_surface
        if not is_pass and not include_review:
            review.append(item)
            continue
        renamed = dict(item)
        renamed["evidence_source"] = evidence_source
        renamed["model_kind"] = {
            "embedded_column": "wall_profile_step",
            "proud_relief_review": "wall_profile_step",
            "beam_or_soffit": "wall_beam_soffit_profile",
            "niche": "wall_niche_profile",
        }[item["kind"]]
        renamed["parent_wall_relationship"] = "merged_same_wall_solid"
        if is_pass:
            renamed["geometry_status"] = "release_two_walk_closed_boundaries"
        else:
            renamed["geometry_status"] = (
                "included_review_two_walk_surface_supported_boundary_uncertain"
            )

        # Prefer the midpoint consensus where both complete fits exist. It
        # minimises model-to-walk residual. Otherwise use complete primary
        # boundaries, or the conservative persistent 50 mm scan interval when
        # a return is occluded. Depth always uses the two-walk midpoint.
        if renamed.get("measured_along_m") and renamed.get("measured_z_m"):
            along = renamed["measured_along_m"]
            z_range = renamed["measured_z_m"]
            bounds_source = "two_walk_precise_midpoint"
        elif primary.get("complete"):
            along = primary["along_m"]
            z_range = primary["z_m"]
            bounds_source = "primary_precise_returns_reference_surface_only"
        elif reference.get("complete"):
            along = reference["along_m"]
            z_range = reference["z_m"]
            bounds_source = "reference_precise_returns_primary_surface_only"
        else:
            deltas = list(map(float, item.get("repeat_edge_delta_mm", [0, 0, 0, 0])))

            def hybrid_boundary(name: str, coarse: float, delta: float) -> float:
                values = []
                for fit in (primary, reference):
                    boundary = fit.get(name)
                    if boundary is not None and boundary.get("value_m") is not None:
                        values.append(float(boundary["value_m"]))
                # No fitted return: midpoint of the two persistent grid edges.
                return float(np.mean(values)) if values else float(coarse - delta / 2000)

            along = [
                hybrid_boundary("left", item["along_m"][0], deltas[0]),
                hybrid_boundary("right", item["along_m"][1], deltas[1]),
            ]
            z_range = [
                hybrid_boundary("bottom", item["z_m"][0], deltas[2]),
                hybrid_boundary("top", item["z_m"][1], deltas[3]),
            ]
            bounds_source = "hybrid_precise_returns_and_two_walk_grid_midpoint"
        renamed["geometry_along_m"] = list(map(float, along))
        renamed["geometry_z_m"] = list(map(float, z_range))
        renamed["geometry_depth_mm"] = (
            round(0.5 * (float(primary["depth_mm"]) + float(reference["depth_mm"])), 3)
            if two_walk_surface else float(primary["depth_mm"])
        )
        renamed["geometry_bounds_source"] = bounds_source

        # The coarse detector also emits smaller 2-D pieces inside a full-height
        # profile. Keep the parent profile once, but retain truly separate local
        # steps that the old structural-only path discarded.
        if item["kind"] == "proud_relief_review":
            a0, a1 = renamed["geometry_along_m"]
            z0, z1 = renamed["geometry_z_m"]
            area = max((a1 - a0) * (z1 - z0), 1e-9)
            duplicate = False
            for existing in included:
                if existing["wall_id"] != item["wall_id"] \
                        or existing["face_index"] != item["face_index"]:
                    continue
                ea0, ea1 = existing["geometry_along_m"]
                ez0, ez1 = existing["geometry_z_m"]
                overlap_area = max(0.0, min(a1, ea1) - max(a0, ea0)) * max(
                    0.0, min(z1, ez1) - max(z0, ez0)
                )
                if overlap_area >= 0.50 * area and abs(
                    renamed["geometry_depth_mm"] - existing["geometry_depth_mm"]
                ) <= 15.0:
                    duplicate = True
                    break
            if duplicate:
                review.append({
                    **item,
                    "metric_status": "review_duplicate_inside_larger_wall_profile",
                })
                continue
        included.append(renamed)

    # Do the same duplicate test once more after collecting every class; coarse
    # evidence ordering is not guaranteed, so a local fragment can precede the
    # full-height profile that absorbs it.
    filtered = []
    for feature in included:
        if feature["kind"] != "proud_relief_review":
            filtered.append(feature)
            continue
        a0, a1 = feature["geometry_along_m"]
        z0, z1 = feature["geometry_z_m"]
        area = max((a1 - a0) * (z1 - z0), 1e-9)
        duplicate = any(
            other is not feature
            and other["kind"] != "proud_relief_review"
            and other["wall_id"] == feature["wall_id"]
            and other["face_index"] == feature["face_index"]
            and max(
                0.0,
                min(a1, other["geometry_along_m"][1])
                - max(a0, other["geometry_along_m"][0]),
            )
            * max(
                0.0,
                min(z1, other["geometry_z_m"][1])
                - max(z0, other["geometry_z_m"][0]),
            ) >= 0.50 * area
            and abs(feature["geometry_depth_mm"] - other["geometry_depth_mm"]) <= 15.0
            for other in included
        )
        if duplicate:
            review.append({
                **feature,
                "metric_status": "review_duplicate_inside_larger_wall_profile",
            })
        else:
            filtered.append(feature)
    return filtered, review


def merge_wall_profile_evidence(features: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge coarse and sensitive profile passes without duplicating relief.

    Release-grade precise features win over a near-identical review candidate.
    For review candidates on the same face and depth plane, a materially larger
    high-resolution interval supersedes the smaller feature; this recovers wall
    junction steps while preserving the precise 10 mm features already solved.
    """
    selected: list[dict] = []
    superseded: list[dict] = []
    for candidate in features:
        ca0, ca1 = candidate["geometry_along_m"]
        cz0, cz1 = candidate["geometry_z_m"]
        candidate_area = max((ca1 - ca0) * (cz1 - cz0), 1e-9)
        match_index = None
        for index, existing in enumerate(selected):
            if existing["wall_id"] != candidate["wall_id"] \
                    or existing["face_index"] != candidate["face_index"] \
                    or existing["model_kind"] != candidate["model_kind"]:
                continue
            ea0, ea1 = existing["geometry_along_m"]
            ez0, ez1 = existing["geometry_z_m"]
            overlap_area = max(0.0, min(ca1, ea1) - max(ca0, ea0)) * max(
                0.0, min(cz1, ez1) - max(cz0, ez0)
            )
            if overlap_area >= 0.35 * min(
                candidate_area, max((ea1 - ea0) * (ez1 - ez0), 1e-9)
            ) and abs(
                candidate["geometry_depth_mm"] - existing["geometry_depth_mm"]
            ) <= 15.0:
                match_index = index
                break
        if match_index is None:
            selected.append(candidate)
            continue
        existing = selected[match_index]
        ea0, ea1 = existing["geometry_along_m"]
        ez0, ez1 = existing["geometry_z_m"]
        existing_area = max((ea1 - ea0) * (ez1 - ez0), 1e-9)
        existing_release = existing["geometry_status"].startswith("release")
        candidate_release = candidate["geometry_status"].startswith("release")
        replace = (
            candidate_release and not existing_release
            or not existing_release and candidate_area >= 1.10 * existing_area
        )
        if replace:
            superseded.append({
                **existing,
                "metric_status": "review_superseded_by_sensitive_profile_extent",
            })
            selected[match_index] = candidate
        else:
            superseded.append({
                **candidate,
                "metric_status": "review_duplicate_of_selected_profile",
            })
    return selected, superseded


def profile_modifier_mesh(feature: dict, wall: dict) -> tuple[str, trimesh.Trimesh]:
    """Create one addition/subtraction interval from primary fitted planes."""
    fit = feature["primary_fit"]
    face_index = int(feature["face_index"])
    base_face = float(wall["faces_m"][face_index])
    outward = -1.0 if face_index == 0 else 1.0
    direction = -1.0 if feature["kind"] == "niche" else 1.0
    # Anchor the validated local depth to the final consensus parent-wall face.
    depth_mm = float(feature.get("geometry_depth_mm", fit["depth_mm"]))
    surface_face = base_face + outward * direction * depth_mm / 1000
    along_values = (
        feature["geometry_along_m"]
        if "geometry_along_m" in feature else fit["along_m"]
    )
    z_values = feature["geometry_z_m"] if "geometry_z_m" in feature else fit["z_m"]
    along_lo, along_hi = map(float, along_values)
    z_lo, z_hi = map(float, z_values)
    cross_lo, cross_hi = sorted((base_face, surface_face))
    # Two millimetres of overlap makes the union topologically unambiguous
    # without changing the measured outer surface.
    if feature["kind"] != "niche":
        if face_index == 0:
            cross_hi += 0.002
        else:
            cross_lo -= 0.002
    if wall["axis"] == "x":
        lo, hi = [cross_lo, along_lo, z_lo], [cross_hi, along_hi, z_hi]
    else:
        lo, hi = [along_lo, cross_lo, z_lo], [along_hi, cross_hi, z_hi]
    operation = "subtract" if feature["kind"] == "niche" else "add"
    return operation, box_mesh(lo, hi)


def mesh_payload(name: str, kind: str, mesh: trimesh.Trimesh) -> dict:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return {
        "name": name,
        "kind": kind,
        "colour": COLOURS[kind],
        "v": np.asarray(mesh.vertices, dtype=float).round(6).tolist(),
        "f": np.asarray(mesh.faces, dtype=int).tolist(),
    }


def _snap_room_rectangle(
    room: dict, frame: dict, walls: list[dict], snap_limit_m: float = 0.22
) -> tuple[Polygon, list[dict]]:
    """Return a clean CAD room loop, never a raster/alpha-shape perimeter.

    ``quad`` is the scan-derived room envelope and is already almost perfectly
    Manhattan-aligned in the stable frame.  Each of its four coordinates is
    re-solved onto a physical room-side wall face when a supported wall run is
    present.  An unmatched exterior/balcony edge stays a straight scan-envelope
    line and is explicitly marked for review; it is never drawn as a jagged
    sequence of occupied floor pixels.
    """
    quad3 = np.column_stack(
        [np.asarray(room["quad"], dtype=float), np.zeros(len(room["quad"]))]
    )
    quad = stable_xy(quad3, frame)[:, :2]
    minimum = quad.min(axis=0)
    maximum = quad.max(axis=0)
    edges = [
        ("x", "min_x", float(minimum[0]), float(minimum[1]), float(maximum[1])),
        ("x", "max_x", float(maximum[0]), float(minimum[1]), float(maximum[1])),
        ("y", "min_y", float(minimum[1]), float(minimum[0]), float(maximum[0])),
        ("y", "max_y", float(maximum[1]), float(minimum[0]), float(maximum[0])),
    ]
    solved: dict[str, float] = {}
    report = []
    for axis, label, coordinate, along_lo, along_hi in edges:
        length = along_hi - along_lo
        required_overlap = min(0.20, 0.12 * length)
        candidates = []
        for wall in walls:
            if wall["axis"] != axis:
                continue
            overlap = max(
                0.0,
                min(along_hi, wall["span_m"][1])
                - max(along_lo, wall["span_m"][0]),
            )
            along_gap = max(
                0.0, wall["span_m"][0] - along_hi, along_lo - wall["span_m"][1]
            )
            if overlap < required_overlap and along_gap > 0.08:
                continue
            for face_index, face in enumerate(wall["faces_m"]):
                distance = abs(float(face) - coordinate)
                if distance <= snap_limit_m:
                    candidates.append(
                        (distance + 1.5 * along_gap, distance, -overlap, wall,
                         face_index, float(face))
                    )
        if candidates:
            _, distance, neg_overlap, wall, face_index, value = min(
                candidates, key=lambda item: item[:3]
            )
            solved[label] = value
            repeatability_mm = None
            face_source = "inferred_hidden_wall_face_review"
            if wall["thickness_source"] == "measured_two_faces":
                values = wall.get("face_repeatability_mm", [])
                if face_index < len(values):
                    repeatability_mm = float(values[face_index])
                    face_source = "measured_wall_face"
            elif abs(value - float(wall.get("measured_face_m", np.inf))) < 1e-5:
                values = wall.get("face_repeatability_mm", [])
                if values:
                    repeatability_mm = float(values[0])
                    face_source = "measured_wall_face"
            report.append(
                {
                    "edge": label,
                    "axis": axis,
                    "source": face_source,
                    "coordinate_m": value,
                    "wall_id": wall["wall_id"],
                    "wall_face_index": face_index,
                    "wall_thickness_source": wall["thickness_source"],
                    "scan_envelope_move_mm": round((value - coordinate) * 1000, 1),
                    "along_overlap_mm": round(-neg_overlap * 1000, 1),
                    "two_walk_face_delta_mm": (
                        None if repeatability_mm is None
                        else round(repeatability_mm, 3)
                    ),
                }
            )
        else:
            solved[label] = coordinate
            report.append(
                {
                    "edge": label,
                    "axis": axis,
                    "source": "straight_scan_envelope_review",
                    "coordinate_m": coordinate,
                    "wall_id": None,
                    "scan_envelope_move_mm": 0.0,
                }
            )
    if solved["min_x"] >= solved["max_x"] or solved["min_y"] >= solved["max_y"]:
        raise ValueError(f"room {room['room']!r} collapsed while snapping to walls")
    return box(solved["min_x"], solved["min_y"], solved["max_x"], solved["max_y"]), report


def room_meshes(
    fused: dict, features: dict, frame: dict, primary_floor_z: float,
    walls: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    feature_map = {item["name"]: item for item in features["ceiling_features"]}
    floors, ceilings, room_report = [], [], []
    counts: Counter[str] = Counter()
    for room in fused["rooms"]:
        counts[room["room"]] += 1
        suffix = counts[room["room"]]
        clean_name = room["room"].lower().replace(" / ", "_").replace(" ", "_")
        polygon, edge_report = _snap_room_rectangle(room, frame, walls)
        if polygon.is_empty or polygon.area < 0.30:
            continue
        try:
            floor = trimesh.creation.extrude_polygon(polygon, height=0.10)
        except Exception:
            floor = trimesh.creation.extrude_polygon(polygon.simplify(0.03), height=0.10)
        floor.apply_translation([0, 0, primary_floor_z - 0.10])
        floors.append(mesh_payload(f"floor_{clean_name}_{suffix:02d}", "floor", floor))

        heights = [
            feature_map[name]["height_mm"] / 1000
            for name in room.get("ceiling_parts", []) if name in feature_map
        ]
        height = float(np.median(heights)) if heights else 2.70
        ceiling_z = primary_floor_z + height
        ceiling = trimesh.creation.extrude_polygon(polygon, height=0.08)
        ceiling.apply_translation([0, 0, ceiling_z])
        payload = mesh_payload(f"ceiling_{clean_name}_{suffix:02d}", "ceiling", ceiling)
        payload["ceiling_z_m"] = round(ceiling_z, 6)
        payload["height_mm"] = round(height * 1000, 1)
        ceilings.append(payload)
        by_edge = {item["edge"]: item for item in edge_report}
        repeat = {
            key: by_edge[key].get("two_walk_face_delta_mm")
            for key in ("min_x", "max_x", "min_y", "max_y")
        }
        dimensions = None
        metric_status = "review_missing_independent_opposing_face"
        if all(value is not None for value in repeat.values()):
            width_delta = float(repeat["max_x"] - repeat["min_x"])
            length_delta = float(repeat["max_y"] - repeat["min_y"])
            dimensions = {
                "width_mm": round((polygon.bounds[2] - polygon.bounds[0]) * 1000, 1),
                "length_mm": round((polygon.bounds[3] - polygon.bounds[1]) * 1000, 1),
                "two_walk_width_delta_mm": round(width_delta, 3),
                "two_walk_length_delta_mm": round(length_delta, 3),
            }
            consensus = any("consensus_face_shift_mm" in wall for wall in walls)
            if consensus:
                dimensions["model_to_each_walk_width_max_mm"] = round(abs(width_delta) / 2, 3)
                dimensions["model_to_each_walk_length_max_mm"] = round(abs(length_delta) / 2, 3)
                metric_status = (
                    "pass_consensus_within_10mm_each_walk"
                    if max(abs(width_delta), abs(length_delta)) / 2 <= 10.0
                    else "fail_consensus_exceeds_10mm_to_a_walk"
                )
            else:
                metric_status = (
                    "pass_two_walk_plan_dimensions"
                    if max(abs(width_delta), abs(length_delta)) <= 10.0
                    else "fail_two_walk_plan_dimensions"
                )
        room_report.append(
            {
                "name": f"{clean_name}_{suffix:02d}",
                "semantic_name": room["room"],
                "area_m2": round(float(polygon.area), 3),
                "bounds_m": [round(float(value), 6) for value in polygon.bounds],
                "edges": edge_report,
                "floor_geometry": "exact_rectilinear_CAD_face",
                "plan_dimensions": dimensions,
                "metric_status": metric_status,
            }
        )
    return floors, ceilings, room_report


def floor_threshold_meshes(
    openings: list[dict], walls: list[dict], floor_z: float,
) -> tuple[list[dict], list[dict]]:
    """Fill the doorway/wide-opening band through the full wall thickness.

    Room slabs finish on their measured wall faces. Once a door is subtracted
    from that wall, the band between those faces otherwise appears as a missing
    strip of floor. A threshold slab restores one continuous finish datum and
    overlaps each adjoining room slab by 20 mm below the visible top surface.
    """
    wall_map = {wall["wall_id"]: wall for wall in walls}
    payloads, report = [], []
    sequence = 0
    for opening in openings:
        z_lo = float(opening["stable_z_m"][0])
        if z_lo > floor_z + 0.08:
            continue
        wall = wall_map[opening["wall_id"]]
        face_lo, face_hi = map(float, wall["faces_m"])
        along_lo, along_hi = map(float, opening["stable_along_m"])
        cross_lo, cross_hi = face_lo - 0.020, face_hi + 0.020
        if wall["axis"] == "x":
            lo = [cross_lo, along_lo - 0.002, floor_z - 0.10]
            hi = [cross_hi, along_hi + 0.002, floor_z]
        else:
            lo = [along_lo - 0.002, cross_lo, floor_z - 0.10]
            hi = [along_hi + 0.002, cross_hi, floor_z]
        sequence += 1
        safe_name = "".join(
            character if character.isalnum() else "_"
            for character in opening["name"]
        ).strip("_")
        name = f"floor_threshold_{sequence:02d}_{safe_name}"
        payloads.append(mesh_payload(name, "floor", box_mesh(lo, hi)))
        report.append({
            "name": name,
            "opening": opening["name"],
            "wall_id": opening["wall_id"],
            "top_z_m": floor_z,
            "geometry": "continuous_floor_through_wall_opening",
        })
    return payloads, report


def recovered_floor_meshes(
    path: Path, floor_z: float, selected_ids: tuple[int, ...] = (1, 5, 13, 14),
) -> tuple[list[dict], list[dict]]:
    """Restore LiDAR floor regions that the semantic room detector omitted.

    The selected occupancy cells are: the complete right L-shaped balcony that
    supersedes two fragmented room rectangles, one enclosed area missed by the
    alternate semantic pass, the recovered left balcony and the recovered
    circulation corridor. Their outlines are already rectilinear and metric;
    use the final consensus datum so they are coplanar with labelled rooms.
    """
    evidence = json.loads(path.read_text())
    candidates = [
        item for item in evidence.get("rooms", [])
        if int(item.get("id", -1)) in selected_ids
    ]
    payloads, report = [], []
    labels = {
        1: "balcony_right_combined",
        5: "room_unclassified",
        13: "balcony_left_recovered",
        14: "corridor_recovered",
    }
    for sequence, item in enumerate(candidates, 1):
        polygon = Polygon(item["polygon_m"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area < 0.30:
            continue
        # These recovered occupancy components have straight Manhattan edges;
        # guard against sub-pixel noise without changing their measured extent.
        polygon = polygon.simplify(0.005, preserve_topology=True)
        slab = trimesh.creation.extrude_polygon(polygon, height=0.10)
        slab.apply_translation([0, 0, floor_z - 0.10])
        label = labels.get(int(item.get("id", -1)), f"area_{sequence:02d}")
        name = f"floor_occupancy_{label}"
        payloads.append(mesh_payload(name, "floor", slab))
        report.append({
            "name": name,
            "source_room_id": item.get("id"),
            "area_m2": round(float(polygon.area), 3),
            "bounds_m": [round(float(value), 6) for value in polygon.bounds],
            "top_z_m": floor_z,
            "geometry": "direct_lidar_floor_occupancy_rectilinear",
            "metric_status": "review_boundary_single_occupancy_solution",
        })
    return payloads, report


def filter_unit_walls(
    walls: list[dict], fused: dict, frame: dict, adjacency_m: float = 0.35
) -> tuple[list[dict], list[dict]]:
    """Reject full-height structure belonging to neighbouring units.

    Unit isolation has to happen before meshing.  A persistent plane can be a
    perfectly real wall while still belonging to the corridor or next flat.
    Keep a wall only when its plan footprint touches the union of this flat's
    scan-derived room envelopes (or lies within a small wall-thickness margin).
    """
    envelopes = []
    for room in fused["rooms"]:
        quad3 = np.column_stack(
            [np.asarray(room["quad"], dtype=float), np.zeros(len(room["quad"]))]
        )
        quad = stable_xy(quad3, frame)[:, :2]
        minimum, maximum = quad.min(axis=0), quad.max(axis=0)
        envelopes.append(box(minimum[0], minimum[1], maximum[0], maximum[1]))
    unit = unary_union(envelopes)
    kept, excluded = [], []
    for wall in walls:
        low, high = wall["faces_m"]
        start, stop = wall["span_m"]
        footprint = (
            box(low, start, high, stop)
            if wall["axis"] == "x"
            else box(start, low, stop, high)
        )
        distance = float(footprint.distance(unit))
        if distance <= adjacency_m:
            kept.append(wall)
        else:
            excluded.append(
                {
                    "wall_id": wall["wall_id"],
                    "distance_to_unit_rooms_mm": round(distance * 1000, 1),
                    "reason": "persistent structure outside isolated flat",
                }
            )
    return kept, excluded


def reference_columns(objects: dict[str, np.ndarray], frame: dict) -> tuple[list[dict], list[dict]]:
    payloads, report = [], []
    for name, raw in objects.items():
        if not name.startswith("column_"):
            continue
        vertices = stable_xy(raw, frame)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        extent = hi - lo
        accepted = (
            extent[2] >= 2.0 and 0.075 <= min(extent[:2])
            and max(extent[:2]) <= 0.85
        )
        report.append(
            {"name": name, "lo": lo.tolist(), "hi": hi.tolist(),
             "extent_mm": (extent * 1000).tolist(),
             "status": "included_review" if accepted else "excluded_small_or_noncolumn"}
        )
        if accepted:
            payloads.append(mesh_payload(name, "column_review", box_mesh(lo, hi)))
    return payloads, report


def reference_beam(build_json: dict, frame: dict) -> tuple[list[dict], list[dict]]:
    payloads, report = [], []
    for part in build_json.get("parts", []):
        if part.get("kind") != "beam":
            continue
        vertices = stable_xy(np.asarray(part["v"], dtype=float), frame)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        report.append({"name": part["name"], "lo": lo.tolist(), "hi": hi.tolist(),
                       "status": "included_review"})
        payloads.append(mesh_payload(part["name"], "beam_review", box_mesh(lo, hi)))
    return payloads, report


def render_plan(walls: list[dict], openings: list[dict], destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 11), dpi=180)
    axis.set_facecolor("#f8fafc")
    for wall in walls:
        low, high = wall["faces_m"]; start, stop = wall["span_m"]
        if wall["axis"] == "x": rectangle = (low, start, high-low, stop-start)
        else: rectangle = (start, low, stop-start, high-low)
        colour = "#334155" if wall["thickness_source"] == "measured_two_faces" else "#b7791f"
        axis.add_patch(Rectangle(rectangle[:2], rectangle[2], rectangle[3],
                                 facecolor=colour, edgecolor=colour, alpha=0.90))
    for opening in openings:
        centre = opening["stable_center_xy"]
        axis.scatter([centre[0]], [centre[1]], c="#f97316", s=10, zorder=5)
    axis.autoscale(); axis.set_aspect("equal"); axis.grid(alpha=0.15)
    axis.set_title(f"Mujammel native solid build preview — {len(walls)} walls, "
                   f"{len(openings)} mapped openings")
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walls", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--annotated-obj", default="output2/koushik_all/skeleton_3d/annotated/annotated_model.obj")
    parser.add_argument("--opening-manifest", default="output2/koushik_all/skeleton_3d/annotated/annotated_manifest.json")
    parser.add_argument("--rooms", default="output2/koushik_all/skeleton_3d/annotated/fused_detections.json")
    parser.add_argument("--features", default="output2/koushik_all/skeleton_3d/detailed_modular/features.json")
    parser.add_argument("--legacy-build", default="Flat Sketchup/Koushik_flat.build.json")
    parser.add_argument("--validated-openings", default="cage_output/metrology/mujammel_openings_validated.json")
    parser.add_argument(
        "--wall-profiles",
        default="cage_output/metrology/mujammel_wall_assemblies_refined_all_profiles.json",
    )
    parser.add_argument(
        "--sensitive-wall-profiles",
        default="cage_output/metrology/mujammel_wall_assemblies_sensitive_refined.json",
    )
    parser.add_argument(
        "--recovered-floors",
        default="cage_output/snapped/snapped_robust_c95_swin_clean_hybrid.json",
    )
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--out-dir", default="output_final/mujammel_asbuilt")
    parser.add_argument("--skip-skp", action="store_true")
    args = parser.parse_args()

    wall_graph = json.loads(resolve(args.walls).read_text())
    frame = json.loads(resolve(args.frame_meta).read_text())
    fused = json.loads(resolve(args.rooms).read_text())
    wall_graph["walls"], excluded_unit_walls = filter_unit_walls(
        wall_graph["walls"], fused, frame
    )
    openings, opening_review = validated_openings(
        resolve(args.validated_openings), wall_graph["walls"]
    )
    wall_profiles, wall_profile_review = validated_wall_profiles(
        resolve(args.wall_profiles), evidence_source="50mm_primary"
    )
    sensitive_profiles, sensitive_review = validated_wall_profiles(
        resolve(args.sensitive_wall_profiles),
        evidence_source="25mm_junction_audit",
        max_coarse_edge_delta_mm=125.0,
    )
    wall_profiles, superseded_profiles = merge_wall_profile_evidence(
        wall_profiles + sensitive_profiles
    )
    wall_profile_review.extend(sensitive_review + superseded_profiles)
    # The drawing trace often stops at a full-height opening because no dark
    # wall stroke crosses the void.  When a mapped LiDAR opening lies just
    # beyond such an endpoint, extend the owning wall body across the opening
    # so its measured header and jambs can exist and the boolean cut is real.
    wall_by_id = {wall["wall_id"]: wall for wall in wall_graph["walls"]}
    for opening in openings:
        wall = wall_by_id[opening["wall_id"]]
        along_lo, along_hi = opening["stable_along_m"]
        wall["span_m"][0] = float(min(wall["span_m"][0], along_lo))
        wall["span_m"][1] = float(max(wall["span_m"][1], along_hi))
        wall["length_mm"] = float((wall["span_m"][1] - wall["span_m"][0]) * 1000)
    openings_by_wall: dict[str, list[dict]] = {}
    for opening in openings:
        openings_by_wall.setdefault(opening["wall_id"], []).append(opening)
    profiles_by_wall: dict[str, list[dict]] = {}
    for profile in wall_profiles:
        profiles_by_wall.setdefault(profile["wall_id"], []).append(profile)

    parts, wall_report = [], []
    floor_z = float(wall_graph["floor_z_m"])
    ceiling_z = float(wall_graph["ceiling_z_m"])
    # A full-height profile shares the parent wall's exact floor/ceiling caps.
    # Averaging the two scan datums left a ~1 mm coplanar sliver at the top of
    # wall_025 which SketchUp correctly rejected as a three-face edge.
    for profile in wall_profiles:
        if profile["kind"] == "embedded_column":
            profile["geometry_z_m"] = [floor_z, ceiling_z]
            profile["geometry_vertical_source"] = "parent_wall_exact_caps"
    for wall in wall_graph["walls"]:
        low, high = wall["faces_m"]; start, stop = wall["span_m"]
        if wall["axis"] == "x": lo, hi = [low, start, floor_z], [high, stop, ceiling_z]
        else: lo, hi = [start, low, floor_z], [stop, high, ceiling_z]
        base = box_mesh(lo, hi)
        additions, profile_cuts = [], []
        for profile in profiles_by_wall.get(wall["wall_id"], []):
            operation, modifier = profile_modifier_mesh(profile, wall)
            (profile_cuts if operation == "subtract" else additions).append(modifier)
        solid = trimesh.boolean.union([base] + additions, engine="manifold") if additions else base
        if profile_cuts:
            solid = trimesh.boolean.difference([solid] + profile_cuts, engine="manifold")
        cuts = [box_mesh(item["cut_lo"], item["cut_hi"])
                for item in openings_by_wall.get(wall["wall_id"], [])]
        solid = trimesh.boolean.difference([solid] + cuts, engine="manifold") if cuts else solid
        has_review_profile = any(
            item.get("geometry_status", "").startswith("included_review")
            for item in profiles_by_wall.get(wall["wall_id"], [])
        )
        kind = (
            "wall_profile_review" if has_review_profile
            else "wall_measured" if wall["thickness_source"] == "measured_two_faces"
            else "wall_inferred"
        )
        profiled = bool(profiles_by_wall.get(wall["wall_id"]))
        group_name = f"{wall['wall_id']}_{kind}_{wall['thickness_mm']:.0f}mm" \
            + ("_profiled_review" if has_review_profile else "_profiled" if profiled else "")
        parts.append(mesh_payload(group_name, kind, solid))
        wall_report.append({**wall, "group_name": group_name,
                            "openings": [item["name"] for item in openings_by_wall.get(wall["wall_id"], [])],
                            "wall_profiles": [
                                {"model_kind": item["model_kind"],
                                 "metric_status": item["metric_status"],
                                 "geometry_status": item["geometry_status"],
                                 "geometry_bounds_source": item["geometry_bounds_source"],
                                 "parent_wall_relationship": item["parent_wall_relationship"],
                                 "primary_fit": item["primary_fit"]}
                                for item in profiles_by_wall.get(wall["wall_id"], [])
                            ],
                            "solid_watertight": bool(solid.is_watertight)})

    features = json.loads(resolve(args.features).read_text())
    floors, ceilings, room_report = room_meshes(
        fused, features, frame, floor_z, wall_graph["walls"]
    )
    # Occupancy room 1 is the full physical L-shaped balcony. It replaces the
    # two semantic rectangles, whose gap was one of the reported missing floors.
    superseded_floor_names = {"floor_balcony_01", "floor_balcony_02"}
    floors = [item for item in floors if item["name"] not in superseded_floor_names]
    floor_thresholds, floor_threshold_report = floor_threshold_meshes(
        openings, wall_graph["walls"], floor_z
    )
    recovered_floors, recovered_floor_report = recovered_floor_meshes(
        resolve(args.recovered_floors), floor_z
    )
    parts.extend(floors + recovered_floors + floor_thresholds + ceilings)

    output_dir = resolve(args.out_dir); output_dir.mkdir(parents=True, exist_ok=True)
    build_json = output_dir / "Mujammel_asbuilt.build.json"
    build_json.write_text(json.dumps({"parts": parts}))
    feature_report = {
        "schema": "mujammel-native-asbuilt-v1",
        "accuracy_status": (
            "release candidate; openings and room dimensions use strict dual-walk gates; "
            "visible structural wall profiles with repeat-supported surface planes are "
            "included on the profile-review tag when their boundary returns are uncertain"
        ),
        "wall_graph": wall_graph["summary"],
        "counts": {
            "parts": len(parts), "walls": len(wall_report),
            "mapped_opening_observations": len(openings) + len(opening_review),
            "unique_opening_cuts": len(openings),
            "opening_review": len(opening_review), "floors": len(floors),
            "occupancy_floor_corrections": len(recovered_floors),
            "floor_thresholds": len(floor_thresholds), "ceilings": len(ceilings),
            "wall_profile_steps_merged": sum(
                item["model_kind"] == "wall_profile_step" for item in wall_profiles
            ),
            "wall_beam_soffit_profiles_merged": sum(
                item["model_kind"] == "wall_beam_soffit_profile" for item in wall_profiles
            ),
            "wall_niche_profiles_merged": sum(
                item["model_kind"] == "wall_niche_profile" for item in wall_profiles
            ),
            "wall_profiles_merged_boundary_review": sum(
                item["geometry_status"].startswith("included_review")
                for item in wall_profiles
            ),
            "wall_profiles_review": len(wall_profile_review),
            "excluded_neighbour_walls": len(excluded_unit_walls),
            "room_edges_measured": sum(
                edge["source"] == "measured_wall_face"
                for room in room_report for edge in room["edges"]
            ),
            "room_edges_review": sum(
                edge["source"] != "measured_wall_face"
                for room in room_report for edge in room["edges"]
            ),
            "rooms_plan_repeatability_pass": sum(
                room["metric_status"].startswith("pass")
                for room in room_report
            ),
            "rooms_plan_repeatability_fail": sum(
                room["metric_status"].startswith("fail")
                for room in room_report
            ),
            "rooms_plan_repeatability_review": sum(
                room["metric_status"] == "review_missing_independent_opposing_face"
                for room in room_report
            ),
        },
        "walls": wall_report,
        "openings": openings,
        "opening_review": opening_review,
        "wall_profiles": wall_profiles,
        "wall_profile_review": wall_profile_review,
        "floor_thresholds": floor_threshold_report,
        "recovered_floors": recovered_floor_report,
        "rooms": room_report,
        "excluded_unit_walls": excluded_unit_walls,
    }
    report_path = output_dir / "feature_manifest.json"
    report_path.write_text(json.dumps(feature_report, indent=2))

    scene = trimesh.Scene()
    for part in parts:
        scene.add_geometry(trimesh.Trimesh(np.asarray(part["v"]), np.asarray(part["f"]), process=False),
                           geom_name=part["name"], node_name=part["name"])
    glb_path = output_dir / "Mujammel_asbuilt.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    render_plan(wall_graph["walls"], openings, output_dir / "Mujammel_asbuilt_plan.png")
    print(json.dumps(feature_report["counts"], indent=2))
    print(f"staged {build_json} ({build_json.stat().st_size / 1e6:.2f} MB)")

    if not args.skip_skp:
        skp_path = output_dir / "Mujammel_asbuilt.skp"
        ruby = ROOT / "scripts" / "export" / "ruby" / "pcm_build.rb"
        code = (
            f'load {json.dumps(str(ruby.resolve()).replace(chr(92), "/"))}; '
            f'PCMBuild.build({json.dumps(str(build_json.resolve()).replace(chr(92), "/"))}, '
            f'{json.dumps(str(skp_path.resolve()).replace(chr(92), "/"))})'
        )
        response = rb(code, timeout=1800)
        if not response.get("ok"):
            raise SystemExit(f"SketchUp build failed: {response.get('error')}\n{response.get('backtrace')}")
        print(response["result"].strip('"').replace('\\"', '"'))


if __name__ == "__main__":
    main()
