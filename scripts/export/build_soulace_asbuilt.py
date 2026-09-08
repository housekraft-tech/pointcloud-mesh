"""Build clean, native-SketchUp Soulace as-built models.

The old Soulace handover deliberately flattened wall relief and exported the
segmented scan floor as dozens of fragments.  This builder instead treats the
scan as metrology evidence for a CAD model:

* wall axes and thicknesses come from the measured solid wall objects;
* doors, windows and arches are clean openings, not jagged scan silhouettes;
* niches are boolean recesses and pilasters are part of their parent wall;
* attached columns are unioned into the wall they modify;
* rooms are the enclosed free-space cells of the wall graph, accepted by both
  the old floor evidence and the raw LAS floor returns;
* doorway thresholds restore one continuous finish-floor datum.

Each level is emitted at finish-floor Z=0, plus a stacked whole-building file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from skp_client import rb  # noqa: E402
from sketchup_pack import write_dae  # noqa: E402
from to_sketchup import read_groups  # noqa: E402


COLOURS = {
    "wall_measured": [207, 200, 187],
    "wall_inferred": [221, 190, 110],
    "floor": [174, 174, 166],
    "ceiling": [210, 210, 205],
    "dropped_ceiling": [190, 195, 204],
    "beam": [153, 147, 140],
    "column": [165, 157, 148],
}

LEVEL_STEMS = {0: "Soulace_L0_ground", 1: "Soulace_L1_first", 2: "Soulace_L2_second"}
# SketchUp's PolygonMesh importer creates non-manifold cap edges on these four
# otherwise-watertight, return-dense opening shells.  Collada imports the same
# BREP as a valid solid component, so only these parts use that native fallback.
SKETCHUP_DAE_WALLS = {0: {"wall_03", "wall_22", "wall_27"}, 2: {"wall_08"}}


def log(message: str) -> None:
    print(message, flush=True)


def snap(value: float, step: float = 0.005) -> float:
    return round(value / step) * step


def polygons(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [item for item in getattr(geometry, "geoms", []) if isinstance(item, Polygon)]


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Earcut intentionally duplicates a few cap vertices around holes.  They
    # are geometrically identical but topologically separate until welded;
    # leaving them apart is what made an otherwise clean arched wall report
    # as non-manifold in SketchUp.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def box_mesh(lo, hi) -> trimesh.Trimesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = np.maximum(hi - lo, 0.001)
    mesh = trimesh.creation.box(extents=ext)
    mesh.apply_translation(0.5 * (lo + hi))
    return mesh


def extrude_along_cross(poly: Polygon, axis: str, cross_lo: float,
                        cross_hi: float) -> trimesh.Trimesh:
    """Extrude an along-by-height polygon through a wall thickness."""
    mesh = trimesh.creation.extrude_polygon(poly, height=cross_hi - cross_lo,
                                            engine="earcut")
    source = np.asarray(mesh.vertices, float)
    target = np.empty_like(source)
    if axis == "x":
        target[:, 0] = source[:, 2] + cross_lo
        target[:, 1] = source[:, 0]
    else:
        target[:, 0] = source[:, 0]
        target[:, 1] = source[:, 2] + cross_lo
    target[:, 2] = source[:, 1]
    mesh.vertices = target
    return clean_mesh(mesh)


def flatten_wall_profiles(source: trimesh.Trimesh, wall: dict,
                          floor_z: float) -> list[trimesh.Trimesh]:
    """Project a supported wall to its silhouette and extrude one thickness.

    Projection preserves true through-openings but fills depth-only niches and
    removes pilasters/profile steps. The extrusion uses the measured thickness,
    not the padded/profile-expanded source-mesh bounds.
    """
    axis_index = 0 if wall["axis"] == "x" else 1
    along_index = 1 - axis_index
    component_shapes = []
    for component in source.split(only_watertight=False):
        triangles = np.asarray(component.vertices, float)[np.asarray(component.faces, int)]
        triangle_shapes = []
        for triangle in triangles:
            points = triangle[:, [along_index, 2]]
            poly = Polygon(points)
            if poly.is_valid and poly.area > 1e-9:
                triangle_shapes.append(poly)
        if triangle_shapes:
            shape = unary_union(triangle_shapes).buffer(0)
            component_shapes.extend(polygons(shape))
    if not component_shapes:
        raise ValueError(f"no projected evidence for {wall['name']}")

    # Remove detached scan/profile flecks while retaining meaningful wall
    # panels and lintels. Through-holes survive because they are polygon rings,
    # not disconnected foreground components.
    largest = max(poly.area for poly in component_shapes)
    keep = [poly for poly in component_shapes
            if poly.area >= max(0.20, largest * 0.025)]
    projected = unary_union(keep).buffer(0)
    along0, along1 = sorted(map(float, wall["along_m"]))
    z0 = floor_z + float(wall["z_mm"][0]) / 1000.0
    z1 = floor_z + float(wall["z_mm"][1]) / 1000.0
    projected = projected.intersection(box(along0, z0, along1, z1)).buffer(0)

    clean_shapes = []
    for poly in polygons(projected):
        simplified = poly.simplify(0.008, preserve_topology=True).buffer(0)
        for item in polygons(simplified):
            holes = [ring.coords for ring in item.interiors
                     if Polygon(ring).area >= 0.10]
            cleaned = Polygon(item.exterior.coords, holes).buffer(0)
            if cleaned.area >= 0.20:
                clean_shapes.extend(polygons(cleaned))
    if not clean_shapes:
        raise ValueError(f"profile flattening removed all of {wall['name']}")

    centre = 0.5 * sum(map(float, wall["across_m"]))
    thickness = float(wall["solid_thickness_mm"]) / 1000.0
    cross_lo, cross_hi = centre - thickness / 2.0, centre + thickness / 2.0
    solids = [extrude_along_cross(poly, wall["axis"], cross_lo, cross_hi)
              for poly in clean_shapes]
    # Keep disconnected silhouette regions as separate SketchUp solids. A CSG
    # union of regions that touch at only one scan-grid vertex produces a
    # non-manifold edge even though every individual extrusion is valid.
    return solids


def boolean_union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = [item for item in meshes if item is not None and len(item.faces)]
    if len(meshes) == 1:
        return meshes[0]
    try:
        return clean_mesh(trimesh.boolean.union(meshes, engine="manifold"))
    except Exception as exc:
        log(f"  union fallback ({exc.__class__.__name__})")
        return clean_mesh(trimesh.util.concatenate(meshes))


def boolean_difference(mesh: trimesh.Trimesh,
                       cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    cutters = [item for item in cutters if item is not None and len(item.faces)]
    if not cutters:
        return mesh
    # Apply recesses one at a time.  Overlapping detections around a window can
    # produce a zero-width CSG ribbon; rejecting only that one ambiguous cut
    # preserves the valid parent wall and every other measured niche.
    result = mesh
    for cutter in cutters:
        try:
            candidate = clean_mesh(trimesh.boolean.difference(
                [result, cutter], engine="manifold"))
            if candidate.is_watertight and abs(candidate.volume) > 1e-8:
                result = candidate
            else:
                log("  skipped one non-manifold niche overlap")
        except Exception as exc:
            log(f"  skipped one niche cut ({exc.__class__.__name__})")
    return result


def mesh_payload(name: str, kind: str, mesh: trimesh.Trimesh) -> dict:
    return {
        "name": name,
        "kind": kind,
        "colour": COLOURS.get(kind, [190, 190, 190]),
        "v": np.asarray(mesh.vertices, float).round(6).tolist(),
        "f": np.asarray(mesh.faces, int).tolist(),
    }


def opening_polygon(feature: dict, floor_z: float) -> Polygon:
    along0, along1 = sorted(map(float, feature["along_m"]))
    bottom = floor_z + float(feature.get("sill_mm", 0)) / 1000.0
    top = floor_z + float(feature.get("head_mm", feature.get("height_mm", 0))) / 1000.0
    rise = float(feature.get("arch_rise_mm", 0)) / 1000.0
    arched = feature.get("kind") == "arch" and 0.045 < rise < 0.6 * (along1 - along0)
    if not arched:
        return box(along0, bottom, along1, top)
    spring = max(bottom + 0.10, top - rise)
    centre = 0.5 * (along0 + along1)
    radius = max(0.01, 0.5 * (along1 - along0))
    points = [(along0, bottom), (along1, bottom), (along1, spring)]
    for angle in np.linspace(0, math.pi, 17):
        points.append((centre + radius * math.cos(angle), spring + rise * math.sin(angle)))
    points.append((along0, spring))
    return Polygon(points)


def accepted_openings(features: list[dict], floor_z: float, outer: Polygon) -> list[dict]:
    candidates = []
    for feature in features:
        kind = feature.get("kind")
        width = float(feature.get("width_mm", 0)) / 1000.0
        height = float(feature.get("height_mm", 0)) / 1000.0
        sill = float(feature.get("sill_mm", 0)) / 1000.0
        real_void = kind == "void" and width >= 0.80 and height >= 0.90 and sill <= 0.25
        if kind not in {"door", "window", "arch", "opening"} and not real_void:
            continue
        if width < 0.35 or height < 0.35:
            continue
        poly = opening_polygon(feature, floor_z).intersection(outer)
        if poly.is_empty or poly.area < 0.10:
            continue
        item = dict(feature)
        item["polygon"] = poly
        candidates.append(item)
    # Remove near-identical holes read twice on overlapping wall parts.
    output = []
    for item in sorted(candidates, key=lambda value: -value["polygon"].area):
        duplicate = False
        for prior in output:
            inter = item["polygon"].intersection(prior["polygon"]).area
            if inter / max(min(item["polygon"].area, prior["polygon"].area), 1e-9) > 0.82:
                duplicate = True
                break
        if not duplicate:
            output.append(item)
    return output


def feature_side(feature: dict, wall: dict, surface: trimesh.Trimesh | None,
                 floor_z: float, cross_lo: float, cross_hi: float) -> str:
    """Choose the wall face on which a niche or pilaster was measured."""
    axis = 0 if wall["axis"] == "x" else 1
    along_axis = 1 - axis
    depth = float(feature.get("depth_mm", 0)) / 1000.0
    kind = feature.get("kind")
    targets = ([cross_lo + depth, cross_hi - depth]
               if kind == "niche" else [cross_lo - depth, cross_hi + depth])
    if surface is None:
        return "low"
    vertices = np.asarray(surface.vertices, float)
    along0, along1 = sorted(map(float, feature["along_m"]))
    base = float(feature.get("base_mm", 0)) / 1000.0 + floor_z
    top = base + float(feature.get("height_mm", 0)) / 1000.0
    mask = ((vertices[:, along_axis] >= along0 - 0.03)
            & (vertices[:, along_axis] <= along1 + 0.03)
            & (vertices[:, 2] >= base - 0.03)
            & (vertices[:, 2] <= top + 0.03))
    values = vertices[mask, axis]
    if len(values) < 12:
        return "low"
    scores = [int((np.abs(values - target) <= 0.025).sum()) for target in targets]
    if scores[0] != scores[1]:
        return "low" if scores[0] > scores[1] else "high"
    distances = [float(np.quantile(np.abs(values - target), 0.1)) for target in targets]
    return "low" if distances[0] <= distances[1] else "high"


def floor_returns(las_path: Path, yaw_deg: float, floor_z: float,
                  cell: float = 0.05) -> np.ndarray:
    """Unique raw floor-return cells in the same local frame as the model."""
    samples = []
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(3_000_000):
            samples.append(np.asarray(chunk.z, float)[::20])
    z_shift = float(np.percentile(np.concatenate(samples), 0.5))
    del samples
    xy = []
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(3_000_000):
            z = np.asarray(chunk.z, float) - z_shift
            take = np.abs(z - floor_z) <= 0.032
            if take.any():
                xy.append(np.column_stack([np.asarray(chunk.x, float)[take],
                                            np.asarray(chunk.y, float)[take]]))
    if not xy:
        return np.empty((0, 2))
    points = np.concatenate(xy)
    angle = math.radians(-yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    points = np.column_stack([points[:, 0] * c - points[:, 1] * s,
                              points[:, 0] * s + points[:, 1] * c])
    keys = np.round(points / cell).astype(np.int32)
    keys = np.unique(keys, axis=0)
    return keys.astype(float) * cell


def room_polygons(wall_footprints: list[Polygon], floor_seed: Polygon,
                  raw_floor_xy: np.ndarray) -> tuple[list[Polygon], list[dict]]:
    """Recover complete room cells bounded by the wall graph."""
    wall_union = unary_union(wall_footprints).buffer(0)
    domain = wall_union.envelope.buffer(0.60, join_style=2)
    # Measured wall segments often stop 20--70 mm shy of the perpendicular
    # face at a noisy scan corner.  Close only those endpoint gaps to discover
    # rooms, then expand the accepted room back to the original measured wall
    # faces so the released floor edge is not shifted by the closing radius.
    close_gap = 0.08
    discovery_walls = wall_union.buffer(close_gap, join_style=2)
    free = domain.difference(discovery_walls)
    rooms, report = [], []
    for index, poly in enumerate(polygons(free)):
        area = poly.area
        if area < 0.30 or area > 160.0:
            continue
        overlap = poly.intersection(floor_seed).area if not floor_seed.is_empty else 0.0
        support = 0
        if len(raw_floor_xy):
            minx, miny, maxx, maxy = poly.bounds
            nearby = ((raw_floor_xy[:, 0] >= minx) & (raw_floor_xy[:, 0] <= maxx)
                      & (raw_floor_xy[:, 1] >= miny) & (raw_floor_xy[:, 1] <= maxy))
            if nearby.any():
                support = int(contains_xy(poly, raw_floor_xy[nearby, 0],
                                          raw_floor_xy[nearby, 1]).sum())
        touches_domain = poly.boundary.intersection(domain.boundary).length > 0.05
        old_ratio = overlap / max(area, 1e-9)
        density = support / max(area, 1e-9)
        accepted = ((not touches_domain and (overlap >= 0.12 or density >= 1.5))
                    or (touches_domain and old_ratio >= 0.30 and density >= 2.0))
        if not accepted:
            continue
        restored = poly.buffer(close_gap, join_style=2).difference(wall_union)
        clean = restored.buffer(-0.002, join_style=2).buffer(0.002, join_style=2)
        for part in polygons(clean):
            if part.area >= 0.25:
                rooms.append(part)
                report.append({"room": len(rooms), "area_m2": round(part.area, 3),
                               "old_floor_overlap_m2": round(overlap, 3),
                               "raw_floor_cells": support,
                               "raw_cells_per_m2": round(density, 2),
                               "bounded": not touches_domain})
    # Fallback is still continuous CAD geometry, never scan triangles.
    if not rooms:
        rooms = [poly for poly in polygons(floor_seed.buffer(0)) if poly.area >= 0.25]
    return rooms, report


def bbox_part_mesh(vertices: np.ndarray, floor_offset: float = 0.0) -> trimesh.Trimesh:
    lo, hi = np.asarray(vertices).min(0), np.asarray(vertices).max(0)
    lo, hi = np.round(lo, 3), np.round(hi, 3)
    lo[2] -= floor_offset
    hi[2] -= floor_offset
    return box_mesh(lo, hi)


def repeated_raw_wall_evidence(las_path: Path, manifest: dict,
                               candidates: list[dict], stride: int = 20) -> dict:
    """Measure raw-return support on both faces, through height and over time.

    The modular classifier is only the proposal stage. This independent pass
    goes back to the aligned LAS returns and tests both proposed wall faces.
    Sampling is deterministic; counts in the report are sampled-return counts.
    """
    z_samples, time_samples = [], []
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(3_000_000):
            z_samples.append(np.asarray(chunk.z, float)[::200])
            time_samples.append(np.asarray(chunk.gps_time, float)[::200])
    z_shift = float(np.percentile(np.concatenate(z_samples), 0.5))
    time_edges = np.quantile(np.concatenate(time_samples), [0.25, 0.50, 0.75])
    del z_samples, time_samples

    stats = {
        item["name"]: {
            "faces": [{"returns": 0, "time": np.zeros(4, dtype=np.int64),
                       "height": np.zeros(6, dtype=np.int64)} for _ in range(2)]
        }
        for item in candidates
    }
    angle = math.radians(-float(manifest["yaw_deg"]))
    c, s = math.cos(angle), math.sin(angle)
    floor_z = float(manifest["floor_z"])
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(3_000_000):
            x = np.asarray(chunk.x, float)[::stride]
            y = np.asarray(chunk.y, float)[::stride]
            z = np.asarray(chunk.z, float)[::stride] - z_shift
            gps_time = np.asarray(chunk.gps_time, float)[::stride]
            xx = x * c - y * s
            yy = x * s + y * c
            time_bin = np.digitize(gps_time, time_edges)
            for item in candidates:
                axis_index = 0 if item["axis"] == "x" else 1
                cross = xx if axis_index == 0 else yy
                along = yy if axis_index == 0 else xx
                along0, along1 = sorted(map(float, item["along_m"]))
                z0 = floor_z + max(0.15, float(item["z_mm"][0]) / 1000.0 + 0.15)
                z1 = floor_z + float(item["z_mm"][1]) / 1000.0 - 0.10
                if z1 <= z0:
                    continue
                interior = ((along >= along0 + 0.03) & (along <= along1 - 0.03)
                            & (z >= z0) & (z <= z1))
                for face_index, target in enumerate(map(float, item["across_m"])):
                    take = interior & (np.abs(cross - target) <= 0.025)
                    if not take.any():
                        continue
                    face = stats[item["name"]]["faces"][face_index]
                    face["returns"] += int(take.sum())
                    face["time"] += np.bincount(time_bin[take], minlength=4)[:4]
                    height_bin = np.clip(
                        ((z[take] - z0) / (z1 - z0) * 6).astype(int), 0, 5)
                    face["height"] += np.bincount(height_bin, minlength=6)[:6]

    output = {}
    for item in candidates:
        raw = stats[item["name"]]
        faces = []
        for face in raw["faces"]:
            faces.append({
                "sampled_returns": int(face["returns"]),
                "time_bins": face["time"].tolist(),
                "height_bins": face["height"].tolist(),
                "height_bins_supported": int((face["height"] >= 10).sum()),
            })
        combined_time = raw["faces"][0]["time"] + raw["faces"][1]["time"]
        repeated_bins = int((combined_time >= 10).sum())
        length_m = float(item.get("length_mm", 0)) / 1000.0
        passed = (length_m >= 1.0
                  and all(face["sampled_returns"] >= 100 for face in faces)
                  and all(face["height_bins_supported"] >= 5 for face in faces)
                  and repeated_bins >= 2)
        output[item["name"]] = {
            "passed": bool(passed),
            "length_m": length_m,
            "repeated_time_bins": repeated_bins,
            "faces": faces,
            "gate": "length>=1m; both faces>=100 sampled returns; both faces cover >=5/6 height bins; >=2 time bins",
        }
    return output


def make_level(level: int, out_dir: Path) -> dict:
    source_dir = ROOT / "output_final" / f"soulace_L{level}"
    manifest = json.loads((source_dir / "modular" / "manifest.json").read_text())
    wall_groups = read_groups(source_dir / "modular" / "modular_solid.obj")
    parts_obj = ROOT / "Soulace Sketchup" / f"{LEVEL_STEMS[level]}_parts.obj"
    part_groups = read_groups(parts_obj)
    surface_scene = trimesh.load(source_dir / "modular" / "modular_view.glb",
                                 force="scene", process=False)
    surface_groups = surface_scene.geometry
    floor_z = float(manifest["floor_z"])
    features_by_wall = defaultdict(list)
    for feature in manifest.get("features", []):
        features_by_wall[feature.get("wall")].append(feature)
    parts_by_name = {item["name"]: item for item in manifest["parts"]}

    # Column boxes are measured parts.  If their footprint touches a wall, the
    # column is a thickness step of that wall and becomes the same SketchUp group.
    column_meshes = {}
    for part in manifest["parts"]:
        if part["kind"] != "column" or part["name"] not in surface_groups:
            continue
        bounds = np.asarray(surface_groups[part["name"]].bounds, float)
        centre = bounds.mean(axis=0)
        footprint = np.asarray(part.get("footprint_m", bounds[1, :2] - bounds[0, :2]), float)
        lo = [centre[0] - footprint[0] / 2, centre[1] - footprint[1] / 2, floor_z]
        hi = [centre[0] + footprint[0] / 2, centre[1] + footprint[1] / 2,
              float(manifest["structural_ceiling_z"])]
        column_meshes[part["name"]] = box_mesh(lo, hi)

    wall_records = []
    wall_footprints = []
    wall_additions = defaultdict(list)
    standalone_columns = dict(column_meshes)
    for wall in manifest["parts"]:
        if wall["kind"] not in {"wall", "parapet"} or wall["name"] not in wall_groups:
            continue
        vertices = np.asarray(wall_groups[wall["name"]][0], float)
        axis_index = 0 if wall["axis"] == "x" else 1
        along_index = 1 - axis_index
        cross_lo, cross_hi = sorted([snap(vertices[:, axis_index].min(), 0.001),
                                     snap(vertices[:, axis_index].max(), 0.001)])
        along_lo, along_hi = sorted([snap(vertices[:, along_index].min()),
                                     snap(vertices[:, along_index].max())])
        z_lo = floor_z
        z_hi = snap(vertices[:, 2].max(), 0.005)
        footprint = (box(cross_lo, along_lo, cross_hi, along_hi)
                     if wall["axis"] == "x"
                     else box(along_lo, cross_lo, along_hi, cross_hi))
        wall_footprints.append(footprint)
        wall_records.append({"wall": wall, "cross": [cross_lo, cross_hi],
                             "along": [along_lo, along_hi], "z": [z_lo, z_hi],
                             "footprint": footprint})

    for column_name, column_mesh in list(column_meshes.items()):
        column_bounds = np.asarray(column_mesh.bounds)
        cb = box(column_bounds[0, 0], column_bounds[0, 1],
                 column_bounds[1, 0], column_bounds[1, 1])
        best = None
        for record in wall_records:
            distance = cb.distance(record["footprint"])
            overlap = cb.intersection(record["footprint"]).area
            score = overlap * 100 - distance
            if distance <= 0.055 and (best is None or score > best[0]):
                best = (score, record["wall"]["name"])
        if best is not None:
            wall_additions[best[1]].append(column_mesh)
            standalone_columns.pop(column_name, None)

    payloads, visible_meshes = [], []
    wall_report, opening_report, relief_report = [], [], []
    plan_walls = []
    for record in wall_records:
        wall = record["wall"]
        cross_lo, cross_hi = record["cross"]
        along_lo, along_hi = record["along"]
        z_lo, z_hi = record["z"]
        outer = box(along_lo, z_lo, along_hi, z_hi)
        openings = accepted_openings(features_by_wall[wall["name"]], floor_z, outer)
        remaining = outer.difference(unary_union([item["polygon"] for item in openings]))
        base_meshes = [extrude_along_cross(poly, wall["axis"], cross_lo, cross_hi)
                       for poly in polygons(remaining) if poly.area > 0.005]
        if not base_meshes:
            continue
        mesh = boolean_union(base_meshes)
        additions = list(wall_additions[wall["name"]])
        cutters = []
        surface = surface_groups.get(wall["name"])
        for feature in features_by_wall[wall["name"]]:
            if feature.get("kind") not in {"niche", "pilaster"}:
                continue
            width = float(feature.get("width_mm", 0)) / 1000.0
            height = float(feature.get("height_mm", 0)) / 1000.0
            depth = float(feature.get("depth_mm", 0)) / 1000.0
            if width < 0.05 or height < 0.10 or depth < 0.035:
                continue
            if feature.get("kind") == "niche" and depth >= cross_hi - cross_lo - 0.025:
                continue
            along0, along1 = sorted(map(float, feature["along_m"]))
            along0, along1 = max(along0, along_lo), min(along1, along_hi)
            fz0 = floor_z + float(feature.get("base_mm", 0)) / 1000.0
            fz1 = min(z_hi, fz0 + height)
            if along1 - along0 < 0.04 or fz1 - fz0 < 0.08:
                continue
            feature_face = box(along0, fz0, along1, fz1)
            if feature["kind"] == "niche":
                # The same window/door can also appear as a deep rectangle in
                # the surface-depth pass.  It is a through opening, not a blind
                # niche; subtracting it twice creates coincident CSG faces.
                overlap = max((feature_face.intersection(item["polygon"]).area
                               for item in openings), default=0.0)
                if overlap / max(feature_face.area, 1e-9) > 0.35:
                    continue
            side = feature_side(feature, wall, surface, floor_z, cross_lo, cross_hi)
            if feature["kind"] == "niche":
                c0, c1 = ((cross_lo - 0.005, cross_lo + depth)
                          if side == "low" else (cross_hi - depth, cross_hi + 0.005))
            else:
                c0, c1 = ((cross_lo - depth, cross_lo)
                          if side == "low" else (cross_hi, cross_hi + depth))
            if wall["axis"] == "x":
                feature_mesh = box_mesh([c0, along0, fz0], [c1, along1, fz1])
            else:
                feature_mesh = box_mesh([along0, c0, fz0], [along1, c1, fz1])
            (cutters if feature["kind"] == "niche" else additions).append(feature_mesh)
            relief_report.append({"wall": wall["name"], "kind": feature["kind"],
                                  "side": side, "along_m": [along0, along1],
                                  "z_m": [fz0, fz1], "depth_mm": round(depth * 1000, 1)})
        mesh = boolean_difference(mesh, cutters)
        mesh = boolean_union([mesh] + additions)
        mesh.apply_translation([0, 0, -floor_z])
        measured = bool(wall.get("solid_thickness_measured"))
        kind = "wall_measured" if measured else "wall_inferred"
        payload = mesh_payload(wall["name"], kind, mesh)
        if wall["name"] in SKETCHUP_DAE_WALLS.get(level, set()):
            dae_dir = out_dir / "dae"
            dae_dir.mkdir(parents=True, exist_ok=True)
            dae_path = dae_dir / f"{LEVEL_STEMS[level]}_{wall['name']}.dae"
            write_dae(mesh, dae_path, wall["name"])
            payload["dae"] = dae_path.resolve().as_posix()
        payloads.append(payload)
        visible_meshes.append((mesh, COLOURS[kind]))
        wall_report.append({"wall": wall["name"], "axis": wall["axis"],
                            "thickness_mm": round((cross_hi - cross_lo) * 1000, 1),
                            "height_mm": round((z_hi - z_lo) * 1000, 1),
                            "openings": len(openings), "niches": len(cutters),
                            "profile_additions": len(additions),
                            "watertight": bool(mesh.is_watertight)})
        for item in openings:
            clean = {key: value for key, value in item.items() if key != "polygon"}
            clean["wall"] = wall["name"]
            opening_report.append(clean)
        cut_shapes = []
        cut_z = floor_z + 1.20
        for item in openings:
            minx, minz, maxx, maxz = item["polygon"].bounds
            if minz <= cut_z <= maxz:
                if wall["axis"] == "x":
                    cut_shapes.append(box(cross_lo, minx, cross_hi, maxx))
                else:
                    cut_shapes.append(box(minx, cross_lo, maxx, cross_hi))
        plan_walls.append(record["footprint"].difference(unary_union(cut_shapes)))

    raw_xy = floor_returns(source_dir / "lidar" / f"L{level}.las",
                           float(manifest["yaw_deg"]), floor_z)
    floor_seed_shapes = []
    for name, (vertices, _) in part_groups.items():
        if not name.startswith("floor_"):
            continue
        bounds = np.asarray(vertices).min(0), np.asarray(vertices).max(0)
        floor_seed_shapes.append(box(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
    floor_seed = unary_union(floor_seed_shapes).buffer(0) if floor_seed_shapes else Polygon()
    rooms, room_report = room_polygons(wall_footprints, floor_seed, raw_xy)
    floor_shapes = []
    for index, room in enumerate(rooms, 1):
        slab = trimesh.creation.extrude_polygon(room, height=0.10, engine="earcut")
        slab.apply_translation([0, 0, floor_z - 0.10])
        slab.apply_translation([0, 0, -floor_z])
        payloads.append(mesh_payload(f"floor_room_{index:02d}", "floor", clean_mesh(slab)))
        visible_meshes.append((slab, COLOURS["floor"]))
        floor_shapes.append(room)

    # Continuous finish floor through every walk-through opening.
    threshold_count = 0
    wall_lookup = {item["wall"]["name"]: item for item in wall_records}
    for opening in opening_report:
        if float(opening.get("sill_mm", 0)) > 150 or opening["wall"] not in wall_lookup:
            continue
        record = wall_lookup[opening["wall"]]
        wall = record["wall"]
        a0, a1 = sorted(map(float, opening["along_m"]))
        c0, c1 = record["cross"]
        if wall["axis"] == "x":
            lo, hi = [c0, a0, -0.10], [c1, a1, 0.0]
            floor_shapes.append(box(c0, a0, c1, a1))
        else:
            lo, hi = [a0, c0, -0.10], [a1, c1, 0.0]
            floor_shapes.append(box(a0, c0, a1, c1))
        threshold = box_mesh(lo, hi)
        threshold_count += 1
        payloads.append(mesh_payload(f"floor_threshold_{threshold_count:02d}", "floor", threshold))
        visible_meshes.append((threshold, COLOURS["floor"]))

    for name, mesh in standalone_columns.items():
        mesh.apply_translation([0, 0, -floor_z])
        payloads.append(mesh_payload(name, "column", mesh))
        visible_meshes.append((mesh, COLOURS["column"]))

    # Keep clean structural/ceiling parts from the previous box stage.  They
    # were not the source of the missing-wall problem and remain useful.
    for name, (vertices, faces) in part_groups.items():
        if name.startswith("beam_"):
            kind = "beam"
        elif name.startswith("dropped_ceiling_"):
            kind = "dropped_ceiling"
        elif name.startswith("ceiling_"):
            kind = "ceiling"
        else:
            continue
        if kind in {"beam", "ceiling", "dropped_ceiling"}:
            # These are rectilinear structural/finish envelopes.  The earlier
            # box handover sometimes stored several coincident surface shards
            # in one group; its measured bounds are sound, but the shell is
            # not.  Recreate the exact envelope as one valid solid.
            mesh = bbox_part_mesh(np.asarray(vertices, float), floor_offset=floor_z)
        else:
            mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, int), process=False)
            mesh.apply_translation([0, 0, -floor_z])
        payloads.append(mesh_payload(name, kind, clean_mesh(mesh)))
        if kind == "beam":
            visible_meshes.append((mesh, COLOURS[kind]))

    build_path = out_dir / f"{LEVEL_STEMS[level]}_asbuilt.build.json"
    build_path.write_text(json.dumps({"parts": payloads}), encoding="utf-8")
    report = {
        "level": level,
        "source_las": str((source_dir / "lidar" / f"L{level}.las").resolve()),
        "floor_datum_m": 0.0,
        "source_floor_z_m": floor_z,
        "source_yaw_deg": manifest["yaw_deg"],
        "parts": len(payloads),
        "walls": wall_report,
        "openings": opening_report,
        "wall_relief": relief_report,
        "rooms": room_report,
        "raw_floor_cells_50mm": len(raw_xy),
        "floor_thresholds": threshold_count,
    }
    report_path = out_dir / f"{LEVEL_STEMS[level]}_asbuilt_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_plan(plan_walls, floor_shapes, out_dir / f"{LEVEL_STEMS[level]}_plan.png")
    render_iso(visible_meshes, out_dir / f"{LEVEL_STEMS[level]}_iso.png")
    return {"level": level, "payloads": payloads, "build": build_path,
            "report": report_path, "visible": visible_meshes}


def make_verified_level(level: int, out_dir: Path) -> dict:
    """Build the conservative release: measured two-face wall solids only.

    This intentionally omits any object whose existence or thickness was
    completed by a class prior. In particular, one-face walls, automatic
    column boxes, recovered room floors, beam envelopes and relief/opening
    classifications remain outside the trusted SketchUp model. The source
    solid for each retained wall is used directly instead of rebuilding a
    full-height rectangle around it, so unobserved wall spans are not added.
    """
    source_dir = ROOT / "output_final" / f"soulace_L{level}"
    manifest = json.loads((source_dir / "modular" / "manifest.json").read_text())
    wall_groups = read_groups(source_dir / "modular" / "modular_solid.obj")
    floor_z = float(manifest["floor_z"])
    payloads, visible_meshes, plan_walls = [], [], []
    included, excluded = [], []
    dae_dir = out_dir / "dae_verified"
    dae_dir.mkdir(parents=True, exist_ok=True)
    wall_candidates = [
        part for part in manifest["parts"]
        if part.get("kind") in {"wall", "parapet"}
        and bool(part.get("solid_thickness_measured"))
        and int(part.get("faces_seen") or 0) >= 2
    ]
    raw_evidence = repeated_raw_wall_evidence(
        source_dir / "lidar" / f"L{level}.las", manifest, wall_candidates)

    for part in manifest["parts"]:
        name = part["name"]
        kind = part.get("kind")
        if kind not in {"wall", "parapet"}:
            if kind == "column":
                excluded.append({
                    "name": name,
                    "kind": kind,
                    "reason": "automatic column cluster has no independent opposing-face validation",
                })
            elif kind in {"beam", "ceiling", "dropped_ceiling", "floor"}:
                excluded.append({
                    "name": name,
                    "kind": kind,
                    "reason": "envelope or boundary is not independently validated",
                })
            continue

        faces_seen = int(part.get("faces_seen") or 0)
        two_face = bool(part.get("solid_thickness_measured")) and faces_seen >= 2
        repeat_pass = bool(raw_evidence.get(name, {}).get("passed"))
        if not two_face or not repeat_pass or name not in wall_groups:
            if not two_face:
                reason = "wall thickness was inferred; opposing LiDAR faces were not accepted"
            elif not repeat_pass:
                reason = "opposing faces did not pass repeat/full-height raw-return validation"
            else:
                reason = "measured wall source solid is missing"
            excluded.append({
                "name": name,
                "kind": kind,
                "faces_seen": faces_seen,
                "raw_evidence": raw_evidence.get(name),
                "reason": reason,
            })
            continue

        vertices, faces = wall_groups[name]
        source_mesh = clean_mesh(trimesh.Trimesh(
            np.asarray(vertices, float), np.asarray(faces, int), process=False))
        meshes = flatten_wall_profiles(source_mesh, part, floor_z)
        if not meshes or not all(mesh.is_watertight for mesh in meshes):
            excluded.append({
                "name": name,
                "kind": kind,
                "faces_seen": faces_seen,
                "raw_evidence": raw_evidence.get(name),
                "reason": "profile-free silhouette was topologically ambiguous; omitted instead of repaired by inference",
            })
            continue
        for part_index, mesh in enumerate(meshes, 1):
            mesh.apply_translation([0, 0, -floor_z])
            payload_name = (name if len(meshes) == 1
                            else f"{name}_part_{part_index:02d}")
            payload = mesh_payload(payload_name, "wall_measured", mesh)
            dae_path = dae_dir / f"{LEVEL_STEMS[level]}_{payload_name}_verified.dae"
            write_dae(mesh, dae_path, payload_name)
            payload["dae"] = dae_path.resolve().as_posix()
            payloads.append(payload)
            visible_meshes.append((mesh, COLOURS["wall_measured"]))
            bounds = np.asarray(mesh.bounds, float)
            plan_walls.append(box(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]))
        included.append({
            "name": name,
            "kind": kind,
            "faces_seen": faces_seen,
            "thickness_mm": part.get("solid_thickness_mm"),
            "all_solid_parts_watertight": all(mesh.is_watertight for mesh in meshes),
            "solid_parts": len(meshes),
            "wall_profiles": "flattened/ignored",
            "raw_evidence": raw_evidence[name],
            "reason": "opposing faces measured and repeated through height in raw LAS returns",
        })

    for feature in manifest.get("features", []):
        excluded.append({
            "name": f"{feature.get('kind', 'feature')}@{feature.get('wall', 'unknown')}",
            "kind": feature.get("kind", "feature"),
            "reason": "feature classification lacks independent ray/temporal confirmation",
        })

    stem = f"{LEVEL_STEMS[level]}_verified"
    build_path = out_dir / f"{stem}.build.json"
    build_path.write_text(json.dumps({"parts": payloads}), encoding="utf-8")
    report = {
        "level": level,
        "release": "verified-only",
        "policy": {
            "walls": "two measured faces plus repeat/full-height raw LAS support",
            "columns": "excluded pending independent opposing-face validation",
            "features": "excluded pending independent ray/temporal validation",
            "floors_beams_ceilings": "excluded because reconstructed envelopes are not direct evidence",
        },
        "source_las": str((source_dir / "lidar" / f"L{level}.las").resolve()),
        "parts": len(payloads),
        "included": included,
        "excluded": excluded,
    }
    report_path = out_dir / f"{stem}_evidence.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_plan(plan_walls, [], out_dir / f"{stem}_plan.png")
    render_iso(visible_meshes, out_dir / f"{stem}_iso.png")
    return {"level": level, "payloads": payloads, "build": build_path,
            "report": report_path, "visible": visible_meshes}


def render_plan(walls: list, floors: list, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 10), dpi=180)
    for shape in floors:
        for poly in polygons(shape):
            x, y = poly.exterior.xy
            axis.fill(x, y, color="#deddd8", edgecolor="#aaa8a0", linewidth=0.35)
    for shape in walls:
        for poly in polygons(shape):
            x, y = poly.exterior.xy
            axis.fill(x, y, color="#2e3943", edgecolor="#182028", linewidth=0.55)
            for ring in poly.interiors:
                x, y = ring.xy
                axis.fill(x, y, color="white")
    axis.set_aspect("equal")
    axis.autoscale()
    axis.margins(0.03)
    axis.axis("off")
    figure.tight_layout(pad=0.1)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(figure)


def render_iso(items: list[tuple[trimesh.Trimesh, list[int]]], destination: Path) -> None:
    figure = plt.figure(figsize=(14, 10), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    bounds = []
    for mesh, colour in items:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if len(faces) > 4500:
            faces = faces[np.linspace(0, len(faces) - 1, 4500).astype(int)]
        collection = Poly3DCollection(vertices[faces], linewidths=0.08,
                                      edgecolors=(0.18, 0.20, 0.22, 0.18))
        collection.set_facecolor(tuple(value / 255 for value in colour) + (0.92,))
        axis.add_collection3d(collection)
        bounds.append(mesh.bounds)
    if bounds:
        lo = np.min([item[0] for item in bounds], axis=0)
        hi = np.max([item[1] for item in bounds], axis=0)
        centre = 0.5 * (lo + hi)
        radius = 0.55 * max(hi[0] - lo[0], hi[1] - lo[1], (hi[2] - lo[2]) * 1.8)
        axis.set_xlim(centre[0] - radius, centre[0] + radius)
        axis.set_ylim(centre[1] - radius, centre[1] + radius)
        axis.set_zlim(min(-0.12, lo[2]), max(hi[2], 0.5))
    axis.view_init(elev=31, azim=-58)
    axis.set_box_aspect((1, 1, 0.42))
    axis.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(figure)


def build_native(build_path: Path, skp_path: Path) -> dict:
    ruby = (ROOT / "scripts" / "export" / "ruby" / "pcm_build.rb").as_posix()
    code = (f'load {json.dumps(ruby)}; PCMBuild.build('
            f'{json.dumps(build_path.resolve().as_posix())}, '
            f'{json.dumps(skp_path.resolve().as_posix())})')
    response = rb(code, timeout=1800)
    if not response.get("ok"):
        raise RuntimeError(f"SketchUp build failed: {response.get('error')}")
    result = response["result"]
    return json.loads(result) if isinstance(result, str) else result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output_final/soulace_asbuilt_v2")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--skip-skp", action="store_true")
    parser.add_argument("--verified-only", action="store_true",
                        help="release only directly supported two-face wall solids")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [int(value) for value in args.levels.split(",")]
    results = []
    for level in wanted:
        if args.verified_only:
            log(f"=== Soulace L{level}: verified evidence only ===")
            results.append(make_verified_level(level, out_dir))
        else:
            log(f"=== Soulace L{level}: clean as-built ===")
            results.append(make_level(level, out_dir))

    # Stacked handover uses the measured 3.20 m and 6.50 m level offsets from
    # the raw SLAM storey split.  Per-level files remain at their own Z=0.
    storeys = json.loads((ROOT / "soulace_output" / "storeys.json").read_text())["storeys"]
    base_floor = float(storeys[0]["floor_z"])
    combined = []
    for result in results:
        level = result["level"]
        target_z = float(storeys[level]["floor_z"]) - base_floor
        for payload in result["payloads"]:
            item = dict(payload)
            item["name"] = f"L{level}_{payload['name']}"
            vertices = np.asarray(payload["v"], float)
            vertices[:, 2] += target_z
            item["v"] = vertices.round(6).tolist()
            if payload.get("dae"):
                dae_dir = out_dir / "dae"
                dae_dir.mkdir(parents=True, exist_ok=True)
                dae_path = dae_dir / f"Soulace_all_L{level}_{payload['name']}.dae"
                dae_mesh = trimesh.Trimesh(vertices, np.asarray(payload["f"], int), process=False)
                write_dae(dae_mesh, dae_path, item["name"])
                item["dae"] = dae_path.resolve().as_posix()
            combined.append(item)
    combined_name = ("Soulace_verified_only" if args.verified_only
                     else "Soulace_all_levels_asbuilt")
    combined_path = out_dir / f"{combined_name}.build.json"
    combined_path.write_text(json.dumps({"parts": combined}), encoding="utf-8")

    if args.verified_only:
        level_audits = [json.loads(result["report"].read_text()) for result in results]
        audit = {
            "release": "verified-only",
            "included_wall_count": sum(len(item["included"]) for item in level_audits),
            "excluded_item_count": sum(len(item["excluded"]) for item in level_audits),
            "levels": [
                {
                    "level": item["level"],
                    "included": item["included"],
                    "excluded": item["excluded"],
                }
                for item in level_audits
            ],
            "important": "Excluded does not mean absent; it means the scan evidence did not satisfy the automatic release gate.",
        }
        (out_dir / "Soulace_verified_evidence_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8")

    build_results = []
    if not args.skip_skp:
        for result in results:
            suffix = "verified" if args.verified_only else "asbuilt"
            skp = out_dir / f"{LEVEL_STEMS[result['level']]}_{suffix}.skp"
            log(f"native SketchUp -> {skp.name}")
            build_results.append(build_native(result["build"], skp))
        all_skp = out_dir / f"{combined_name}.skp"
        log(f"native SketchUp -> {all_skp.name}")
        build_results.append(build_native(combined_path, all_skp))
    summary = {"levels": wanted, "level_parts": {str(r["level"]): len(r["payloads"]) for r in results},
               "release": "verified-only" if args.verified_only else "review",
               "combined_parts": len(combined), "native_builds": build_results}
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
