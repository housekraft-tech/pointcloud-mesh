"""Measure variable-thickness wall assemblies in two registered LiDAR walks.

The base wall graph contains architectural planes.  This stage unfolds every
wall into local ``along x height x depth`` evidence.  Full-height thickness
steps become embedded columns; upper-only proud regions become beams/soffits;
bounded recesses become niches.  A candidate is retained only when the same
region is present in both independently registered scans.

This script produces evidence and a review plot.  It deliberately does not put
unvalidated relief into the SketchUp model; the build stage consumes only
``metric_status == 'pass'`` features.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))
from snap_to_planes import load_frame  # noqa: E402


CELL_M = 0.05
DEPTH_BIN_M = 0.005
MIN_RELIEF_M = 0.035


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _face_is_measured(wall: dict, face_index: int) -> bool:
    if wall["thickness_source"] == "measured_two_faces":
        return True
    measured = wall.get("measured_face_m")
    return measured is not None and abs(wall["faces_m"][face_index] - measured) < 1e-5


def _surface_mode(values: np.ndarray) -> tuple[float, int] | None:
    """Dominant surface depth in one elevation cell."""
    if len(values) < 2:
        return None
    edges = np.arange(-0.325, 0.356, DEPTH_BIN_M)
    hist, edges = np.histogram(values, edges)
    index = int(np.argmax(hist))
    if hist[index] < 2:
        return None
    lo, hi = edges[index], edges[index + 1]
    selected = values[(values >= lo) & (values < hi)]
    return float(np.median(selected)), int(len(selected))


def relief_map(
    xyz: np.ndarray, wall: dict, face_index: int, floor_z: float, ceiling_z: float
) -> dict | None:
    """Unfold one physical wall face into dominant surface depth cells."""
    axis = 0 if wall["axis"] == "x" else 1
    along_axis = 1 - axis
    face = float(wall["faces_m"][face_index])
    outward = -1.0 if face_index == 0 else 1.0
    along0, along1 = map(float, wall["span_m"])
    z0, z1 = floor_z + 0.06, ceiling_z - 0.04
    depth = (xyz[:, axis] - face) * outward  # + into room, - into masonry
    mask = (
        (xyz[:, along_axis] >= along0)
        & (xyz[:, along_axis] <= along1)
        & (xyz[:, 2] >= z0)
        & (xyz[:, 2] <= z1)
        & (depth >= -0.32)
        & (depth <= 0.35)
    )
    if mask.sum() < 300:
        return None
    along = xyz[mask, along_axis]
    height = xyz[mask, 2]
    depth = depth[mask]
    nu = max(2, int(np.ceil((along1 - along0) / CELL_M)))
    nz = max(2, int(np.ceil((z1 - z0) / CELL_M)))
    iu = np.clip(((along - along0) / CELL_M).astype(np.int32), 0, nu - 1)
    iz = np.clip(((height - z0) / CELL_M).astype(np.int32), 0, nz - 1)
    key = iu.astype(np.int64) * nz + iz
    order = np.argsort(key, kind="stable")
    key, depth = key[order], depth[order]
    surface = np.full(nu * nz, np.nan, dtype=np.float32)
    support = np.zeros(nu * nz, dtype=np.uint16)
    splits = np.flatnonzero(np.diff(key)) + 1
    for group in np.split(np.arange(len(key)), splits):
        result = _surface_mode(depth[group])
        if result is None:
            continue
        value, count = result
        cell = int(key[group[0]])
        surface[cell] = value
        support[cell] = min(count, np.iinfo(np.uint16).max)
    surface = surface.reshape(nu, nz)
    support = support.reshape(nu, nz)
    have = np.isfinite(surface)
    if have.sum() < 30:
        return None
    # Recess into masonry is positive; proud into room is negative.
    relief = -surface
    baseline_cells = have & (np.abs(relief) <= 0.035)
    baseline = float(np.median(relief[baseline_cells])) if baseline_cells.any() else 0.0
    relief -= baseline
    return {
        "relief_m": relief,
        "support": support,
        "have": have,
        "along0": along0,
        "z0": z0,
        "nu": nu,
        "nz": nz,
        "face_m": face,
        "face_index": face_index,
        "outward": outward,
    }


def _component_rectangles(mask: np.ndarray, relief: np.ndarray) -> list[dict]:
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((2, 2), bool))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    output = []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        if len(cells) < 10:
            continue
        u0, z0 = cells.min(axis=0)
        u1, z1 = cells.max(axis=0) + 1
        width = (u1 - u0) * CELL_M
        height = (z1 - z0) * CELL_M
        area = len(cells) * CELL_M * CELL_M
        if width < 0.10 or height < 0.12 or area < 0.025:
            continue
        values = np.abs(relief[labels == label])
        output.append(
            {
                "u_cells": [int(u0), int(u1)],
                "z_cells": [int(z0), int(z1)],
                "width_m": float(width),
                "height_m": float(height),
                "area_m2": float(area),
                "depth_m": float(np.median(values)),
            }
        )
    return output


def detect_relief(grid: dict, wall: dict, floor_z: float, ceiling_z: float) -> list[dict]:
    relief = grid["relief_m"]
    have = grid["have"]
    candidates = []
    masks = {
        "niche": have & (relief >= MIN_RELIEF_M),
        "proud": have & (relief <= -MIN_RELIEF_M),
    }
    for initial_kind, mask in masks.items():
        for item in _component_rectangles(mask, relief):
            along = [
                grid["along0"] + item["u_cells"][0] * CELL_M,
                grid["along0"] + item["u_cells"][1] * CELL_M,
            ]
            z = [
                grid["z0"] + item["z_cells"][0] * CELL_M,
                min(grid["z0"] + item["z_cells"][1] * CELL_M, ceiling_z),
            ]
            depth = item["depth_m"]
            if initial_kind == "niche":
                # A recess deeper than the remaining wall body is a void or an
                # incorrectly associated opposite face, never a blind niche.
                if depth >= wall["thickness_mm"] / 1000 - 0.025 or item["area_m2"] > 3.0:
                    continue
                kind = "niche"
            elif z[1] >= ceiling_z - 0.16 and item["height_m"] <= 1.20 \
                    and item["width_m"] >= 0.25:
                kind = "beam_or_soffit"
            elif z[0] <= floor_z + 0.20 and z[1] >= ceiling_z - 0.25 \
                    and 0.10 <= item["width_m"] <= 1.80:
                kind = "embedded_column"
            else:
                kind = "proud_relief_review"
            candidates.append(
                {
                    "kind": kind,
                    "wall_id": wall["wall_id"],
                    "face_index": grid["face_index"],
                    "face_m": grid["face_m"],
                    "outward": grid["outward"],
                    "along_m": [float(value) for value in along],
                    "z_m": [float(value) for value in z],
                    "width_mm": round(item["width_m"] * 1000, 1),
                    "height_mm": round((z[1] - z[0]) * 1000, 1),
                    "depth_mm": round(depth * 1000, 1),
                    "area_m2": round(item["area_m2"], 3),
                }
            )
    # Connected 2-D relief fragments easily split a true column where furniture
    # or an open door occludes a few height cells.  Aggregate a second profile
    # along the wall: a column is a bounded proud run whose evidence spans most
    # of the storey, even if that evidence is not one unbroken component.
    proud = have & (relief <= -MIN_RELIEF_M)
    hot = np.zeros(grid["nu"], dtype=bool)
    depth_profile = np.full(grid["nu"], np.nan)
    height_required = max(1.45, 0.60 * (ceiling_z - floor_z))
    for u in range(grid["nu"]):
        cells = np.flatnonzero(proud[u])
        available = int(have[u].sum())
        if len(cells) < max(6, int(0.22 * max(available, 1))):
            continue
        vertical_span = (cells.max() - cells.min() + 1) * CELL_M
        values = -relief[u, cells]
        depth = float(np.median(values))
        if vertical_span >= height_required and 0.035 <= depth <= 0.18:
            hot[u] = True
            depth_profile[u] = depth
    hot = ndimage.binary_closing(hot, structure=np.ones(3, bool))
    hot = ndimage.binary_opening(hot, structure=np.ones(2, bool))
    labels, count = ndimage.label(hot)
    for label in range(1, count + 1):
        cells = np.flatnonzero(labels == label)
        if not len(cells):
            continue
        u0, u1 = int(cells.min()), int(cells.max()) + 1
        width = (u1 - u0) * CELL_M
        if not 0.12 <= width <= 1.50:
            continue
        values = depth_profile[cells]
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        candidates.append(
            {
                "kind": "embedded_column",
                "wall_id": wall["wall_id"],
                "face_index": grid["face_index"],
                "face_m": grid["face_m"],
                "outward": grid["outward"],
                "along_m": [grid["along0"] + u0 * CELL_M,
                              grid["along0"] + u1 * CELL_M],
                "z_m": [float(floor_z), float(ceiling_z)],
                "width_mm": round(width * 1000, 1),
                "height_mm": round((ceiling_z - floor_z) * 1000, 1),
                "depth_mm": round(float(np.median(values)) * 1000, 1),
                "area_m2": round(width * (ceiling_z - floor_z), 3),
                "detection": "full_height_thickness_profile",
            }
        )
    return candidates


def scan_features(
    xyz: np.ndarray, walls: list[dict], floor_z: float, ceiling_z: float,
    keep_junctions: bool = False,
) -> list[dict]:
    output = []
    for wall in walls:
        for face_index in (0, 1):
            if not _face_is_measured(wall, face_index):
                continue
            grid = relief_map(xyz, wall, face_index, floor_z, ceiling_z)
            if grid is not None:
                output.extend(detect_relief(grid, wall, floor_z, ceiling_z))
    # A short proud return exactly where a perpendicular wall meets this face is
    # already represented by that wall prism.  It is a T-junction, not a column.
    filtered = []
    for feature in output:
        if not keep_junctions and feature["kind"] == "embedded_column" \
                and feature["width_mm"] < 700:
            wall = next(item for item in walls if item["wall_id"] == feature["wall_id"])
            junction = any(
                other["axis"] != wall["axis"]
                and feature["along_m"][0] - 0.10 <= np.mean(other["faces_m"]) <= feature["along_m"][1] + 0.10
                and other["span_m"][0] - 0.25 <= np.mean(wall["faces_m"]) <= other["span_m"][1] + 0.25
                for other in walls
            )
            if junction:
                continue
        # Prefer the full-height profile over a duplicate 2-D component.
        duplicate = any(
            existing["kind"] == feature["kind"]
            and existing["wall_id"] == feature["wall_id"]
            and existing["face_index"] == feature["face_index"]
            and overlap(existing["along_m"], feature["along_m"]) > 0.5 * min(
                np.diff(existing["along_m"])[0], np.diff(feature["along_m"])[0]
            )
            for existing in filtered
        )
        if duplicate:
            if feature.get("detection") == "full_height_thickness_profile":
                filtered = [existing for existing in filtered if not (
                    existing["kind"] == feature["kind"]
                    and existing["wall_id"] == feature["wall_id"]
                    and existing["face_index"] == feature["face_index"]
                    and overlap(existing["along_m"], feature["along_m"]) > 0
                )]
                filtered.append(feature)
            continue
        filtered.append(feature)
    return filtered


def overlap(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def match_features(primary: list[dict], reference: list[dict]) -> tuple[list[dict], list[dict]]:
    matched, review = [], []
    used = set()
    for feature in primary:
        choices = []
        for index, candidate in enumerate(reference):
            if index in used or candidate["wall_id"] != feature["wall_id"] \
                    or candidate["face_index"] != feature["face_index"] \
                    or candidate["kind"] != feature["kind"]:
                continue
            u_overlap = overlap(feature["along_m"], candidate["along_m"])
            z_overlap = overlap(feature["z_m"], candidate["z_m"])
            short_u = min(np.diff(feature["along_m"])[0], np.diff(candidate["along_m"])[0])
            short_z = min(np.diff(feature["z_m"])[0], np.diff(candidate["z_m"])[0])
            if short_u <= 0 or short_z <= 0 or u_overlap < 0.35 * short_u \
                    or z_overlap < 0.35 * short_z:
                continue
            centre_delta = abs(np.mean(feature["along_m"]) - np.mean(candidate["along_m"])) \
                + abs(np.mean(feature["z_m"]) - np.mean(candidate["z_m"]))
            choices.append((centre_delta, index, candidate))
        if not choices:
            review.append({**feature, "metric_status": "review_not_repeated_in_second_walk"})
            continue
        _, index, candidate = min(choices, key=lambda item: item[0])
        used.add(index)
        edge_deltas = [
            1000 * (feature["along_m"][0] - candidate["along_m"][0]),
            1000 * (feature["along_m"][1] - candidate["along_m"][1]),
            1000 * (feature["z_m"][0] - candidate["z_m"][0]),
            1000 * (feature["z_m"][1] - candidate["z_m"][1]),
        ]
        depth_delta = feature["depth_mm"] - candidate["depth_mm"]
        # Detection is on a 50 mm grid.  This is a persistence gate, not yet a
        # millimetre boundary fit; passing features proceed to jamb/edge fitting.
        status = (
            "pass_persistent_candidate"
            if max(abs(value) for value in edge_deltas) <= 75.0
            and abs(depth_delta) <= 20.0
            else "review_repeat_geometry"
        )
        item = {
            **feature,
            "metric_status": status,
            "repeat_edge_delta_mm": [round(value, 1) for value in edge_deltas],
            "repeat_depth_delta_mm": round(depth_delta, 1),
            "reference": candidate,
        }
        (matched if status.startswith("pass") else review).append(item)
    return matched, review


def render(walls: list[dict], features: list[dict], review: list[dict], destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 12), dpi=180)
    for wall in walls:
        low, high = wall["faces_m"]
        start, stop = wall["span_m"]
        rect = (low, start, high-low, stop-start) if wall["axis"] == "x" \
            else (start, low, stop-start, high-low)
        axis.add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor="#d6d3d1",
                                 edgecolor="#57534e", linewidth=0.6))
    colours = {"niche": "#2563eb", "embedded_column": "#dc2626",
               "beam_or_soffit": "#7c3aed", "proud_relief_review": "#f59e0b"}
    for item, alpha in [(item, 0.95) for item in features] + [(item, 0.35) for item in review]:
        wall = next(w for w in walls if w["wall_id"] == item["wall_id"])
        centre = np.mean(item["along_m"])
        if wall["axis"] == "x": x, y = item["face_m"], centre
        else: x, y = centre, item["face_m"]
        axis.scatter([x], [y], s=max(10, item["width_mm"] / 12),
                     c=colours.get(item["kind"], "#f59e0b"), alpha=alpha,
                     edgecolors="black", linewidths=0.3)
    axis.set_aspect("equal"); axis.autoscale(); axis.grid(alpha=0.15)
    axis.set_title(f"Dual-walk wall assembly candidates — {len(features)} persistent, "
                   f"{len(review)} review")
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")


def main() -> None:
    global CELL_M, MIN_RELIEF_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walls", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument("--primary-transform", default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json")
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--max-points", type=int, default=6_000_000)
    parser.add_argument("--cell-mm", type=float, default=CELL_M * 1000)
    parser.add_argument("--min-relief-mm", type=float, default=MIN_RELIEF_M * 1000)
    parser.add_argument("--keep-junctions", action="store_true")
    parser.add_argument("--out", default="cage_output/metrology/mujammel_wall_assemblies.json")
    parser.add_argument("--plot", default="cage_output/metrology/mujammel_wall_assemblies.png")
    args = parser.parse_args()

    CELL_M = float(args.cell_mm) / 1000
    MIN_RELIEF_M = float(args.min_relief_mm) / 1000

    graph = json.loads(resolve(args.walls).read_text())
    walls = graph["walls"]
    print("loading primary wall-elevation evidence", flush=True)
    primary_xyz, _, _, _ = load_frame(
        args.primary_las, args.max_points, args.primary_transform, args.frame_meta
    )
    primary = scan_features(
        primary_xyz, walls, graph["floor_z_m"], graph["ceiling_z_m"],
        keep_junctions=args.keep_junctions,
    )
    del primary_xyz; gc.collect()
    print(f"primary candidates: {len(primary)}", flush=True)
    print("loading reference wall-elevation evidence", flush=True)
    reference_xyz, _, _, _ = load_frame(
        args.reference_las, args.max_points, args.reference_transform, args.frame_meta
    )
    reference = scan_features(
        reference_xyz, walls, graph["floor_z_m"], graph["ceiling_z_m"],
        keep_junctions=args.keep_junctions,
    )
    del reference_xyz; gc.collect()
    print(f"reference candidates: {len(reference)}", flush=True)
    passed, review = match_features(primary, reference)
    result = {
        "schema": "dual-walk-wall-assemblies-v1",
        "grid_mm": CELL_M * 1000,
        "role": "persistent candidates; pass features require a later <=10 mm boundary fit",
        "summary": {"primary": len(primary), "reference": len(reference),
                    "persistent_candidates": len(passed), "review": len(review)},
        "features": passed,
        "review": review,
    }
    destination = resolve(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    render(walls, passed, review, resolve(args.plot))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
