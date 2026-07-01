"""LAS -> 2D floor plan + clean 3D walls/openings (Phase 0 + Phase 1).

Usage:
    python scripts/floorplan_reconstruct.py [input.las] [output_dir]
"""
import sys
import json
from pathlib import Path

import numpy as np

try:
    from floorplan_geometry import (
        crop_to_percentile_bounds, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls, merge_duplicate_walls,
        select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
        plane_normal, wall_uv_basis, points_to_wall_uv,
        detect_openings_on_wall_face, cross_check_opening_both_faces,
        render_floorplan_image,
    )
    from floorplan_schema import Wall, Opening, new_wall_id, wall_to_dict
except ImportError:
    # bare sibling import only resolves when scripts/ itself is on sys.path
    # (direct execution: `python scripts/floorplan_reconstruct.py`). When this
    # module is imported package-qualified instead (`from scripts.floorplan_reconstruct
    # import ...`, the path every test in this repo uses), only the repo root is on
    # sys.path, so fall back to the qualified form -- confirmed necessary during
    # Task 12's review (a bare-only import raised ModuleNotFoundError under that
    # exact import path).
    from scripts.floorplan_geometry import (
        crop_to_percentile_bounds, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls, merge_duplicate_walls,
        select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
        plane_normal, wall_uv_basis, points_to_wall_uv,
        detect_openings_on_wall_face, cross_check_opening_both_faces,
        render_floorplan_image,
    )
    from scripts.floorplan_schema import Wall, Opening, new_wall_id, wall_to_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "floorplan"

DEFAULT_CONFIG = {
    "crop_low_pct": 1.0,
    "crop_high_pct": 99.0,
    "crop_margin_m": 0.5,
    "ceiling_band_m": 0.1,  # slice thickness for the top-down wall detection
    "ceiling_offset_m": 0.15,  # how far below the detected ceiling to slice
    "cell_size_m": 0.02,
    "density_min_count": 2,
    "morph_kernel": 3,
    "dp_epsilon_cells": 2.0,
    "min_span_cells": 3.0,
    "min_segment_length_cells": 5.0,
    "pair_min_thickness_m": 0.06,
    "pair_max_thickness_m": 0.35,
    "pair_max_angle_deg": 5.0,
    "pair_min_overlap_frac": 0.5,
    "snap_tolerance_m": 0.08,
    "min_wall_length_m": 0.3,
    "refit_coarse_band_m": 0.025,
    "refit_fine_band_m": 0.008,
    "refit_corner_margin_m": 0.5,
    "opening_cell_m": 0.05,
    "opening_min_w": 0.45,
    "opening_min_h": 0.45,
}


def _detect_wall_segments(xyz, config):
    # Use a robust high/low percentile rather than raw min/max: Phase 0's
    # crop_to_percentile_bounds filters per-axis independently (kept if x, y, AND z
    # each individually fall within their own percentile+margin bound), so a rare
    # point can coincidentally satisfy all three axis bounds and survive the crop
    # even though it isn't part of the real room -- confirmed with
    # tests/fixtures.py's two_room_house: exactly 1 of its 500 synthetic
    # stray/SLAM-drift points survives at z=3.15m, well above the true ~2.7m
    # ceiling. Using raw xyz[:, 2].max() there skews the ceiling-band slice
    # entirely above every real wall point, yielding 0 wall segments.
    z_max = float(np.percentile(xyz[:, 2], 99.5))
    z_min = float(np.percentile(xyz[:, 2], 0.5))
    band_hi = z_max - config["ceiling_offset_m"]
    band_lo = band_hi - config["ceiling_band_m"]
    ceiling_slice = xyz[(xyz[:, 2] >= band_lo) & (xyz[:, 2] <= band_hi)]
    if len(ceiling_slice) < 100:
        return [], z_min, z_max

    xy = ceiling_slice[:, :2]
    bmin, bmax = xy.min(axis=0) - 0.1, xy.max(axis=0) + 0.1
    image, origin = points_to_density_image(xy, config["cell_size_m"], bmin, bmax)
    binary = threshold_density_image(image, config["density_min_count"], config["morph_kernel"])
    segments = extract_wall_segments(
        binary, origin, config["cell_size_m"],
        epsilon_cells=config["dp_epsilon_cells"], min_span_cells=config["min_span_cells"],
    )
    min_len = config["min_segment_length_cells"] * config["cell_size_m"]
    segments = [s for s in segments if s["length"] >= min_len]
    return segments, z_min, z_max


def _build_walls_from_segments(segments, config):
    walls_raw = pair_wall_surfaces(
        segments,
        min_thickness_m=config["pair_min_thickness_m"],
        max_thickness_m=config["pair_max_thickness_m"],
        max_angle_deg=config["pair_max_angle_deg"],
        min_overlap_frac=config["pair_min_overlap_frac"],
    )
    walls_raw = apply_modal_thickness_fallback(walls_raw)
    walls_raw, _clusters = snap_wall_endpoints(walls_raw, tolerance_m=config["snap_tolerance_m"])
    walls_raw = drop_short_walls(walls_raw, min_length_m=config["min_wall_length_m"])
    walls_raw = merge_duplicate_walls(walls_raw)
    return walls_raw


def _refine_wall(wall_raw, xyz, floor_z, ceiling_z, config, index):
    d = (wall_raw["p1"] - wall_raw["p0"])
    d = d / np.linalg.norm(d)
    normal2d = np.array([-d[1], d[0]])

    full_height = xyz  # region selector below restricts by perpendicular band + U-range anyway
    band_pts = select_wall_band_points(
        full_height, wall_raw,
        corner_margin_m=config["refit_corner_margin_m"],
        band_m=max(config["pair_max_thickness_m"], 0.3),
    )
    if len(band_pts) < 20:
        return None  # not enough data to refine this wall; drop it rather than report a wrong number

    mid = band_pts[:, 0] * normal2d[0] + band_pts[:, 1] * normal2d[1]
    med = float(np.median(mid))
    side_a_pts = band_pts[mid < med]
    side_b_pts = band_pts[mid >= med]

    coarse_a = [normal2d[0], normal2d[1], 0.0, -float(np.dot(normal2d, side_a_pts[:, :2].mean(axis=0)))]
    plane_a = refine_wall_plane_two_pass(
        side_a_pts, coarse_a, config["refit_coarse_band_m"], config["refit_fine_band_m"],
    )

    plane_b = None
    thickness_m = wall_raw["thickness_m"]
    thickness_source = wall_raw["thickness_source"]
    if len(side_b_pts) >= 20:
        coarse_b = [normal2d[0], normal2d[1], 0.0, -float(np.dot(normal2d, side_b_pts[:, :2].mean(axis=0)))]
        plane_b = refine_wall_plane_two_pass(
            side_b_pts, coarse_b, config["refit_coarse_band_m"], config["refit_fine_band_m"],
        )
        thickness_m = abs(plane_a[3] - plane_b[3])
        thickness_source = "measured"

    origin_xyz = np.array([wall_raw["p0"][0], wall_raw["p0"][1], floor_z])
    u_axis = np.array([d[0], d[1], 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])

    return Wall(
        wall_id=new_wall_id(index),
        p0=tuple(wall_raw["p0"]), p1=tuple(wall_raw["p1"]), length_m=wall_raw["length_m"],
        thickness_m=float(thickness_m), thickness_source=thickness_source,
        plane_front=plane_a, plane_back=plane_b,
        origin_xyz=tuple(origin_xyz), u_axis=tuple(u_axis), v_axis=tuple(v_axis),
        floor_z_m=floor_z, ceiling_z_m=ceiling_z,
        region_band_m=config["pair_max_thickness_m"], region_corner_margin_m=config["refit_corner_margin_m"],
    ), side_a_pts, side_b_pts


def _detect_wall_openings(wall, side_a_pts, side_b_pts, config):
    plane_a = wall.plane_front
    uv_a = points_to_wall_uv(side_a_pts, plane_a, np.array(wall.origin_xyz), np.array(wall.u_axis))
    openings_raw = detect_openings_on_wall_face(
        uv_a, wall.length_m, cell_m=config["opening_cell_m"],
        min_opening_w=config["opening_min_w"], min_opening_h=config["opening_min_h"],
    )

    uv_b = None
    if wall.plane_back is not None and len(side_b_pts):
        uv_b = points_to_wall_uv(side_b_pts, wall.plane_back, np.array(wall.origin_xyz), np.array(wall.u_axis))

    openings = []
    for i, op in enumerate(openings_raw):
        both_faces = True
        if uv_b is not None:
            both_faces = cross_check_opening_both_faces(op, uv_b, cell_m=config["opening_cell_m"])
        if not both_faces:
            continue
        openings.append(Opening(
            opening_id=f"{wall.wall_id}_op_{i:02d}", wall_id=wall.wall_id, type=op["type"],
            u_min_m=op["u_min"], u_max_m=op["u_max"], sill_m=op["sill_m"],
            height_m=op["height_m"], width_m=op["width_m"],
            edge_method="density_half_max", both_faces_confirmed=(uv_b is not None),
        ))
    return openings


def build_floorplan_outputs(xyz, config=None):
    config = config or DEFAULT_CONFIG
    xyz = np.asarray(xyz, dtype=np.float64)

    _lo, _hi, keep_mask, _crop_stats = crop_to_percentile_bounds(
        xyz, config["crop_low_pct"], config["crop_high_pct"], config["crop_margin_m"],
    )
    cropped = xyz[keep_mask]

    segments, floor_z, ceiling_z = _detect_wall_segments(cropped, config)
    if not segments:
        return {"wall_count": 0, "walls": []}, []

    walls_raw = _build_walls_from_segments(segments, config)

    walls = []
    for i, wall_raw in enumerate(walls_raw):
        result = _refine_wall(wall_raw, cropped, floor_z, ceiling_z, config, i)
        if result is None:
            continue
        wall, side_a_pts, side_b_pts = result
        wall.openings = _detect_wall_openings(wall, side_a_pts, side_b_pts, config)
        walls.append(wall)

    manifest = {
        "wall_count": len(walls),
        "floor_z_m": floor_z,
        "ceiling_z_m": ceiling_z,
        "walls": [_wall_manifest_entry(w) for w in walls],
    }
    return manifest, walls


def _wall_manifest_entry(wall):
    return wall_to_dict(wall)
