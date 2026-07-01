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
        crop_to_percentile_bounds, find_dense_z_band, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls, merge_duplicate_walls,
        select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
        plane_normal, wall_uv_basis, points_to_wall_uv,
        detect_openings_on_wall_face, cross_check_opening_both_faces,
        refine_opening_edges, render_floorplan_image,
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
        crop_to_percentile_bounds, find_dense_z_band, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls, merge_duplicate_walls,
        select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
        plane_normal, wall_uv_basis, points_to_wall_uv,
        detect_openings_on_wall_face, cross_check_opening_both_faces,
        refine_opening_edges, render_floorplan_image,
    )
    from scripts.floorplan_schema import Wall, Opening, new_wall_id, wall_to_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "floorplan"

DEFAULT_CONFIG = {
    "crop_low_pct": 1.0,
    "crop_high_pct": 99.0,
    "crop_margin_m": 0.5,
    "z_band_bin_m": 0.1,  # histogram bin size for find_dense_z_band
    "z_band_density_ratio": 0.1,  # min density (fraction of peak bin) to keep expanding the z-band
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
    "opening_edge_search_band_m": 0.15,
    "opening_edge_bin_m": 0.004,
    "refit_z_margin_m": 0.1,  # excludes floor/ceiling points from the wall-thickness refit band
}


def _detect_wall_segments(xyz, config):
    # Use a density-based Z-band rather than a fixed percentile: Phase 0's
    # crop_to_percentile_bounds filters per-axis independently (kept if x, y, AND z
    # each individually fall within their own percentile+margin bound), so a rare
    # point can coincidentally satisfy all three axis bounds and survive the crop
    # even though it isn't part of the real room -- confirmed with
    # tests/fixtures.py's two_room_house: exactly 1 of its 500 synthetic
    # stray/SLAM-drift points survives at z=3.15m, well above the true ~2.7m
    # ceiling. A fixed 99.9th/0.1st percentile handled that sparse-outlier case,
    # but running against the real koushikexport.las scan revealed a second
    # failure mode a fixed percentile can't handle: real scan data can contain
    # a second dense Z-band (another floor, a stairwell, a tall atrium) with
    # substantial point mass, not just sparse noise, which skewed the ceiling-band
    # slice and yielded 0 wall segments. find_dense_z_band instead expands
    # outward from the peak-density histogram bin and stops at a real density
    # cliff, correctly isolating the primary room's wall-height band in both cases.
    z_min, z_max = find_dense_z_band(
        xyz[:, 2], bin_m=config["z_band_bin_m"], density_ratio_threshold=config["z_band_density_ratio"],
    )
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

    # Exclude floor/ceiling points from the refit band: select_wall_band_points only
    # filters by perpendicular distance + along-wall U-range, so on real scan data
    # (unlike this pipeline's wall-face-only synthetic test fixtures) horizontal
    # floor/ceiling points sitting within the wall's perpendicular band would bias
    # the least-squares plane fit -- flagged in the final whole-branch review.
    z_margin = config["refit_z_margin_m"]
    z_lo, z_hi = floor_z + z_margin, ceiling_z - z_margin
    full_height = xyz[(xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)]
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
        # Refine the coarse 50mm-grid-detected rectangle to sub-cell accuracy using
        # a density half-max crossing on the original points -- never trust the
        # coarse detection as the final number (this pipeline's core accuracy rule).
        refined = refine_opening_edges(
            op, uv_a,
            search_band_m=config["opening_edge_search_band_m"],
            bin_m=config["opening_edge_bin_m"],
        )
        openings.append(Opening(
            opening_id=f"{wall.wall_id}_op_{i:02d}", wall_id=wall.wall_id, type=refined["type"],
            u_min_m=refined["u_min"], u_max_m=refined["u_max"], sill_m=refined["sill_m"],
            height_m=refined["height_m"], width_m=refined["width_m"],
            edge_method=refined["edge_method"], both_faces_confirmed=(uv_b is not None),
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
        return {"wall_count": 0, "walls": [], "floor_z_m": floor_z, "ceiling_z_m": ceiling_z}, []

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


def _walls_to_obj_mesh(walls):
    import open3d as _o3d  # local import: keeps build_floorplan_outputs/pure functions open3d-free
    combined = _o3d.geometry.TriangleMesh()
    for wall in walls:
        d = np.array(wall.p1) - np.array(wall.p0)
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        d = d / length
        n = np.array([-d[1], d[0]])
        half_t = wall.thickness_m / 2.0
        p0, p1 = np.array(wall.p0), np.array(wall.p1)
        z0, z1 = wall.floor_z_m, wall.ceiling_z_m
        corners_2d = [p0 - n * half_t, p1 - n * half_t, p1 + n * half_t, p0 + n * half_t]
        verts = []
        for cx, cy in corners_2d:
            verts.append([cx, cy, z0])
        for cx, cy in corners_2d:
            verts.append([cx, cy, z1])
        mesh = _o3d.geometry.TriangleMesh()
        mesh.vertices = _o3d.utility.Vector3dVector(verts)
        # side faces (4 walls of the box) + top/bottom caps, 2 triangles each
        faces = [
            [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
            [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7],
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        ]
        mesh.triangles = _o3d.utility.Vector3iVector(faces)
        combined += mesh
    if len(combined.triangles) > 0:
        combined.remove_duplicated_vertices()
        combined.compute_vertex_normals()
    return combined


def main(input_path, output_dir, config=None):
    try:
        from mesh_common import load_las_as_o3d, recenter_pcd, log
    except ImportError:
        from scripts.mesh_common import load_las_as_o3d, recenter_pcd, log
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or DEFAULT_CONFIG

    pcd = load_las_as_o3d(Path(input_path))
    recenter_pcd(pcd)
    xyz = np.asarray(pcd.points)

    manifest, walls = build_floorplan_outputs(xyz, config)
    log(f"Detected {manifest['wall_count']} walls")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    openings_by_wall_id = {i: [op.__dict__ if hasattr(op, "__dict__") else op for op in w.openings]
                            for i, w in enumerate(walls)}
    wall_dicts = [{"p0": np.array(w.p0), "p1": np.array(w.p1),
                   "thickness_m": w.thickness_m, "length_m": w.length_m} for w in walls]
    render_floorplan_image(wall_dicts, openings_by_wall_id, str(output_dir / "floorplan.png"))

    obj_mesh = _walls_to_obj_mesh(walls)
    obj_path = output_dir / "reconstructed.obj"
    if len(obj_mesh.triangles) > 0:
        o3d.io.write_triangle_mesh(str(obj_path), obj_mesh)
    else:
        obj_path.write_text("")  # empty but present, so downstream tooling doesn't error on a missing file
    log(f"Wrote {manifest_path}, floorplan.png, {obj_path}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    main(inp, outp)
