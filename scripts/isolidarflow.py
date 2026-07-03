"""isolidarflow: one-command end-to-end plane-first modular reconstruction.

Wires every module built in Tasks 1-9 into a single pipeline that turns a raw
SLAM point cloud (LAS/LAZ) into a sharp-edged modular 3D model (GLB), a 2D CAD
floor plan (DXF + SVG), a JSON manifest, and a human-readable report.

Stage order (each stage is an existing, tested function -- this module is thin
orchestration only):

  load_scan -> percentile_crop -> remove_outliers -> approx_trajectory
  -> select_z_band -> isolate_unit
  -> estimate_normals -> dominant_axes -> axis_align  (+ rotate trajectory)
  -> detect_planes(seed)
  -> group_wall_runs -> extract_columns_beams -> extract_unclassified (report)
  -> snap_walls -> pair_thickness -> recenter_walls -> resolve_corners
     -> snap_endpoints_to_lines
  -> remove_furniture
  -> wall_crossings -> detect_openings
  -> build_room_polygons
  -> build_manifest
  -> build_room_model -> build_scene -> write_glb
  -> write_dxf + write_svg + manifest.json + report.txt
  -> (optional) debug PNG in the diag_floorplan2d_v3 solid-wall style

Usage:
    venv311\\Scripts\\python.exe scripts\\isolidarflow.py <scan.las> <out_dir> \\
        [--keep-furniture] [--seed N] [--debug-png] [--max-points N] \\
        [--plane-max-points N] [--trajectory PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

# Make the repo root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.recon import io_las, clean, frame, planes, structure, regularize  # noqa: E402
from scripts.recon.trajectory import approx_trajectory, load_trajectory, wall_crossings  # noqa: E402
from scripts.recon.isolate import select_z_band, isolate_unit  # noqa: E402
from scripts.recon.openings import detect_openings  # noqa: E402
from scripts.recon.floorplan2d import build_room_polygons, write_dxf, write_svg  # noqa: E402
from scripts.recon.schema import build_manifest, new_wall_id, new_opening_id  # noqa: E402
from scripts.recon.model import build_room_model  # noqa: E402
from scripts.recon.assemble import build_scene, write_glb  # noqa: E402


# ---------------------------------------------------------------------------
# Every threshold from Tasks 1-9, one place, one-line comment each.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- reproducibility (Task 1) ---
    "seed": 0,                       # RNG seed for Open3D RANSAC + numpy subsampling
    # --- ingest (Task 1) ---
    "max_points": None,              # uniform subsample cap at load (None = full resolution)
    # --- clean: percentile bbox crop (Task 1) ---
    "crop_lo_pct": 1.0,              # lower per-axis percentile kept
    "crop_hi_pct": 99.0,             # upper per-axis percentile kept
    "crop_margin_m": 0.5,            # bbox expansion margin
    # --- clean: statistical outlier removal (Task 1, Open3D) ---
    "remove_outliers": True,         # toggle Open3D statistical outlier removal
    "outlier_nb": 20,                # neighbours considered per point
    "outlier_std_ratio": 2.0,        # std-dev multiplier cutoff
    # --- trajectory recovery (Task 5) ---
    "traj_dt_s": 0.25,               # time-slice width for the approx walk-path
    # --- isolate: primary-storey z-band (Task 4) ---
    "z_bin_m": 0.05,                 # histogram bin for floor/ceiling density spikes
    "z_min_height_m": 1.8,           # min plausible storey height
    "z_max_height_m": 4.5,           # max plausible storey height
    # --- isolate: XY-connected unit (Task 4) ---
    "iso_cell_m": 0.25,              # XY occupancy cell for connectivity
    "iso_max_gap_cells": 1,          # bridge doorway-scale gaps up to this many cells
    "iso_max_dist_m": None,          # optional distance-from-path cap (off by default)
    # --- working-cloud cap for plane/structure/opening stages ---
    "plane_max_points": 1_500_000,   # subsample the isolated+aligned cloud above this
    # --- framing (Task 3, Open3D normals) ---
    "normals_radius_m": 0.06,        # hybrid KD-tree search radius for normal estimation
    "normals_max_nn": 30,            # max neighbours per normal
    "normals_max_points": 800_000,   # subsample used only to estimate dominant axes
    # --- z reference for detection (percentile on isolated+aligned cloud) ---
    "z_floor_pct": 1.0,              # percentile taken as the floor height
    "z_ceiling_pct": 99.0,           # percentile taken as the ceiling height
    # --- plane detection (Task 2, Open3D RANSAC + DBSCAN) ---
    "plane_dist_thresh_m": 0.02,     # RANSAC inlier distance
    "plane_min_inliers": 1500,       # min inliers to accept a plane
    "plane_max_planes": 60,          # max RANSAC peels
    "plane_dbscan_eps_m": 0.15,      # DBSCAN face-splitting radius
    "plane_dbscan_min": 50,          # DBSCAN min points per face
    "plane_horizontal_min": 0.85,    # |n_z| above this => floor/ceiling
    "plane_ransac_iters": 1000,      # RANSAC iterations per plane
    # --- wall-run grouping (Task 2, incl. rescue defaults) ---
    "wr_merge_offset_m": 0.15,       # collapse colinear faces into one WallStep below this
    "wr_max_relief_m": 0.35,         # max protrusion still counted as same-wall relief
    "wr_min_run_length_m": 0.5,      # drop runs shorter than this (rescue-loosened)
    "wr_angle_tol_deg": 10.0,        # normal-to-axis tolerance for run eligibility
    "wr_full_height_frac": 0.4,      # min height fraction of tallest plane (rescue-loosened)
    "wr_u_gap_m": 2.8,               # split colinear runs across holes wider than this
    "wr_min_run_inliers": 1200,      # furniture guard: min combined inliers per run
    # --- column / beam extraction (Task 8) ---
    "cb_min_size_m": 0.1,            # min compact-face extent for a column
    "cb_max_size_m": 0.6,            # max compact-face extent (column vs wall split)
    "cb_height_tol_m": 0.3,          # floor/ceiling reach tolerance for a column
    "cb_beam_elevation_m": 1.0,      # a beam face must sit at least this above the floor
    "cb_beam_ceiling_tol_m": 0.5,    # a beam face must reach within this of the ceiling
    "cb_beam_offset_gap_m": 1.0,     # offset-chaining gap for grouping beam side faces
    # --- unclassified plane reporting (Task 2) ---
    "unclassified_dir_tol_deg": 20.0,  # looser axis tolerance for best-guess direction
    # --- axis snap (Task 3) ---
    "snap_angle_tol_deg": 8.0,       # snap a wall to a Manhattan axis only within this
    # --- thickness pairing (Task 3) ---
    "pair_default_m": 0.10,          # assumed thickness when no back face found
    "pair_min_thickness_m": 0.03,    # min plausible measured thickness
    "pair_max_thickness_m": 0.6,     # max plausible measured thickness
    "pair_min_overlap_frac": 0.5,    # min back-face / main-face u-overlap fraction
    # --- midline recentre (Task 3) ---
    "recenter_min_thickness_m": 0.04,  # skip recentre below this (no evidence of a side)
    # --- corner resolution (Task 3) ---
    "corner_tol_m": 0.25,            # endpoint-to-endpoint clustering radius
    # --- T-junction endpoint-to-line snap (Task 3) ---
    "line_snap_reach_m": 0.7,        # how far a dangling endpoint may extend to a line
    "line_snap_dangling_tol_m": 0.15,  # endpoint within this of a wall = not dangling
    # --- furniture removal (Task 6) ---
    "furn_dist_m": 0.15,             # corridor half-width around each wall step face
    "furn_floor_band_m": 0.12,       # floor band kept everywhere
    "furn_ceiling_band_m": 0.20,     # ceiling band kept everywhere
    # --- wall crossings (Task 5) ---
    "crossing_end_margin_m": 0.15,   # ignore crossings within this of a wall end
    # --- 2D arrangement (Task 9) ---
    "room_eps_primary_m": 0.30,      # polygonize gap-closing epsilon (primary pass)
    "room_eps_recovery_m": 0.50,     # looser epsilon for room recovery pass
    "room_min_area_m2": 1.0,         # ignore polygons smaller than this
    "room_simplify_m": 0.02,         # Douglas-Peucker tolerance for room polygons
    # --- opening priors (Task 7) -- never hard rejections, only classification/flags ---
    "priors": {
        "door_h_m": 2.13,            # 7 ft door height prior (user requirement)
        "door_h_tol_m": 0.25,        # tolerance band around the door-height prior
        "window_min_sill_m": 0.30,   # min sill for a void to read as a window
        "balcony_min_w_m": 1.50,     # min width for a floor-touching void to read balcony
        "ceiling_m": 2.75,           # 2750 mm ceiling prior (user requirement)
        "ceiling_tol_m": 0.35,       # tolerance band around the ceiling-height prior
    },
}


# ---------------------------------------------------------------------------
# small adapters (dict wall-run -> attribute objects the 2D exporters expect)
# ---------------------------------------------------------------------------

def _wall_length(w) -> float:
    p0 = np.asarray(w["p0"], dtype=float)
    p1 = np.asarray(w["p1"], dtype=float)
    return float(np.linalg.norm(p1 - p0))


def _wall_namespaces(walls):
    """floorplan2d.write_dxf/write_svg/build_room_polygons only read
    wall_id/p0/p1/length_m/thickness_m -- give them lightweight namespaces."""
    out = []
    for i, w in enumerate(walls):
        out.append(SimpleNamespace(
            wall_id=new_wall_id(i),
            p0=tuple(w["p0"]),
            p1=tuple(w["p1"]),
            length_m=float(w.get("length_m", _wall_length(w))),
            thickness_m=float(w.get("thickness_m", 0.1)),
        ))
    return out


def _opening_namespaces(openings_by_wall):
    """floorplan2d exporters read wall_id/u_min_m/u_max_m off each opening."""
    out = []
    idx = 0
    for wi, ops in (openings_by_wall or {}).items():
        for op in ops:
            out.append(SimpleNamespace(
                opening_id=new_opening_id(idx),
                wall_id=new_wall_id(wi),
                u_min_m=float(op.get("u0", op.get("u_min_m"))),
                u_max_m=float(op.get("u1", op.get("u_max_m"))),
            ))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _write_report(out_dir, in_path, info, error=None):
    lines = [f"isolidarflow report", f"input: {in_path}", ""]
    for key in ("loaded", "after_crop", "after_denoise", "isolated_kept",
                "isolated_dropped", "working_points", "trajectory_vertices"):
        if key in info:
            lines.append(f"{key}: {info[key]}")
    if "z_floor" in info:
        lines.append(
            f"z-band (floor, ceiling): ({info['z_floor']:.3f}, {info['z_ceiling']:.3f}) m "
            f"=> height {info['z_ceiling'] - info['z_floor']:.3f} m")
    for key in ("n_planes", "n_vertical", "n_walls", "n_columns", "n_beams",
                "n_unclassified", "n_crossings", "n_openings", "n_rooms",
                "furniture_dropped"):
        if key in info:
            lines.append(f"{key}: {info[key]}")
    if "opening_types" in info:
        lines.append(f"opening_types: {info['opening_types']}")
    if "room_areas_m2" in info:
        lines.append(f"room_areas_m2: {info['room_areas_m2']}")
    if error is not None:
        lines += ["", f"ERROR: partial pipeline -- failed with: {error}"]
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(str(ln) for ln in lines) + "\n")
    return lines


# ---------------------------------------------------------------------------
# main orchestration
# ---------------------------------------------------------------------------

def run(in_path, out_dir, config=None):
    """End-to-end reconstruction. Returns a summary dict pointing at the
    written artifacts (model.glb, floorplan.dxf/.svg, manifest.json, report.txt)
    plus the parsed manifest and element counts.

    On a stage failure the partial report.txt is written and the exception is
    re-raised (every stage is an existing tested function; this layer only
    sequences them).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg["seed"])
    info = {}
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    try:
        # --- 1. ingest ---
        scan = io_las.load_scan(in_path, max_points=cfg["max_points"], rng_seed=seed)
        info["loaded"] = scan.n

        # --- 2. clean: percentile crop ---
        scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"],
                                     margin_m=cfg["crop_margin_m"])
        info["after_crop"] = scan.n

        # --- 3. clean: outlier removal ---
        if cfg["remove_outliers"]:
            scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"],
                                         std_ratio=cfg["outlier_std_ratio"])
        info["after_denoise"] = scan.n

        # --- 4. trajectory ---
        traj_path = cfg.get("trajectory_path")
        if traj_path:
            traj = load_trajectory(traj_path)
        elif scan.gps_time is not None:
            traj = approx_trajectory(scan.gps_time, scan.xyz, dt_s=cfg["traj_dt_s"])
        else:
            traj = np.zeros((0, 3))

        # --- 5. isolate the storey ---
        z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                               min_height_m=cfg["z_min_height_m"],
                               max_height_m=cfg["z_max_height_m"])
        scan, iso_stats = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                                       max_gap_cells=cfg["iso_max_gap_cells"],
                                       max_dist_m=cfg["iso_max_dist_m"])
        info["isolated_kept"] = iso_stats["kept"]
        info["isolated_dropped"] = iso_stats["dropped"]
        if scan.n == 0:
            raise RuntimeError("isolate_unit produced an empty cloud (no storey found)")

        # --- 6. framing: normals -> dominant axes -> axis align (also rotate traj) ---
        rng = np.random.default_rng(seed)
        sub = (rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
               if scan.n > 0 else np.arange(0))
        normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"],
                                         max_nn=cfg["normals_max_nn"])
        R = frame.dominant_axes(normals)
        scan = frame.axis_align(scan, R)
        if traj.shape[0]:
            traj = traj @ np.asarray(R, dtype=float).T

        # --- working cloud (cap density for the heavy detection stages) ---
        if cfg["plane_max_points"] and scan.n > cfg["plane_max_points"]:
            keep = rng.choice(scan.n, size=cfg["plane_max_points"], replace=False)
            scan = scan.subset(keep)
        xyz = scan.xyz
        info["working_points"] = scan.n
        info["trajectory_vertices"] = int(traj.shape[0])

        z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
        z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
        info["z_floor"], info["z_ceiling"] = z_floor, z_ceiling

        # --- 7. plane detection ---
        all_planes = planes.detect_planes(
            xyz, dist_thresh=cfg["plane_dist_thresh_m"], min_inliers=cfg["plane_min_inliers"],
            max_planes=cfg["plane_max_planes"], dbscan_eps=cfg["plane_dbscan_eps_m"],
            dbscan_min=cfg["plane_dbscan_min"], z_floor=z_floor, z_ceiling=z_ceiling,
            horizontal_min=cfg["plane_horizontal_min"], ransac_iters=cfg["plane_ransac_iters"],
            seed=seed)
        verticals = [p for p in all_planes if p.label == "vertical"]
        info["n_planes"], info["n_vertical"] = len(all_planes), len(verticals)

        # --- 8. structure: wall runs + columns/beams + unclassified report ---
        runs, used_w = structure.group_wall_runs(
            verticals, xyz, np.eye(3), merge_offset_m=cfg["wr_merge_offset_m"],
            max_relief_m=cfg["wr_max_relief_m"], min_run_length_m=cfg["wr_min_run_length_m"],
            angle_tol_deg=cfg["wr_angle_tol_deg"], full_height_frac=cfg["wr_full_height_frac"],
            u_gap_m=cfg["wr_u_gap_m"], min_run_inliers=cfg["wr_min_run_inliers"],
            return_used_indices=True)
        columns, beams, used_cb = structure.extract_columns_beams(
            verticals, xyz, z_floor, z_ceiling, min_size_m=cfg["cb_min_size_m"],
            max_size_m=cfg["cb_max_size_m"], height_tol_m=cfg["cb_height_tol_m"],
            beam_elevation_m=cfg["cb_beam_elevation_m"], beam_ceiling_tol_m=cfg["cb_beam_ceiling_tol_m"],
            beam_offset_gap_m=cfg["cb_beam_offset_gap_m"], return_used_indices=True)
        unclassified = structure.extract_unclassified(
            verticals, used_w | used_cb, xyz, direction_tol_deg=cfg["unclassified_dir_tol_deg"])
        info["n_columns"], info["n_beams"] = len(columns), len(beams)
        info["n_unclassified"] = len(unclassified)

        # --- 9. regularize (strict order: snap -> pair -> recenter -> corners -> lines) ---
        walls = regularize.snap_walls(runs, np.eye(3), angle_tol_deg=cfg["snap_angle_tol_deg"])
        walls = regularize.pair_thickness(walls, xyz, default_m=cfg["pair_default_m"],
                                          min_thickness_m=cfg["pair_min_thickness_m"],
                                          max_thickness_m=cfg["pair_max_thickness_m"],
                                          min_overlap_frac=cfg["pair_min_overlap_frac"])
        walls = regularize.recenter_walls(walls, xyz, z_floor, z_ceiling,
                                          min_thickness_m=cfg["recenter_min_thickness_m"])
        walls = regularize.resolve_corners(walls, tol_m=cfg["corner_tol_m"])
        walls = regularize.snap_endpoints_to_lines(walls, reach_m=cfg["line_snap_reach_m"],
                                                   dangling_tol_m=cfg["line_snap_dangling_tol_m"])
        info["n_walls"] = len(walls)
        # carry the storey Z onto each wall so per-segment meshing places
        # openings/features in the same absolute Z frame as detection.
        for w in walls:
            w["floor_z_m"] = z_floor
            w["ceiling_z_m"] = z_ceiling

        # --- 10. furniture removal (needs direction/offset_m/steps -- all present) ---
        if cfg.get("_keep_furniture"):
            clean_scan, furn_dropped = scan, 0
        else:
            clean_scan, furn_dropped = clean.remove_furniture(
                scan, walls, z_floor, z_ceiling, columns=columns, beams=beams,
                dist_m=cfg["furn_dist_m"], floor_band_m=cfg["furn_floor_band_m"],
                ceiling_band_m=cfg["furn_ceiling_band_m"])
        info["furniture_dropped"] = int(furn_dropped)
        clean_xyz = clean_scan.xyz

        # --- 11. openings: crossings -> detect ---
        crossings = wall_crossings(traj, walls, end_margin_m=cfg["crossing_end_margin_m"])
        info["n_crossings"] = int(sum(len(v) for v in crossings.values()))
        openings_by_wall = detect_openings(walls, clean_xyz, traj, crossings,
                                           z_floor, z_ceiling, cfg["priors"])
        n_openings = sum(len(v) for v in openings_by_wall.values())
        info["n_openings"] = n_openings
        info["opening_types"] = sorted({o["type"] for v in openings_by_wall.values() for o in v})

        # --- 12. 2D arrangement: room polygons (primary eps + recovery eps) ---
        wall_ns = _wall_namespaces(walls)
        rooms = [p.simplify(cfg["room_simplify_m"])
                 for p in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_primary_m"])
                 if p.area >= cfg["room_min_area_m2"]]
        for cell in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_recovery_m"]):
            if cell.area >= cfg["room_min_area_m2"] and not any(r.contains(cell.centroid) for r in rooms):
                rooms.append(cell.simplify(cfg["room_simplify_m"]))
        info["n_rooms"] = len(rooms)
        info["room_areas_m2"] = sorted(round(r.area, 1) for r in rooms)
        room_coords = [list(r.exterior.coords) for r in rooms]

        # --- 13. manifest ---
        manifest = build_manifest(walls, openings_by_wall, columns, beams, room_coords,
                                  z_floor, z_ceiling, cfg)

        # --- 14. 3D model -> GLB ---
        model = build_room_model(walls, openings_by_wall, columns, beams, rooms,
                                 z_floor, z_ceiling)
        glb_path = os.path.join(out_dir, "model.glb")
        write_glb(build_scene(model), glb_path)

        # --- 15. 2D CAD exports ---
        opening_ns = _opening_namespaces(openings_by_wall)
        dxf_path = os.path.join(out_dir, "floorplan.dxf")
        svg_path = os.path.join(out_dir, "floorplan.svg")
        write_dxf(wall_ns, opening_ns, rooms, dxf_path)
        write_svg(wall_ns, opening_ns, rooms, svg_path)

        # --- 16. manifest.json + report.txt ---
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=_json_default)
        _write_report(out_dir, in_path, info)

        # --- 17. optional debug PNG ---
        png_path = None
        if cfg.get("debug_png"):
            try:
                png_path = _write_debug_png(out_dir, xyz, walls, rooms, columns,
                                            openings_by_wall, z_floor, z_ceiling, scan)
            except Exception as exc:  # a diagnostic image must never sink the run
                print(f"[isolidarflow] debug PNG skipped: {exc}", file=sys.stderr)

        return {
            "out_dir": out_dir,
            "glb": glb_path,
            "dxf": dxf_path,
            "svg": svg_path,
            "manifest_path": manifest_path,
            "report_path": os.path.join(out_dir, "report.txt"),
            "debug_png": png_path,
            "manifest": manifest,
            "model": model,
            "z_floor": z_floor,
            "z_ceiling": z_ceiling,
            "counts": {
                "walls": len(walls), "columns": len(columns), "beams": len(beams),
                "openings": n_openings, "rooms": len(rooms),
                "unclassified": len(unclassified),
            },
            "info": info,
        }
    except Exception as exc:
        _write_report(out_dir, in_path, info, error=repr(exc))
        raise


def _json_default(obj):
    """Serialize numpy scalars / arrays and dataclasses that slip into the
    config echo or manifest."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ---------------------------------------------------------------------------
# debug PNG (diag_floorplan2d_v3 solid-wall style + RGB-per-wall-band note)
# ---------------------------------------------------------------------------

def _write_debug_png(out_dir, xyz, walls, rooms, columns, openings_by_wall,
                     z_floor, z_ceiling, scan, ppm=60):
    import cv2

    xy = xyz[:, :2]
    xmin, ymin = np.percentile(xy, 0.2, axis=0) - 0.5
    xmax, ymax = np.percentile(xy, 99.8, axis=0) + 0.5
    W, H = max(1, int((xmax - xmin) * ppm)), max(1, int((ymax - ymin) * ppm))

    def to_px(pt):
        return (int((pt[0] - xmin) * ppm), int((ymax - pt[1]) * ppm))

    cols_i = np.clip(((xy[:, 0] - xmin) * ppm).astype(int), 0, W - 1)
    rows_i = np.clip(((ymax - xy[:, 1]) * ppm).astype(int), 0, H - 1)
    density = np.zeros((H, W))
    np.add.at(density, (rows_i, cols_i), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 60).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])

    overlay = img.copy()
    for poly in rooms:
        pts = np.array([to_px(c) for c in poly.exterior.coords], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (120, 60, 10))
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    def wall_rect(w, u0=None, u1=None, extra=0.0):
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        d = d / L if L else np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]])
        ht = w.get("thickness_m", 0.1) / 2 + extra
        a = p0 + d * (0.0 if u0 is None else u0)
        b = p0 + d * (L if u1 is None else u1)
        return np.array([to_px(a + n * ht), to_px(b + n * ht),
                         to_px(b - n * ht), to_px(a - n * ht)], dtype=np.int32)

    # solid white walls
    for w in walls:
        cv2.fillPoly(img, [wall_rect(w)], (235, 235, 235))
    # openings tinted by type
    type_color = {"door": (0, 150, 255), "balcony_door": (0, 220, 255),
                  "window": (255, 180, 0), "unknown_opening": (160, 130, 110)}
    for wi, ops in (openings_by_wall or {}).items():
        w = walls[wi]
        p0 = np.asarray(w["p0"], float)
        d = np.asarray(w["p1"], float) - p0
        L = float(np.linalg.norm(d))
        if L == 0:
            continue
        d = d / L
        u_i = 1 if w["direction"] == "x" else 0
        if abs(d[u_i]) < 1e-6:
            continue
        for op in ops:
            t0 = (op["u0"] - p0[u_i]) / d[u_i] if abs(d[u_i]) > 1e-9 else 0.0
            t1 = (op["u1"] - p0[u_i]) / d[u_i] if abs(d[u_i]) > 1e-9 else 0.0
            lo, hi = sorted((t0, t1))
            lo, hi = max(lo, 0.0), min(hi, L)
            if hi - lo < 0.02:
                continue
            cv2.fillPoly(img, [wall_rect(w, u0=lo, u1=hi, extra=0.02)],
                         type_color.get(op["type"], (200, 200, 200)))
    for w in walls:
        cv2.polylines(img, [wall_rect(w)], True, (90, 90, 90), 1)
    for c in columns:
        pts = np.array([to_px(p) for p in c.footprint], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 165, 255))
    for poly in rooms:
        cen = poly.centroid
        cv2.putText(img, f"{poly.area:.1f}m2", to_px((cen.x, cen.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 150), 1, cv2.LINE_AA)

    # RGB-per-wall-band annotation (evidence for a future colour-assisted
    # opening/frame detector -- printed, and drawn as a swatch when RGB exists)
    if getattr(scan, "rgb", None) is not None:
        rgb_note = _rgb_per_wall_band(walls, xy, scan.rgb)
        for wi, (mean_rgb, npts) in rgb_note.items():
            w = walls[wi]
            mid = (np.asarray(w["p0"], float) + np.asarray(w["p1"], float)) / 2
            b, g, r = int(mean_rgb[2]), int(mean_rgb[1]), int(mean_rgb[0])
            cv2.circle(img, to_px(mid), 5, (b, g, r), -1)
        print("[isolidarflow] RGB mean per wall band (wall_idx -> (R,G,B), n_pts):")
        for wi, (mean_rgb, npts) in sorted(rgb_note.items()):
            print(f"  wall_{wi:03d}: ({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})  n={npts}")

    banner = (f"isolidarflow | {len(walls)} walls | {len(columns)} cols | "
              f"{sum(len(v) for v in (openings_by_wall or {}).values())} openings | "
              f"{len(rooms)} rooms")
    cv2.putText(img, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    png_path = os.path.join(out_dir, "floorplan_debug.png")
    cv2.imwrite(png_path, img)
    return png_path


def _rgb_per_wall_band(walls, xy, rgb, band_m=0.2):
    """Mean RGB of points within band_m of each wall's midline plane."""
    rgb = np.asarray(rgb, dtype=float)
    out = {}
    for wi, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = xy - p0
        along = rel @ u
        perp = rel @ n
        band = (np.abs(perp) <= band_m + w.get("thickness_m", 0.1) / 2) & (along >= 0) & (along <= L)
        if int(band.sum()) < 20:
            continue
        out[wi] = (rgb[band].mean(axis=0), int(band.sum()))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="isolidarflow: end-to-end modular reconstruction.")
    ap.add_argument("in_path")
    ap.add_argument("out_dir")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="skip the furniture/clutter corridor filter")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default 0)")
    ap.add_argument("--debug-png", action="store_true", help="write floorplan_debug.png")
    ap.add_argument("--max-points", type=int, default=None, help="subsample cap at load")
    ap.add_argument("--plane-max-points", type=int, default=None,
                    help="subsample cap for the plane/structure/opening stages")
    ap.add_argument("--no-outliers", action="store_true", help="skip statistical outlier removal")
    ap.add_argument("--trajectory", default=None, help="explicit trajectory file")
    args = ap.parse_args(argv)

    cfg = dict(DEFAULT_CONFIG)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_points is not None:
        cfg["max_points"] = args.max_points
    if args.plane_max_points is not None:
        cfg["plane_max_points"] = args.plane_max_points
    if args.no_outliers:
        cfg["remove_outliers"] = False
    cfg["debug_png"] = args.debug_png
    cfg["trajectory_path"] = args.trajectory
    if args.keep_furniture:
        # keep-furniture => disable the corridor filter by making it keep all
        cfg["_keep_furniture"] = True

    result = run(args.in_path, args.out_dir, cfg)
    print(f"walls={result['counts']['walls']} columns={result['counts']['columns']} "
          f"beams={result['counts']['beams']} openings={result['counts']['openings']} "
          f"rooms={result['counts']['rooms']}")
    print(f"wrote: {result['glb']}, {result['dxf']}, {result['svg']}, {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
