"""Sweep plane_max_points across several levels on the SAME isolated+aligned
cloud (loaded/isolated once -- expensive, shared) to find where room/wall
closure actually plateaus, rather than guessing a single cap.

Mirrors scripts/isolidarflow.py's run() stage-for-stage from "working cloud"
(stage 6) through rooms/openings (stage 12), using the SAME DEFAULT_CONFIG so
results are directly comparable to CLI runs at those same point counts.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\sweep_plane_points.py <scan.las> <out_dir>
Writes <out_dir>/sweep_results.json and one debug PNG per level.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame, planes, structure, regularize
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory, wall_crossings
from scripts.recon.openings import detect_openings
from scripts.recon.floorplan2d import build_room_polygons
from scripts.isolidarflow import DEFAULT_CONFIG, _wall_length, _wall_namespaces, _write_debug_png

LEVELS = [1_500_000, 3_000_000, 6_000_000, 9_000_000, 12_000_000, 16_000_000, 0]  # 0 = full/uncapped


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(in_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])

    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    t0 = time.time()
    log("loading + isolating (shared across all levels)...")
    scan = load_scan(in_path, max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"],
                                 margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"],
                                     std_ratio=cfg["outlier_std_ratio"])
    traj = (approx_trajectory(scan.gps_time, scan.xyz, dt_s=cfg["traj_dt_s"])
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, iso_stats = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                                   max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)
    if traj.shape[0]:
        traj = traj @ np.asarray(R, dtype=float).T
    log(f"shared prep done in {time.time()-t0:.0f}s: {scan.n:,} isolated+aligned points "
        f"(dropped {iso_stats['dropped']:,})")

    results = []
    for level in LEVELS:
        label = "full" if level == 0 else f"{level/1e6:.1f}M"
        t1 = time.time()
        if level and scan.n > level:
            keep = rng.choice(scan.n, size=level, replace=False)
            level_scan = scan.subset(keep)
        else:
            level_scan = scan
        xyz = level_scan.xyz
        n_working = level_scan.n
        log(f"--- level {label} ({n_working:,} points) ---")

        z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
        z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))

        all_planes = planes.detect_planes(
            xyz, dist_thresh=cfg["plane_dist_thresh_m"], min_inliers=cfg["plane_min_inliers"],
            max_planes=cfg["plane_max_planes"], dbscan_eps=cfg["plane_dbscan_eps_m"],
            dbscan_min=cfg["plane_dbscan_min"], z_floor=z_floor, z_ceiling=z_ceiling,
            horizontal_min=cfg["plane_horizontal_min"], ransac_iters=cfg["plane_ransac_iters"], seed=seed)
        verticals = [p for p in all_planes if p.label == "vertical"]
        t_planes = time.time() - t1
        log(f"  detect_planes: {len(all_planes)} planes ({len(verticals)} vertical) in {t_planes:.0f}s")

        runs, used_w = structure.group_wall_runs(
            verticals, xyz, np.eye(3), merge_offset_m=cfg["wr_merge_offset_m"],
            max_relief_m=cfg["wr_max_relief_m"], min_run_length_m=cfg["wr_min_run_length_m"],
            angle_tol_deg=cfg["wr_angle_tol_deg"], full_height_frac=cfg["wr_full_height_frac"],
            u_gap_m=cfg["wr_u_gap_m"], min_run_inliers=cfg["wr_min_run_inliers"], return_used_indices=True)
        columns, beams, used_cb = structure.extract_columns_beams(
            verticals, xyz, z_floor, z_ceiling, min_size_m=cfg["cb_min_size_m"],
            max_size_m=cfg["cb_max_size_m"], height_tol_m=cfg["cb_height_tol_m"],
            beam_elevation_m=cfg["cb_beam_elevation_m"], beam_ceiling_tol_m=cfg["cb_beam_ceiling_tol_m"],
            beam_offset_gap_m=cfg["cb_beam_offset_gap_m"], return_used_indices=True)
        unclassified = structure.extract_unclassified(verticals, used_w | used_cb, xyz,
                                                       direction_tol_deg=cfg["unclassified_dir_tol_deg"])

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
        walls = [w for w in walls if _wall_length(w) > cfg["min_wall_length_m"]]
        for w in walls:
            w["floor_z_m"] = z_floor
            w["ceiling_z_m"] = z_ceiling

        crossings = wall_crossings(traj, walls, end_margin_m=cfg["crossing_end_margin_m"])
        openings_by_wall = detect_openings(walls, xyz, traj, crossings, z_floor, z_ceiling, cfg["priors"])
        n_openings = sum(len(v) for v in openings_by_wall.values())
        opening_types = sorted({o["type"] for v in openings_by_wall.values() for o in v})

        wall_ns = _wall_namespaces(walls)
        rooms = [p.simplify(cfg["room_simplify_m"])
                 for p in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_primary_m"])
                 if p.area >= cfg["room_min_area_m2"]]
        for cell in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_recovery_m"]):
            if cell.area >= cfg["room_min_area_m2"] and not any(r.contains(cell.centroid) for r in rooms):
                rooms.append(cell.simplify(cfg["room_simplify_m"]))

        elapsed = time.time() - t1
        result = {
            "level": label, "n_working_points": n_working,
            "n_planes": len(all_planes), "n_vertical": len(verticals),
            "n_walls": len(walls), "n_columns": len(columns), "n_beams": len(beams),
            "n_unclassified": len(unclassified), "n_crossings": int(sum(len(v) for v in crossings.values())),
            "n_openings": n_openings, "opening_types": opening_types,
            "n_rooms": len(rooms), "room_areas_m2": sorted(round(r.area, 1) for r in rooms),
            "seconds": round(elapsed, 1),
        }
        results.append(result)
        log(f"  -> walls={result['n_walls']} cols={result['n_columns']} beams={result['n_beams']} "
            f"openings={n_openings} rooms={result['n_rooms']} unclassified={result['n_unclassified']} "
            f"({elapsed:.0f}s)")

        try:
            png = out_dir / f"debug_{label.replace('.', '_')}.png"
            _write_debug_png(str(out_dir), xyz, walls, rooms, columns, openings_by_wall,
                             z_floor, z_ceiling, level_scan)
            default_png = out_dir / "floorplan_debug.png"
            if default_png.exists():
                default_png.replace(png)
        except Exception as exc:
            log(f"  debug PNG failed: {exc}")

        with open(out_dir / "sweep_results.json", "w") as fh:
            json.dump(results, fh, indent=2)

    log("sweep done.")
    for r in results:
        print(f"{r['level']:>6}  pts={r['n_working_points']:>9,}  walls={r['n_walls']:>3}  "
              f"cols={r['n_columns']}  beams={r['n_beams']:>2}  openings={r['n_openings']:>3}  "
              f"rooms={r['n_rooms']}  unclassified={r['n_unclassified']:>3}  "
              f"time={r['seconds']:>5.0f}s  areas={r['room_areas_m2']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
