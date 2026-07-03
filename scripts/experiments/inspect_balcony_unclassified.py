"""Render accepted walls (green) vs UNCLASSIFIED vertical planes (bright red)
on the koushik isolated cloud, at the default 1.5M working-point level, to
see whether the two suspected balcony regions (top-right room, right-side
strip -- both show a railing-like grid pattern in the raw density image) have
real detected vertical planes that fail to become wall runs, or have no
detected planes there at all.

Writes output/koushik_gap_check/balcony_unclassified.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame, planes, structure
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.isolidarflow import DEFAULT_CONFIG

PPM = 80


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

    log("loading + isolating...")
    scan = load_scan(in_path, max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"],
                                 margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (approx_trajectory(scan.gps_time, scan.xyz, dt_s=cfg["traj_dt_s"])
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, _ = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                           max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)

    if scan.n > cfg["plane_max_points"]:
        keep = rng.choice(scan.n, size=cfg["plane_max_points"], replace=False)
        scan = scan.subset(keep)
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    log(f"working cloud: {scan.n:,} points, z-band ({z_floor:.2f}, {z_ceiling:.2f})")

    t0 = time.time()
    all_planes = planes.detect_planes(
        xyz, dist_thresh=cfg["plane_dist_thresh_m"], min_inliers=cfg["plane_min_inliers"],
        max_planes=cfg["plane_max_planes"], dbscan_eps=cfg["plane_dbscan_eps_m"],
        dbscan_min=cfg["plane_dbscan_min"], z_floor=z_floor, z_ceiling=z_ceiling,
        horizontal_min=cfg["plane_horizontal_min"], ransac_iters=cfg["plane_ransac_iters"], seed=seed)
    verticals = [p for p in all_planes if p.label == "vertical"]
    log(f"detect_planes: {len(all_planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

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
    used = used_w | used_cb
    unclassified = structure.extract_unclassified(verticals, used, xyz,
                                                   direction_tol_deg=cfg["unclassified_dir_tol_deg"])
    log(f"runs={len(runs)} columns={len(columns)} beams={len(beams)} "
        f"unclassified={len(unclassified)}/{len(verticals)}")

    # ---------- render ----------
    xy = xyz[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    W, H = int((xmax - xmin) * PPM) + 1, int((ymax - ymin) * PPM) + 1

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    cols = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)
    density = np.zeros((H, W), dtype=np.float64)
    np.add.at(density, (rows, cols), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 100).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])

    # unclassified vertical planes: look up each by plane_index into the
    # ORIGINAL verticals list (extract_unclassified returns summary dicts,
    # not the points) -- render their real inlier points bright red, and also
    # tag n_points>=200 ones (real support, not noise) even brighter/thicker.
    n_big_unclassified = 0
    for u in unclassified:
        p = verticals[u["plane_index"]]
        pts = xyz[p.inlier_idx]
        pc = np.clip(((pts[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
        pr = np.clip(((ymax - pts[:, 1]) * PPM).astype(int), 0, H - 1)
        big = u["n_points"] >= 200
        n_big_unclassified += int(big)
        color = (60, 60, 255) if big else (0, 0, 140)  # BGR: bright red vs dim red
        img[pr, pc] = color
    log(f"unclassified planes with >=200 points (real support, not noise): {n_big_unclassified}/{len(unclassified)}")

    # accepted wall runs: green centerlines (drawn AFTER unclassified so they
    # stay visible on top where a run and an unclassified plane overlap)
    for r in runs:
        cv2.line(img, to_px(r["p0"]), to_px(r["p1"]), (0, 220, 0), 3)

    # columns: orange outline; beams: cyan centerline
    for c in columns:
        pts = np.array([to_px(p) for p in c.footprint], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 165, 255), 2)  # BGR orange
    for b in beams:
        cv2.line(img, to_px(b.p0), to_px(b.p1), (255, 255, 0), 2)  # BGR cyan

    banner = (f"green=accepted wall runs ({len(runs)})  red=UNCLASSIFIED vertical planes "
              f"({len(unclassified)}/{len(verticals)})  orange=columns/beams")
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_png = out_dir / "balcony_unclassified.png"
    cv2.imwrite(str(out_png), img)
    log(f"wrote {out_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
