"""Zoom into the persistent unenclosed region (top-right / right side of the
koushik debug renders, present at every plane_max_points level tried) and
render it two ways: plain top-down density, and colored by Z-height, to
distinguish "real missed structure at our own storey height" from "a
different building glimpsed through the balcony at another elevation" (per
this project's known scan characteristics: the balcony is kept, the
neighbouring building across the air gap normally drops out on Z/connectivity
during isolation -- but a partial bleed-through is possible).

Uses the SAME isolated+aligned cloud stage as isolidarflow.run() (no plane
detection needed -- this is a raw-data question).
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.isolidarflow import DEFAULT_CONFIG

PPM = 80  # pixels per metre -- higher res than the debug PNG for a zoomed crop


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
    scan, iso_stats = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                                   max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)
    xyz = scan.xyz
    log(f"isolated+aligned: {scan.n:,} points")

    xy = xyz[:, :2]
    z = xyz[:, 2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    W, H = int((xmax - xmin) * PPM) + 1, int((ymax - ymin) * PPM) + 1
    log(f"full footprint: {xmax-xmin:.1f} x {ymax-ymin:.1f} m -> {W}x{H}px")

    cols = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)

    # 1) plain density (log-scaled), full footprint -- so the gap's position
    # relative to the whole unit is visible, not just a pre-guessed crop.
    density = np.zeros((H, W), dtype=np.float64)
    np.add.at(density, (rows, cols), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "gap_density_full.png"), gray)

    # 2) height-colored: mean Z per occupied cell, so a different building at
    # a different absolute elevation (or a large offset within the storey)
    # reads as a visibly different color from the rest of the unit.
    zsum = np.zeros((H, W), dtype=np.float64)
    zcount = np.zeros((H, W), dtype=np.float64)
    np.add.at(zsum, (rows, cols), z)
    np.add.at(zcount, (rows, cols), 1.0)
    mean_z = np.divide(zsum, zcount, out=np.full_like(zsum, np.nan), where=zcount > 0)

    z_lo, z_hi = np.nanpercentile(mean_z, 1), np.nanpercentile(mean_z, 99)
    norm = np.clip((mean_z - z_lo) / max(z_hi - z_lo, 1e-6), 0, 1)
    norm_u8 = np.nan_to_num(norm * 255, nan=0).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    colored[zcount == 0] = (0, 0, 0)
    cv2.imwrite(str(out_dir / "gap_height_colored.png"), colored)
    log(f"z range plotted: {z_lo:.2f}m (blue) .. {z_hi:.2f}m (red), storey z-band was {z_band}")

    # 3) trajectory overlay on the height-colored image -- did the scanner
    # ever walk into the gap region at all? If not, it's outside the walked
    # unit almost by definition.
    overlay = colored.copy()
    if traj.shape[0]:
        traj_r = traj @ np.asarray(R, dtype=float).T if False else traj  # already rotated in isolidarflow;
        # NOTE: this script's traj was computed pre-align; rotate it now to match scan's frame.
        traj_rot = traj @ np.asarray(R, dtype=float).T
        tcols = np.clip(((traj_rot[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
        trows = np.clip(((ymax - traj_rot[:, 1]) * PPM).astype(int), 0, H - 1)
        for c, r in zip(tcols, trows):
            cv2.circle(overlay, (c, r), 2, (255, 255, 255), -1)
    cv2.imwrite(str(out_dir / "gap_height_with_trajectory.png"), overlay)

    log("wrote gap_density_full.png, gap_height_colored.png, gap_height_with_trajectory.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
