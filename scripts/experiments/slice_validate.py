"""Cheap validation step before building the full slice-stacked wall
detector: take 4 representative horizontal Z-slices (floor-level,
waist-height where a railing should show, mid-wall, near-ceiling), build a
literal point-count OCCUPANCY grid for each (no brightness thresholding --
sidesteps the whole Otsu/Canny struggle from the flat top-down pass, since a
thin slice has no floor-reobservation confound: real air should be
genuinely empty), and render each as its own PNG so we can visually check
whether the railing appears at low height and disappears higher up before
investing in the full multi-slice merge logic.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\slice_validate.py <scan.las> <out_dir>
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

PPM = 80
CELL_M = 1.0 / PPM
MIN_PTS_OCCUPIED = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def occupancy_image(xy, xmin, ymin, xmax, ymax, ppm, min_pts):
    W, H = int((xmax - xmin) * ppm) + 1, int((ymax - ymin) * ppm) + 1
    if xy.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8), W, H
    cols = np.clip(((xy[:, 0] - xmin) * ppm).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) * ppm).astype(int), 0, H - 1)
    count = np.zeros((H, W), dtype=np.int32)
    np.add.at(count, (rows, cols), 1)
    occ = (count >= min_pts).astype(np.uint8) * 255
    return occ, W, H


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
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
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
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    storey = z_ceiling - z_floor
    log(f"{scan.n:,} points, z-band ({z_floor:.2f}, {z_ceiling:.2f}), storey={storey:.2f}m")

    xy_all = xyz[:, :2]
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)

    slices = [
        ("floor_level", z_floor + 0.02, z_floor + 0.17),
        ("waist_height_railing_zone", z_floor + 0.70, z_floor + 1.10),
        ("mid_wall", z_floor + 1.30, z_floor + 1.60),
        ("near_ceiling", z_ceiling - 0.30, z_ceiling - 0.05),
    ]

    for name, z_lo, z_hi in slices:
        mask = (xyz[:, 2] >= z_lo) & (xyz[:, 2] < z_hi)
        xy = xy_all[mask]
        occ, W, H = occupancy_image(xy, xmin, ymin, xmax, ymax, PPM, MIN_PTS_OCCUPIED)
        n_occ_px = int(np.count_nonzero(occ))
        log(f"  {name} [{z_lo:.2f},{z_hi:.2f}]: {xy.shape[0]:,} points, "
            f"{n_occ_px:,} occupied px ({100*n_occ_px/(W*H):.1f}% of frame)")
        img = cv2.merge([occ, occ, occ])
        cv2.putText(img, f"{name}  z=[{z_lo:.2f},{z_hi:.2f}]  n={xy.shape[0]:,}  "
                          f"occ_px={n_occ_px:,}",
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"slice_{name}.png"), img)

    log(f"wrote 4 slice PNGs to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
