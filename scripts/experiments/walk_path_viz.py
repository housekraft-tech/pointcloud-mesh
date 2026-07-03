"""walk_path_viz.py
----------------
A clear standalone visualization of the operator's WALK PATH, recovered from
gps_time (native-tick median poses -- see freespace_floorplan.
sensor_trajectory_from_gpstime). The path is drawn as a connected line
colored by TIME (start->end), over a faint wall/occupancy background, with
START/END markers and periodic direction arrows, so the actual route through
the apartment is legible: where you entered, the order rooms were visited,
where you lingered vs. passed through.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\walk_path_viz.py <scan.las> <scan_name>
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
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime

PPM = 70


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las_path, scan_name, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"walkpath_{scan_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log(f"[{scan_name}] load + isolate + align ...")
    scan = load_scan(str(las_path), max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    gps_full = scan.gps_time
    traj = sensor_trajectory_from_gpstime(scan.xyz, gps_full) if gps_full is not None else np.zeros((0, 3))
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
    # re-derive the trajectory in the ALIGNED frame directly from the isolated
    # scan's own gps_time so it lines up with the rendered points exactly
    if scan.gps_time is not None:
        traj = sensor_trajectory_from_gpstime(xyz, scan.gps_time)
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    log(f"aligned {scan.n:,} pts | {len(traj)} walk poses")

    xy = xyz[:, :2]
    xmin, ymin = xy.min(axis=0) - 0.3
    xmax, ymax = xy.max(axis=0) + 0.3
    W, H = int((xmax - xmin) * PPM), int((ymax - ymin) * PPM)

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    # faint wall/occupancy background (mid-wall band)
    band = (xyz[:, 2] >= z_floor + 0.9) & (xyz[:, 2] <= z_floor + 1.6)
    cols = np.clip(((xy[band, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[band, 1]) * PPM).astype(int), 0, H - 1)
    occ = np.zeros((H, W), np.uint8)
    occ[rows, cols] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img = cv2.merge([occ * 55] * 3)  # dim gray walls

    # walk path as a TIME-colored connected polyline (start -> end)
    n = len(traj)
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        # turbo-ish gradient: green(start) -> yellow -> red(end)
        col = cv2.applyColorMap(np.uint8([[int(t * 255)]]), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(img, to_px(traj[i][:2]), to_px(traj[i + 1][:2]), col, 2, cv2.LINE_AA)

    # direction arrows every ~20 poses
    for i in range(0, n - 1, 20):
        p0, p1 = to_px(traj[i][:2]), to_px(traj[min(i + 3, n - 1)][:2])
        cv2.arrowedLine(img, p0, p1, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.5)

    if n:
        cv2.circle(img, to_px(traj[0][:2]), 9, (0, 255, 0), -1)
        cv2.putText(img, "START", (to_px(traj[0][:2])[0] + 11, to_px(traj[0][:2])[1] + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, to_px(traj[-1][:2]), 9, (0, 0, 255), -1)
        cv2.putText(img, "END", (to_px(traj[-1][:2])[0] + 11, to_px(traj[-1][:2])[1] + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # color-time legend bar
    bar_w = 260
    for i in range(bar_w):
        col = cv2.applyColorMap(np.uint8([[int(i / bar_w * 255)]]), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(img, (20 + i, H - 30), (20 + i, H - 18), col, 1)
    cv2.putText(img, "START", (20, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "END", (20 + bar_w - 26, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{scan_name} WALK PATH  ({n} poses, colored by time; arrows=direction; "
                     f"gray=walls)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "walk_path.png"), img)
    log(f"wrote {out_dir / 'walk_path.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
