"""rgb_section_sweep.py
--------------------
Dense vertical-section sweep, each cut colored by TRUE RGB (for scans that
carry color, e.g. mujammel). Cuts many parallel vertical planes along X and
along Y and renders each as an elevation panel painted with the real point
colors -- so at any cut you can see exactly what material is there: a matte
wall, a glossy window/glass pane, a wooden door, a plant on the balcony, etc.
Each panel is labeled with its exact cut coordinate so you can locate it on
the plan.

Produces (in output/rgb_sections_<name>/):
  sweepX_rgb.png   N cuts at increasing X, painted RGB, elevation view
  sweepY_rgb.png   N cuts at increasing Y, painted RGB, elevation view

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\rgb_section_sweep.py <scan.las> <scan_name> [n_sections]
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

PPM = 70
BAND_M = 0.06
COLS = 2          # montage columns (panels laid out in a grid, not one tall strip)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las_path, scan_name, n_sections=20):
    n_sections = int(n_sections)
    out_dir = ROOT / "output" / f"rgb_sections_{scan_name}"
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
    log(f"loaded {scan.n:,} pts | rgb={scan.rgb is not None}")
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
    rgb = scan.rgb
    have_rgb = rgb is not None
    if not have_rgb:
        log("no RGB in this scan -- panels will be grayscale occupancy instead")
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    z_lo_view, z_hi_view = z_floor - 0.25, z_ceiling + 0.25
    log(f"aligned {scan.n:,} pts | storey z=[{z_floor:.2f},{z_ceiling:.2f}]")

    def panel(pos_vals, z_vals, colors, pos_lo, pos_hi, label):
        Wp = int((pos_hi - pos_lo) * PPM) + 1
        Hp = int((z_hi_view - z_lo_view) * PPM) + 1
        img = np.zeros((Hp, Wp, 3), np.uint8)
        if pos_vals.size:
            cols = np.clip(((pos_vals - pos_lo) * PPM).astype(int), 0, Wp - 1)
            rows = np.clip(((z_hi_view - z_vals) * PPM).astype(int), 0, Hp - 1)
            if colors is not None:
                # paint BGR from RGB (later points overwrite -- fine for a section)
                img[rows, cols] = colors[:, ::-1]
            else:
                img[rows, cols] = (210, 210, 210)
        # floor/ceiling reference lines
        for zc in (z_floor, z_ceiling):
            r = int((z_hi_view - zc) * PPM)
            cv2.line(img, (0, r), (Wp, r), (0, 180, 0), 1)
        lab_h = 16
        framed = np.full((Hp + lab_h, Wp, 3), 25, np.uint8)
        framed[lab_h:, :] = img
        cv2.putText(framed, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
        return framed

    def sweep(axis):
        if axis == "x":
            cut_lo, cut_hi, pos_lo, pos_hi = xlo, xhi, ylo, yhi
        else:
            cut_lo, cut_hi, pos_lo, pos_hi = ylo, yhi, xlo, xhi
        cut_positions = np.linspace(cut_lo, cut_hi, n_sections + 2)[1:-1]
        panels = []
        for cp in cut_positions:
            m = (np.abs(x - cp) <= BAND_M) if axis == "x" else (np.abs(y - cp) <= BAND_M)
            pos_vals = (y[m] if axis == "x" else x[m])
            colors = rgb[m] if have_rgb else None
            lbl = f"{axis.upper()}={cp:.2f}m"
            panels.append(panel(pos_vals, z[m], colors, pos_lo, pos_hi, lbl))
        # grid montage
        Wmax = max(p.shape[1] for p in panels)
        Hmax = max(p.shape[0] for p in panels)
        norm = []
        for p in panels:
            pp = cv2.copyMakeBorder(p, 0, Hmax - p.shape[0], 0, Wmax - p.shape[1],
                                    cv2.BORDER_CONSTANT, value=(25, 25, 25))
            norm.append(pp)
        rows_of = []
        for i in range(0, len(norm), COLS):
            row = norm[i:i + COLS]
            while len(row) < COLS:
                row.append(np.full((Hmax, Wmax, 3), 15, np.uint8))
            row_img = []
            for pp in row:
                row_img.append(pp)
                row_img.append(np.full((Hmax, 6, 3), 45, np.uint8))
            rows_of.append(np.hstack(row_img))
            rows_of.append(np.full((6, rows_of[-1].shape[1], 3), 45, np.uint8))
        montage = np.vstack(rows_of)
        header = np.full((26, montage.shape[1], 3), 15, np.uint8)
        cv2.putText(header, f"{scan_name}  {n_sections} RGB VERTICAL SECTIONS sweeping {axis.upper()} "
                            f"(elevation; green=floor/ceiling; read openings, railings, materials by color)",
                   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, montage])

    cv2.imwrite(str(out_dir / "sweepX_rgb.png"), sweep("x"))
    log(f"wrote {out_dir / 'sweepX_rgb.png'} ({n_sections} cuts)")
    cv2.imwrite(str(out_dir / "sweepY_rgb.png"), sweep("y"))
    log(f"wrote {out_dir / 'sweepY_rgb.png'} ({n_sections} cuts)")
    log(f"[{scan_name}] rgb section sweep complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 20)
