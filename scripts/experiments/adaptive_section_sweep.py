"""adaptive_section_sweep.py
-------------------------
Smart (importance-sampled) vertical-section sweep. Instead of uniformly
spacing cuts -- which wastes most of them on empty room interiors -- this:

1. Builds a top-down occupancy map and finds WHERE the walls actually are
   (peaks in the X- and Y-marginal occupancy profiles: a wall running along
   Y shows as a tall column of occupied cells at some X, and vice-versa).
2. Concentrates section cuts ON and tightly AROUND each detected wall line
   (a cluster of closely-spaced cuts per wall), because that is exactly
   where the interesting structure lives -- door/window openings show as
   gaps, the wall's two faces and any reveal show across the surrounding
   cuts. Empty interior space gets no cuts.
3. Each cut is a vertical section rendered as an elevation, RGB-painted when
   the scan has color, labeled with its exact coordinate and whether it sits
   on a detected wall line.

~n_target cuts total, distributed by wall count. Produces sweepX / sweepY
montages in output/adaptive_sections_<name>/.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\adaptive_section_sweep.py <scan.las> <scan_name> [n_target]
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.isolidarflow import DEFAULT_CONFIG

PPM = 70
BAND_M = 0.05
GRID_COLS = 4
CLUSTER_HALF_M = 0.12       # cuts span +-this around each wall line
CLUSTER_STEP_M = 0.04       # spacing of cuts within a wall cluster


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_wall_positions(occ, origin, cell_m, axis, min_prominence_frac=0.15):
    """Peaks in the marginal occupancy profile = wall lines.
    axis 'x': sum over rows(Y) -> profile over X columns -> X of Y-running walls.
    axis 'y': sum over cols(X) -> profile over Y rows   -> Y of X-running walls.
    Returns list of world coordinates of detected wall lines."""
    if axis == "x":
        profile = occ.sum(axis=0).astype(float)   # length = nx (over X)
    else:
        profile = occ.sum(axis=1).astype(float)   # length = ny (over Y)
    if profile.max() <= 0:
        return []
    peaks, _ = find_peaks(profile, prominence=profile.max() * min_prominence_frac,
                          distance=int(0.3 / cell_m))
    coords = []
    for p in peaks:
        if axis == "x":
            coords.append(origin[0] + p * cell_m)       # world X
        else:
            # image row 0 = ymax; convert row index back to world Y
            coords.append(origin[1] - p * cell_m)
    return coords


def main(las_path, scan_name, n_target=100):
    n_target = int(n_target)
    out_dir = ROOT / "output" / f"adaptive_sections_{scan_name}"
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
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    z_lo_view, z_hi_view = z_floor - 0.25, z_ceiling + 0.25
    log(f"aligned {scan.n:,} pts | rgb={have_rgb} | storey z=[{z_floor:.2f},{z_ceiling:.2f}]")

    # ---- top-down occupancy at wall-reliable height band (waist+mid) ----
    cell = 0.04
    xmin, ymin = x.min(), y.min()
    xmax, ymax = x.max(), y.max()
    nx = int((xmax - xmin) / cell) + 1
    ny = int((ymax - ymin) / cell) + 1
    band = (z >= z_floor + 0.45) & (z <= z_floor + 1.35)
    ix = np.clip(((x[band] - xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((ymax - y[band]) / cell).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), np.uint8)
    occ[iy, ix] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    walls_x = find_wall_positions(occ, (xmin, ymax), cell, "x")
    walls_y = find_wall_positions(occ, (xmin, ymax), cell, "y")
    log(f"detected wall lines: {len(walls_x)} along-Y walls (at X), {len(walls_y)} along-X walls (at Y)")

    # ---- build adaptive cut list: cluster of cuts around each wall line ----
    def clusters(wall_coords, budget):
        cuts = []
        offsets = np.arange(-CLUSTER_HALF_M, CLUSTER_HALF_M + 1e-6, CLUSTER_STEP_M)
        for wc in wall_coords:
            for off in offsets:
                cuts.append((wc + off, abs(off) < 1e-6))  # (coord, is_wall_center)
        # if that overshoots the budget, thin uniformly but keep every center
        if len(cuts) > budget:
            centers = [c for c in cuts if c[1]]
            others = [c for c in cuts if not c[1]]
            keep = centers + others[:: max(1, len(others) // max(1, budget - len(centers)))]
            cuts = sorted(keep[:budget])
        return sorted(cuts)

    xlo_v, xhi_v = np.percentile(x, [0.5, 99.5])
    ylo_v, yhi_v = np.percentile(y, [0.5, 99.5])

    def render_sweep(axis, wall_coords, pos_lo, pos_hi, budget):
        cut_list = clusters(wall_coords, budget)
        panels = []
        for coord, is_center in cut_list:
            m = (np.abs(x - coord) <= BAND_M) if axis == "x" else (np.abs(y - coord) <= BAND_M)
            pos_vals = (y[m] if axis == "x" else x[m])
            Wp = int((pos_hi - pos_lo) * PPM) + 1
            Hp = int((z_hi_view - z_lo_view) * PPM) + 1
            img = np.zeros((Hp, Wp, 3), np.uint8)
            if pos_vals.size:
                cc = np.clip(((pos_vals - pos_lo) * PPM).astype(int), 0, Wp - 1)
                rr = np.clip(((z_hi_view - z[m]) * PPM).astype(int), 0, Hp - 1)
                img[rr, cc] = (rgb[m][:, ::-1] if have_rgb else (210, 210, 210))
            for zc in (z_floor, z_ceiling):
                r = int((z_hi_view - zc) * PPM)
                cv2.line(img, (0, r), (Wp, r), (0, 150, 0), 1)
            border_col = (0, 255, 255) if is_center else (70, 70, 70)
            img = cv2.copyMakeBorder(img, 15, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_col)
            tag = "WALL " if is_center else "     "
            cv2.putText(img, f"{tag}{axis.upper()}={coord:.2f}", (4, 11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0) if is_center else (200, 200, 200), 1, cv2.LINE_AA)
            panels.append(img)

        Wmax = max(p.shape[1] for p in panels)
        Hmax = max(p.shape[0] for p in panels)
        norm = [cv2.copyMakeBorder(p, 0, Hmax - p.shape[0], 0, Wmax - p.shape[1],
                                   cv2.BORDER_CONSTANT, value=(25, 25, 25)) for p in panels]
        rows_of = []
        for i in range(0, len(norm), GRID_COLS):
            row = norm[i:i + GRID_COLS]
            while len(row) < GRID_COLS:
                row.append(np.full((Hmax, Wmax, 3), 15, np.uint8))
            rows_of.append(np.hstack(row))
        montage = np.vstack(rows_of)
        header = np.full((26, montage.shape[1], 3), 15, np.uint8)
        cv2.putText(header, f"{scan_name}  {len(panels)} ADAPTIVE sections sweeping {axis.upper()} "
                            f"(clustered on {len(wall_coords)} detected walls; yellow border=on wall centerline; "
                            f"gaps=openings)",
                   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, montage]), len(panels)

    budget_each = n_target // 2
    imgX, nX = render_sweep("x", walls_x, ylo_v, yhi_v, budget_each)
    cv2.imwrite(str(out_dir / "sweepX_adaptive.png"), imgX)
    log(f"wrote sweepX_adaptive.png ({nX} cuts on {len(walls_x)} walls)")
    imgY, nY = render_sweep("y", walls_y, xlo_v, xhi_v, budget_each)
    cv2.imwrite(str(out_dir / "sweepY_adaptive.png"), imgY)
    log(f"wrote sweepY_adaptive.png ({nY} cuts on {len(walls_y)} walls)")

    # also save the top-down occupancy with detected wall lines marked
    dbg = cv2.merge([occ * 120] * 3)
    for wc in walls_x:
        col = int((wc - xmin) / cell)
        cv2.line(dbg, (col, 0), (col, ny), (0, 255, 255), 1)
    for wc in walls_y:
        row = int((ymax - wc) / cell)
        cv2.line(dbg, (0, row), (nx, row), (255, 0, 255), 1)
    cv2.putText(dbg, f"{scan_name} detected walls: yellow=along-Y (X-cut) magenta=along-X (Y-cut)",
               (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "detected_wall_lines.png"), cv2.resize(dbg, (nx * 2, ny * 2), interpolation=cv2.INTER_NEAREST))
    log(f"[{scan_name}] adaptive section sweep complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 100)
