"""openings_from_sections.py
-------------------------
USE the vertical sections for reconstruction, not just diagnostics.

The walk-path method finds DOORS (you walk through them) but never WINDOWS
(you don't). A vertical section cut ALONG a wall is exactly the signal for
both: the wall's elevation, where an opening is a gap in the floor->ceiling
occupancy. Classify each gap by its height band:
   DOOR / pass-through : reaches the floor, tall
   WINDOW             : sill above the floor, top below the ceiling
   (a beam soffit -- solid at top, open below -- is handled elsewhere)

For each detected wall line this builds the (along-wall u  x  height z)
occupancy, finds the gaps, classifies them, and marks doors/windows on the
floorplan + saves per-wall elevation strips with openings outlined.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\openings_from_sections.py <isolated_or_raw.las> <scan_name>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime

CELL = 0.03
WALL_BAND_M = 0.20      # points within this of a wall line contribute to its elevation
DOOR_MIN_H = 1.6        # a floor-touching gap this tall is a door/pass-through
WIN_MIN_SILL = 0.3      # a gap whose bottom is this far above the floor is a window
MIN_OPENING_W = 0.4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_wall_lines(occ, xmin, ymax, cell):
    """Peaks in the X- and Y-marginal occupancy -> wall lines, returned as
    ('x'|'y', coord) where 'y'-walls run along X (normal along Y) etc."""
    walls = []
    colsum = occ.sum(axis=0).astype(float)
    for p in find_peaks(colsum, prominence=colsum.max() * 0.15, distance=int(0.4 / cell))[0]:
        walls.append(("y", xmin + p * cell))       # vertical wall at X=coord
    rowsum = occ.sum(axis=1).astype(float)
    for p in find_peaks(rowsum, prominence=rowsum.max() * 0.15, distance=int(0.4 / cell))[0]:
        walls.append(("x", ymax - p * cell))        # horizontal wall at Y=coord
    return walls


def main(las_path, scan_name, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"openings_{scan_name}"
    (out_dir / "elevations").mkdir(parents=True, exist_ok=True)
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
    traj = (sensor_trajectory_from_gpstime(scan.xyz, scan.gps_time)
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
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    nx = int((xmax - xmin) / CELL) + 1
    ny = int((ymax - ymin) / CELL) + 1
    log(f"aligned {scan.n:,} pts | storey [{z_floor:.2f},{z_ceiling:.2f}]")

    # wall lines from the mid-height occupancy
    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cxi = np.clip(((x[band] - xmin) / CELL).astype(int), 0, nx - 1)
    cyi = np.clip(((ymax - y[band]) / CELL).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), np.uint8); occ[cyi, cxi] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    walls = detect_wall_lines(occ, xmin, ymax, CELL)
    log(f"{len(walls)} wall lines")

    # ---- per-wall elevation -> opening detection ----
    ez = np.arange(z_floor - 0.1, z_ceiling + 0.1, CELL)
    nz = len(ez)
    all_openings = []  # (axis, offset, u0, u1, z0, z1, type)
    for wi, (axis, off) in enumerate(walls):
        if axis == "y":     # vertical wall at X=off; along-axis is Y
            sel = np.abs(x - off) <= WALL_BAND_M
            u = y[sel]
        else:               # horizontal wall at Y=off; along-axis is X
            sel = np.abs(y - off) <= WALL_BAND_M
            u = x[sel]
        zz = z[sel]
        if u.size < 500:
            continue
        u0a, u1a = np.percentile(u, [1, 99])
        nu = int((u1a - u0a) / CELL) + 1
        if nu < 10:
            continue
        elev = np.zeros((nz, nu), np.uint8)
        ui = np.clip(((u - u0a) / CELL).astype(int), 0, nu - 1)
        zi = np.clip(((ez[-1] - zz) / CELL).astype(int), 0, nz - 1)
        elev[zi, ui] = 1
        elev = cv2.morphologyEx(elev, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        # openings = empty regions bounded left/right by wall (interior gaps)
        empty = (elev == 0).astype(np.uint8)
        # pad occupied border so exterior empty doesn't count
        padded = np.pad(elev, 1, constant_values=1)
        holes = ndimage.binary_fill_holes(padded) & (padded == 0)
        holes = holes[1:-1, 1:-1]
        lblh, nh = ndimage.label(holes, structure=np.ones((3, 3)))
        strip = cv2.merge([elev * 200] * 3)
        for hi in range(1, nh + 1):
            ys, xs = np.where(lblh == hi)
            if xs.size < 20:
                continue
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            w_m = (c1 - c0) * CELL
            top_z = ez[-1] - r0 * CELL
            bot_z = ez[-1] - r1 * CELL
            if w_m < MIN_OPENING_W:
                continue
            sill = bot_z - z_floor
            height = top_z - bot_z
            if sill <= 0.15 and height >= DOOR_MIN_H:
                typ, col = "door", (0, 140, 255)
            elif sill >= WIN_MIN_SILL and top_z <= z_ceiling - 0.15:
                typ, col = "window", (255, 160, 0)
            else:
                typ, col = "opening", (160, 130, 110)
            uu0 = u0a + c0 * CELL; uu1 = u0a + c1 * CELL
            all_openings.append((axis, off, uu0, uu1, bot_z, top_z, typ))
            cv2.rectangle(strip, (c0, r0), (c1, r1), col, 2)
            cv2.putText(strip, f"{typ} {w_m:.1f}x{height:.1f}m s{sill:.1f}", (c0, max(r0 - 3, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        # floor/ceiling refs
        for zc in (z_floor, z_ceiling):
            r = int((ez[-1] - zc) / CELL)
            cv2.line(strip, (0, r), (nu, r), (0, 180, 0), 1)
        cv2.imwrite(str(out_dir / "elevations" / f"wall_{wi:02d}_{axis}{off:.2f}.png"),
                    cv2.resize(strip, (nu * 2, nz * 2), interpolation=cv2.INTER_NEAREST))

    n_door = sum(1 for o in all_openings if o[6] == "door")
    n_win = sum(1 for o in all_openings if o[6] == "window")
    log(f"openings from sections: {n_door} doors, {n_win} windows, "
        f"{len(all_openings)-n_door-n_win} other")

    # ---- floorplan with section-detected openings marked ----
    img = cv2.merge([occ * 60] * 3)

    def to_px(px, py):
        return int((px - xmin) / CELL), int((ymax - py) / CELL)
    for axis, off, u0, u1, z0, z1, typ in all_openings:
        col = {"door": (0, 140, 255), "window": (255, 160, 0)}.get(typ, (160, 130, 110))
        if axis == "y":
            p0 = to_px(off, u0); p1 = to_px(off, u1)
        else:
            p0 = to_px(u0, off); p1 = to_px(u1, off)
        cv2.line(img, p0, p1, col, 3)
    img = cv2.resize(img, (nx * 2, ny * 2), interpolation=cv2.INTER_NEAREST)
    for i, (lab, col) in enumerate([("DOOR (from section)", (0, 140, 255)),
                                    ("WINDOW (from section)", (255, 160, 0)),
                                    ("other opening", (160, 130, 110))]):
        cv2.rectangle(img, (10, 20 + i * 20 - 9), (24, 20 + i * 20 + 1), col, -1)
        cv2.putText(img, lab, (30, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, f"{scan_name} openings from VERTICAL SECTIONS: {n_door} doors + {n_win} windows "
                     f"(windows are what walk-path can't find)",
               (10, ny * 2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "openings_floorplan.png"), img)
    log(f"[{scan_name}] complete -> {out_dir} (floorplan + {len(walls)} wall elevation strips)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
