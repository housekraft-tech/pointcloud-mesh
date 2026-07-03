"""Measure wall thickness directly from the occupancy IMAGE's pixel width,
not from a point-cloud perpendicular-offset histogram/peak-detection.

Motivation: slice_validate.py's waist/mid-height occupancy renders show
walls as bands with real, visible pixel width -- because they're built
directly from raw point density in a height band with no statistical
fitting at all. A wall's true thickness is directly readable there as "how
wide is the continuous occupied band crossing this wall's line", which is
a much more direct and robust measurement than fitting two density peaks
to noisy/sparse points (which fails outright on thin data, as found on the
balcony/railing wall).

For each registered fused wall: sample several perpendicular profiles
along its length through the mid-wall-height occupancy grid, measure the
width of the continuous occupied run crossing the wall's expected offset
at each sample, and take the median across samples as the thickness.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\wall_thickness_from_occupancy.py
"""
import json
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

FUSED_WALLS_JSON = ROOT / "output/koushik_fusion_rectified/fused_walls.json"
LAS_PATH = ROOT / "koushikexport.las"
OUT_DIR = ROOT / "output/koushik_fusion_occupancy_thickness"

CELL_M = 0.02          # occupancy grid resolution (matches slice_validate.py)
MIN_PTS_OCCUPIED = 2
HEIGHT_BANDS_REL = [(0.45, 0.85), (1.05, 1.35)]  # waist + mid -- most reliable per slice_validate.py
N_SAMPLES_PER_WALL = 9
MAX_SEARCH_M = 0.5     # how far perpendicular to scan for the occupied band's true edges


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def occupancy_grid(xy, xmin, ymin, xmax, ymax, cell_m, min_pts, close_kernel=3):
    """Binary occupancy grid, morphologically CLOSED to bridge the small
    (1-2 cell) gaps that ordinary point sparsity leaves even within a truly
    solid wall's footprint. Without this, a strict cell-by-cell width walk
    hits a false "empty" cell almost immediately and reports near-zero
    thickness for real, solid walls -- confirmed on this data (nearly every
    wall came back 0.000-0.020m before closing, which is not physically
    plausible for an interior partition)."""
    W = int((xmax - xmin) / cell_m) + 1
    H = int((ymax - ymin) / cell_m) + 1
    if xy.shape[0] == 0:
        return np.zeros((H, W), dtype=bool), W, H
    cols = np.clip(((xy[:, 0] - xmin) / cell_m).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) / cell_m).astype(int), 0, H - 1)
    count = np.zeros((H, W), dtype=np.int32)
    np.add.at(count, (rows, cols), 1)
    occ = (count >= min_pts).astype(np.uint8)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
    return occ.astype(bool), W, H


def sample_perpendicular_width(occ, xmin, ymax, cell_m, W, H, point_xy, normal_xy, max_search_m):
    """From point_xy, walk outward along +normal and -normal in cell steps;
    find the continuous run of occupied cells that CONTAINS or is NEAREST to
    point_xy, return its width in meters (None if no occupied cell found
    within max_search_m on either side)."""
    max_steps = int(max_search_m / cell_m)

    def to_cell(p):
        col = int((p[0] - xmin) / cell_m)
        row = int((ymax - p[1]) / cell_m)
        return row, col

    def occupied(row, col):
        return 0 <= row < H and 0 <= col < W and occ[row, col]

    # find nearest occupied cell along the normal line (either side)
    center_row, center_col = to_cell(point_xy)
    nearest_offset = None
    for step in range(max_steps + 1):
        for sign in (1, -1) if step > 0 else (1,):
            p = point_xy + normal_xy * (sign * step * cell_m)
            row, col = to_cell(p)
            if occupied(row, col):
                nearest_offset = sign * step
                break
        if nearest_offset is not None:
            break
    if nearest_offset is None:
        return None

    # walk outward from that first hit to find the full continuous run's extent
    def extent_from(start_step, direction):
        s = start_step
        while True:
            p = point_xy + normal_xy * ((s + direction) * cell_m)
            row, col = to_cell(p)
            if not occupied(row, col):
                return s
            s += direction

    lo = extent_from(nearest_offset, -1)
    hi = extent_from(nearest_offset, 1)
    return (hi - lo) * cell_m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log("loading + isolating...")
    scan = load_scan(str(LAS_PATH), max_points=cfg["max_points"], rng_seed=seed)
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
    log(f"{scan.n:,} points, z_floor={z_floor:.3f}")

    xy_all = xyz[:, :2]
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)

    occ_grids = []
    for lo_rel, hi_rel in HEIGHT_BANDS_REL:
        mask = (xyz[:, 2] >= z_floor + lo_rel) & (xyz[:, 2] < z_floor + hi_rel)
        occ, W, H = occupancy_grid(xy_all[mask], xmin, ymin, xmax, ymax, CELL_M, MIN_PTS_OCCUPIED)
        occ_grids.append(occ)
        log(f"height band [{lo_rel},{hi_rel}]: {int(mask.sum()):,} points, "
            f"{int(occ.sum()):,} occupied cells")

    with open(FUSED_WALLS_JSON) as f:
        fused_walls = json.load(f)

    results = []
    for w in fused_walls:
        m = w["measured"]
        p0 = np.array(m["p0"] if m else w["floorplan_p0"])
        p1 = np.array(m["p1"] if m else w["floorplan_p1"])
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        d = d / L
        n = np.array([-d[1], d[0]])

        widths = []
        for t in np.linspace(0.1, 0.9, N_SAMPLES_PER_WALL):
            pt = p0 + d * (t * L)
            for occ in occ_grids:
                width = sample_perpendicular_width(occ, xmin, ymax, CELL_M, occ.shape[1], occ.shape[0],
                                                   pt, n, MAX_SEARCH_M)
                if width is not None:
                    widths.append(width)

        thickness_occupancy_m = float(np.median(widths)) if len(widths) >= 3 else None
        raw_point_thickness = m["thickness_m"] if m else None
        is_balcony = any("Balcony" in a for a in w["area_names"])
        results.append(dict(area_names=w["area_names"], is_balcony=is_balcony,
                            raw_point_thickness_m=raw_point_thickness,
                            occupancy_thickness_m=thickness_occupancy_m,
                            n_samples_hit=len(widths)))
        tag = " [BALCONY]" if is_balcony else ""
        rp = f"{raw_point_thickness:.3f}m" if raw_point_thickness is not None else "UNMEASURED"
        oc = f"{thickness_occupancy_m:.3f}m" if thickness_occupancy_m is not None else "unmeasured"
        log(f"{tag} raw-point={rp}  occupancy-width={oc} ({len(widths)}/{N_SAMPLES_PER_WALL*len(occ_grids)} "
            f"samples hit)  areas={w['area_names']}")

    with open(OUT_DIR / "occupancy_thickness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT_DIR / 'occupancy_thickness_results.json'}")

    # ---------- render: mid-wall occupancy with wall thickness bands overlaid ----------
    ppm = 1.0 / CELL_M
    img = cv2.merge([occ_grids[1].astype(np.uint8) * 200] * 3)

    def to_px(pt):
        return (int((pt[0] - xmin) * ppm), int((ymax - pt[1]) * ppm))

    for w, r in zip(fused_walls, results):
        m = w["measured"]
        p0 = np.array(m["p0"] if m else w["floorplan_p0"])
        p1 = np.array(m["p1"] if m else w["floorplan_p1"])
        color = (0, 165, 255) if r["is_balcony"] else (0, 255, 0)
        cv2.line(img, to_px(p0), to_px(p1), color, 2)
        mid = (p0 + p1) / 2
        label = f"{r['occupancy_thickness_m']*100:.0f}cm" if r["occupancy_thickness_m"] else "?"
        cv2.putText(img, label, to_px(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT_DIR / "occupancy_thickness_overlay.png"), img)
    log(f"wrote {OUT_DIR / 'occupancy_thickness_overlay.png'}")


if __name__ == "__main__":
    main()
