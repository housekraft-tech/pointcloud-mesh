"""las_information_mining.py
-------------------------
Mine EVERY usable signal in the LAS beyond raw XYZ, and turn each into a
map that improves the reconstruction. The raw analysis showed three strong,
barely-used signals:

  INTENSITY   (both scans) full 0-255, bimodal-ish: ~41% very-high, ~2%
              very-low. LiDAR return strength is material-dependent -- glossy
              glass / polished metal / matte plaster / wood all differ. This
              is the signal that can flag a GLASS balcony door or a window
              (anomalous intensity) that geometry alone misses.
  GPS_TIME    356-406 ticks, points-per-tick 27k..118k. Distinct-tick count
              per cell = how many times an area was revisited = COVERAGE
              CONFIDENCE. Low = glimpsed once ("ran through the balcony") =
              flag as low-confidence, not silently wrong.
  RGB         (mujammel) ~10% of points are colour-saturated -- wood doors,
              accent walls, balcony plants -- segmentable from the ~90% grey
              painted walls.
  DENSITY     points-per-cell ~ 1/range^2 for a fixed-resolution spinning
              LiDAR: a direct measurement-quality / near-vs-far proxy without
              any range field.

Produces top-down maps of each (in output/lasmining_<name>/), plus an
opening-evidence map that fuses intensity-anomaly + low-density to highlight
likely glass/window locations.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\las_information_mining.py <scan.las> <scan_name>
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
CELL = 1.0 / PPM


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las_path, scan_name, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"lasmining_{scan_name}"
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
    inten = np.asarray(scan.intensity, float) if scan.intensity is not None else None
    rgb = scan.rgb
    gt = np.asarray(scan.gps_time, float) if scan.gps_time is not None else None
    log(f"aligned {scan.n:,} pts | intensity={inten is not None} rgb={rgb is not None} gps_time={gt is not None}")

    xy = xyz[:, :2]
    z = xyz[:, 2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    W = int((xmax - xmin) * PPM) + 1
    H = int((ymax - ymin) * PPM) + 1
    # analyze at the wall/opening band (waist->mid) where windows/doors live
    band = (z >= z_floor + 0.4) & (z <= z_floor + 1.8)
    bxy = xy[band]
    cols = np.clip(((bxy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((ymax - bxy[:, 1]) * PPM).astype(int), 0, H - 1)
    lin = rows * W + cols

    def cmap(mean, cm=cv2.COLORMAP_TURBO, lo=1, hi=99, title=""):
        valid = np.isfinite(mean)
        img = np.zeros((H, W, 3), np.uint8)
        if valid.sum():
            a, b = np.nanpercentile(mean, [lo, hi])
            norm = np.clip((mean - a) / max(b - a, 1e-6), 0, 1)
            u8 = np.nan_to_num(norm * 255, nan=0).astype(np.uint8)
            img = cv2.applyColorMap(u8, cm)
            img[~valid] = 0
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def per_cell(values, agg="mean"):
        vsum = np.bincount(lin, weights=values, minlength=W * H)
        cnt = np.bincount(lin, minlength=W * H).astype(float)
        if agg == "mean":
            out = np.divide(vsum, cnt, out=np.full(W * H, np.nan), where=cnt > 0)
        elif agg == "count":
            out = np.where(cnt > 0, cnt, np.nan)
        return out.reshape(H, W)

    # ---- 1. COVERAGE CONFIDENCE: distinct gps_time ticks per cell ----
    if gt is not None:
        bgt = gt[band]
        # map each point's tick to a small int id
        _, tick_id = np.unique(bgt, return_inverse=True)
        # distinct ticks per cell via set-size: use (cell, tick) unique pairs
        pair = lin.astype(np.int64) * (tick_id.max() + 1) + tick_id
        ucell = np.unique(pair) // (tick_id.max() + 1)
        distinct = np.bincount(ucell.astype(np.int64), minlength=W * H).astype(float)
        distinct = np.where(distinct > 0, distinct, np.nan).reshape(H, W)
        cv2.imwrite(str(out_dir / "1_coverage_confidence.png"),
                    cmap(distinct, cv2.COLORMAP_TURBO, title=f"{scan_name} 1 COVERAGE (distinct gps ticks/cell): "
                         f"red=well-scanned, blue=glimpsed-once=LOW CONFIDENCE"))
        log(f"coverage: median {np.nanmedian(distinct):.1f} ticks/cell, "
            f"{np.nanmean(distinct < 3)*100:.0f}% of cells seen <3 times")

    # ---- 2. DENSITY (range/quality proxy) ----
    dens = per_cell(np.ones(band.sum()), agg="count")
    cv2.imwrite(str(out_dir / "2_density_quality.png"),
                cmap(np.log1p(dens), cv2.COLORMAP_VIRIDIS,
                     title=f"{scan_name} 2 DENSITY (pts/cell ~ 1/range^2): bright=near/accurate, dark=far/sparse"))

    # ---- 3. INTENSITY material ----
    if inten is not None:
        mi = per_cell(inten[band], agg="mean")
        cv2.imwrite(str(out_dir / "3_intensity_material.png"),
                    cmap(mi, cv2.COLORMAP_INFERNO,
                         title=f"{scan_name} 3 INTENSITY (material): distinguishes plaster/wood/metal/glass"))
        # ---- 4. INTENSITY ANOMALY -> glass/specular/opening evidence ----
        mstd = per_cell((inten[band] - inten[band].mean()) ** 2, agg="mean")
        mstd = np.sqrt(mstd)
        cv2.imwrite(str(out_dir / "4_intensity_variance_glass.png"),
                    cmap(mstd, cv2.COLORMAP_INFERNO,
                         title=f"{scan_name} 4 INTENSITY VARIANCE: high=glass/specular/edge (windows, glass doors)"))

    # ---- 5. RGB material (mujammel) ----
    if rgb is not None:
        brgb = rgb[band].astype(float)
        img_rgb = np.zeros((H, W, 3), np.uint8)
        for ch in range(3):
            m = per_cell(brgb[:, ch], agg="mean")
            img_rgb[:, :, 2 - ch] = np.nan_to_num(m, nan=0).astype(np.uint8)
        cv2.putText(img_rgb, f"{scan_name} 5 RGB true colour (material: wood doors/plants vs grey walls)",
                   (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "5_rgb_material.png"), img_rgb)
        # saturation = colouredness (doors/plants pop from grey walls)
        mx = brgb.max(axis=1); mn = brgb.min(axis=1)
        sat = np.where(mx > 0, (mx - mn) / mx, 0)
        ms = per_cell(sat, agg="mean")
        cv2.imwrite(str(out_dir / "6_colour_saturation.png"),
                    cmap(ms, cv2.COLORMAP_TURBO,
                         title=f"{scan_name} 6 COLOUR SATURATION: bright=coloured (wood/plants/accent), dark=grey wall"))

    log(f"[{scan_name}] las information mining complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
