"""diagnostic_atlas.py
--------------------
One comprehensive diagnostic-visualization generator, run per scan, that
mines EVERY field available in the LAS (XYZ + gps_time + intensity + RGB
when present) and produces a full atlas of top-down views so nothing in the
raw data is left unlooked-at between runs. Consolidates the one-off views
that were scattered across earlier experiments (slice_validate,
inspect_gap_region, rescue's hough binary) into a single reproducible pass.

Views produced (each a PNG in <out_dir>/atlas_<scan>/):
  01_height_slices          floor / waist / mid / near-ceiling occupancy
                            (gap-closed) -- walls as clean bands, height-resolved
  02_hough_walls            waist+mid occupancy -> morphological-closed binary
                            (walls as continuous lines, the "rescue2" view)
  03_height_colored         top-down density colored by mean Z (turbo)
  04_trajectory_overlay     height-colored + the gps_time-derived walk path
  05_gpstime_heatmap        top-down colored by mean gps_time (walk order in time)
  06_intensity              top-down colored by mean LiDAR return intensity
  07_rgb                    top-down mean true color (only if the scan has RGB)
  08_elevation_profile      Z-histogram (floor/ceiling spikes, storey band)

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\diagnostic_atlas.py <scan.las> <scan_name>
e.g.
  venv311\\Scripts\\python.exe scripts\\experiments\\diagnostic_atlas.py koushikexport.las koushik
  venv311\\Scripts\\python.exe scripts\\experiments\\diagnostic_atlas.py mujammelexport.las mujammel
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _grid_dims(xy):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    W = int((xmax - xmin) * PPM) + 1
    H = int((ymax - ymin) * PPM) + 1
    return xmin, ymin, xmax, ymax, W, H


def _cell_index(xy, xmin, ymax, W, H):
    cols = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)
    return rows, cols


def _mean_per_cell(xy, values, xmin, ymax, W, H):
    rows, cols = _cell_index(xy, xmin, ymax, W, H)
    vsum = np.zeros((H, W), dtype=np.float64)
    vcnt = np.zeros((H, W), dtype=np.float64)
    np.add.at(vsum, (rows, cols), values)
    np.add.at(vcnt, (rows, cols), 1.0)
    mean = np.divide(vsum, vcnt, out=np.full_like(vsum, np.nan), where=vcnt > 0)
    return mean, vcnt


def _colormap_with_mask(mean, cmap=cv2.COLORMAP_TURBO, lo_pct=1, hi_pct=99):
    valid = np.isfinite(mean)
    if valid.sum() == 0:
        return np.zeros((*mean.shape, 3), np.uint8)
    lo, hi = np.nanpercentile(mean, [lo_pct, hi_pct])
    norm = np.clip((mean - lo) / max(hi - lo, 1e-6), 0, 1)
    u8 = np.nan_to_num(norm * 255, nan=0).astype(np.uint8)
    img = cv2.applyColorMap(u8, cmap)
    img[~valid] = (0, 0, 0)
    return img, lo, hi


def _banner(img, text):
    cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_atlas(scan, traj, z_floor, z_ceiling, out_dir, scan_name,
                 walls=None, openings_by_wall=None):
    """Render the full diagnostic atlas from an ALREADY isolated+axis-aligned
    ScanData (carrying whatever of gps_time/intensity/rgb the source LAS had),
    its aligned trajectory, and the storey z-band. Reusable so the main
    pipeline can drop the whole atlas into every run's output folder without
    re-doing the expensive load/isolate. `walls`/`openings_by_wall` are
    optional -- when the caller already has reconstructed geometry, it is
    overlaid on the height-colored and hough views for direct comparison.

    Produces PNGs 01-08 (see module docstring) in out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = scan.xyz
    z = xyz[:, 2]
    xy = xyz[:, :2]
    xmin, ymin, xmax, ymax, W, H = _grid_dims(xy)

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    def draw_traj(img):
        if traj.shape[0]:
            for c, r in zip(*_cell_index(traj[:, :2], xmin, ymax, W, H)):
                cv2.circle(img, (int(c), int(r)), 2, (255, 255, 255), -1)
        return img

    # ---- 01: height slices (gap-closed occupancy) ----
    storey = z_ceiling - z_floor
    bands = [("floor", 0.02, 0.17), ("waist", 0.45, 0.85),
             ("mid", 1.05, 1.35), ("near_ceiling", storey - 0.30, storey - 0.05)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    slice_occ = {}
    for name, lo, hi in bands:
        mask = (z >= z_floor + lo) & (z < z_floor + hi)
        rows, cols = _cell_index(xy[mask], xmin, ymax, W, H)
        occ = np.zeros((H, W), np.uint8)
        occ[rows, cols] = 255
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
        slice_occ[name] = occ
        img = cv2.merge([occ, occ, occ])
        _banner(img, f"{scan_name} 01 slice [{name}] z=floor+[{lo:.2f},{hi:.2f}]m  n={int(mask.sum()):,}")
        cv2.imwrite(str(out_dir / f"01_slice_{name}.png"), img)
    log("wrote 01 height slices")

    # ---- 02: hough-clean wall lines (waist+mid combined, closed) ----
    combined = cv2.bitwise_or(slice_occ["waist"], slice_occ["mid"])
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    lines = cv2.HoughLinesP(combined, 1, np.pi / 180, threshold=50,
                            minLineLength=int(0.6 * PPM), maxLineGap=int(0.2 * PPM))
    hough_img = cv2.merge([combined // 2] * 3)
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    _banner(hough_img, f"{scan_name} 02 hough walls: gray=waist+mid occupancy  green={n_lines} Hough lines")
    cv2.imwrite(str(out_dir / "02_hough_walls.png"), hough_img)
    cv2.imwrite(str(out_dir / "02_hough_binary.png"), combined)
    log(f"wrote 02 hough walls ({n_lines} lines)")

    # ---- 03: height-colored density ----
    mean_z, cnt = _mean_per_cell(xy, z, xmin, ymax, W, H)
    hc, zlo, zhi = _colormap_with_mask(mean_z)
    _banner(hc, f"{scan_name} 03 height-colored: blue={zlo:.2f}m -> red={zhi:.2f}m")
    cv2.imwrite(str(out_dir / "03_height_colored.png"), hc)

    # ---- 04: height-colored + trajectory ----
    cv2.imwrite(str(out_dir / "04_trajectory_overlay.png"),
                _banner(draw_traj(hc.copy()), f"{scan_name} 04 height + walk path (white)  "
                                             f"{traj.shape[0]} traj vertices"))
    log("wrote 03-04 height-colored + trajectory")

    # ---- 05: gps_time heatmap (walk order in time) ----
    if scan.gps_time is not None:
        t = np.asarray(scan.gps_time, float)
        mean_t, _ = _mean_per_cell(xy, t - t.min(), xmin, ymax, W, H)
        gt, tlo, thi = _colormap_with_mask(mean_t, cmap=cv2.COLORMAP_JET)
        _banner(gt, f"{scan_name} 05 gps_time heatmap: blue=early -> red=late  span={t.max()-t.min():.0f}s")
        cv2.imwrite(str(out_dir / "05_gpstime_heatmap.png"), gt)
        log("wrote 05 gps_time heatmap")

    # ---- 06: intensity ----
    if scan.intensity is not None:
        inten = np.asarray(scan.intensity, float)
        mean_i, _ = _mean_per_cell(xy, inten, xmin, ymax, W, H)
        it, ilo, ihi = _colormap_with_mask(mean_i, cmap=cv2.COLORMAP_VIRIDIS)
        _banner(it, f"{scan_name} 06 intensity: low={ilo:.0f} -> high={ihi:.0f} (glass/metal/matte differ)")
        cv2.imwrite(str(out_dir / "06_intensity.png"), it)
        log("wrote 06 intensity")

    # ---- 07: RGB true color (mujammel only) ----
    if scan.rgb is not None:
        rgb = np.asarray(scan.rgb, float)
        img_rgb = np.zeros((H, W, 3), np.uint8)
        for ch in range(3):
            m, _ = _mean_per_cell(xy, rgb[:, ch], xmin, ymax, W, H)
            img_rgb[:, :, 2 - ch] = np.nan_to_num(m, nan=0).astype(np.uint8)  # RGB->BGR
        _banner(img_rgb, f"{scan_name} 07 true color (mean RGB per cell)")
        cv2.imwrite(str(out_dir / "07_rgb.png"), img_rgb)
        log("wrote 07 rgb")
    else:
        log("07 rgb: skipped (no RGB in this scan)")

    # ---- 08: elevation profile (Z histogram) ----
    hist_h, hist_w = 400, 700
    prof = np.full((hist_h, hist_w, 3), 20, np.uint8)
    zh, edges = np.histogram(z, bins=120)
    zh = (zh / zh.max() * (hist_h - 40)).astype(int)
    for i, v in enumerate(zh):
        x = int(i / len(zh) * hist_w)
        cv2.line(prof, (x, hist_h - 20), (x, hist_h - 20 - v), (0, 200, 255), max(1, hist_w // len(zh)))
    for zc, lbl in [(z_floor, "floor"), (z_ceiling, "ceiling")]:
        xx = int((zc - edges[0]) / (edges[-1] - edges[0]) * hist_w)
        cv2.line(prof, (xx, 0), (xx, hist_h), (0, 255, 0), 1)
        cv2.putText(prof, f"{lbl} {zc:.2f}m", (xx + 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    _banner(prof, f"{scan_name} 08 elevation profile (Z histogram, floor->ceiling)")
    cv2.imwrite(str(out_dir / "08_elevation_profile.png"), prof)
    log("wrote 08 elevation profile")

    # ---- 09/10: VERTICAL SECTIONS (side-angle cuts -- architectural
    # "sections", not top-down "plans"). A vertical plane sweeps through the
    # building; each cut is rendered as an elevation (horizontal = position
    # along the cut, vertical = Z height, ceiling at top). Openings show as
    # gaps at their real height, sills/headers as band edges, beams as solid
    # bands at the top, and a waist-high railing as a band that STOPS partway
    # up instead of reaching the ceiling -- directly distinguishing a real
    # wall from a railing/half-wall, which no top-down plan slice can show.
    render_sections(scan, z_floor, z_ceiling, out_dir, scan_name)

    log(f"[{scan_name}] atlas complete -> {out_dir}")


def render_sections(scan, z_floor, z_ceiling, out_dir, scan_name,
                    n_sections=8, band_m=0.08):
    """Vertical section montages along X and along Y. Each montage stacks
    n_sections parallel cuts as elevation panels (position x height)."""
    out_dir = Path(out_dir)
    xyz = scan.xyz
    z = xyz[:, 2]
    x, y = xyz[:, 0], xyz[:, 1]
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    z_lo_view, z_hi_view = z_floor - 0.25, z_ceiling + 0.25
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pad = 8
    label_h = 18

    def elevation_panel(pos_vals, z_vals, pos_lo, pos_hi):
        """One elevation image: horizontal=pos, vertical=z (ceiling at top)."""
        Wp = int((pos_hi - pos_lo) * PPM) + 1
        Hp = int((z_hi_view - z_lo_view) * PPM) + 1
        occ = np.zeros((Hp, Wp), np.uint8)
        if pos_vals.size:
            cols = np.clip(((pos_vals - pos_lo) * PPM).astype(int), 0, Wp - 1)
            rows = np.clip(((z_hi_view - z_vals) * PPM).astype(int), 0, Hp - 1)
            occ[rows, cols] = 255
            occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
        img = cv2.merge([occ, occ, occ])
        # floor + ceiling reference lines (green), 1m grid (dim)
        for zc, col in [(z_floor, (0, 180, 0)), (z_ceiling, (0, 180, 0))]:
            r = int((z_hi_view - zc) * PPM)
            cv2.line(img, (0, r), (Wp, r), col, 1)
        for zg in np.arange(np.ceil(z_lo_view), z_hi_view, 1.0):
            r = int((z_hi_view - zg) * PPM)
            cv2.line(img, (0, r), (Wp, r), (60, 60, 60), 1)
        return img

    def montage(cut_axis):
        """cut_axis 'x': vertical planes at increasing X, project onto (Y,Z).
        cut_axis 'y': vertical planes at increasing Y, project onto (X,Z)."""
        if cut_axis == "x":
            cut_lo, cut_hi = xlo, xhi
            pos_lo, pos_hi = ylo, yhi
        else:
            cut_lo, cut_hi = ylo, yhi
            pos_lo, pos_hi = xlo, xhi
        cut_positions = np.linspace(cut_lo, cut_hi, n_sections + 2)[1:-1]
        panels = []
        for cp in cut_positions:
            if cut_axis == "x":
                m = np.abs(x - cp) <= band_m
                panel = elevation_panel(y[m], z[m], pos_lo, pos_hi)
                lbl = f"X={cp:.2f}m  (view along X, horiz=Y)"
            else:
                m = np.abs(y - cp) <= band_m
                panel = elevation_panel(x[m], z[m], pos_lo, pos_hi)
                lbl = f"Y={cp:.2f}m  (view along Y, horiz=X)"
            Hp, Wp = panel.shape[:2]
            framed = np.full((Hp + label_h, Wp, 3), 25, np.uint8)
            framed[label_h:, :] = panel
            cv2.putText(framed, lbl, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
            panels.append(framed)
        Wmax = max(p.shape[1] for p in panels)
        rows_img = []
        for p in panels:
            if p.shape[1] < Wmax:
                p = cv2.copyMakeBorder(p, 0, 0, 0, Wmax - p.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25))
            rows_img.append(p)
            rows_img.append(np.full((pad, Wmax, 3), 40, np.uint8))
        full = np.vstack(rows_img)
        header = np.full((26, Wmax, 3), 15, np.uint8)
        cv2.putText(header, f"{scan_name}  VERTICAL SECTIONS sweeping {cut_axis.upper()} "
                            f"(elevation: ceiling top, floor bottom; gaps=openings, "
                            f"short bands=railing/half-wall)",
                   (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, full])

    cv2.imwrite(str(out_dir / "09_sections_sweepX.png"), montage("x"))
    cv2.imwrite(str(out_dir / "10_sections_sweepY.png"), montage("y"))
    log("wrote 09/10 vertical sections (side-angle elevation cuts)")


def main(las_path, scan_name, out_dir=None):
    """Standalone entry: load + isolate + align a raw LAS, then render_atlas."""
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"atlas_{scan_name}"
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log(f"[{scan_name}] loading {las_path} ...")
    scan = load_scan(str(las_path), max_points=cfg["max_points"], rng_seed=seed)
    log(f"loaded {scan.n:,} pts | gps_time={scan.gps_time is not None} "
        f"intensity={scan.intensity is not None} rgb={scan.rgb is not None}")
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
    if traj.shape[0]:
        traj = traj @ np.asarray(R, dtype=float).T
    z_floor = float(np.percentile(scan.xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(scan.xyz[:, 2], cfg["z_ceiling_pct"]))
    log(f"isolated+aligned: {scan.n:,} pts | storey z=[{z_floor:.2f},{z_ceiling:.2f}]")
    render_atlas(scan, traj, z_floor, z_ceiling, out_dir, scan_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
