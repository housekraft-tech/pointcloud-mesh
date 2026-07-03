"""tomographic_floorplan.py
------------------------
CT-scan-style reconstruction: build a 3D occupancy VOLUME from the isolated
point cloud, then classify every vertical (x,y) COLUMN by its height-
occupancy signature -- combining the "plan" (top-down) and "section"
(elevation) information into one structural map instead of looking at them
separately.

Each column's signature (which Z bands are occupied) directly names the
element, which no single top-down slice can do:

  WALL       occupied across most of floor -> ceiling
  BEAM/header occupied only in the upper band, empty below  (soffit over open
             space, or a door/window header)
  RAILING    occupied floor -> waist only, does NOT reach the ceiling
             (the balcony railing -- the exact thing every top-down method
             kept missing)
  LOW CLUTTER a short floor-anchored blob that is neither wall-tall nor a
             clean railing line (furniture)

Output: a reconstructed floorplan colored by element class, plus a per-class
mask stack and a summary JSON.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\tomographic_floorplan.py <scan.las> <scan_name>
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

CELL_XY = 0.04          # voxel XY size (m)
CELL_Z = 0.05           # voxel Z size (m)
MIN_PTS_PER_VOXEL = 2   # a voxel counts as occupied at this many points
MIN_PTS_PER_COLUMN = 12 # a column below this total is noise, ignored
FLOOR_BAND_M = 0.30     # "reaches floor" = occupied within this of z_floor
CEIL_BAND_M = 0.30      # "reaches ceiling" = occupied within this of z_ceiling
WALL_FILL_FRAC = 0.55   # WALL: >= this fraction of the storey's z-bins occupied
RAILING_MAX_TOP_M = 1.40  # RAILING top must stay below this height above floor
RAILING_MIN_TOP_M = 0.50  # ...and above this (else it's just floor clutter)
BEAM_MIN_BOTTOM_M = 1.40   # BEAM bottom must start above this height


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las_path, scan_name, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"tomographic_{scan_name}"
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
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    storey = z_ceiling - z_floor
    log(f"isolated+aligned {scan.n:,} pts | storey z=[{z_floor:.2f},{z_ceiling:.2f}] ({storey:.2f}m)")

    # ---- build the 3D occupancy volume ----
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, ymin = x.min(), y.min()
    z0 = z_floor - 0.15
    z1 = z_ceiling + 0.15
    nx = int((x.max() - xmin) / CELL_XY) + 1
    ny = int((y.max() - ymin) / CELL_XY) + 1
    nz = int((z1 - z0) / CELL_Z) + 1
    ix = np.clip(((x - xmin) / CELL_XY).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / CELL_XY).astype(int), 0, ny - 1)
    iz = np.clip(((z - z0) / CELL_Z).astype(int), 0, nz - 1)

    vcount = np.zeros((ny, nx, nz), dtype=np.int32)
    np.add.at(vcount, (iy, ix, iz), 1)
    occ = vcount >= MIN_PTS_PER_VOXEL
    col_pts = vcount.sum(axis=2)
    log(f"volume {ny}x{nx}x{nz} = {ny*nx*nz/1e6:.1f}M voxels; "
        f"{int(occ.sum()):,} occupied; {int((col_pts>=MIN_PTS_PER_COLUMN).sum()):,} real columns")

    # ---- per-column height-signature classification (vectorized) ----
    z_bins = np.arange(nz)
    floor_hi = int((z_floor + FLOOR_BAND_M - z0) / CELL_Z)
    ceil_lo = int((z_ceiling - CEIL_BAND_M - z0) / CELL_Z)
    railing_top_hi = int((z_floor + RAILING_MAX_TOP_M - z0) / CELL_Z)
    railing_top_lo = int((z_floor + RAILING_MIN_TOP_M - z0) / CELL_Z)
    beam_bottom_lo = int((z_floor + BEAM_MIN_BOTTOM_M - z0) / CELL_Z)

    has_col = col_pts >= MIN_PTS_PER_COLUMN
    n_occ_z = occ.sum(axis=2)
    fill_frac = n_occ_z / max(int((z_ceiling - z_floor) / CELL_Z), 1)
    reaches_floor = occ[:, :, :floor_hi + 1].any(axis=2)
    reaches_ceiling = occ[:, :, ceil_lo:].any(axis=2)
    # top / bottom occupied z-index per column
    topz = np.where(occ, z_bins[None, None, :], -1).max(axis=2)
    botz = np.where(occ, z_bins[None, None, :], nz).min(axis=2)
    top_m = z0 + topz * CELL_Z - z_floor   # top height above floor
    bot_m = z0 + botz * CELL_Z - z_floor   # bottom height above floor

    # class codes: 0 none, 1 wall, 2 beam, 3 railing, 4 furniture, 5 open room
    cls = np.zeros((ny, nx), dtype=np.uint8)
    wall = has_col & reaches_floor & reaches_ceiling & (fill_frac >= WALL_FILL_FRAC)
    # OPEN ROOM: floor AND ceiling present but the middle is empty -- this is
    # negative space (a room interior seen from above), NOT structure or
    # clutter. Separating it is what makes the wall skeleton legible.
    open_room = has_col & reaches_floor & reaches_ceiling & (~wall) & (fill_frac < WALL_FILL_FRAC)
    beam_raw = has_col & (~wall) & (~open_room) & reaches_ceiling & (~reaches_floor) & (bot_m >= BEAM_MIN_BOTTOM_M)
    railing = (has_col & (~wall) & (~open_room) & (~beam_raw) & reaches_floor & (~reaches_ceiling)
               & (top_m <= RAILING_MAX_TOP_M) & (top_m >= RAILING_MIN_TOP_M))

    # BEAM shape filter: a real beam is a THIN LINE spanning space; tall
    # furniture (wardrobe top) is a compact BLOB that also reaches near the
    # ceiling. Keep only elongated connected components as beams; demote
    # blobby ones to furniture. Thinness = min bbox side <= ~0.5 m and
    # aspect ratio >= 3.
    beam = np.zeros_like(beam_raw)
    furniture_from_beam = np.zeros_like(beam_raw)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(beam_raw.astype(np.uint8), connectivity=8)
    thin_cells = int(0.5 / CELL_XY)
    for lab in range(1, nlab):
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]
        comp = labels == lab
        aspect = max(w, h) / max(min(w, h), 1)
        if min(w, h) <= thin_cells and aspect >= 3.0:
            beam |= comp
        else:
            furniture_from_beam |= comp
    furniture = (has_col & (~wall) & (~open_room) & (~beam) & (~railing)) | furniture_from_beam

    cls[open_room] = 5
    cls[furniture] = 4
    cls[railing] = 3
    cls[beam] = 2
    cls[wall] = 1

    counts = {k: int(v) for k, v in zip(
        ["wall", "beam", "railing", "furniture", "open_room"],
        [wall.sum(), beam.sum(), railing.sum(), furniture.sum(), open_room.sum()])}
    log(f"column classes: {counts}")

    # ---- render reconstructed floorplan colored by class ----
    UP = 3  # upscale
    img = np.zeros((ny, nx, 3), np.uint8)
    img[cls == 5] = (28, 28, 28)      # OPEN room -- dark negative space
    img[cls == 4] = (40, 70, 40)      # furniture dim green
    img[cls == 3] = (255, 200, 0)     # RAILING cyan
    img[cls == 2] = (60, 60, 255)     # BEAM red (BGR)
    img[cls == 1] = (235, 235, 235)   # WALL white
    img = cv2.resize(img, (nx * UP, ny * UP), interpolation=cv2.INTER_NEAREST)
    # legend
    y0 = 20
    for i, (lbl, col) in enumerate([("WALL (full height)", (235, 235, 235)),
                                    ("BEAM (thin, top-only)", (60, 60, 255)),
                                    ("RAILING (waist-high)", (255, 200, 0)),
                                    ("furniture", (40, 70, 40)),
                                    ("open room", (28, 28, 28))]):
        cv2.rectangle(img, (10, y0 + i * 22 - 10), (26, y0 + i * 22 + 2), col, -1)
        cv2.putText(img, lbl, (32, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"{scan_name} tomographic reconstruction: walls={counts['wall']} "
                     f"beam={counts['beam']} railing={counts['railing']} cells",
               (10, ny * UP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "tomographic_floorplan.png"), img)
    log(f"wrote {out_dir / 'tomographic_floorplan.png'}")

    # ---- also dump each class as its own mask for downstream use ----
    for name, m in [("walls", wall), ("beams", beam), ("railings", railing)]:
        mask_img = cv2.resize((m.astype(np.uint8) * 255), (nx * UP, ny * UP), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"mask_{name}.png"), mask_img)

    with open(out_dir / "tomographic_summary.json", "w") as f:
        json.dump(dict(scan=scan_name, z_floor=z_floor, z_ceiling=z_ceiling,
                       cell_xy=CELL_XY, cell_z=CELL_Z, counts=counts,
                       volume_dims=[int(ny), int(nx), int(nz)]), f, indent=2)
    log(f"[{scan_name}] tomographic reconstruction complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
