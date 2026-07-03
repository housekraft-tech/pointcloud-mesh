"""poisson_freespace.py
--------------------
Use the Poisson mesh the RIGHT way -- with air/free-space logic, not the old
surface logic that failed.

The earlier mesh attempt (floorplan_fusion_poisson) looked for where mesh
SURFACES are near a wall line -- the same wrong idea as "walls = where points
are dense". It failed on the sparse balcony because there was no clean
surface to find.

The fix: a Poisson mesh is WATERTIGHT, so a horizontal cross-section through
it is a set of CLOSED, gap-filled contours -- and the area INSIDE each
contour is free air (a room). So instead of hunting surfaces, we:
  1. slice the mesh at mid-wall height -> closed room-interior polygons (AIR)
  2. free space = union of those polygon interiors
  3. walls = building footprint MINUS the free air (the solid complement)
This is exactly the free-space carving that works, but the "free air" comes
from Poisson's clean gap-filled contours instead of raw-point occupancy --
Poisson doing what it is actually good at (continuous boundaries), with the
air logic doing what it is good at (turning enclosure into walls).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\poisson_freespace.py <raw_scan.las> <scan_name> <mesh.obj>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import trimesh
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime

CELL = 0.03
UP = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(raw_las, scan_name, mesh_path):
    out_dir = ROOT / "output" / f"poisson_freespace_{scan_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    # derive the same alignment R + z-band the other outputs use (so this
    # overlays with them), from the raw scan.
    log(f"[{scan_name}] deriving alignment from {raw_las} ...")
    scan = load_scan(str(raw_las), max_points=cfg["max_points"], rng_seed=seed)
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
    if traj.shape[0]:
        traj = traj @ np.asarray(R, dtype=float).T
    z_floor = float(np.percentile(scan.xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(scan.xyz[:, 2], cfg["z_ceiling_pct"]))

    log(f"loading mesh {mesh_path} ...")
    mesh = trimesh.load(str(mesh_path), process=False)
    # NOTE: this mesh (mesh_isolated_v*_clean.obj) was already saved in the
    # ALIGNED frame by strip_furniture_v2 (it axis-aligns before meshing), and
    # dominant_axes is deterministic (seed 0) so that frame == the one derived
    # here. Rotating by R again would DOUBLE-rotate it (the bug behind the
    # earlier fusion offsets). Use the mesh vertices as-is.
    log(f"mesh: {len(mesh.vertices):,} verts (already aligned); slicing at mid-wall height")

    # ---- slice the watertight mesh -> wall-surface polylines (world XY) ----
    # Real rooms have doorways, so these contours are OPEN (broken at each
    # door), not closed -- which is exactly why "fill the closed contour"
    # fails. Instead rasterize the surface lines as occupancy (cleaner /
    # gap-filled vs raw points) and carve free space from the trajectory,
    # which flows through the open doorways naturally.
    segs = []
    for zc in (z_floor + 1.0, z_floor + 1.3, z_floor + 1.6):
        section = mesh.section(plane_origin=[0, 0, zc], plane_normal=[0, 0, 1])
        if section is None:
            continue
        for poly in section.discrete:          # 3D world-frame polylines
            xy2 = np.asarray(poly)[:, :2]
            for i in range(len(xy2) - 1):
                segs.append((xy2[i], xy2[i + 1]))
    if not segs:
        log("no mesh cross-section at these heights -- mesh open/too sparse")
        return
    allpts = np.array([p for s in segs for p in s])
    minx, miny = allpts.min(axis=0) - 0.4
    maxx, maxy = allpts.max(axis=0) + 0.4
    nx = int((maxx - minx) / CELL) + 1
    ny = int((maxy - miny) / CELL) + 1

    def to_px(x, y):
        return int((x - minx) / CELL), int((maxy - y) / CELL)

    occ = np.zeros((ny, nx), np.uint8)
    for a, b in segs:
        cv2.line(occ, to_px(a[0], a[1]), to_px(b[0], b[1]), 1, 1)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    log(f"mesh cross-section: {len(segs):,} surface segments -> occupancy")

    # free space = flood-fill from the walk path through non-occupied cells
    free = np.zeros((ny, nx), np.uint8)
    empty = (occ == 0).astype(np.uint8)
    if traj.shape[0]:
        seed_mask = np.zeros((ny, nx), np.uint8)
        for px, py in traj[:, :2]:
            c, r = to_px(px, py)
            if 0 <= r < ny and 0 <= c < nx and empty[r, c]:
                seed_mask[r, c] = 1
        lbl0, _ = ndimage.label(empty, structure=np.ones((3, 3)))
        keep0 = set(lbl0[seed_mask == 1].tolist()) - {0}
        free = np.isin(lbl0, list(keep0)).astype(np.uint8)

    footprint = ndimage.binary_fill_holes((free | occ) > 0)
    walls = (footprint & (free == 0)).astype(np.uint8)
    # tidy: fill wall holes, drop specks
    walls = ndimage.binary_fill_holes(walls).astype(np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    lbl, nlab = ndimage.label(walls, structure=np.ones((3, 3)))
    sz = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, nlab + 1))
    walls[np.isin(lbl, [i for i, s in enumerate(sz, 1) if s < (0.05 / CELL / CELL)])] = 0

    # ---- walk-path entrances (through-crossing), same as freespace_floorplan ----
    entrances = np.zeros((ny, nx), np.uint8)
    if traj.shape[0] >= 2:
        tp = [to_px(px, py) for px, py in traj[:, :2]]

        def is_free(c, r):
            return 0 <= r < ny and 0 <= c < nx and free[r, c] == 1
        seed_e = np.zeros((ny, nx), np.uint8)
        for i in range(len(tp) - 1):
            seg = np.zeros((ny, nx), np.uint8)
            cv2.line(seg, tp[i], tp[i + 1], 1, 1)
            through = seg & walls
            if int(through.sum()) >= 2 and is_free(*tp[i]) and is_free(*tp[i + 1]):
                seed_e |= through
        dh = int(0.45 / CELL)
        seed_e = cv2.dilate(seed_e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dh + 1, 2 * dh + 1)))
        entrances = (seed_e & walls).astype(np.uint8)
        walls[entrances.astype(bool)] = 0

    # ---- render ----
    img = np.zeros((ny, nx, 3), np.uint8)
    img[free == 1] = (40, 25, 15)
    img[walls == 1] = (235, 235, 235)
    img[entrances.astype(bool)] = (0, 140, 255)
    img = cv2.resize(img, (nx * UP, ny * UP), interpolation=cv2.INTER_NEAREST)
    if traj.shape[0]:
        for c, r in tp:
            cv2.circle(img, (int(c * UP), int(r * UP)), 1, (0, 200, 255), -1)
    for i, (lab, col) in enumerate([("SOLID WALL (mesh-air complement)", (235, 235, 235)),
                                    ("ENTRANCE (walked)", (0, 140, 255)),
                                    ("free room air (mesh contour interior)", (40, 25, 15)),
                                    ("walk path", (0, 200, 255))]):
        cv2.rectangle(img, (10, 18 + i * 22 - 10), (26, 18 + i * 22 + 2), col, -1)
        cv2.putText(img, lab, (32, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"{scan_name} POISSON free-space floorplan: rooms=mesh-slice contour interiors (air), "
                     f"walls=their complement", (10, ny * UP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "poisson_freespace_floorplan.png"), img)
    log(f"[{scan_name}] poisson free-space floorplan complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
