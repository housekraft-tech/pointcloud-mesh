"""freespace_floorplan.py
----------------------
Wall reconstruction by FREE-SPACE CARVING, not point density.

Key insight (from inspecting the vertical sections): a solid wall returns NO
points -- the laser cannot penetrate it, so the interior of a wall is BLACK
(empty) in the data, exactly like open air. Point density alone cannot tell
"empty because it's open air" from "empty because it's solid wall".

The disambiguator is REACHABILITY. The scanner physically walked the rooms,
and had line-of-sight across open air to every surface it hit. So:
  FREE space  = cells reachable from the walked trajectory through
                non-occupied cells (definitely air).
  SOLID/WALL  = cells INSIDE the building footprint that are NOT free and
                NOT a furniture island -- i.e. the enclosed "black" volumes
                that the laser could never reach. These are walls, and this
                gives their FULL THICKNESS (the whole masonry volume), not
                just the thin visible face.

At mid-wall height this yields clean, solid, correctly-thick walls as the
complement of the carved-out room free-space.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\freespace_floorplan.py <scan.las> <scan_name>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.isolidarflow import DEFAULT_CONFIG

CELL = 0.03
UP = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sensor_trajectory_from_gpstime(xyz, gps_time, smooth=5):
    """Recover the real walk path (sensor position over time) from gps_time.

    A handheld SLAM scanner spins, sampling a full sphere around itself many
    times per gps_time tick, so all points sharing ONE gps_time value were
    seen from ONE sensor position. Its robust center estimates that position.

    Uses the NATIVE gps_time discretization (not an arbitrary dt bin -- the
    data here has ~361 real timestamps over ~286s, so a finer bin would just
    fragment a single sensor pose into noise) and the per-group MEDIAN XY
    (not mean -- the mean is dragged toward whichever far wall was open,
    biasing the estimate by 1-2 m; the median resists that). Returns an
    (M,3) path ordered by time, lightly moving-average smoothed."""
    gt = np.asarray(gps_time)
    order = np.argsort(gt, kind="stable")
    gt_s = gt[order]
    xyz_s = xyz[order]
    # split at each change of gps_time value
    bounds = np.flatnonzero(np.diff(gt_s) != 0) + 1
    groups = np.split(np.arange(len(gt_s)), bounds)
    path = np.array([np.median(xyz_s[g], axis=0) for g in groups if len(g) >= 20])
    if smooth > 1 and len(path) >= smooth:
        k = np.ones(smooth) / smooth
        path = np.column_stack([np.convolve(path[:, i], k, mode="same") for i in range(3)])
    return path


def main(las_path, scan_name, out_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "output" / f"freespace_{scan_name}"
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
    # real walk path from the NATIVE gps_time grouping (median per timestamp)
    if scan.gps_time is not None:
        traj = sensor_trajectory_from_gpstime(scan.xyz, scan.gps_time)
        log(f"sensor trajectory: {len(traj)} poses from {len(np.unique(scan.gps_time))} gps_time ticks")
    else:
        traj = np.zeros((0, 3))
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
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    nx = int((xmax - xmin) / CELL) + 1
    ny = int((ymax - ymin) / CELL) + 1
    log(f"aligned {scan.n:,} pts | grid {ny}x{nx}")

    def cells(px, py):
        cx = np.clip(((px - xmin) / CELL).astype(int), 0, nx - 1)
        cy = np.clip(((ymax - py) / CELL).astype(int), 0, ny - 1)
        return cy, cx

    # ---- occupancy of SURFACES at a mid-wall height band ----
    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cy, cx = cells(x[band], y[band])
    occ = np.zeros((ny, nx), np.uint8)
    occ[cy, cx] = 1
    # close scan-gap breaks in the wall SURFACE before carving so free space
    # can't leak through a small hole and eat the wall there (5x5 bridges
    # ~15cm gaps; real doorways are ~0.9m so they stay open).
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # ---- FREE space: flood fill from the walked trajectory through empty ----
    free = np.zeros((ny, nx), np.uint8)
    empty = (occ == 0).astype(np.uint8)
    if traj.shape[0]:
        ty, tx = cells(traj[:, 0], traj[:, 1])
        seed_mask = np.zeros((ny, nx), np.uint8)
        for r, c in zip(ty, tx):
            if empty[r, c]:
                seed_mask[r, c] = 1
        # connected-component label of empty; keep components touching a seed
        lbl, nlab = ndimage.label(empty, structure=np.ones((3, 3)))
        seed_labels = set(lbl[seed_mask == 1].tolist()) - {0}
        free = np.isin(lbl, list(seed_labels)).astype(np.uint8)
    log(f"free-space cells: {int(free.sum()):,} of {int(empty.sum()):,} empty")

    # ---- building footprint: free rooms + their bounding walls ----
    # fill holes of (free | occ) so enclosed wall-interiors become part of the
    # footprint, then footprint solid = footprint minus free space.
    reach_or_surface = ((free | occ) > 0)
    footprint = ndimage.binary_fill_holes(reach_or_surface)
    solid = footprint & (free == 0)          # occupied surfaces + enclosed no-return volumes
    # remove small furniture islands sitting inside rooms (fully surrounded by
    # free space, compact): keep only solid connected to the footprint border
    # (real walls form the room boundaries), OR large enough to be structural.
    lbl_s, n_s = ndimage.label(solid, structure=np.ones((3, 3)))
    border = np.zeros_like(solid)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    keep = set(lbl_s[border & (lbl_s > 0)].tolist())
    sizes = ndimage.sum(np.ones_like(lbl_s), lbl_s, index=np.arange(1, n_s + 1))
    for i, sz in enumerate(sizes, start=1):
        if sz >= (0.5 / CELL) ** 2:  # >= ~0.5m^2 -> structural, keep
            keep.add(i)
    walls = np.isin(lbl_s, list(keep - {0})).astype(np.uint8)

    # ---- FILL UP the walls: fill enclosed holes and bridge small breaks so
    # every wall reads as one continuous solid band, not a thin/dotted line.
    walls = ndimage.binary_fill_holes(walls).astype(np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    walls = (walls.astype(bool) & footprint).astype(np.uint8)  # stay inside the footprint
    # drop tiny leftover specks (< ~0.05 m^2) that survived
    lbl_w, n_w = ndimage.label(walls, structure=np.ones((3, 3)))
    wsz = ndimage.sum(np.ones_like(lbl_w), lbl_w, index=np.arange(1, n_w + 1))
    small = {i for i, s in enumerate(wsz, start=1) if s < (0.05 / (CELL * CELL))}
    walls[np.isin(lbl_w, list(small))] = 0

    # ---- ENTRANCES via a THROUGH-CROSSING test on the walk path ----
    # A real doorway is where the operator walked FROM free space in one room,
    # THROUGH a wall, INTO free space in another room. This rejects "grazes"
    # (the path running parallel to a wall, brushing its edge but never
    # passing through) which the earlier "any wall cell the path touched"
    # rule wrongly kept -- while catching every genuine crossing.
    entrance_seed = np.zeros((ny, nx), np.uint8)
    n_cross = 0
    if traj.shape[0] >= 2:
        ty2, tx2 = cells(traj[:, 0], traj[:, 1])

        def is_free(r, c):
            return 0 <= r < ny and 0 <= c < nx and free[r, c] == 1

        for i in range(len(tx2) - 1):
            seg = np.zeros((ny, nx), np.uint8)
            cv2.line(seg, (int(tx2[i]), int(ty2[i])), (int(tx2[i + 1]), int(ty2[i + 1])), 1, 1)
            through = seg & walls
            # through-crossing: segment passes through >=2 wall cells AND both
            # endpoints sit in free space (i.e. it went room -> wall -> room)
            if int(through.sum()) >= 2 and is_free(ty2[i], tx2[i]) and is_free(ty2[i + 1], tx2[i + 1]):
                entrance_seed |= through
                n_cross += 1
    # widen each crossing to a clean doorway rectangle and merge nearby ones
    door_half = int(0.45 / CELL)
    seed_d = cv2.dilate(entrance_seed, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * door_half + 1, 2 * door_half + 1)))
    entrances = np.zeros((ny, nx), np.uint8)
    lbl_e, n_e = ndimage.label((seed_d & walls).astype(np.uint8), structure=np.ones((3, 3)))
    n_kept = 0
    for i in range(1, n_e + 1):
        comp = lbl_e == i
        if comp.sum() < (0.08 / (CELL * CELL)):   # >= ~0.08 m^2 (looser than before)
            continue
        ys, xs = np.where(comp)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        rect = np.zeros_like(entrances)
        rect[r0:r1 + 1, c0:c1 + 1] = 1
        entrances |= (rect & walls).astype(np.uint8)   # carve only where wall exists
        n_kept += 1
    entrances = entrances.astype(bool)
    walls[entrances] = 0
    log(f"walk-path entrances: {n_kept} doorways from {n_cross} through-crossings "
        f"({int(traj.shape[0])} walk poses)")

    # ---- render ----
    img = np.zeros((ny, nx, 3), np.uint8)
    img[free == 1] = (40, 25, 15)        # free room space (dark blue-ish)
    furniture_islands = solid & (walls == 0) & (~entrances)
    img[furniture_islands == 1] = (40, 70, 40)  # removed furniture islands, dim green
    img[walls == 1] = (235, 235, 235)    # solid walls (full thickness) white
    img[entrances] = (0, 140, 255)       # ENTRANCE (walked through) orange
    img = cv2.resize(img, (nx * UP, ny * UP), interpolation=cv2.INTER_NEAREST)
    if traj.shape[0]:
        for r, c in zip(ty, tx):
            cv2.circle(img, (int(c * UP), int(r * UP)), 1, (0, 200, 255), -1)
    for i, (lbl_txt, col) in enumerate([("SOLID WALL (carved, full thickness)", (235, 235, 235)),
                                        ("ENTRANCE (walked through)", (0, 140, 255)),
                                        ("free room space", (40, 25, 15)),
                                        ("furniture island (removed)", (40, 70, 40)),
                                        ("walk path", (0, 200, 255))]):
        cv2.rectangle(img, (10, 18 + i * 22 - 10), (26, 18 + i * 22 + 2), col, -1)
        cv2.putText(img, lbl_txt, (32, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"{scan_name} free-space carved floorplan: walls=complement of reachable air",
               (10, ny * UP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "freespace_floorplan.png"), img)
    cv2.imwrite(str(out_dir / "walls_mask.png"),
                cv2.resize(walls * 255, (nx * UP, ny * UP), interpolation=cv2.INTER_NEAREST))
    log(f"[{scan_name}] free-space floorplan complete -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
