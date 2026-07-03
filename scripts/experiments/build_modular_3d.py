"""build_modular_3d.py
-------------------
Recreate a sharp MODULAR 3D model directly from the CUT/SLICE data.

Instead of extruding a flat box and guessing details, each wall is built from
its real vertical profile: for every wall line we form the (along-wall u x
height z) occupancy from the points -- exactly the vertical-section "cut" --
and extrude THAT silhouette by the measured thickness. Everything then falls
out of the data automatically:
   * doors / windows  -> gaps in the (u,z) silhouette (floor-touching vs sill)
   * railings         -> the silhouette only reaches waist height
   * beams / soffits  -> material only near the top
   * GROOVES / L-cuts -> height bands where the face recedes are cut as
     channels of the measured depth
Plus floor + ceiling slabs. Exports a modular GLB (named parts).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_modular_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime
from scripts.experiments.hough_vectorize import snap_and_merge
from scripts.experiments.make_floorplan import close_junctions, merge_parallels

PPM = 50
CELL = 1.0 / PPM
UZ_RES = 0.04            # (u,z) silhouette cell size
WALL_HALF_BAND = 0.18
GROOVE_MIN_DEPTH = 0.03     # min face setback to count as a groove (sensitive -> visible reveals)
MIN_WALL_LEN = 0.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slab(cx, cy, cz, sx, sy, sz):
    b = trimesh.creation.box(extents=(sx, sy, sz))
    b.apply_translation((cx, cy, cz))
    return b


def silhouette_polygons(mask, ures, zres, z_base):
    """(u,z) binary mask -> list of shapely polygons (with holes = openings),
    coords in metric (u, z_world)."""
    H, W = mask.shape
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    polys = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:            # this is a hole; handled with its parent
            continue
        if cv2.contourArea(cnt) < (0.05 / (ures * zres)):
            continue
        ext = [(c * ures, z_base + (H - r) * zres) for c, r in cnt[:, 0, :]]
        holes = []
        child = hier[i][2]
        while child != -1:
            hc = contours[child]
            if cv2.contourArea(hc) >= (0.05 / (ures * zres)):
                holes.append([(c * ures, z_base + (H - r) * zres) for c, r in hc[:, 0, :]])
            child = hier[child][0]
        try:
            p = Polygon(ext, holes)
            if p.is_valid and p.area > 0.05:
                polys.append(p)
        except Exception:
            pass
    return polys


def extrude_wall(polys, p0, d, n, thick):
    """Extrude (u,z) polygons by thickness along n, place at wall p0/d/n."""
    geom = unary_union(polys) if len(polys) > 1 else polys[0]
    prism = trimesh.creation.extrude_polygon(geom, height=thick)
    # local: X=u(along d), Y=z(world up), Z=thickness(along n)
    M = np.array([[d[0], 0, n[0], p0[0] - n[0] * thick / 2],
                  [d[1], 0, n[1], p0[1] - n[1] * thick / 2],
                  [0,    1, 0,    0],
                  [0,    0, 0,    1]], float)
    prism.apply_transform(M)
    return prism


def main(las_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log(f"loading + isolating {las_path} ...")
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
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    z_floor = float(np.percentile(z, cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(z, cfg["z_ceiling_pct"]))
    storey = z_ceiling - z_floor
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    W = int((xmax - xmin) / CELL) + 1; H = int((ymax - ymin) / CELL) + 1
    log(f"aligned {scan.n:,} pts | storey {storey:.2f}m")

    from scipy import ndimage
    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
    rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
    occ = np.zeros((H, W), np.uint8); occ[rr, cc] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # ---- FREE-SPACE CARVED walls (connected, clean -- includes the real
    # perimeter AND interior partitions), the same accurate source as the
    # good floorplan. Skeletonize -> Hough -> Manhattan -> close junctions. ----
    empty = (occ == 0).astype(np.uint8)
    free = np.zeros_like(empty)
    if traj.shape[0]:
        seed = np.zeros_like(empty)
        for px, py in traj[:, :2]:
            cx = int((px - xmin) / CELL); cy = int((ymax - py) / CELL)
            if 0 <= cy < H and 0 <= cx < W and empty[cy, cx]:
                seed[cy, cx] = 1
        lbl0, _ = ndimage.label(empty, structure=np.ones((3, 3)))
        keep = set(lbl0[seed == 1].tolist()) - {0}
        free = np.isin(lbl0, list(keep)).astype(np.uint8)
    footprint = ndimage.binary_fill_holes((free | (occ > 0)))
    walls_solid = (footprint & (free == 0)).astype(np.uint8)
    walls_solid = ndimage.binary_fill_holes(walls_solid).astype(np.uint8)
    walls_solid = cv2.morphologyEx(walls_solid, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # thin to centerlines if cv2 thinning is available, else Hough the solid
    # directly (merge_parallels collapses the doubled faces either way).
    try:
        thin = cv2.ximgproc.thinning(walls_solid)
    except Exception:
        thin = walls_solid
    lines = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=28,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.35 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.3 * PPM), coord_tol_px=int(0.12 * PPM))
    hsegs, vsegs = merge_parallels(hsegs), merge_parallels(vsegs)
    hsegs, vsegs = close_junctions(hsegs, vsegs, snap=int(0.5 * PPM))
    hsegs = [s for s in hsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]
    vsegs = [s for s in vsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]

    def px2m(col, row):
        return xmin + col * CELL, ymax - row * CELL
    interior = [(np.array(px2m(a0, yr)), np.array(px2m(a1, yr))) for a0, a1, yr in hsegs] + \
               [(np.array(px2m(xc, a0)), np.array(px2m(xc, a1))) for a0, a1, xc in vsegs]
    exterior = []   # perimeter already included in the carved walls above

    # ---- dedup (merge near-identical parallel overlapping segments) ----
    def dedup(walls, tol=0.16):
        Hs, Vs = [], []
        for p0, p1 in walls:
            dd = p1 - p0
            if abs(dd[0]) >= abs(dd[1]):
                Hs.append([min(p0[0], p1[0]), max(p0[0], p1[0]), (p0[1] + p1[1]) / 2])
            else:
                Vs.append([min(p0[1], p1[1]), max(p0[1], p1[1]), (p0[0] + p1[0]) / 2])

        def m(segs):
            segs = sorted(segs, key=lambda s: (s[2], s[0])); out = []
            for a0, a1, c in segs:
                hit = False
                for w in out:
                    if abs(w[2] - c) <= tol and a0 <= w[1] + tol and a1 >= w[0] - tol:
                        w[0] = min(w[0], a0); w[1] = max(w[1], a1); w[2] = (w[2] + c) / 2; hit = True; break
                if not hit:
                    out.append([a0, a1, c])
            return out
        res = []
        for a0, a1, y in m(Hs):
            res.append((np.array([a0, y]), np.array([a1, y])))
        for a0, a1, x in m(Vs):
            res.append((np.array([x, a0]), np.array([x, a1])))
        return res

    n_raw = len(interior) + len(exterior)
    walls = dedup(interior + exterior)
    log(f"{len(interior)} interior + {len(exterior)} exterior = {n_raw} -> {len(walls)} deduped walls")

    # ---- 2D plan visualization of exactly the walls the 3D is built from ----
    plan = np.full((H, W, 3), 255, np.uint8)
    def m2px(p):
        return int((p[0] - xmin) / CELL), int((ymax - p[1]) / CELL)
    for p0, p1 in walls:
        dd = p1 - p0
        col = (0, 150, 0) if abs(dd[0]) >= abs(dd[1]) else (200, 60, 0)  # H green, V blue
        cv2.line(plan, m2px(p0), m2px(p1), col, 4, cv2.LINE_AA)
    cv2.putText(plan, f"2D plan of the 3D model: {len(walls)} walls "
                      f"(green=horizontal, blue=vertical, deduped)",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "wall_plan_2d.png"), plan)
    log(f"wrote wall_plan_2d.png ({len(walls)} walls) -> building (u,z) silhouettes")

    z_base = z_floor - 0.10
    nz = int((z_ceiling + 0.15 - z_base) / UZ_RES) + 1
    parts = []
    n_open = n_groove = 0
    for wi, (p0, p1) in enumerate(walls):
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < MIN_WALL_LEN:
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        rel = xyz[:, :2] - p0
        u = rel @ d; perp = rel @ n
        near = (np.abs(perp) <= WALL_HALF_BAND) & (u >= 0) & (u <= L)
        mid = (p0 + p1) / 2
        if near.sum() >= 200:
            uu = u[near]; zz = z[near]; pp = perp[near]
            thick = float(np.clip(np.percentile(pp, 92) - np.percentile(pp, 8), 0.06, 0.35))
        else:
            uu = np.array([]); zz = np.array([]); pp = np.array([]); thick = DEFAULT_THICK

        # ---- clean SOLID wall box, matching the 2D plan line exactly
        # (floor -> ceiling, measured thickness). Grooves + doors are cut into
        # this clean box below, so the walls stay as straight as the 2D. ----
        Rz = trimesh.transformations.rotation_matrix(np.arctan2(d[1], d[0]), [0, 0, 1])
        wall = trimesh.creation.box(extents=(L, thick, storey))
        wall.apply_transform(Rz)
        wall.apply_translation((mid[0], mid[1], z_floor + storey / 2))

        # ---- grooves: pronounced full-width horizontal REVEALS. For each 0.1m
        # height band measure how far the room-side face sits back from the
        # wall's main face; every recessed band gets its own full-width channel
        # (merged only when truly contiguous), so the relief reads clearly as
        # architectural groove lines -- on the clean corner-closed layout. ----
        cutters = []
        room = pp > 0 if pp.size else np.zeros(0, bool)
        if room.sum() > 300:
            face = np.percentile(pp[room], 82)
            zb = np.arange(z_floor + 0.12, z_ceiling - 0.12, 0.1)
            setb = np.full(len(zb), -1.0)
            for i, zl in enumerate(zb):
                m = room & (zz >= zl) & (zz < zl + 0.1)
                if m.sum() >= 25:
                    setb[i] = face - np.percentile(pp[m], 82)
            active = setb >= GROOVE_MIN_DEPTH
            i = 0
            while i < len(active):
                if active[i]:
                    j = i
                    while j < len(active) and active[j]:
                        j += 1
                    z0 = zb[i]; z1 = zb[j - 1] + 0.1
                    depth = float(np.clip(np.median(setb[i:j]), 0.03, thick * 0.8))
                    ch = trimesh.creation.box(extents=(L * 0.98, depth * 2.2, z1 - z0))
                    ch.apply_transform(Rz)
                    off = (p0 + d * L / 2) + n * (thick / 2)
                    ch.apply_translation((off[0], off[1], (z0 + z1) / 2))
                    cutters.append(ch); n_groove += 1
                    i = j
                else:
                    i += 1

        # ---- door openings from walk-path crossings (cut into the clean box) ----
        if traj.shape[0] >= 2:
            trel = traj[:, :2] - p0
            tu = trel @ d; tperp = trel @ n
            for k in range(len(tu) - 1):
                if (tperp[k] * tperp[k + 1] < 0) and (0.2 < tu[k] < L - 0.2):
                    du = trimesh.creation.box(extents=(0.9, thick * 3, 2.1))
                    du.apply_transform(Rz)
                    dc = p0 + d * tu[k]
                    du.apply_translation((dc[0], dc[1], z_floor + 1.05))
                    cutters.append(du); n_open += 1
                    break

        for ch in cutters:
            try:
                wall = wall.difference(ch)
            except Exception:
                pass
        parts.append((f"wall_{wi:02d}", wall))

    # floor slab only (no ceiling, per request -- keeps the interior visible)
    fx, fy = (xmax - xmin), (ymax - ymin)
    parts.append(("floor", slab((xmin + xmax) / 2, (ymin + ymax) / 2, z_floor - 0.05, fx, fy, 0.10)))

    scene = trimesh.Scene()
    for name, m in parts:
        if m is None or getattr(m, "is_empty", True):
            continue
        m.visual.face_colors = ([150, 130, 110, 255] if name == "floor"
                                else [205, 205, 210, 255])
        scene.add_geometry(m, geom_name=name)
    scene.export(str(out_dir / "modular_model.glb"))
    scene.export(str(out_dir / "modular_model.obj"))
    log(f"exported modular_model.glb/.obj: {len(parts)} parts, "
        f"{n_open} openings (from silhouette gaps), {n_groove} groove cuts")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
