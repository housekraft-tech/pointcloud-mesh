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
GROOVE_MIN_DEPTH = 0.05     # min face setback to count as a groove
GROOVE_MIN_BANDS = 2        # groove must span >= this many 0.1m bands (>=0.2m)
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

    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
    rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
    occ = np.zeros((H, W), np.uint8); occ[rr, cc] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    lines = cv2.HoughLinesP(occ, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.6 * PPM), maxLineGap=int(0.3 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * PPM), coord_tol_px=int(0.12 * PPM))
    # perfect the wall lines: collapse doubled faces + close corners/T-junctions
    # (extend endpoints to meet perpendicular walls) so panels meet cleanly.
    hsegs, vsegs = merge_parallels(hsegs), merge_parallels(vsegs)
    hsegs, vsegs = close_junctions(hsegs, vsegs, snap=int(0.4 * PPM))
    hsegs = [s for s in hsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]
    vsegs = [s for s in vsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]

    def px2m(col, row):
        return xmin + col * CELL, ymax - row * CELL
    walls = [(np.array(px2m(a0, yr)), np.array(px2m(a1, yr))) for a0, a1, yr in hsegs] + \
            [(np.array(px2m(xc, a0)), np.array(px2m(xc, a1))) for a0, a1, xc in vsegs]
    log(f"{len(walls)} wall segments -> building (u,z) silhouettes")

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
        if near.sum() < 300:
            continue
        uu = u[near]; zz = z[near]; pp = perp[near]
        thick = float(np.clip(np.percentile(pp, 92) - np.percentile(pp, 8), 0.06, 0.35))

        # (u,z) occupancy silhouette = the vertical-section cut of this wall
        nu = int(L / UZ_RES) + 1
        sil = np.zeros((nz, nu), np.uint8)
        ui = np.clip((uu / UZ_RES).astype(int), 0, nu - 1)
        zi = np.clip(((z_ceiling + 0.15 - zz) / UZ_RES).astype(int), 0, nz - 1)
        sil[zi, ui] = 255
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        sil = cv2.morphologyEx(sil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        polys = silhouette_polygons(sil, UZ_RES, UZ_RES, z_base)
        if not polys:
            continue
        try:
            wall = extrude_wall(polys, p0, d, n, thick)
        except Exception as exc:
            continue
        n_open += sum(len(list(p.interiors)) for p in polys)

        # ---- grooves: 2D (along-wall u  x  height z) face-recess map. Build
        # the room-side face position per (u,z) cell; where the surface is set
        # BACK from the wall's main face, that (u,z) region is a groove/L-cut.
        # Each connected recess region -> one channel at its real u/z extent and
        # measured depth (so grooves match the actual relief shape, not a full
        # horizontal band). ----
        cutters = []
        Rz = trimesh.transformations.rotation_matrix(np.arctan2(d[1], d[0]), [0, 0, 1])
        room = pp > 0
        if room.sum() > 300:
            ur, zr, pr = uu[room], zz[room], pp[room]
            gu = 0.07
            nu_g = max(int(L / gu) + 1, 2)
            nz_g = max(int(storey / gu) + 1, 2)
            ui2 = np.clip((ur / gu).astype(int), 0, nu_g - 1)
            zi2 = np.clip(((zr - z_floor) / gu).astype(int), 0, nz_g - 1)
            lin = zi2 * nu_g + ui2
            face2 = np.full(nz_g * nu_g, -1e9)
            np.maximum.at(face2, lin, pr)                     # outer room-side surface per cell
            cnt = np.bincount(lin, minlength=nz_g * nu_g)
            face2 = face2.reshape(nz_g, nu_g)
            occ2 = (cnt.reshape(nz_g, nu_g) >= 4)
            main_face = float(np.median(face2[occ2])) if occ2.any() else thick / 2
            setback = np.where(occ2, main_face - face2, 0.0)   # +ve = recessed
            groove = ((setback >= GROOVE_MIN_DEPTH) & occ2).astype(np.uint8)
            groove = cv2.morphologyEx(groove, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            groove = cv2.morphologyEx(groove, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            ncomp, lbl = cv2.connectedComponents(groove)
            for ci in range(1, ncomp):
                ys, xs = np.where(lbl == ci)
                uext = (xs.max() - xs.min() + 1) * gu
                zext = (ys.max() - ys.min() + 1) * gu
                if uext < 0.15 or zext < 0.15 or xs.size < 6:
                    continue
                depth = float(np.clip(np.median(setback[ys, xs]), GROOVE_MIN_DEPTH, thick * 0.8))
                u_mid = (xs.min() + xs.max() + 1) / 2 * gu
                z_mid = z_floor + (ys.min() + ys.max() + 1) / 2 * gu
                ch = trimesh.creation.box(extents=(uext, depth * 2.2, zext))
                ch.apply_transform(Rz)
                c = p0 + d * u_mid + n * (thick / 2)
                ch.apply_translation((c[0], c[1], z_mid))
                cutters.append(ch); n_groove += 1
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
