"""skeleton_3d.py
--------------
Build a clean 3D straight from the wall SKELETON (centerline network). The
skeleton is the neatest wall representation -- one 1px line per wall, no doubled
faces -- so extruding it gives clean single-thickness walls following the exact
network. Optionally cut real openings from the (u,z) silhouette so it isn't a
solid slab.

  wall footprint -> skeletonize -> thicken to wall thickness -> extrude to ceiling

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\skeleton_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint, rectilinear
from scripts.experiments.build_modular_3d import footprint_polygons
from scipy import ndimage as _ndi

WALL_T = 0.11          # uniform wall thickness (m)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cluster(vals, tol):
    s = sorted(vals); g = [[s[0]]]
    for v in s[1:]:
        (g[-1].append(v) if v - g[-1][-1] <= tol else g.append([v]))
    return [float(np.mean(c)) for c in g]


def rectify_ring(coords, thr=0.025):
    """Square off 45-degree chamfers into 90-degree L corners WITHOUT snapping
    coordinates (snapping would collapse the ~11cm-thin walls). Each clearly
    diagonal edge is replaced by two axis-aligned edges via an inserted corner."""
    pts = [(float(x), float(y)) for x, y in coords]
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    out = []
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        out.append((x0, y0))
        if abs(x1 - x0) > thr and abs(y1 - y0) > thr:         # a real chamfer -> square L
            out.append((x1, y0) if abs(x1 - x0) >= abs(y1 - y0) else (x0, y1))
    clean = []
    for p in out:
        if not clean or abs(clean[-1][0] - p[0]) > 1e-6 or abs(clean[-1][1] - p[1]) > 1e-6:
            clean.append(p)
    return clean


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    xmin, ymax = R["xmin"], R["ymax"]
    H, W = R["H"], R["W"]
    x, y, z = R["x"], R["y"], R["z"]; zf = R["z_floor"]; zc = R["z_ceiling"]
    storey = zc - zf

    # ---- recover SINGLE-SIDED walls the free-space carve misses: a wall has
    # FULL-HEIGHT material (points near floor AND near ceiling at the same x,y);
    # furniture does not. Add those columns to the wall mask. ----
    def raster(mask):
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[mask] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[mask]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1
        return o
    lo = raster((z > zf + 0.25) & (z < zf + 0.75))
    hi = raster((z > zc - 0.75) & (z < zc - 0.25))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.10 / CELL) | 1,) * 2)
    fh = cv2.morphologyEx((cv2.dilate(lo, dk) & cv2.dilate(hi, dk)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    wallmask = ((ws > 0) | (fh > 0)).astype(np.uint8)
    # keep the wall network + long thin pieces; drop compact furniture blobs
    lbl, n = _ndi.label(wallmask, structure=np.ones((3, 3)))
    if n:
        szs = _ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        big = 1 + int(np.argmax(szs)); keep = {big}
        for k, s in enumerate(szs, 1):
            if k == big or s < (0.3 / (CELL * CELL)):
                continue
            ys, xs = np.where(lbl == k)
            if max(np.ptp(xs), np.ptp(ys)) * CELL >= 0.8:
                keep.add(k)
        wallmask = np.isin(lbl, list(keep)).astype(np.uint8)
    wallmask = _ndi.binary_fill_holes(wallmask).astype(np.uint8)

    # 1. skeleton (centerline network) of the FULL wall mask
    skel = skeletonize(wallmask > 0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "skeleton.png"), skel * 255)

    # 2. thicken to uniform wall thickness, fill, clean
    t = max(3, int(round(WALL_T / CELL)) | 1)
    thick = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))
    thick = ndimage.binary_fill_holes(thick).astype(np.uint8)
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    plan = np.full((*thick.shape, 3), 255, np.uint8)
    plan[thick > 0] = (40, 40, 40)
    cv2.imwrite(str(out_dir / "skeleton_walls_plan.png"), plan)

    # 3. extrude the thickened skeleton footprint to ceiling -- rectified to
    #    strict 90-degree (L) corners first.
    from shapely.geometry import Polygon as _Poly
    polys = []
    for pg in footprint_polygons(thick, xmin, ymax):
        ext = rectify_ring(pg.exterior.coords)
        holes = [h for h in (rectify_ring(r.coords) for r in pg.interiors) if len(h) >= 4]
        if len(ext) < 4:
            continue
        try:
            p = _Poly(ext, holes)
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0.05:
                polys.append(p)
        except Exception:
            pass
    wall_solid = None
    for pg in polys:
        for g in (pg.geoms if hasattr(pg, "geoms") else [pg]):
            try:
                pr = trimesh.creation.extrude_polygon(g, height=storey)
                pr.apply_translation((0, 0, zf))
                wall_solid = pr if wall_solid is None else trimesh.util.concatenate([wall_solid, pr])
            except Exception:
                pass
    # ---- cut door/passage openings as their real (u,z) SILHOUETTE, so ARCHED
    # heads (material spanning the opening only near the top) are kept. The mid-
    # level skeleton has the gap; the arch lives in the top slices -> the
    # silhouette cutter stops at the arch soffit instead of going full height. ----
    from shapely.geometry import Polygon as _Poly2
    from scripts.experiments.hough_vectorize import snap_and_merge
    ppm = 1.0 / CELL
    lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, threshold=20,
                            minLineLength=int(0.4 * ppm), maxLineGap=int(0.3 * ppm))
    segs = []
    if lines is not None:
        hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
        segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL]))
                for a0, a1, yr in hs] + \
               [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL]))
                for a0, a1, xc in vs]
    xy = np.column_stack([x, y]); ncut = 0
    UR = 0.02; z0w = zf - 0.05; z1w = zc + 0.05
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.20) & (u >= 0) & (u <= L)
        if near.sum() < 200:
            continue
        uu, zz = u[near], z[near]
        nu2 = max(int(L / UR) + 1, 4); nz2 = max(int((z1w - z0w) / UR) + 1, 4)
        mat = np.zeros((nz2, nu2), np.uint8)
        mat[np.clip(((zz - z0w) / UR).astype(int), 0, nz2 - 1),
            np.clip((uu / UR).astype(int), 0, nu2 - 1)] = 1
        mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        core = np.zeros_like(mat)
        core[int((zf + 0.08 - z0w) / UR):int((zc - 0.12 - z0w) / UR), :] = 1
        empty = cv2.morphologyEx(((core > 0) & (mat == 0)).astype(np.uint8),
                                 cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ncE, lblE, st, _ = cv2.connectedComponentsWithStats(empty, 8)
        for i in range(1, ncE):
            bx, by, bw, bh, _ = st[i]
            if not (0.5 <= bw * UR <= 3.0) or bh * UR < 0.6 or bx <= 1 or bx + bw >= nu2 - 1:
                continue
            cs, _ = cv2.findContours((lblE == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.02 / UR, True)
            if len(cnt) < 3:
                continue
            poly = _Poly2([(c * UR, z0w + r * UR) for c, r in cnt[:, 0, :]])
            if not poly.is_valid or poly.area < 0.3:
                continue
            try:
                cutter = trimesh.creation.extrude_polygon(poly, height=WALL_T * 4)
                cutter.apply_transform(np.array(
                    [[dd[0], 0, nn[0], p0[0] - nn[0] * WALL_T * 2],
                     [dd[1], 0, nn[1], p0[1] - nn[1] * WALL_T * 2],
                     [0, 1, 0, 0], [0, 0, 0, 1]], float))
                wall_solid = wall_solid.difference(cutter); ncut += 1
            except Exception:
                pass
    log(f"cut {ncut} openings (arched heads kept)")

    # ---- shallow REVEALS / shadow-gaps: NOT openings -- a small-depth recess
    # over a HEIGHT RANGE (starts at a certain z, runs partway). Map via the
    # (u,z) recess silhouette: per (along-wall u, height z) cell measure how far
    # the face sets back; cut a shallow shell of the measured depth over the
    # recessed (u, z) region so the height extent is preserved. ----
    nrev = 0
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.20) & (u >= 0) & (u <= L)
        if near.sum() < 200:
            continue
        uu, zz, pp = u[near], z[near], perp[near]
        thick = float(np.clip(np.percentile(pp, 92) - np.percentile(pp, 8), 0.06, 0.35))
        Rz = trimesh.transformations.rotation_matrix(np.arctan2(dd[1], dd[0]), [0, 0, 1])
        for side in (+1.0, -1.0):
            sel = (pp * side) > 0
            if sel.sum() < 200:
                continue
            sp = pp[sel] * side; sz = zz[sel]; su = uu[sel]
            UB = 0.10; zb = np.arange(zf + 0.15, zc - 0.15, 0.1); nu = max(int(L / UB), 1)
            surf = np.full((len(zb), nu), np.nan)
            for iz, zl in enumerate(zb):
                zm = (sz >= zl) & (sz < zl + 0.1)
                if zm.sum() < 8:
                    continue
                ui = np.clip((su[zm] / UB).astype(int), 0, nu - 1); spz = sp[zm]
                for iu in np.unique(ui):
                    cm = ui == iu
                    if cm.sum() >= 3:
                        surf[iz, iu] = np.median(spz[cm])
            valid = ~np.isnan(surf)
            if valid.sum() < 6:
                continue
            face = float(np.nanmedian(surf))
            noise = 1.4826 * float(np.nanmedian(np.abs(surf[valid] - face))) + 1e-6
            thr = max(0.02, 3.0 * noise)
            recess = np.where(valid, face - surf, 0.0)
            groove = (recess >= thr) & valid
            lbl2, nc2 = ndimage.label(groove, structure=np.ones((3, 3)))
            for cc in range(1, nc2 + 1):
                ys, xs = np.where(lbl2 == cc)
                if ys.size < 3:
                    continue
                bb = (np.ptp(ys) + 1) * (np.ptp(xs) + 1)
                if ys.size / bb < 0.5:
                    continue
                z0 = zb[ys.min()]; z1 = zb[ys.max()] + 0.1
                u0 = xs.min() * UB; u1 = (xs.max() + 1) * UB
                depth = float(np.clip(np.median(recess[ys, xs]), 0.005, min(0.035, thick * 0.4)))
                box_th = depth + 0.02; cn = thick / 2 - depth + box_th / 2
                ch = trimesh.creation.box(extents=(max(u1 - u0, UB) * 0.98, box_th, z1 - z0))
                ch.apply_transform(Rz)
                off = (p0 + dd * (u0 + u1) / 2) + nn * side * cn
                ch.apply_translation((off[0], off[1], (z0 + z1) / 2))
                try:
                    wall_solid = wall_solid.difference(ch); nrev += 1
                except Exception:
                    pass
    log(f"cut {nrev} shallow reveals / shadow-gaps (height-mapped)")

    fx = R["x"].max() - R["x"].min(); fy = R["y"].max() - R["y"].min()
    floor = trimesh.creation.box(extents=(fx, fy, 0.08))
    floor.apply_translation(((R["x"].min() + R["x"].max()) / 2, (R["y"].min() + R["y"].max()) / 2, zf - 0.05))
    scene = trimesh.Scene()
    wall_solid.visual.face_colors = [205, 205, 210, 255]
    floor.visual.face_colors = [150, 130, 110, 255]
    scene.add_geometry(wall_solid, geom_name="walls")
    scene.add_geometry(floor, geom_name="floor")
    scene.export(str(out_dir / "skeleton_model.glb"))
    scene.export(str(out_dir / "skeleton_model.obj"))
    log(f"skeleton_model: {len(wall_solid.vertices):,}v / {len(wall_solid.faces):,}f -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
