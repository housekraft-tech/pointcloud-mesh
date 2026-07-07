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
        # OUTER perimeter (exterior ring): keep SMOOTH (light simplify).
        # INTERIOR walls (room-outline holes): rectify to crisp 90-degree corners.
        ext = [(float(x), float(y)) for x, y in pg.exterior.simplify(0.06).coords]
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
