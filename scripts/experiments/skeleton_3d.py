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

    # 3. extrude the thickened skeleton footprint to ceiling
    polys = footprint_polygons(thick, xmin, ymax)
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
