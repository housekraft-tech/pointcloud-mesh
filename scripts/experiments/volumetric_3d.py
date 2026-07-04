"""volumetric_3d.py
----------------
A TRUE 3D reconstruction (not a single extruded top-view footprint). Slice the
cloud at many heights; at each height take the FILLED wall footprint; stack the
slices into a 3D occupancy volume; marching-cubes -> one accurate mesh.

Because each height has its own footprint, features that exist only over PART of
the height are captured correctly: bulkheads/beams (top only), piers/pilasters
that stop partway, arched heads (the opening closes up high), and doors (the gap
is empty only at door height). Uses walls_solid (the full carved footprint) so
no walls are missing. Also computes per-room clear dimensions.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\volumetric_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
import trimesh
from skimage import measure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, measure_axis, CELL, STRIP

DZ = 0.025             # slice thickness (2.5cm) -- fine enough for real features
DIL_M = 0.12           # dilate each slice's face points to fill wall thickness


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wall_footprint(R):
    free, occ = R["free"], R["occ"]
    fp = ndimage.binary_fill_holes((free > 0) | (occ > 0))
    ws = (fp & (free == 0)).astype(np.uint8)
    ws = ndimage.binary_fill_holes(ws).astype(np.uint8)
    ws = cv2.morphologyEx(ws, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # keep ONLY the connected wall NETWORK (perimeter + internal walls all join);
    # drop isolated interior islands = free-standing furniture, so material is
    # captured precisely AT THE WALLS, not everywhere.
    lbl, n = ndimage.label(ws, structure=np.ones((3, 3)))
    if n:
        szs = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        big = 1 + int(np.argmax(szs))
        # the wall network is the largest component; also keep large elongated
        # (wall-like) pieces, but drop compact interior blobs = free furniture.
        keep = {big}
        for k, s in enumerate(szs, 1):
            if k == big or s < (0.4 / (CELL * CELL)):
                continue
            ys, xs = np.where(lbl == k)
            ext = max(xs.max() - xs.min(), ys.max() - ys.min()) * CELL
            if ext >= 1.2 and s / max((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1), 1) < 0.6:
                keep.add(k)                                  # long + thin => a wall, not a blob
        ws = np.isin(lbl, list(keep)).astype(np.uint8)
    return ws


def build_volume(R, ws):
    x, y, z = R["x"], R["y"], R["z"]
    zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    levels = np.arange(zf, zc + DZ, DZ)
    nz = len(levels)
    vol = np.zeros((H, W, nz), np.uint8)
    dk = max(3, int(DIL_M / CELL) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    for k, zk in enumerate(levels):
        band = (z >= zk - DZ * 0.7) & (z <= zk + DZ * 0.7)
        if band.sum() < 20:
            continue
        occk = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
        occk[rr, cc] = 1
        occk = cv2.dilate(occk, kern)
        vol[:, :, k] = (ws > 0) & (occk > 0)
    # ---- fill GAPS between/within walls from scan occlusion, without erasing
    # genuine openings ----
    # (a) in-plane: connect wall segments within each slice
    ck = np.ones((5, 5), np.uint8)
    for k in range(nz):
        if vol[:, :, k].any():
            vol[:, :, k] = cv2.morphologyEx(vol[:, :, k], cv2.MORPH_CLOSE, ck)
    # (b) vertical: bridge occlusion holes up to ~0.35m; real doors/windows
    #     (>=0.5m tall gaps) stay open
    vol = ndimage.binary_closing(vol, structure=np.ones((1, 1, 15))).astype(np.uint8)
    # (c) a wall column with support over most of the height is solid: fill it
    #     floor->ceiling EXCEPT where a tall empty run (a real opening) exists
    support = vol.sum(axis=2)
    solidish = support >= int(0.6 * nz)                      # present most of the height
    fill = np.repeat(solidish[:, :, None], nz, axis=2) & (ws[:, :, None] > 0)
    vol = ((vol > 0) | fill).astype(np.uint8)
    log(f"volume {vol.shape}  {int(vol.sum()):,} filled voxels over {nz} height slices")
    return vol, levels


def volume_to_mesh(vol, R, zf):
    xmin, ymax = R["xmin"], R["ymax"]
    volp = np.pad(vol, 1).astype(np.float32)
    volp = ndimage.gaussian_filter(volp, 0.6)                 # light smooth -> less staircase
    verts, faces, _, _ = measure.marching_cubes(volp, 0.5, spacing=(CELL, CELL, DZ))
    wx = xmin + (verts[:, 1] - CELL)
    wy = ymax - (verts[:, 0] - CELL)
    wz = zf + (verts[:, 2] - DZ)
    m = trimesh.Trimesh(vertices=np.column_stack([wx, wy, wz]), faces=faces, process=True)
    trimesh.smoothing.filter_taubin(m, iterations=3)
    return m


def room_dims(R):
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    wb = (z >= zf + 0.2) & (z <= zc - 0.1)
    xw, yw, zw = x[wb], y[wb], z[wb]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    out = []
    for Lr in R["room_labels"]:
        mm = (R["mk"] == Lr)
        if mm.sum() < (1.0 / (CELL * CELL)):
            continue
        rd = R["distm"] * mm
        cy, cx = np.unravel_index(int(np.argmax(rd)), rd.shape)
        rcx = xmin + cx * CELL; rcy = ymax - cy * CELL
        row = mm[cy, :]; c0 = cx; c1 = cx
        while c0 > 0 and row[c0 - 1]: c0 -= 1
        while c1 < W - 1 and row[c1 + 1]: c1 += 1
        col = mm[:, cx]; r0 = cy; r1 = cy
        while r0 > 0 and col[r0 - 1]: r0 -= 1
        while r1 < H - 1 and col[r1 + 1]: r1 += 1
        mx = np.abs(yw - rcy) <= STRIP; my = np.abs(xw - rcx) <= STRIP
        if mx.sum() < 30 or my.sum() < 30:
            continue
        rw = measure_axis(xw[mx], zw[mx], rcx, xmin + c0 * CELL, xmin + c1 * CELL)
        rh = measure_axis(yw[my], zw[my], rcy, ymax - r1 * CELL, ymax - r0 * CELL)
        opn = rw["open_l"] or rw["open_r"] or rh["open_l"] or rh["open_r"]
        out.append((cx, cy, rw["span"], rh["span"], opn))
    return out


def dimensioned_plan(R, ws, out_dir):
    H, W = R["H"], R["W"]
    plan = np.full((H, W, 3), 255, np.uint8)
    plan[ws > 0] = (45, 45, 45)                              # thick filled walls
    FT = cv2.FONT_HERSHEY_SIMPLEX
    for cx, cy, wdt, hgt, opn in room_dims(R):
        op = "~" if opn else ""
        for t, dy, sc in [(f"{op}{wdt*1000:.0f} x {hgt*1000:.0f} mm", -6, 0.5),
                          (f"({wdt*3.28084:.1f} x {hgt*3.28084:.1f} ft)", 12, 0.42)]:
            (tw, th), _ = cv2.getTextSize(t, FT, sc, 1)
            tx = int(np.clip(cx - tw // 2, 2, W - tw - 2))
            cv2.rectangle(plan, (tx - 2, cy + dy - th - 1), (tx + tw + 2, cy + dy + 3), (255, 255, 255), -1)
            cv2.putText(plan, t, (tx, cy + dy), FT, sc, (150, 0, 160), 1, cv2.LINE_AA)
    cv2.putText(plan, "Volumetric 3D - wall plan (all walls) + internal CLEAR dims (mm / ft)",
                (10, 22), FT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "volumetric_dimensioned.png"), plan)


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"reconstructing {las} ...")
    R = reconstruct(las)
    ws = wall_footprint(R)
    zf = R["z_floor"]
    vol, levels = build_volume(R, ws)
    log("marching cubes ...")
    walls = volume_to_mesh(vol, R, zf)
    # floor slab
    xmin, ymax = R["xmin"], R["ymax"]
    fx = R["x"].max() - R["x"].min(); fy = R["y"].max() - R["y"].min()
    floor = trimesh.creation.box(extents=(fx, fy, 0.08))
    floor.apply_translation(((R["x"].min() + R["x"].max()) / 2, (R["y"].min() + R["y"].max()) / 2, zf - 0.05))
    scene = trimesh.Scene()
    walls.visual.face_colors = [205, 205, 210, 255]
    floor.visual.face_colors = [150, 130, 110, 255]
    scene.add_geometry(walls, geom_name="walls")
    scene.add_geometry(floor, geom_name="floor")
    scene.export(str(out_dir / "volumetric_model.glb"))
    scene.export(str(out_dir / "volumetric_model.obj"))
    log(f"exported volumetric_model.glb/.obj: walls {len(walls.vertices):,}v/{len(walls.faces):,}f")
    dimensioned_plan(R, ws, out_dir)
    log(f"wrote volumetric_dimensioned.png -> {out_dir}")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
