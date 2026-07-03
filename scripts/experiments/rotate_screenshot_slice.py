"""rotate_screenshot_slice.py
--------------------------
Load the Poisson mesh ONCE, rotate it 180 degrees about the up axis, then
(a) save the rotated OBJ, (b) render screenshots, (c) take the top-down slice
and Hough it -- all from the single in-memory mesh (the mesh is 570MB / ~5min
to read, so we never reload it).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\rotate_screenshot_slice.py <mesh.obj> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import trimesh
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.hough_vectorize import snap_and_merge

PPM = 80


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(mesh_path, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    log(f"loading {mesh_path} (once) ...")
    mesh = trimesh.load(str(mesh_path), process=False)
    v = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    spans = v.max(axis=0) - v.min(axis=0)
    up = int(np.argmin(spans))
    plane = [a for a in (0, 1, 2) if a != up]
    log(f"loaded {len(v):,} verts; up axis = {'XYZ'[up]}")

    # ---- rotate 180 deg about the up axis (yaw flip): negate the two
    # floor-plane axes, keep up. ----
    center = v.mean(axis=0)
    vr = v - center
    for a in plane:
        vr[:, a] = -vr[:, a]
    vr = vr + center
    log("rotated mesh 180 deg about up axis")

    # (a) save rotated OBJ
    rot = trimesh.Trimesh(vertices=vr, faces=faces, process=False)
    rot.export(str(out_dir / "poisson_mesh_rot180.obj"))
    log(f"saved {out_dir / 'poisson_mesh_rot180.obj'}")

    # ---- (b) screenshots via Open3D (build from arrays, no reload) ----
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(vr)
    om.triangles = o3d.utility.Vector3iVector(faces)
    om.compute_vertex_normals()
    om.paint_uniform_color([0.75, 0.76, 0.78])
    # Open3D cameras are Z-up; our mesh is Y-up -> rotate Y-up->Z-up for framing
    if up == 1:
        om.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), center=(0, 0, 0))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1500, height=1100)
    vis.add_geometry(om)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.10]); opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    views = [("top_down", [0, 0, 1], [0, 1, 0], 0.72),
             ("perspective", [0.5, 0.6, 0.6], [0, 0, 1], 0.62),
             ("bottom_up", [0, 0, -1], [0, 1, 0], 0.72)]
    for name, front, upv, zoom in views:
        ctr.set_front(front); ctr.set_up(upv); ctr.set_lookat(om.get_center()); ctr.set_zoom(zoom)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(str(out_dir / "screenshots" / f"rot180_{name}.png"), do_render=True)
        log(f"wrote screenshot rot180_{name}.png")
    vis.destroy_window()

    # ---- (c) top-down slice (vertex band) + Hough on the rotated mesh ----
    up_lo, up_hi = vr[:, up].min(), vr[:, up].max()
    floor = up_lo + 0.10 * (up_hi - up_lo)
    band = (vr[:, up] >= floor + 0.5) & (vr[:, up] <= floor + 2.0)
    vb = vr[band][:, plane]
    minx, miny = vb.min(axis=0) - 0.4
    maxx, maxy = vb.max(axis=0) + 0.4
    W = int((maxx - minx) * PPM) + 1
    H = int((maxy - miny) * PPM) + 1
    raster = np.zeros((H, W), np.uint8)
    cols = np.clip(((vb[:, 0] - minx) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((maxy - vb[:, 1]) * PPM).astype(int), 0, H - 1)
    raster[rows, cols] = 255
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    lines = cv2.HoughLinesP(raster, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.25 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.2 * PPM), coord_tol_px=int(0.12 * PPM))
    canvas = cv2.merge([raster // 4] * 3)
    for a0, a1, y in hsegs:
        cv2.line(canvas, (a0, y), (a1, y), (0, 235, 0), 2)
    for a0, a1, x in vsegs:
        cv2.line(canvas, (x, a0), (x, a1), (0, 235, 0), 2)
    cv2.putText(canvas, f"rotated-180 mesh top-down Hough: {len(hsegs)+len(vsegs)} wall lines",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "rot180_hough_topdown.png"), canvas)
    cv2.imwrite(str(out_dir / "rot180_slice_raster.png"), raster)
    log(f"rot180 top-down slice: {int(band.sum()):,} verts -> {len(hsegs)+len(vsegs)} wall lines")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
