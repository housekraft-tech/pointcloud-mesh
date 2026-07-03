"""render_mesh_png.py
------------------
Render an OBJ mesh (or the isolated point cloud) to PNG snapshots from a few
viewpoints, headless -- so you can see how the reconstructed geometry looks
without opening Blender. Uses Open3D's offscreen screen-capture.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_mesh_png.py <mesh.obj> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(mesh_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading {mesh_path} ...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        log("no triangles -- loading as point cloud instead")
        geom = o3d.io.read_point_cloud(str(mesh_path))
    else:
        mesh.compute_vertex_normals()
        # light gray surface so shading reads
        mesh.paint_uniform_color([0.75, 0.76, 0.78])
        geom = mesh
    log(f"loaded: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1500, height=1100)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.10])
    opt.light_on = True
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()

    # a few viewpoints: perspective corners + straight top-down (plan)
    views = [
        ("perspective_1", dict(front=[0.5, 0.6, 0.6], up=[0, 0, 1], zoom=0.62)),
        ("perspective_2", dict(front=[-0.6, 0.5, 0.55], up=[0, 0, 1], zoom=0.62)),
        ("top_down_plan", dict(front=[0, 0, 1], up=[0, 1, 0], zoom=0.72)),
        ("front_elevation", dict(front=[0, 1, 0.15], up=[0, 0, 1], zoom=0.7)),
    ]
    for name, cam in views:
        ctr.set_front(cam["front"])
        ctr.set_up(cam["up"])
        ctr.set_lookat(geom.get_center())
        ctr.set_zoom(cam["zoom"])
        vis.poll_events()
        vis.update_renderer()
        p = out_dir / f"mesh_{name}.png"
        vis.capture_screen_image(str(p), do_render=True)
        log(f"wrote {p}")

    vis.destroy_window()
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
