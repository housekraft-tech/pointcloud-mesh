"""render_colored.py
------------------
Render a GLB/OBJ scene KEEPING its per-part colours (unlike render_mesh_png which
paints a uniform grey). Used to show the measured-vs-synthesised colour key:
grey = measured shell, amber = synthesised fittings, cyan = glass.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_colored.py <scene.glb> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sc = trimesh.load(path)
    geoms = sc.geometry.values() if isinstance(sc, trimesh.Scene) else [sc]

    combined = o3d.geometry.TriangleMesh()
    for g in geoms:
        if not hasattr(g, "vertices") or len(g.faces) == 0:
            continue
        vc = g.visual.vertex_colors[:, :3].astype(float) / 255.0
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(g.vertices)),
            o3d.utility.Vector3iVector(np.asarray(g.faces)))
        m.vertex_colors = o3d.utility.Vector3dVector(vc)
        combined += m
    combined.compute_vertex_normals()

    v = np.asarray(combined.vertices)
    spans = v.max(0) - v.min(0)
    if np.argmin(spans) == 1:
        combined.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1500, height=1100)
    vis.add_geometry(combined)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.10])
    opt.light_on = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    views = [
        ("perspective_1", dict(front=[0.5, 0.6, 0.6], up=[0, 0, 1], zoom=0.62)),
        ("perspective_2", dict(front=[-0.6, 0.5, 0.55], up=[0, 0, 1], zoom=0.62)),
        ("top_down_plan", dict(front=[0, 0, 1], up=[0, 1, 0], zoom=0.72)),
    ]
    for name, cam in views:
        ctr.set_front(cam["front"]); ctr.set_up(cam["up"])
        ctr.set_lookat(combined.get_center()); ctr.set_zoom(cam["zoom"])
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(str(out_dir / f"color_{name}.png"), do_render=True)
        log(f"wrote color_{name}.png")
    vis.destroy_window()
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
