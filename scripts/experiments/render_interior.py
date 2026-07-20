"""render_interior.py
--------------------
Render INTERIOR views of a mesh: place the camera physically inside the flat
(eye ~1.4 m above floor, looking horizontally) and also produce a
ceiling-removed cutaway from above. Lets us compare how much real surface
relief (grooves / beams / columns) a mesh carries vs a clean skeleton model.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_interior.py <mesh.obj> <out_dir> <tag>
"""
import sys, time
from pathlib import Path
import numpy as np
import open3d as o3d

W, H = 1500, 1100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def look_at_extrinsic(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f])           # world -> camera rotation (rows)
    ext = np.eye(4)
    ext[:3, :3] = R
    ext[:3, 3] = -R @ eye
    return ext


def main(mesh_path, out_dir, tag):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log(f"loading {mesh_path} ...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        log("no triangles; abort"); return
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.73, 0.75])
    log(f"loaded {len(mesh.vertices):,} v / {len(mesh.triangles):,} f")

    # Y-up (Blender) -> Z-up so height is Z
    v = np.asarray(mesh.vertices)
    spans = v.max(0) - v.min(0)
    if np.argmin(spans) == 1:
        Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        mesh.rotate(Rz, center=(0, 0, 0))
        log("rotated Y-up -> Z-up")

    v = np.asarray(mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    zfloor = lo[2]
    eye_z = zfloor + 1.4
    log(f"bbox {lo.round(2)} .. {hi.round(2)}  eye_z={eye_z:.2f}")

    # -------- interior camera views --------
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.10])
    opt.light_on = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()   # keep intrinsic

    eye = [cx, cy, eye_z]
    interior_views = {
        "interior_look_x": [cx + 5, cy, eye_z],
        "interior_look_y": [cx, cy + 5, eye_z],
        "interior_look_nx": [cx - 5, cy, eye_z],
        "interior_look_ceiling": [cx + 2, cy, eye_z + 3],   # tilt up -> beams
    }
    for name, tgt in interior_views.items():
        ext = look_at_extrinsic(eye, tgt)
        param.extrinsic = ext
        try:
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events(); vis.update_renderer()
        p = out / f"{tag}_{name}.png"
        vis.capture_screen_image(str(p), do_render=True)
        log(f"wrote {p.name}")
    vis.destroy_window()

    # -------- ceiling-removed cutaway (see into all rooms) --------
    keep = v[:, 2] <= (hi[2] - 0.35)     # drop top 35 cm (ceiling)
    m2 = o3d.geometry.TriangleMesh(mesh)
    m2.remove_vertices_by_mask(~keep)
    m2.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(m2)
    o2 = vis.get_render_option()
    o2.background_color = np.array([0.08, 0.09, 0.10]); o2.light_on = True
    o2.mesh_show_back_face = True
    c2 = vis.get_view_control()
    c2.set_front([0.35, 0.45, 0.82]); c2.set_up([0, 0, 1])
    c2.set_lookat(m2.get_center()); c2.set_zoom(0.62)
    vis.poll_events(); vis.update_renderer()
    p = out / f"{tag}_cutaway_no_ceiling.png"
    vis.capture_screen_image(str(p), do_render=True)
    log(f"wrote {p.name}")
    vis.destroy_window()
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
