"""Render the solid model to PNGs with Open3D's offscreen-capable visualiser."""
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "model3d"

VIEWS = [
    # name, front (camera direction), up, zoom
    # `front` runs from the target TOWARD the camera, so +z looks down on it
    ("iso",      (-0.55, -0.62, 0.56), (0, 0, 1), 0.70),
    ("iso_back", (0.60, 0.55, 0.58),   (0, 0, 1), 0.70),
    ("plan",     (0.0, 0.0, 1.0),      (0, 1, 0), 0.72),
    ("eye",      (-0.72, -0.68, 0.16), (0, 0, 1), 0.60),
]


def render(path, out_prefix, views=VIEWS, bg=(1, 1, 1), size=(1400, 1050)):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not mesh.has_triangles():
        print(f"  !! {path.name} has no triangles"); return
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=size[0], height=size[1], visible=True)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg)
    opt.light_on = True
    opt.mesh_show_back_face = True

    for name, front, up, zoom in views:
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_lookat(mesh.get_center())
        ctr.set_zoom(zoom)
        for _ in range(6):
            vis.poll_events()
            vis.update_renderer()
        time.sleep(0.25)
        p = MODEL / f"{out_prefix}_{name}.png"
        vis.capture_screen_image(str(p), do_render=True)
        print(f"  wrote {p.name}", flush=True)
    vis.destroy_window()


def main():
    render(MODEL / "v2_cutaway.glb", "v3",
           views=[v for v in VIEWS if v[0] in ("iso", "iso_back", "plan")])
    render(MODEL / "v2_marked.glb", "v3marked",
           views=[v for v in VIEWS if v[0] in ("iso", "eye")])
    return
    render(MODEL / "flat_openings_cutaway.glb", "openings",
           views=[v for v in VIEWS if v[0] in ("iso", "iso_back", "eye")])
    render(MODEL / "flat_openings_marked.glb", "marked",
           views=[v for v in VIEWS if v[0] in ("iso", "iso_back")])
    render(MODEL / "flat_openings.glb", "opensolid",
           views=[v for v in VIEWS if v[0] in ("eye",)])
    render(MODEL / "flat_cutaway.glb", "cutaway",
           views=[v for v in VIEWS if v[0] in ("iso", "plan", "iso_back")])
    render(MODEL / "flat_rooms.glb", "rooms",
           views=[v for v in VIEWS if v[0] in ("iso", "plan")])
    render(MODEL / "flat_solid.glb", "solid",
           views=[v for v in VIEWS if v[0] in ("iso", "eye")])


if __name__ == "__main__":
    main()
