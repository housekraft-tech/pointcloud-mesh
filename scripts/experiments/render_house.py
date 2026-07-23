"""render_house.py
--------------
Views of the rebuilt house, coloured by WHAT EACH PART IS.

The modular renderer gives every object its own hue, which proves the parts are
separate but says nothing about what they are. Here walls, pillars, beams,
parapets and relief each get a fixed colour, so the structure is readable: you
can see at a glance where the beams run, which walls are balustrades, and where
the grooves and niches sit.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_house.py \\
      <house_open.obj> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import open3d as o3d

W, H = 1600, 1200

KIND_COLOR = {
    "wall":           [0.86, 0.86, 0.84],   # plaster
    "parapet":        [0.72, 0.66, 0.55],   # balustrade
    "pillar":         [0.95, 0.42, 0.12],   # structural column
        "beamsoffit":     [0.85, 0.20, 0.25],
    "droppedceiling": [0.55, 0.35, 0.75],   # dropped slab (wet rooms)
    "overhead":       [0.85, 0.20, 0.25],
    "beam":           [0.85, 0.20, 0.25],
    "arch":           [0.20, 0.55, 0.95],
    "furniture":      [0.45, 0.75, 0.35],
    "pilaster":       [0.98, 0.72, 0.10],
    "niche":          [0.20, 0.55, 0.95],
    "duct":           [0.35, 0.75, 0.85],
    "protrusion":     [0.30, 0.80, 0.50],
    "floor":          [0.42, 0.44, 0.47],
    "ceiling":        [0.62, 0.64, 0.66],
    "ENTRANCE":       [1.00, 0.10, 0.45],
    "WALK":           [1.00, 0.25, 0.85],
}
VIEWS = [
    ("house_perspective.png", [0.55, 0.62, 0.58], 0.62),
    ("house_plan.png",        [0.02, 0.05, 0.999], 0.70),
    ("house_structure.png",   [-0.60, -0.55, 0.58], 0.62),
    ("house_eye.png",         [0.92, 0.30, 0.16], 0.42),
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse(obj_path):
    verts = []; groups = []
    for ln in open(obj_path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), []])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            verts.append((float(x), float(y), float(z)))
        elif ln.startswith("f ") and groups:
            groups[-1][1].append([int(t.split("/")[0]) - 1
                                  for t in ln.split()[1:4]])
    return np.array(verts), groups


def kind_of(name):
    # CAD hierarchy: "wall_11__beam_07" is a beam, not a wall. Take the last
    # component of a parented name, otherwise the child inherits its parent's
    # colour and every beam disappears into the wall it hangs on.
    return name.split("__")[-1].split("_")[0]


def build(V, groups, keep=None):
    mesh = o3d.geometry.TriangleMesh()
    vc = np.tile([0.5, 0.5, 0.5], (len(V), 1))
    tris = []
    for name, f in groups:
        if not f:
            continue
        k = kind_of(name)
        if keep is not None and k not in keep:
            continue
        f = np.array(f)
        vc[np.unique(f)] = KIND_COLOR.get(k, [0.7, 0.7, 0.7])
        tris.append(f)
    if not tris:
        return None
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack(tris))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    mesh.compute_vertex_normals()
    return mesh


def render(mesh, front, zoom, out_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.99, 0.99, 0.99])
    opt.light_on = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front(front); ctr.set_up([0, 0, 1])
    ctr.set_lookat(mesh.get_center()); ctr.set_zoom(zoom)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    log(f"wrote {out_path.name}")


def main(obj_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    V, groups = parse(obj_path)
    from collections import Counter
    log(f"{len(groups)} objects, {len(V):,} verts: "
        + ", ".join(f"{k} {v}" for k, v in
                    Counter(kind_of(n) for n, _ in groups).most_common()))

    full = build(V, groups)
    for name, front, zoom in VIEWS:
        render(full, front, zoom, out / name)

    # structure only: what holds the building up, without the relief clutter
    hard = build(V, groups, keep={"wall", "parapet", "pillar", "beam",
                                  "beamsoffit", "droppedceiling", "overhead", "arch",
                                  "floor", "ENTRANCE"})
    if hard is not None:
        render(hard, [0.55, 0.62, 0.58], 0.62, out / "house_shell_only.png")
        render(hard, [0.02, 0.05, 0.999], 0.70, out / "house_shell_plan.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
