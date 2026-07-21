"""render_modular.py -- colour each object in the modular OBJ distinctly and
render a perspective + a ceiling-removed cutaway so the per-wall modularity is
visible. Parses `o <name>` groups from the OBJ.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_modular.py <modular.obj> <out_dir>
"""
import sys, time, colorsys
from pathlib import Path
import numpy as np
import open3d as o3d

W, H = 1500, 1100


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse(obj_path):
    verts = []; groups = []; cur = None
    for ln in open(obj_path):
        if ln.startswith("o "):
            cur = ln[2:].strip(); groups.append([cur, [], []])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]; verts.append((float(x), float(y), float(z)))
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:4]]
            groups[-1][2].append(idx)
    V = np.array(verts)
    return V, groups


def color_for(name, i):
    if name == "floor":   return [0.55, 0.55, 0.58]
    if name.startswith("ceiling"):
        # each ceiling plateau is its own object now -- vary the grey slightly so
        # the separate parts (main / dropped / beam soffits) read apart
        g = 0.62 + 0.10 * ((i * 0.61803) % 1.0)
        return [g, g + 0.02, g + 0.04]
    if name.startswith("column"): return [0.95, 0.35, 0.1]
    # measured openings get fixed, meaningful colours (see build_annotated_model)
    if name.startswith("balcony_door"): return [0.00, 1.00, 0.55]
    if name.startswith("door"):         return [0.00, 0.90, 1.00]
    if name.startswith("window"):       return [1.00, 0.85, 0.00]
    if name.startswith("archway"):      return [1.00, 0.55, 0.00]
    if name.startswith("opening"):      return [1.00, 0.30, 0.55]
    h = (i * 0.61803) % 1.0
    return list(colorsys.hsv_to_rgb(h, 0.55, 0.95))


def build(V, groups, drop_ceiling=False, zcut=None, ceiling_only=False):
    mesh = o3d.geometry.TriangleMesh()
    allv = o3d.utility.Vector3dVector(V)
    vc = np.tile([0.5, 0.5, 0.5], (len(V), 1))
    tris = []
    zc = [np.median(V[np.unique(np.array(f)), 2]) for n, _, f in groups
          if f and n.startswith("ceiling")]
    zlo_c, zhi_c = (min(zc), max(zc)) if zc else (0.0, 1.0)
    for i, (name, _, f) in enumerate(groups):
        if not f:
            continue
        if drop_ceiling and name.startswith("ceiling"):
            continue
        if ceiling_only and not name.startswith("ceiling"):
            continue
        f = np.array(f)
        col = color_for(name, i)
        if ceiling_only:
            z = V[np.unique(f), 2]
            t = np.clip((np.median(z) - zlo_c) / max(zhi_c - zlo_c, 1e-6), 0, 1)
            col = list(colorsys.hsv_to_rgb(0.75 * (1 - t), 0.75, 0.95))
        vc[np.unique(f)] = col
        tris.append(f)
    if not tris:
        return None
    mesh.vertices = allv
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack(tris))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    if zcut is not None:
        keep = V[:, 2] <= zcut
        mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()
    return mesh


def render(mesh, front, out_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.10])
    opt.light_on = True; opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front(front); ctr.set_up([0, 0, 1])
    ctr.set_lookat(mesh.get_center()); ctr.set_zoom(0.62)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    log(f"wrote {out_path.name}")


def main(obj_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    V, groups = parse(obj_path)
    log(f"parsed {len(groups)} objects, {len(V):,} verts")
    m1 = build(V, groups)
    render(m1, [0.5, 0.6, 0.6], out / "modular_perspective.png")
    zcut = V[:, 2].max() - 0.6
    m2 = build(V, groups, drop_ceiling=True, zcut=zcut)
    render(m2, [0.35, 0.45, 0.82], out / "modular_cutaway.png")
    # looking UP at the underside: the only view where the beams and dropped
    # ceilings actually show, since the cutaway removes the ceiling
    m3 = build(V, groups, ceiling_only=True)
    if m3 is not None:
        render(m3, [0.25, 0.35, -0.90], out / "modular_soffit.png")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
