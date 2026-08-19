"""Offscreen renders of the segmented model, one colour per part.

Colour per PART, not per triangle: a wall that has shattered into fragments
shows up immediately as confetti, and a wall that has swallowed a wardrobe
shows up as one colour spilling off the plane. Both are invisible in a
single-colour render.
"""
import sys, json
import numpy as np
import open3d as o3d

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/model/poisson_modular"
CACHE = sys.argv[2] if len(sys.argv) > 2 else "output/model/poisson_koushik.npz"
W, H = 1600, 1200

d = np.load(CACHE)
T = d["T"].astype(np.int64)
V = np.load(f"{OUT}/verts.npy").astype(np.float64)
label = np.load(f"{OUT}/labels.npy")
names = json.load(open(f"{OUT}/names.json"))
C = V[T].mean(axis=1)

rng = np.random.default_rng(3)
pal = rng.integers(45, 240, (len(names)+1, 3))/255.0

def mesh_of(mask, keep_ceiling=True, zmax=None):
    idx = np.where(mask)[0]
    if zmax is not None:
        idx = idx[C[idx, 2] < zmax]
    t = T[idx]
    used = np.unique(t)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V[used])
    m.triangles = o3d.utility.Vector3iVector(remap[t])
    col = np.zeros((len(used), 3))
    lab = np.clip(label[idx], 0, None)
    col[remap[t].reshape(-1)] = np.repeat(pal[lab], 3, axis=0)
    m.vertex_colors = o3d.utility.Vector3dVector(col)
    m.compute_vertex_normals()
    return m

def shot(mesh, path, front, up=(0, 0, 1), zoom=0.62):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(mesh)
    o = vis.get_render_option()
    o.background_color = np.array([1., 1., 1.])
    o.light_on = True
    vc = vis.get_view_control()
    vc.set_front(front); vc.set_up(up); vc.set_zoom(zoom)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()
    print("wrote", path, flush=True)

kept = label >= 0
zc = np.percentile(C[kept, 2], 99.0)
ceil_names = {i for i, n in enumerate(names) if n.startswith("ceiling")}
no_ceil = kept & ~np.isin(label, list(ceil_names))

shot(mesh_of(kept), f"{OUT}/render_iso.png", (0.6, 0.6, 0.45))
shot(mesh_of(no_ceil, zmax=zc-0.15), f"{OUT}/render_cutaway.png", (0.35, 0.45, 0.82), zoom=0.55)
shot(mesh_of(kept), f"{OUT}/render_plan.png", (0.0, 0.0, 1.0), up=(0, 1, 0), zoom=0.5)
shot(mesh_of(~kept), f"{OUT}/render_dropped.png", (0.6, 0.6, 0.45))
