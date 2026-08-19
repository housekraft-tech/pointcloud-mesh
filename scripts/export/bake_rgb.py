"""Put the scan's colour back onto the model.

mujammel's export carries RGB; koushik's is all zero. Where it exists it is
worth carrying, because colour answers questions geometry cannot:

  what a surface IS   tile, paint, timber, glass and stone are the same shape
                      and different colours, so the finish schedule is readable
                      off the model rather than off site photographs
  where glazing is    a window is a hole to the walk-path detector and a dark,
                      low-return, blue-grey patch to the colour -- the one cue
                      that survives when the scanner saw straight through it
  what changed        two scans of one flat differ in geometry by millimetres;
                      a repaint or a re-tile is invisible in geometry and
                      obvious in colour

The mesh has no colour of its own -- Poisson threw it away -- so each vertex
takes the colour of the nearest scanned point. The model was rotated onto the
building's axes and the mesh was shifted in z when it was built, so the point
cloud is put through the same two transforms before the lookup, and the residual
distance is reported: if it is not a few millimetres, the frames do not match
and the colours would be a lie.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load_las(path, keep=6_000_000):
    import laspy
    with laspy.open(path) as r:
        p = r.read()
    P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
    try:
        C = np.column_stack([p.red, p.green, p.blue]).astype(np.float64)
    except Exception:
        return P, None
    if C.max() <= 0:
        return P, None
    C /= 65535.0 if C.max() > 255 else 255.0
    if len(P) > keep:
        i = np.random.default_rng(0).choice(len(P), keep, replace=False)
        P, C = P[i], C[i]
    return P, C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="output/model/poisson_modular_mujammel")
    ap.add_argument("--cache", default="output/model/poisson_mujammel.npz")
    ap.add_argument("--las", default="output/mujammel_all/isolated.las")
    ap.add_argument("--lite", type=int, default=900_000,
                    help="triangles in the coloured GLB")
    a = ap.parse_args()
    d = Path(a.dir)
    man = json.load(open(d/"manifest.json"))
    T = np.load(a.cache)["T"].astype(np.int64)
    V = np.load(d/"verts.npy").astype(np.float64)
    L = np.load(d/"labels.npy")
    names = json.load(open(d/"names.json"))

    P, C = load_las(a.las)
    if C is None:
        log(f"{a.las} carries no colour -- nothing to bake")
        return 1
    log(f"{len(P):,} coloured points")

    # the same two transforms the mesh went through
    P[:, 2] -= float(np.percentile(P[:, 2], 0.5))
    yaw = np.radians(-man["yaw_deg"])
    c, s = np.cos(yaw), np.sin(yaw)
    P[:, :2] = np.column_stack([P[:, 0]*c - P[:, 1]*s, P[:, 0]*s + P[:, 1]*c])

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    dist, idx = tree.query(V, workers=-1)
    log(f"nearest scanned point: median {np.median(dist)*1000:.1f} mm, "
        f"90th pct {np.percentile(dist, 90)*1000:.1f} mm")
    if np.median(dist) > 0.05:
        log("WARNING: the cloud and the model are not in the same frame; "
            "the colours below are not to be trusted")
    col = C[idx]

    np.save(d/"vertex_rgb.npy", (col*255).astype(np.uint8))
    # per-part colour, which is what a finish schedule reads
    tri_col = col[T].mean(axis=1)
    for p in man["parts"]:
        pid = names.index(p["name"])
        m = L == pid
        if not m.any():
            continue
        cc = tri_col[m]
        p["rgb"] = [int(round(x*255)) for x in np.median(cc, axis=0)]
        p["rgb_spread"] = int(round(float(np.median(np.std(cc, axis=0)))*255))
    json.dump(man, open(d/"manifest.json", "w"), indent=1)
    log("manifest updated with a colour per part")

    # a coloured, decimated GLB, so the finishes can actually be looked at
    import open3d as o3d, trimesh
    keep = np.where(L >= 0)[0]
    ratio = min(1.0, a.lite/len(keep))
    sc = trimesh.Scene()
    for pid, nm in enumerate(names):
        sel = np.where(L == pid)[0]
        if sel.size == 0:
            continue
        t = T[sel]
        used = np.unique(t)
        rm = np.full(len(V), -1, np.int64); rm[used] = np.arange(len(used))
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(V[used])
        m.triangles = o3d.utility.Vector3iVector(rm[t])
        m.vertex_colors = o3d.utility.Vector3dVector(col[used])
        want = max(200, int(len(t)*ratio))
        if want < len(t):
            m = m.simplify_quadric_decimation(want)
        vv = np.asarray(m.vertices); tt = np.asarray(m.triangles)
        if len(tt) == 0:
            continue
        cc = np.asarray(m.vertex_colors)
        tm = trimesh.Trimesh(vv, tt, process=False)
        rgba = np.zeros((len(vv), 4), np.uint8)
        rgba[:, :3] = np.clip(cc*255, 0, 255).astype(np.uint8); rgba[:, 3] = 255
        tm.visual.vertex_colors = rgba
        sc.add_geometry(tm, geom_name=nm, node_name=nm)
    sc.export(str(d/"modular_rgb.glb"))
    log(f"wrote {d/'modular_rgb.glb'} "
        f"({(d/'modular_rgb.glb').stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
