"""colorize_model.py
-----------------
Bake mujammel's colour onto the koushik modular model.

koushik carries the geometry but exported RGB as all zeros; mujammel has colour
in its own frame. register_scans.py supplies the rigid transform, so every model
vertex can take the colour of the nearest mujammel point.

Worth being clear about what this does and does not buy. The colour is ~76%
greyscale (chroma < 0.05, channels correlated 0.92-0.96), so it will not carry
hue-based semantics -- that is why the door-vs-balcony attempt failed. What it
does carry is TONE: tiling against paint, skirting, staining, and the wet-room
finishes, which is exactly the interior-finish signal the deviation product
wants and which pure geometry cannot see.

Per-vertex chroma is written out too, so the genuinely coloured minority can be
found rather than assumed.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\colorize_model.py \\
      <model.obj> <rgb_scan.las> <scan_transform.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np

VOXEL = 0.02          # m: downsample the colour cloud before the KD-tree
MAX_DIST = 0.12       # m: further than this and the vertex has no colour
UNCOLOURED = (0.45, 0.45, 0.47)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_obj(path):
    V = []; groups = []
    for ln in open(path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), []])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
        elif ln.startswith("f ") and groups:
            groups[-1][1].append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, np.float64), groups


def load_colour_cloud(las_path, T):
    import laspy
    las = laspy.read(las_path)
    P = np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)].astype(np.float32)
    C = (np.c_[np.asarray(las.red), np.asarray(las.green),
               np.asarray(las.blue)].astype(np.float32) / 65535.0)
    T = np.asarray(T)
    P = (P @ T[:3, :3].T.astype(np.float32)) + T[:3, 3].astype(np.float32)
    log(f"{len(P):,} colour points transformed")

    # voxel-average: cheaper than a 27M-point tree and it denoises the colour
    key = np.floor(P / VOXEL).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    n = len(cnt)
    Pv = np.zeros((n, 3), np.float32); Cv = np.zeros((n, 3), np.float32)
    np.add.at(Pv, inv, P); np.add.at(Cv, inv, C)
    Pv /= cnt[:, None]; Cv /= cnt[:, None]
    log(f"voxel-averaged to {n:,} points at {VOXEL} m")
    return Pv, Cv


def main(obj_path, las_path, tf_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = json.load(open(tf_path))
    if not tf.get("usable"):
        raise SystemExit("registration unusable -- refusing to bake colour")
    log(f"transform fitness {tf['fitness']:.3f}, rmse {tf['rmse_mm']:.1f} mm")

    V, groups = parse_obj(obj_path)
    log(f"model: {len(V):,} vertices, {len(groups)} objects")
    P, C = load_colour_cloud(las_path, tf["transform"])

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    log("querying nearest colour per vertex ...")
    d, idx = tree.query(V.astype(np.float32), k=1, workers=-1,
                        distance_upper_bound=MAX_DIST)
    hit = np.isfinite(d) & (idx < len(P))
    col = np.tile(np.array(UNCOLOURED, np.float32), (len(V), 1))
    col[hit] = C[idx[hit]]
    log(f"{hit.sum():,}/{len(V):,} vertices coloured "
        f"({100*hit.mean():.1f}%), median distance "
        f"{np.median(d[hit])*1000:.0f} mm")

    chroma = col.max(1) - col.min(1)
    log(f"chroma: median {np.median(chroma[hit]):.3f}, "
        f"{100*(chroma[hit] > 0.10).mean():.1f}% of coloured vertices above 0.10 "
        f"(the genuinely coloured minority)")

    ply = out / "model_coloured.ply"
    with open(ply, "w") as fh:
        tris = [t for _, f in groups for t in f]
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(V)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {len(tris)}\n")
        fh.write("property list uchar int vertex_index\nend_header\n")
        rgb = np.clip(col * 255, 0, 255).astype(np.uint8)
        for p, c in zip(V, rgb):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for t in tris:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    log(f"wrote {ply}")

    try:
        import trimesh
        scene = trimesh.Scene()
        for name, f in groups:
            if not f:
                continue
            f = np.asarray(f)
            used = np.unique(f)
            remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
            m = trimesh.Trimesh(V[used], remap[f], process=False)
            m.visual.vertex_colors = np.clip(col[used] * 255, 0, 255).astype(np.uint8)
            scene.add_geometry(m, geom_name=name, node_name=name)
        scene.export(str(out / "model_coloured.glb"))
        log(f"wrote {out/'model_coloured.glb'}")
    except Exception as e:
        log(f"glb skipped: {e}")

    np.save(out / "vertex_colours.npy", col.astype(np.float32))
    json.dump(dict(vertices=int(len(V)), coloured=int(hit.sum()),
                   coloured_frac=round(float(hit.mean()), 4),
                   median_match_mm=round(float(np.median(d[hit]) * 1000), 1),
                   median_chroma=round(float(np.median(chroma[hit])), 4),
                   frac_chroma_gt_010=round(float((chroma[hit] > 0.10).mean()), 4)),
              open(out / "colour_stats.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main(*sys.argv[1:5])
