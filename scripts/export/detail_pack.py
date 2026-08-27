"""The handover model at full relief, not boxed.

The box model is a deliberate simplification: flat faces, sharp edges, about
4 mm from the scan. Everything that is neither -- an arch soffit, a fillet, a
curved reveal, the actual wobble of a plastered wall -- is gone from it by
construction.

This writes the other end: the modular surface itself, decimated only as far as
the target asks, so the relief survives. It is a SURFACE, not a solid: it has
the two faces of every wall and the seams between parts, but you cannot cut a
hole in it and expect a closed volume.

Accuracy is measured against the Poisson mesh it came from and printed, so the
number is on the record rather than assumed.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load_modular(d, cache):
    T = np.load(cache)["T"].astype(np.int64)
    V = np.load(Path(d)/"verts.npy").astype(np.float64)
    lab = np.load(Path(d)/"labels.npy")
    return V, T[lab >= 0], T


def decimate(V, F, target):
    """Quadric decimation, but through a library that can take 27 M triangles.

    open3d's simplify_quadric_decimation is single threaded and does not finish
    on a mesh this size -- it ran for ten minutes on one storey without getting
    to the first result. fast_simplification does the same job in seconds.
    """
    import fast_simplification
    if not target or len(F) <= target:
        return V, F
    Vd, Fd = fast_simplification.simplify(
        np.ascontiguousarray(V, np.float32),
        np.ascontiguousarray(F, np.int32),
        target_reduction=1.0 - target/len(F))
    m = trimesh.Trimesh(Vd, Fd, process=False)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.faces)


def error_mm(Vp, Tp, V, F, n=120000):
    ref = trimesh.Trimesh(Vp, Tp, process=False)
    P, _ = trimesh.sample.sample_surface(ref, n)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(V, o3d.core.float32), o3d.core.Tensor(F, o3d.core.uint32)))
    d = sc.compute_distance(o3d.core.Tensor(np.asarray(P, np.float32))).numpy()*1000
    return float(np.median(d)), float(np.percentile(d, 90)), float(np.percentile(d, 99))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--targets", default="60000,250000",
                    help="comma separated triangle budgets; the first is the "
                         "one the viewer and the README point at")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    V, F, Tall = load_modular(a.dir, a.cache)
    log(f"{a.stem}: modular surface {len(F):,} triangles")
    rows = []
    for tgt in [int(x) for x in a.targets.split(",")]:
        Vd, Fd = decimate(V, F, tgt)
        med, p90, p99 = error_mm(V, F, Vd, Fd)
        m = trimesh.Trimesh(Vd, Fd, process=False)
        # a stable name, so the viewer and the README do not chase the exact
        # triangle count: the first budget is _detail, the rest are _detail_hi
        name = f"{a.stem}_detail" if not rows else f"{a.stem}_detail_hi"
        m.export(out/f"{name}.stl")
        m.export(out/f"{name}.obj")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from sketchup_pack import write_dae
            write_dae(m, out/f"{name}.dae", name)
        except Exception as e:
            log(f"  (no dae: {e})")
        sz = (out/f"{name}.stl").stat().st_size/1e6
        log(f"  {len(Fd):,} tris -> median {med:.2f} mm, 90th {p90:.1f}, 99th {p99:.1f}"
            f"  ({sz:.0f} MB stl)")
        rows.append(dict(name=name, tris=int(len(Fd)), median_mm=med,
                         p90_mm=p90, p99_mm=p99, stl_mb=sz))
    (out/f"{a.stem}_detail.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
