"""The Poisson surface, closed and light enough to work with.

The box model is watertight but flat-faced. The relief model keeps every
arch, reveal and ripple but is an open surface. This is the third thing: the
Poisson mesh itself, decimated to a workable size and then SEALED, so it has
all the relief AND a real inside.

It is nearly there to begin with -- Poisson produces a closed surface and only
the density trim opens it, leaving about 0.11% of edges on a boundary. So the
work is: keep the one big shell, decimate, close what the trim opened, and
check the result rather than assume it.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import trimesh

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="poisson.npz")
    ap.add_argument("--out", required=True, help="output mesh (.glb/.obj/.ply)")
    ap.add_argument("--target", type=int, default=800_000, help="triangle budget")
    ap.add_argument("--refresh", action="store_true",
                    help="re-decimate instead of reusing the cached decimation")
    a = ap.parse_args()

    z = np.load(a.cache)
    V, T = z["V"], z["T"].astype(np.int64)
    m = trimesh.Trimesh(V, T, process=False)
    log(f"{len(m.faces):,} triangles in")

    # one shell: the floaters the trim leaves behind can never be closed
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
        log(f"{len(parts)} shells -> keeping the largest, {len(m.faces):,} triangles")

    tmp = Path(a.out).with_suffix(".decimated.ply")
    if tmp.exists() and not a.refresh:
        m = trimesh.load(tmp, process=False)
        log(f"reusing {tmp.name}: {len(m.faces):,} triangles")
    elif a.target and len(m.faces) > a.target:
        import fast_simplification
        Vd, Fd = fast_simplification.simplify(
            np.ascontiguousarray(m.vertices, np.float32),
            np.ascontiguousarray(m.faces, np.int32),
            target_reduction=1.0 - a.target/len(m.faces))
        m = trimesh.Trimesh(Vd, Fd, process=False)
        m.export(tmp)
        log(f"decimated to {len(m.faces):,}")

    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    log(f"welded: watertight={m.is_watertight}, "
        f"{len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1)):,} boundary edges")

    # trimesh's fill_holes only closes a hole it can span with one fan -- it
    # left 5,148 boundary edges here, because the trim tears are long and
    # ragged, not triangular. MeshFix closes arbitrary holes and removes the
    # self-intersections that come with them, which is the actual job.
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        m.remove_unreferenced_vertices()
        log(f"  simple fill: {len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1)):,} "
            f"boundary edges left")
    if not m.is_watertight:
        try:
            import pymeshfix
            fix = pymeshfix.MeshFix(np.asarray(m.vertices, float),
                                    np.asarray(m.faces, np.int32))
            fix.repair(joincomp=True, remove_smallest_components=False)
            m = trimesh.Trimesh(fix.points, fix.faces, process=False)
            log(f"  meshfix: {len(m.faces):,} triangles, watertight={m.is_watertight}")
        except ImportError:
            log("  pymeshfix not installed -- leaving the holes")

    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_winding(m)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    m.export(out)
    log(f"wrote {out}  {out.stat().st_size/1e6:.1f} MB")
    log(f"FINAL: {len(m.faces):,} triangles, watertight={m.is_watertight}, "
        f"winding consistent={m.is_winding_consistent}, volume={m.volume:.1f} m3")


if __name__ == "__main__":
    main()
