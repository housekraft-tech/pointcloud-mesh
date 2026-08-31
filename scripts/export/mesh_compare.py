"""How far apart are two meshes, and how sharp is the second one.

Both directions matter and they say different things. raw -> rebuilt asks
whether anything in the scan has been LOST: a hole in the rebuild shows up
here and nowhere else. rebuilt -> raw asks whether anything has been INVENTED.
A rebuild can score well one way and badly the other, so neither number alone
is an answer.

Sharpness is the edge-length-weighted dihedral histogram: a rolled fillet is a
run of 5-40 degree edges where a real corner is a single edge over 60.
"""
import argparse

import numpy as np
import open3d as o3d
import trimesh


def load_ref(path, every=1):
    """A mesh OR a point cloud. mesh_compare could only read meshes, so every
    measurement it has ever produced was against a RECONSTRUCTION, not against
    the scan -- which is the error that made a 4.4 mm model look like 9.9 mm."""
    if str(path).lower().endswith(".las") or str(path).lower().endswith(".laz"):
        import laspy
        keep = []
        with laspy.open(path) as r:
            for ch in r.chunk_iterator(8_000_000):
                q = np.column_stack([ch.x, ch.y, ch.z]).astype(np.float32)
                keep.append(q[::every] if every > 1 else q)
        return np.vstack(keep)
    return trimesh.load(path, force="mesh")


def dist(src, dst, n=200000):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(dst.vertices),
                                  o3d.utility.Vector3iVector(dst.faces))))
    pts = src.sample(min(n, 200000))
    d = scene.compute_distance(o3d.core.Tensor(np.asarray(pts, np.float32))).numpy()
    return d * 1000.0


def dist_to_points(rtree, m, n=300000):
    """Model surface -> nearest scan point."""
    d, _ = rtree.query(m.sample(n), workers=-1)
    return d * 1000.0


def dist_from_points(P, m, n=300000):
    """Scan point -> model surface: what the model has LOST."""
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                  o3d.utility.Vector3iVector(m.faces))))
    s = P[np.linspace(0, len(P)-1, min(n, len(P))).astype(int)]
    return sc.compute_distance(o3d.core.Tensor(s.astype(np.float32))).numpy() * 1000.0


def buckets(m):
    ang = np.degrees(m.face_adjacency_angles)
    w = np.linalg.norm(m.vertices[m.face_adjacency_edges[:, 0]] -
                       m.vertices[m.face_adjacency_edges[:, 1]], axis=1)
    tot = w.sum()
    return (100*w[ang < 5].sum()/tot, 100*w[(ang >= 5) & (ang < 40)].sum()/tot,
            100*w[ang >= 60].sum()/tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rebuilt", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=1,
                    help="subsample the reference cloud. Leave at 1: taking every "
                         "10th point tripled the spacing and turned a 4.4 mm "
                         "result into 9.9 mm")
    a = ap.parse_args()

    raw = load_ref(a.raw, a.every)
    pointref = isinstance(raw, np.ndarray)
    if pointref:
        from scipy.spatial import cKDTree
        rtree = cKDTree(raw)
        print(f"reference: {len(raw):,} scan POINTS "
              f"(subsampling every {a.every} would inflate every distance -- "
              f"currently every {a.every})")
    else:
        print(f"raw: {len(raw.faces):,} tris, {raw.area:.0f} m2")
    for path in a.rebuilt:
        m = trimesh.load(path, force="mesh")
        f, r, c = buckets(m)
        print(f"\n{path}")
        ref_area = None if pointref else raw.area
        print(f"  {len(m.faces):,} tris, {m.area:.0f} m2"
              + ("" if ref_area is None else f" ({100*m.area/ref_area:.0f}% of raw)"))
        print(f"  flat<5 {f:.1f}%  rolled 5-40 {r:.1f}%  crease>60 {c:.1f}%")
        if pointref:
            x = dist_to_points(rtree, m)
            print(f"  model -> nearest scan point: median {np.median(x):7.2f} mm  "
                  f"90th {np.percentile(x, 90):7.1f}  <5mm {100*(x < 5).mean():.0f}%"
                  f"  <10mm {100*(x < 10).mean():.0f}%")
            y = dist_from_points(raw, m)
            print(f"  scan point -> model:         median {np.median(y):7.2f} mm  "
                  f"90th {np.percentile(y, 90):7.1f}  <5mm {100*(y < 5).mean():.0f}%"
                  f"  <10mm {100*(y < 10).mean():.0f}%")
        else:
            for name, s_, d_ in (("raw -> rebuilt", raw, m), ("rebuilt -> raw", m, raw)):
                x = dist(s_, d_)
                print(f"  {name}: median {np.median(x):7.2f} mm  "
                      f"90th {np.percentile(x, 90):7.1f}  <10mm {100*(x < 10).mean():.0f}%")


if __name__ == "__main__":
    main()
