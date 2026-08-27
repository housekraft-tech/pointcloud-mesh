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


def dist(src, dst, n=200000):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(dst.vertices),
                                  o3d.utility.Vector3iVector(dst.faces))))
    pts = src.sample(min(n, 200000))
    d = scene.compute_distance(o3d.core.Tensor(np.asarray(pts, np.float32))).numpy()
    return d * 1000.0


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
    a = ap.parse_args()

    raw = trimesh.load(a.raw, force="mesh")
    print(f"raw: {len(raw.faces):,} tris, {raw.area:.0f} m2")
    for path in a.rebuilt:
        m = trimesh.load(path, force="mesh")
        f, r, c = buckets(m)
        print(f"\n{path}")
        print(f"  {len(m.faces):,} tris, {m.area:.0f} m2 "
              f"({100*m.area/raw.area:.0f}% of raw)")
        print(f"  flat<5 {f:.1f}%  rolled 5-40 {r:.1f}%  crease>60 {c:.1f}%")
        for name, s, d in (("raw -> rebuilt", raw, m), ("rebuilt -> raw", m, raw)):
            x = dist(s, d)
            print(f"  {name}: median {np.median(x):7.2f} mm  90th {np.percentile(x, 90):7.1f}"
                  f"  <10mm {100*(x < 10).mean():.0f}%")


if __name__ == "__main__":
    main()
