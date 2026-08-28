"""Give the point-fitted surfaces a shape, so they can be looked at.

The planes are equations and point sets; a viewer needs faces. Each surface's
own points are projected into its plane, rasterised at --cell, and the occupied
region is traced -- so the outline is the SUPPORT: the extent that was actually
measured, holes and all. A window opening stays an opening because no point
landed in it, and nothing is invented to make the outline tidy.

The mask is closed then opened before tracing. One stray cell of speckle
otherwise becomes a spike on the outline, and a one-cell gap between two runs
of points becomes a slit -- neither is a building.

This is deliberately NOT the rectangular model. It is the measured shape of
each fitted surface, which is the thing to look at before deciding what a
regularised version is allowed to move.
"""
import argparse

import cv2
import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.03, help="m")
    ap.add_argument("--close", type=int, default=3,
                    help="cells: bridge gaps in the point coverage")
    ap.add_argument("--open", type=int, default=2,
                    help="cells: drop speckle")
    ap.add_argument("--min-area", type=float, default=0.02, help="m2")
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D = z["n"], z["d"]
    pts, lab = z["pts"], z["lab"]
    V, F = [], []
    kept = 0
    for i in range(len(N)):
        P = pts[lab == i]
        if len(P) < 30:
            continue
        n = N[i] / np.linalg.norm(N[i])
        t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(n, t)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uu, vv = P @ u, P @ v
        lo = np.array([uu.min(), vv.min()]) - 3 * a.cell
        hi = np.array([uu.max(), vv.max()]) + 3 * a.cell
        w, h = np.maximum(np.ceil((hi - lo) / a.cell).astype(int), 2)
        if w * h > 6_000_000:
            continue
        m = np.zeros((h, w), np.uint8)
        ij = np.floor((np.c_[uu, vv] - lo) / a.cell).astype(int)
        m[np.clip(ij[:, 1], 0, h-1), np.clip(ij[:, 0], 0, w-1)] = 255
        k1 = np.ones((a.close*2+1,)*2, np.uint8)
        k2 = np.ones((a.open*2+1,)*2, np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)
        cont, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cont:
            continue
        hier = hier[0]
        for ci, c in enumerate(cont):
            if hier[ci][3] != -1:            # a hole; handled with its parent
                continue
            outer = c.reshape(-1, 2).astype(float)
            if len(outer) < 3:
                continue
            holes = [cont[k].reshape(-1, 2).astype(float)
                     for k in range(len(cont)) if hier[k][3] == ci
                     and len(cont[k]) >= 3]
            from shapely.geometry import Polygon
            try:
                poly = Polygon(outer, holes)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            for g in getattr(poly, "geoms", [poly]):
                if g.is_empty or g.area * a.cell**2 < a.min_area:
                    continue
                g = g.simplify(1.0)
                try:
                    vv2, ff2 = trimesh.creation.triangulate_polygon(g, engine="earcut")
                except Exception:
                    continue
                base = len(V)
                for x, y in vv2:
                    p = lo + np.array([x, y]) * a.cell
                    V.append(u * p[0] + v * p[1] + n * D[i])
                F.extend((np.array(ff2) + base).tolist())
                kept += 1

    out = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"{kept:,} faces from {len(N):,} surfaces -> "
          f"{len(out.faces):,} triangles, {out.area:.0f} m2 -> {a.out}")


if __name__ == "__main__":
    main()
