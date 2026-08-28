"""Faces made of intersection lines, with support only choosing which to keep.

Every previous attempt traced a surface's outline from where its points stopped
and then tried to tidy the result. That is why they came out jagged: the outline
was the scan's shape, not the building's, and no amount of rectilinearising or
snapping fixes a boundary that was never in the right place.

Here a plane has no outline of its own. Its neighbours' intersection lines are
drawn IN the plane, they cut it into cells, and the face is whichever cells the
points support. So every boundary is a plane-plane intersection by construction
and every corner is a three-plane point -- computed from the same three
equations no matter which of the three faces asks for it, so the corners are
shared exactly rather than welded by proximity afterwards.

Support is demoted to the only job it can do honestly: deciding whether a cell
is surface or a hole. A window is a cell with no points in it. Unscanned wall
is a cell with no points in it too, and the two are told apart by whether the
cell is enclosed by supported cells -- an opening is surrounded by wall, a
missing corner of the scan is on the boundary.

Geometry comes from the planes (2.82 mm RMS, offsets to 0.019 mm). No vertex of
any reconstruction enters this mesh.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union


def frame_of(n):
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--touch", type=float, default=0.12,
                    help="m: two planes are adjacent where their points come "
                         "this close")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="m: how far the face may reach past its own points")
    ap.add_argument("--cell", type=float, default=0.05,
                    help="m: grid for asking whether a cell has support")
    ap.add_argument("--support", type=float, default=0.35,
                    help="fraction of a cell's area that must carry points")
    ap.add_argument("--min-area", type=float, default=0.01, help="m2")
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]
    member = {i: pts[lab == i] for i in range(len(N)) if (lab == i).sum() > 60}
    trees = {i: cKDTree(P) for i, P in member.items()}
    ids = sorted(member)
    print(f"{len(ids):,} planes with points")

    adj = defaultdict(set)
    for k, i in enumerate(ids):
        for j in ids[k+1:]:
            if abs(N[i] @ N[j]) > 0.98:
                continue
            if trees[i].count_neighbors(trees[j], a.touch) > 30:
                adj[i].add(j)
                adj[j].add(i)
    print(f"{sum(len(v) for v in adj.values())//2:,} adjacent pairs")

    V, F, kept, holes = [], [], 0, 0
    for i in ids:
        n, d = N[i], D[i]
        u, v = frame_of(n)
        P = member[i]
        uu, vv = P @ u, P @ v
        lo = np.array([uu.min(), vv.min()]) - a.pad
        hi = np.array([uu.max(), vv.max()]) + a.pad
        fr = box(lo[0], lo[1], hi[0], hi[1])
        R = float(np.linalg.norm(hi - lo))
        lines = []
        for j in adj[i]:
            dirv = np.cross(n, N[j])
            L = np.linalg.norm(dirv)
            if L < 1e-9:
                continue
            dirv /= L
            A = np.vstack([n, N[j], dirv])
            try:
                p0 = np.linalg.solve(A, np.array([d, D[j], 0.0]))
            except np.linalg.LinAlgError:
                continue
            p2 = np.array([p0 @ u, p0 @ v])
            d2 = np.array([dirv @ u, dirv @ v])
            nd = np.linalg.norm(d2)
            if nd < 1e-9:
                continue
            d2 /= nd
            g = LineString([p2 - d2*R, p2 + d2*R]).intersection(fr)
            if g.geom_type == "LineString" and g.length > 0.05:
                lines.append(g)
        cells = list(polygonize(unary_union(lines + [fr.boundary])))
        if not cells:
            cells = [fr]
        tree = trees[i]
        good = []
        for c in cells:
            if c.area < a.min_area:
                continue
            x0, y0, x1, y1 = c.bounds
            gx = np.arange(x0 + a.cell/2, x1, a.cell)
            gy = np.arange(y0 + a.cell/2, y1, a.cell)
            if not len(gx) or not len(gy):
                gx, gy = np.array([(x0+x1)/2]), np.array([(y0+y1)/2])
            G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
            inside = np.array([c.contains(Polygon([(p[0]-1e-6, p[1]-1e-6),
                                                   (p[0]+1e-6, p[1]-1e-6),
                                                   (p[0], p[1]+1e-6)]).centroid)
                               for p in G]) if len(G) < 4000 else np.ones(len(G), bool)
            G = G[inside]
            if not len(G):
                continue
            W = (u[None, :]*G[:, :1] + v[None, :]*G[:, 1:2] + n*d)
            hit = tree.query_ball_point(W, a.cell*0.75, return_length=True) > 0
            if hit.mean() >= a.support:
                good.append(c)
            else:
                holes += 1
        if not good:
            continue
        face = unary_union(good)
        for g in getattr(face, "geoms", [face]):
            if g.is_empty or g.area < a.min_area:
                continue
            try:
                vv2, ff2 = trimesh.creation.triangulate_polygon(g, engine="earcut")
            except Exception:
                continue
            b = len(V)
            for x, y in vv2:
                V.append(u*x + v*y + n*d)
            F.extend((np.array(ff2)+b).tolist())
            kept += 1

    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.export(a.out)
    ang = np.degrees(m.face_adjacency_angles)
    w = np.linalg.norm(m.vertices[m.face_adjacency_edges[:, 0]] -
                       m.vertices[m.face_adjacency_edges[:, 1]], axis=1)
    print(f"{kept:,} faces, {holes:,} cells rejected for lack of support")
    print(f"out: {len(m.faces):,} tris, {m.area:.0f} m2, "
          f"flat<5deg {100*w[ang < 5].sum()/w.sum():.1f}%, "
          f"rolled {100*w[(ang >= 5) & (ang < 40)].sum()/w.sum():.1f}% -> {a.out}")


if __name__ == "__main__":
    main()
