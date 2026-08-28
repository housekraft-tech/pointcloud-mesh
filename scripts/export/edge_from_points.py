"""The edge rebuild, sourced from the POINTS instead of from a reconstruction.

The existing edge model is excellent -- 3.17 mm, sharp junctions -- but it is
3.17 mm from the POISSON MESH, and that mesh is itself about 9 mm off the points
where they support it and entirely invented across 30% of its area. It also
lives in a different frame from the LAS, which is how the arrangement came to be
overlaid 14 degrees out.

So this rebuilds the same idea on the plane set fitted directly to the points:

  * a surface's shape starts as its SUPPORT -- where points actually landed,
    rasterised and traced, holes and all;
  * wherever another surface is adjacent, that traced boundary is thrown away
    and replaced by the line where the two planes intersect. The traced edge
    only survives where nothing else is there, because a scan boundary is the
    scan's shape, not the building's;
  * adjacency is decided by the points too: two surfaces touch where their own
    points come within --touch of each other, and the contact stretch is where
    that happens, not the whole line.

Everything is in the LAS frame, so it is directly comparable with the
arrangement built from the same planes.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union


def frame_of(n):
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def support_polygon(P, u, v, cell, close, open_, min_area):
    uu, vv = P @ u, P @ v
    lo = np.array([uu.min(), vv.min()]) - 3*cell
    hi = np.array([uu.max(), vv.max()]) + 3*cell
    w, h = np.maximum(np.ceil((hi - lo)/cell).astype(int), 2)
    if w*h > 8_000_000:
        return None, lo
    m = np.zeros((h, w), np.uint8)
    ij = np.floor((np.c_[uu, vv] - lo)/cell).astype(int)
    m[np.clip(ij[:, 1], 0, h-1), np.clip(ij[:, 0], 0, w-1)] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close*2+1,)*2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_*2+1,)*2, np.uint8))
    cont, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return None, lo
    hier = hier[0]
    polys = []
    for ci, c in enumerate(cont):
        if hier[ci][3] != -1 or len(c) < 3:
            continue
        holes = [cont[k].reshape(-1, 2) for k in range(len(cont))
                 if hier[k][3] == ci and len(cont[k]) >= 3]
        try:
            g = Polygon(c.reshape(-1, 2), holes)
            if not g.is_valid:
                g = g.buffer(0)
        except Exception:
            continue
        if not g.is_empty:
            polys.append(g)
    if not polys:
        return None, lo
    g = unary_union(polys)
    # into metres, in the plane's own frame
    from shapely.affinity import affine_transform
    g = affine_transform(g, [cell, 0, 0, cell, lo[0], lo[1]])
    if g.area < min_area:
        return None, lo
    return g, lo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.03, help="m")
    ap.add_argument("--touch", type=float, default=0.10,
                    help="m: two surfaces are adjacent where their points come "
                         "this close")
    ap.add_argument("--bite", type=float, default=0.05,
                    help="m: material this close to the intersection line, on "
                         "the wrong side, is the rounded band")
    ap.add_argument("--grow", type=float, default=0.15,
                    help="m: how far a surface may be extended to reach a seam")
    ap.add_argument("--pad", type=float, default=0.05, help="m")
    ap.add_argument("--min-area", type=float, default=0.02, help="m2")
    ap.add_argument("--weld", type=float, default=0.006, help="m")
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]
    member = {i: pts[lab == i] for i in range(len(N))}
    trees = {i: cKDTree(P) for i, P in member.items() if len(P) > 30}
    print(f"{len(N):,} surfaces, {len(trees):,} with enough points")

    # adjacency and contact, from the points
    adj = defaultdict(set)
    contact = {}
    ids = sorted(trees)
    for a_i, i in enumerate(ids):
        Pi = member[i]
        for j in ids[a_i+1:]:
            if abs(N[i] @ N[j]) > 0.98:
                continue
            pairs = trees[i].query_ball_tree(trees[j], a.touch)
            hit = [k for k, v in enumerate(pairs) if v]
            if len(hit) < 20:
                continue
            adj[i].add(j)
            adj[j].add(i)
            contact[(i, j)] = contact[(j, i)] = Pi[hit]
    print(f"{sum(len(v) for v in adj.values())//2:,} adjacent pairs")

    V, F = [], []
    vp = []
    kept = 0
    for i in ids:
        n = N[i]
        u, v = frame_of(n)
        patch, _ = support_polygon(member[i], u, v, a.cell, 3, 2, a.min_area)
        if patch is None:
            continue
        for j in adj[i]:
            dirv = np.cross(n, N[j])
            L = np.linalg.norm(dirv)
            if L < 1e-6:
                continue
            dirv /= L
            A = np.vstack([n, N[j], dirv])
            try:
                pt = np.linalg.solve(A, np.array([D[i], D[j], 0.0]))
            except np.linalg.LinAlgError:
                continue
            p2 = np.array([pt @ u, pt @ v])
            d2 = np.array([dirv @ u, dirv @ v])
            Ld = np.linalg.norm(d2)
            if Ld < 1e-9:
                continue
            d2 /= Ld
            nrm2 = np.array([-d2[1], d2[0]])
            cen = np.array(patch.representative_point().coords[0])
            s = np.sign((cen - p2) @ nrm2)
            if s == 0:
                continue
            C = contact[(i, j)]
            t = (C - pt) @ dirv
            t0, t1 = float(t.min()) - a.pad, float(t.max()) + a.pad
            b0, b1 = p2 + d2*t0, p2 + d2*t1
            far = 1e3
            half = Polygon([b0, b1, b1 + nrm2*s*far, b0 + nrm2*s*far])
            strip = Polygon([b0 - nrm2*a.bite, b1 - nrm2*a.bite,
                             b1 + nrm2*a.bite, b0 + nrm2*a.bite])
            cut = strip.difference(half)
            grown = strip.intersection(half).intersection(patch.buffer(a.grow))
            new = unary_union([patch.difference(cut), grown])
            if new.is_empty or new.area < 0.5*patch.area:
                continue
            patch = new
        for g in getattr(patch, "geoms", [patch]):
            if g.is_empty or g.area < a.min_area:
                continue
            g = g.simplify(0.004)
            try:
                vv, ff = trimesh.creation.triangulate_polygon(g, engine="earcut")
            except Exception:
                continue
            base = len(V)
            for x, y in vv:
                V.append(u*x + v*y + n*D[i])
                vp.append(i)
            F.extend((np.array(ff) + base).tolist())
            kept += 1

    P = np.array(V)
    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for x, y in tree.query_pairs(a.weld, output_type="ndarray"):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    grp = defaultdict(list)
    for k in range(len(P)):
        grp[find(k)].append(k)
    for mem in grp.values():
        if len(mem) < 2:
            continue
        pl = sorted({vp[k] for k in mem})
        c = P[mem].mean(0)
        if len(pl) >= 2:
            Nn = np.array([N[p] for p in pl])
            dd = np.array([D[p] for p in pl])
            keep = [0]
            for k in range(1, len(Nn)):
                if all(abs(Nn[k] @ Nn[q]) < 0.985 for q in keep):
                    keep.append(k)
            Nn, dd = Nn[keep], dd[keep]
            lhs = np.vstack([np.eye(3)*1e-2, Nn])
            rhs = np.concatenate([c*1e-2, dd])
            c, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        for k in mem:
            P[k] = c

    out = trimesh.Trimesh(P, np.array(F), process=False)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"{kept:,} faces -> {len(out.faces):,} triangles, {out.area:.0f} m2 "
          f"-> {a.out}")


if __name__ == "__main__":
    main()
