"""Grow the surface into real regions, put each where its points say, and let
the regions meet in an edge.

Clustering normals and offsets independently produced 50,758 planes -- one per
triangle in the noisy parts -- and faceted the model. A surface is not a set of
triangles that happen to share an orientation: it is a CONNECTED patch. So the
mesh is grown into regions across face adjacency, a region only accepting a
neighbour that agrees with the plane it has so far.

Each region's plane is then the area-weighted mean of the points in it, which
is what makes the position accurate: the scan's noise is 1.7 mm per point, so
the mean over ten thousand points is known to a hundredth of a millimetre. The
surface comes out straight because every vertex is projected onto that plane,
and the junctions come out sharp because a vertex belonging to two regions is
put where the two planes intersect.
"""
import argparse
from collections import defaultdict, deque
import numpy as np
import trimesh


def grow(mesh, ang, dist, min_area):
    N = mesh.face_normals; A = mesh.area_faces
    C = mesh.triangles.mean(axis=1)
    adj = defaultdict(list)
    for a_, b_ in mesh.face_adjacency:
        adj[a_].append(b_); adj[b_].append(a_)
    cos_tol = np.cos(np.radians(ang))
    label = np.full(len(N), -1)
    regions = []
    for seed in np.argsort(-A):
        if label[seed] >= 0:
            continue
        n = N[seed].copy(); d = float(C[seed] @ n)
        members = [seed]; label[seed] = len(regions)
        q = deque(adj[seed])
        wsum = A[seed]
        while q:
            f = q.popleft()
            if label[f] >= 0:
                continue
            if N[f] @ n < cos_tol:
                continue
            if abs(C[f] @ n - d) > dist:
                continue
            label[f] = len(regions); members.append(f)
            # keep the plane honest as the region grows: area-weighted mean
            wsum += A[f]
            n = n + (N[f] - n) * (A[f]/wsum)
            n /= max(np.linalg.norm(n), 1e-12)
            d = d + (float(C[f] @ n) - d) * (A[f]/wsum)
            q.extend(adj[f])
        regions.append((members, n, d))
    keep = [i for i, (mem, _, _) in enumerate(regions) if A[mem].sum() >= min_area]
    return label, regions, set(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ang", type=float, default=12.0, help="deg")
    ap.add_argument("--dist", type=float, default=0.020, help="m")
    ap.add_argument("--min-area", type=float, default=0.05, help="m2")
    ap.add_argument("--merge", action="store_true",
                    help="merge regions that lie on the same plane")
    ap.add_argument("--merge-ang", type=float, default=6.0, help="deg")
    ap.add_argument("--merge-off", type=float, default=0.020, help="m")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    V = np.array(m.vertices, float); F = np.array(m.faces)
    print(f"in: {len(F):,} triangles, {m.area:.0f} m2")
    label, regions, keep = grow(m, a.ang, a.dist, a.min_area)
    A = m.area_faces
    kept_area = sum(A[regions[i][0]].sum() for i in keep)
    print(f"{len(regions):,} regions grown; {len(keep):,} are over {a.min_area} m2 "
          f"and carry {100*kept_area/m.area:.1f}% of the area")

    # each kept region: the plane its own points average to
    planes = {}
    for i in keep:
        mem, n, d = regions[i]
        vs = np.unique(F[mem])
        w = np.zeros(len(V)); np.add.at(w, F[mem].ravel(), np.repeat(A[mem], 3))
        ww = w[vs]
        c = np.average(V[vs], axis=0, weights=ww)
        X = (V[vs] - c) * np.sqrt(ww)[:, None]
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        nn = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
        if nn @ n < 0:
            nn = -nn
        planes[i] = (nn, float(c @ nn), len(vs))
    print(f"planes fitted from {sum(p[2] for p in planes.values()):,} vertex samples")

    # Merge regions that are the same plane.
    #
    # A wall is not one region: doorways, pipes and noise cut it into twenty,
    # and each gets its own slightly different plane, so the model keeps a
    # seam wherever two of them meet. They are the same wall, so they should
    # be the same plane -- and the merged plane, fitted over all of their
    # points at once, is better known than any of them alone.
    if a.merge:
        ids = list(planes.keys())
        cos_tol = np.cos(np.radians(a.merge_ang))
        groups = []
        for i in sorted(ids, key=lambda k: -planes[k][2]):
            n_i, d_i, w_i = planes[i]
            placed = False
            for g in groups:
                n_g, d_g, w_g, mem = g
                if abs(n_i @ n_g) < cos_tol:
                    continue
                sgn = 1.0 if n_i @ n_g > 0 else -1.0
                if abs(sgn*d_i - d_g) > a.merge_off:
                    continue
                nn = n_g*w_g + sgn*n_i*w_i
                nn /= max(np.linalg.norm(nn), 1e-12)
                g[0] = nn
                g[1] = (d_g*w_g + sgn*d_i*w_i)/(w_g + w_i)
                g[2] = w_g + w_i
                mem.append(i)
                placed = True
                break
            if not placed:
                groups.append([n_i, d_i, w_i, [i]])
        remap = {}
        for gi, (n_g, d_g, w_g, mem) in enumerate(groups):
            for i in mem:
                remap[i] = (n_g, d_g, w_g)
        planes = {i: remap[i] for i in planes}
        big = sum(1 for g in groups if g[2] > 200)
        print(f"regions merged into {len(groups):,} distinct planes "
              f"({big} of them carrying more than 200 vertices)")

    vp = defaultdict(dict)
    for fi in range(len(F)):
        li = label[fi]
        if li in planes:
            for vi in F[fi]:
                vp[vi][li] = vp[vi].get(li, 0.0) + A[fi]
    moved = 0
    for vi, pw in vp.items():
        top = sorted(pw.items(), key=lambda kv: -kv[1])[:3]
        Nn = np.array([planes[p][0] for p, _ in top])
        dd = np.array([planes[p][1] for p, _ in top])
        if len(top) == 1:
            v = V[vi] - Nn[0]*(V[vi] @ Nn[0] - dd[0])
        else:
            lhs = np.vstack([np.eye(3), Nn*1e3]); rhs = np.concatenate([V[vi], dd*1e3])
            v, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if np.linalg.norm(v - V[vi]) > 1e-9:
            V[vi] = v; moved += 1
    print(f"{moved:,} vertices put on their region's plane (or where planes meet)")
    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} triangles -> {a.out}")


if __name__ == "__main__":
    main()
