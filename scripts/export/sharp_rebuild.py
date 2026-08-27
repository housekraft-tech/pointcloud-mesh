"""Clean surfaces and true edges, built the way the literature says to.

Six attempts at this failed in instructive ways, and the diagnosis was clear:

  * growing regions on RAW face normals fragments, because decimation leaves
    slivers whose normals are unstable well beyond the point noise;
  * refitting a three-face region by SVD gives a garbage normal, after which
    every candidate fails the angle test and the region freezes at three faces
    -- which is exactly the 3-faces-per-region statistic that came out;
  * deciding membership by DISTANCE means the fillet band belongs to nothing,
    so it can never be sharpened. Membership has to be decided by a vote that
    prefers agreeing with the neighbours.

So: filter the normals first, grow from flat seeds with the acceptance test on
the candidate's vertices, refit only when a region doubles, merge regions that
lie on one plane, then let each unlabelled face take the label its neighbours
mostly carry -- which pulls the fillet strip onto one side or the other. Each
plane is refitted robustly, ignoring the fillet vertices as outliers, and every
vertex is placed where its own planes intersect.
"""
import argparse
from collections import defaultdict, deque

import numpy as np
import trimesh


def filter_normals(mesh, iters=12, sigma_r=0.35):
    """Bilateral normal filtering: clean normals without touching geometry."""
    N = mesh.face_normals.copy()
    A = mesh.area_faces
    C = mesh.triangles.mean(axis=1)
    adj = defaultdict(list)
    for a_, b_ in mesh.face_adjacency:
        adj[a_].append(b_)
        adj[b_].append(a_)
    nbr = [np.array(adj[i], dtype=np.int64) if adj[i] else np.array([], np.int64)
           for i in range(len(N))]
    fa = mesh.face_adjacency
    sigma_s = float(np.median(np.linalg.norm(C[fa[:, 0]] - C[fa[:, 1]], axis=1))) * 2.0
    for _ in range(iters):
        M = N.copy()
        for i in range(len(N)):
            k = nbr[i]
            if not len(k):
                continue
            ws = np.exp(-np.sum((C[k] - C[i]) ** 2, axis=1) / (2 * sigma_s ** 2))
            wr = np.exp(-np.sum((N[k] - N[i]) ** 2, axis=1) / (2 * sigma_r ** 2))
            w = A[k] * ws * wr
            v = (N[k] * w[:, None]).sum(0) + N[i] * A[i]
            n = np.linalg.norm(v)
            if n > 1e-12:
                M[i] = v / n
        N = M
    return N


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ang", type=float, default=20.0, help="deg")
    ap.add_argument("--dist", type=float, default=0.005, help="m: about 3 sigma")
    ap.add_argument("--merge-ang", type=float, default=4.0)
    ap.add_argument("--merge-off", type=float, default=0.004)
    ap.add_argument("--smooth", type=int, default=3, help="label smoothing rounds")
    ap.add_argument("--reach", type=float, default=0.06,
                    help="m: how far a face may be from the plane it is voted onto")
    ap.add_argument("--min-area", type=float, default=0.01, help="m2")
    ap.add_argument("--save-labels", default=None,
                    help="write the segmentation for edge_rebuild.py")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    V = np.array(m.vertices, float)
    V0 = V.copy()                     # before projection: the edge rebuild wants both
    F = np.array(m.faces)
    A = m.area_faces
    print(f"in: {len(F):,} triangles, {m.area:.0f} m2")

    NF = filter_normals(m)
    print("normals filtered")

    adj = defaultdict(list)
    for x, y in m.face_adjacency:
        adj[x].append(y)
        adj[y].append(x)

    # seed order: flattest neighbourhoods first, not face order
    var = np.zeros(len(F))
    for i in range(len(F)):
        k = adj[i]
        var[i] = 0.0 if not k else float(np.mean(1.0 - NF[k] @ NF[i]))
    order = np.lexsort((-A, var))

    cos_tol = np.cos(np.radians(a.ang))
    label = np.full(len(F), -1)
    regions = []
    for seed in order:
        if label[seed] >= 0:
            continue
        mem = [seed]
        label[seed] = len(regions)
        n = NF[seed].copy()
        d = float(V[F[seed]].mean(0) @ n)
        next_refit = 8
        q = deque(adj[seed])
        while q:
            f = q.popleft()
            if label[f] >= 0 or NF[f] @ n < cos_tol:
                continue
            if np.abs(V[F[f]] @ n - d).max() > a.dist:   # vertices, not the centroid
                continue
            label[f] = len(regions)
            mem.append(f)
            q.extend(adj[f])
            if len(mem) >= next_refit:                   # refit only on doubling
                next_refit *= 2
                vs = np.unique(F[mem])
                c = V[vs].mean(0)
                _, _, vt = np.linalg.svd(V[vs] - c, full_matrices=False)
                nn = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
                if nn @ n < 0:
                    nn = -nn
                n, d = nn, float(c @ nn)
        regions.append(mem)
    big = [i for i, mem in enumerate(regions) if A[mem].sum() >= a.min_area]
    print(f"{len(regions):,} regions, {len(big):,} over {a.min_area} m2 "
          f"({100*sum(A[regions[i]].sum() for i in big)/m.area:.1f}% of area)")

    def fit(mem):
        vs = np.unique(F[mem])
        c = V[vs].mean(0)
        _, _, vt = np.linalg.svd(V[vs] - c, full_matrices=False)
        n = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
        d = float(c @ n)
        for _ in range(2):                               # robust: drop the outliers
            r = np.abs(V[vs] @ n - d)
            keep = r < max(2.5 * float(np.median(r)), 0.002)
            if keep.sum() < 8:
                break
            c = V[vs[keep]].mean(0)
            _, _, vt = np.linalg.svd(V[vs[keep]] - c, full_matrices=False)
            n = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
            d = float(c @ n)
        return n, d

    planes = {i: fit(regions[i]) for i in big}

    cos_m = np.cos(np.radians(a.merge_ang))
    groups, gof = [], {}
    for i in sorted(big, key=lambda k: -A[regions[k]].sum()):
        n_i, d_i = planes[i]
        w_i = float(A[regions[i]].sum())
        hit = False
        for gi, (n_g, d_g, w_g) in enumerate(groups):
            s = 1.0 if n_i @ n_g > 0 else -1.0
            if abs(n_i @ n_g) >= cos_m and abs(s * d_i - d_g) <= a.merge_off:
                nn = n_g * w_g + s * n_i * w_i
                groups[gi] = (nn / max(np.linalg.norm(nn), 1e-12),
                              (d_g * w_g + s * d_i * w_i) / (w_g + w_i), w_g + w_i)
                gof[i] = gi
                hit = True
                break
        if not hit:
            gof[i] = len(groups)
            groups.append((n_i, d_i, w_i))
    print(f"merged into {len(groups):,} planes")

    flab = np.full(len(F), -1)
    for i in big:
        flab[regions[i]] = gof[i]
    for _ in range(a.smooth):
        upd = flab.copy()
        for f in np.flatnonzero(flab < 0):
            k = [flab[x] for x in adj[f] if flab[x] >= 0]
            if not k:
                continue
            vals, cnt = np.unique(k, return_counts=True)
            best = int(vals[np.argmax(cnt)])
            n_b, d_b, _ = groups[best]
            if np.abs(V[F[f]] @ n_b - d_b).max() < a.reach:
                upd[f] = best
        flab = upd
    print(f"{100*(flab >= 0).mean():.1f}% of faces carry a plane after smoothing")

    vp = defaultdict(dict)
    for f in np.flatnonzero(flab >= 0):
        for vi in F[f]:
            vp[vi][flab[f]] = vp[vi].get(flab[f], 0.0) + A[f]
    moved = 0
    for vi, pw in vp.items():
        top = sorted(pw.items(), key=lambda kv: -kv[1])[:3]
        Nn = np.array([groups[p][0] for p, _ in top])
        dd = np.array([groups[p][1] for p, _ in top])
        if len(top) > 1:
            g = np.abs(Nn @ Nn.T) - np.eye(len(Nn))
            if g.max() > 0.985:                # near-parallel: ill-conditioned
                Nn, dd, top = Nn[:1], dd[:1], top[:1]
        if len(top) == 1:
            v = V[vi] - Nn[0] * (V[vi] @ Nn[0] - dd[0])
        else:
            lhs = np.vstack([np.eye(3), Nn * 1e3])
            rhs = np.concatenate([V[vi], dd * 1e3])
            v, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if np.linalg.norm(v - V[vi]) > 1e-9:
            V[vi] = v
            moved += 1
    print(f"{moved:,} vertices placed on their planes")

    if a.save_labels:
        # the segmentation itself, so the edge rebuild can work from it rather
        # than segmenting a second time
        np.savez_compressed(a.save_labels, V=V0, F=F, flab=flab,
                            plane_n=np.array([g[0] for g in groups]),
                            plane_d=np.array([g[1] for g in groups]))
        print(f"labels -> {a.save_labels}")

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} triangles -> {a.out}")


if __name__ == "__main__":
    main()
