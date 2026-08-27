"""Put every face on a plane, and every vertex where its planes meet.

Snapping to axis planes takes the rolled edges from 31.5% of the mesh down to
5%, and then stops: what is left is not a fillet between two known surfaces, it
is surface the detector never covered -- 65% of the remaining roll has NEITHER
of its faces on a plane. Those are slanted soffits, the undersides of flights,
sills that are not quite level, and the scan's own noise.

So the planes are no longer assumed to lie along the axes. Face normals are
clustered into directions, offsets are clustered within each direction, and
every face that fits gets a plane. A vertex then sits at the meeting of the
planes of its own faces: one plane and it projects, two and it goes onto their
line, three and it goes to the corner -- solved as least squares so it works
for any directions, not just perpendicular ones.
"""
import argparse
from collections import defaultdict
import numpy as np
import trimesh


def cluster_dirs(N, A, tol_deg):
    """Group face normals into directions, biggest first."""
    order = np.argsort(-A)
    dirs, assign = [], np.full(len(N), -1)
    cos_tol = np.cos(np.radians(tol_deg))
    for i in order:
        if assign[i] >= 0:
            continue
        d = N[i]
        dot = N @ d
        near = (dot > cos_tol) & (assign < 0)
        if A[near].sum() < 1e-9:
            continue
        d = (N[near] * A[near, None]).sum(0)
        d /= max(np.linalg.norm(d), 1e-12)
        dot = N @ d
        near = (dot > cos_tol) & (assign < 0)
        assign[near] = len(dirs)
        dirs.append(d)
    return np.array(dirs), assign


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ang", type=float, default=7.0, help="deg: one direction")
    ap.add_argument("--off", type=float, default=0.012, help="m: one plane")
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    V = np.array(m.vertices, float); F = np.array(m.faces)
    print(f"in: {len(F):,} triangles")

    for it in range(a.passes):
        tm = trimesh.Trimesh(V, F, process=False)
        N, A = tm.face_normals, tm.area_faces
        C = tm.triangles.mean(axis=1)
        dirs, dassign = cluster_dirs(N, A, a.ang)
        planes = []                      # (normal, offset)
        fplane = np.full(len(F), -1)
        for di, d in enumerate(dirs):
            sel = np.flatnonzero(dassign == di)
            if not len(sel):
                continue
            t = C[sel] @ d
            order = np.argsort(t)
            ts, sels = t[order], sel[order]
            start = 0
            for k in range(1, len(ts)+1):
                if k == len(ts) or ts[k] - ts[start] > a.off:
                    grp = sels[start:k]
                    w = A[grp]
                    off = float(np.average(ts[start:k], weights=w))
                    fplane[grp] = len(planes)
                    planes.append((d, off))
                    start = k
        planes_n = np.array([p[0] for p in planes])
        planes_d = np.array([p[1] for p in planes])
        print(f"  pass {it+1}: {len(dirs)} directions, {len(planes):,} planes, "
              f"{100*(fplane >= 0).mean():.1f}% of faces assigned")

        # every vertex sits where its own faces' planes meet
        vp = defaultdict(dict)
        for fi in range(len(F)):
            pi = fplane[fi]
            if pi < 0:
                continue
            for vi in F[fi]:
                vp[vi][pi] = vp[vi].get(pi, 0.0) + A[fi]
        moved = 0
        for vi, pw in vp.items():
            top = sorted(pw.items(), key=lambda kv: -kv[1])[:3]
            Nn = planes_n[[p for p, _ in top]]
            dd = planes_d[[p for p, _ in top]]
            if len(top) == 1:
                v = V[vi] - Nn[0]*(V[vi] @ Nn[0] - dd[0])
            else:
                # least squares: closest point to V[vi] satisfying n.v = d
                lhs = np.vstack([np.eye(3), Nn*1e3])
                rhs = np.concatenate([V[vi], dd*1e3])
                v, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            if np.linalg.norm(v - V[vi]) > 1e-9:
                V[vi] = v; moved += 1
        print(f"           {moved:,} vertices placed on their planes")

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} triangles -> {a.out}")


if __name__ == "__main__":
    main()
