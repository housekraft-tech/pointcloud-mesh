"""Close the seams, so the junctions are joined and not just adjacent.

The clipped-plane rebuild puts every surface within about 3 mm of the scan, but
it emits each plane as its own polygon, so the model is 5,755 islands with
8,922 m of open edge. Where a wall meets a floor the two polygons both run to
the intersection line and stop -- and because each was triangulated on its own,
the vertices along that line do not line up. The result is a hairline crack you
can see from inside the room, which is what "there is a gap near the floor"
is: not a hole in the geometry, a seam in the topology.

Closing it is two passes, repeated until nothing moves:

  * WELD. Vertices within --tol are one vertex. This alone joins the corners
    where two triangulations happened to agree.
  * SPLIT. A vertex sitting partway along someone else's open edge is a
    T-junction: the edge passes through it but does not use it, so the two
    sides stay separate. That edge is cut at the vertex. The face is then
    re-fanned from its own centroid rather than from a corner, because fanning
    a triangle with points inserted on its edges from a corner produces
    zero-area slivers, and a sliver is a crack that has learnt to hide.

Only OPEN edges are considered. An edge already shared by two faces is joined
and splitting it would only add work.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def corner(pts, planes):
    """Where a welded cluster belongs: the intersection of the planes it is on.

    Averaging the cluster instead pulls the corner off both planes by up to the
    weld tolerance, which tilts every triangle that touches it -- that is how a
    stitch pass turns 95% flat into 83% flat while claiming to have changed
    nothing.
    """
    c = pts.mean(0)
    if len(planes) < 2:
        if len(planes) == 1:
            n, d = planes[0]
            return c - n * (c @ n - d)
        return c
    Nn = np.array([p[0] for p in planes])
    dd = np.array([p[1] for p in planes])
    keep = [0]
    for k in range(1, len(Nn)):
        if all(abs(Nn[k] @ Nn[j]) < 0.985 for j in keep):
            keep.append(k)
    Nn, dd = Nn[keep], dd[keep]
    lhs = np.vstack([np.eye(3) * 1e-2, Nn])
    rhs = np.concatenate([c * 1e-2, dd])
    x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return x


def vertex_planes(V, F):
    """The distinct planes meeting at each vertex, from its own faces."""
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    L = np.linalg.norm(n, axis=1)
    ok = L > 1e-12
    n[ok] /= L[ok][:, None]
    d = (n * tri[:, 0]).sum(1)
    per = defaultdict(list)
    for fi, f in enumerate(F):
        if not ok[fi]:
            continue
        for vi in f:
            per[int(vi)].append(fi)
    out = {}
    for vi, fs in per.items():
        got = []
        for fi in sorted(fs, key=lambda k: -1.0):
            if all(abs(n[fi] @ g[0]) < 0.999 or abs(d[fi] - np.sign(n[fi] @ g[0])*g[1]) > 0.003
                   for g in got):
                got.append((n[fi], d[fi]))
            if len(got) == 3:
                break
        out[vi] = got
    return out


def weld(V, F, tol):
    """Vertices within tol of each other become one, placed on their planes."""
    vp = vertex_planes(V, F)
    tree = cKDTree(V)
    parent = np.arange(len(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(tol, output_type="ndarray"):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    root = np.array([find(i) for i in range(len(V))])
    keep, inv = np.unique(root, return_inverse=True)
    members = defaultdict(list)
    for i, k in enumerate(inv):
        members[int(k)].append(i)
    nv = np.zeros((len(keep), 3))
    for k, mem in members.items():
        if len(mem) == 1:
            nv[k] = V[mem[0]]
            continue
        pl = []
        for i in mem:
            pl.extend(vp.get(i, []))
        nv[k] = corner(V[mem], pl[:6])
    nf = inv[F]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    return nv, nf


def open_edges(F):
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    u, idx, c = np.unique(e, axis=0, return_index=True, return_counts=True)
    return u[c == 1]


def split(V, F, tol):
    """Cut every open edge at any vertex sitting on it. Returns the new faces."""
    be = open_edges(F)
    if not len(be):
        return V, F, 0
    onb = np.unique(be)
    tree = cKDTree(V[onb])

    # for each open edge, which vertices lie along its interior
    cuts = defaultdict(list)
    hits = 0
    for a, b in be:
        pa, pb = V[a], V[b]
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        mid, r = (pa + pb) / 2, L / 2 + tol
        for k in tree.query_ball_point(mid, r):
            vi = int(onb[k])
            if vi == a or vi == b:
                continue
            t = float((V[vi] - pa) @ d) / (L * L)
            if not (tol / L < t < 1 - tol / L):
                continue
            foot = pa + t * d
            if np.linalg.norm(foot - V[vi]) > tol:
                continue
            # onto the edge, exactly. A split point that is merely NEAR the
            # edge is off the face's plane, and inserting it tilts every
            # triangle in the fan -- the seam closes and the surface stops
            # being flat, which trades one defect for a worse one.
            V[vi] = foot
            cuts[(int(a), int(b))].append((t, vi))
            hits += 1
    if not hits:
        return V, F, 0

    V = np.array(V, float)
    V = list(V)
    out = []
    for f in F:
        ins = []                       # the face's boundary, corner by corner
        touched = False
        for i in range(3):
            x, y = int(f[i]), int(f[(i + 1) % 3])
            ins.append(x)
            key = (x, y) if x < y else (y, x)
            cc = cuts.get(key)
            if not cc:
                continue
            seq = sorted(cc, key=lambda tv: tv[0] if key == (x, y) else -tv[0])
            ins.extend(vi for _, vi in seq)
            touched = True
        if not touched:
            out.append(list(f))
            continue
        # fan from the centroid: a corner fan would make zero-area slivers
        c = len(V)
        V.append(np.mean([V[i] for i in ins], axis=0))
        for i in range(len(ins)):
            out.append([ins[i], ins[(i + 1) % len(ins)], c])
    return np.array(V), np.array(out), hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.008, help="m")
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    V, F = np.array(m.vertices, float), np.array(m.faces)

    def report(tag):
        mm = trimesh.Trimesh(V, F, process=False)
        be = open_edges(F)
        L = np.linalg.norm(V[be[:, 0]] - V[be[:, 1]], axis=1).sum() if len(be) else 0
        cc = trimesh.graph.connected_components(mm.face_adjacency,
                                                nodes=np.arange(len(F)))
        print(f"{tag}: {len(F):,} tris, {mm.area:.0f} m2, "
              f"{len(cc):,} pieces, {L:.0f} m open edge")

    report("in ")
    for r in range(a.rounds):
        V, F = weld(V, F, a.tol)
        V, F, hits = split(V, F, a.tol)
        V, F = weld(V, F, a.tol)
        report(f"r{r+1}")
        if not hits:
            break

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out -> {a.out}")


if __name__ == "__main__":
    main()
