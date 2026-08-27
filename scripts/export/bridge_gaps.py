"""Build the return face where a surface steps, instead of leaving a slot.

The dark stripe running along a wall is not a seam and not a tolerance: it is
a STEP. Where a wall thickens, or a recess starts, the scan has two parallel
surfaces 50 to 200 mm apart. Two parallel planes have no intersection line, so
the clipped-plane rebuild has nothing to clip against and simply stops at the
edge of each -- leaving a slot exactly as wide as the step is deep, running the
whole length of the wall.

What belongs there is the return: the little perpendicular face that carries
the surface from one offset to the other. So every open edge looks across the
gap for an open edge facing it, and the strip between them is filled.

Two guards keep this from closing things that are meant to be open:

  * --close bounds how far it will reach. A doorway is not a step, and the
    default is well under any opening a building has;
  * the two edges have to be the two sides of a step -- surfaces parallel and
    facing the same way, with the crossing along their shared normal. Two
    edges that merely come close, like the scan's ragged outer fringe, do not
    qualify; without that test this pass welds 49,300 "steps" and adds 230 m2
    of surface that is not there.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def open_edges(F):
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--close", type=float, default=0.22,
                    help="m: the widest step it will build a return across")
    ap.add_argument("--min-run", type=int, default=8,
                    help="edges: a reveal is a long run of facing boundary, "
                         "a fringe coincidence is one or two edges")
    ap.add_argument("--min-gap", type=float, default=0.004,
                    help="m: below this the seam is a weld, not a step")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    m.merge_vertices()
    V = np.array(m.vertices, float)
    F = np.array(m.faces)
    be = open_edges(F)
    print(f"in: {len(F):,} tris, {m.area:.0f} m2, {len(be):,} open edges")

    # each boundary vertex's outward sense: the average normal of its faces
    vn = np.zeros((len(V), 3))
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    L = np.linalg.norm(fn, axis=1)
    fn[L > 1e-12] /= L[L > 1e-12][:, None]
    for fi, f in enumerate(F):
        vn[f] += fn[fi]
    L = np.linalg.norm(vn, axis=1)
    vn[L > 1e-12] /= L[L > 1e-12][:, None]

    bv = np.unique(be)
    # which boundary edges each boundary vertex belongs to, for the facing test
    inc = defaultdict(list)
    for k, (x, y) in enumerate(be):
        inc[int(x)].append(k)
        inc[int(y)].append(k)

    tree = cKDTree(V[bv])
    pairs = tree.query_pairs(a.close, output_type="ndarray")
    partner = {}
    best = defaultdict(lambda: np.inf)
    for i, j in pairs:
        vi, vj = int(bv[i]), int(bv[j])
        d = float(np.linalg.norm(V[vi] - V[vj]))
        if d < a.min_gap:
            continue
        # A STEP, and only a step: the two surfaces are parallel and face the
        # same way, and the crossing runs along their shared normal. That is a
        # wall that thickens. Anything else -- the scan's ragged fringe, a pipe
        # passing a wall, two edges that merely come close -- fails one of the
        # two and is left alone, which is the difference between building 700
        # return faces and welding the whole model shut.
        u = (V[vj] - V[vi]) / d
        if vn[vi] @ vn[vj] < 0.85:
            continue
        if abs(u @ vn[vi]) < 0.7 or abs(u @ vn[vj]) < 0.7:
            continue
        if d < best[vi]:
            best[vi], partner[vi] = d, vj
        if d < best[vj]:
            best[vj], partner[vj] = d, vi

    # A recess is a COLLAR: a long unbroken run of boundary, all of it facing
    # its opposite number across the same offset. The scan's fringe throws up
    # matches too, but they are isolated -- one edge here, two there. Requiring
    # a run is what separates a window reveal from noise, and without it this
    # pass fires 5,523 times and fragments the model instead of closing it.
    cand = [k for k, (x, y) in enumerate(be)
            if partner.get(int(x)) is not None and partner.get(int(y)) is not None
            and partner[int(x)] != partner[int(y)]]
    at = defaultdict(list)
    for k in cand:
        x, y = be[k]
        at[int(x)].append(k)
        at[int(y)].append(k)
    seenk, runs = set(), []
    for k0 in cand:
        if k0 in seenk:
            continue
        run, stack = [], [k0]
        seenk.add(k0)
        while stack:
            k = stack.pop()
            run.append(k)
            for v in be[k]:
                for k2 in at[int(v)]:
                    if k2 not in seenk:
                        seenk.add(k2)
                        stack.append(k2)
        runs.append(run)
    keep = [k for r in runs if len(r) >= a.min_run for k in r]
    print(f"  {len(cand):,} edges face a step; {len(runs):,} runs, "
          f"{len([r for r in runs if len(r) >= a.min_run]):,} long enough to be a reveal")

    add = []
    seen = set()
    for k in keep:
        x, y = int(be[k][0]), int(be[k][1])
        px, py = partner[x], partner[y]
        key = tuple(sorted((x, y, px, py)))
        if key in seen:
            continue
        seen.add(key)
        add.append([x, y, py])
        add.append([x, py, px])

    if not add:
        print("  nothing to bridge")
    else:
        F = np.vstack([F, np.array(add)])
        print(f"  bridged {len(seen):,} steps with {len(add):,} triangles")

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    be2 = open_edges(np.array(out.faces))
    print(f"out: {len(out.faces):,} tris, {out.area:.0f} m2, "
          f"{len(be2):,} open edges -> {a.out}")


if __name__ == "__main__":
    main()
