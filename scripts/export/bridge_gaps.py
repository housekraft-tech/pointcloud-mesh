"""Close the slots, by asking whether a gap is a SLOT rather than what made it.

The dark stripe running along a wall is not a seam and not a tolerance. Sample
the scan where one appears and four planes meet there, all of them parallel --
normals along Y, offsets at -5.85, -5.80, -5.66 and +5.79 m. It is a recess.
Parallel planes have no intersection line, so the clipped-plane rebuild has
nothing to clip against, each patch stops at its own edge, and what is left is
a slot exactly as deep as the recess and as long as the wall.

The first version of this tested the geometry of each pair of edges: parallel
surfaces, same facing, crossing along the shared normal. That describes a step,
and it closed the steps -- but the slots left over were not steps. Measured at
the six longest, the two sides' normals ran from -0.80 to +0.99, because a slot
can equally be the end of a pier, the side of a jamb, or a reveal turning a
corner. No test written in terms of normals covers them all.

What they have in common is not their shape but their CONSISTENCY. A slot is a
long run of boundary whose opposite side stays the same distance away: two
edges of one thing that should have been joined. The scan's ragged fringe also
throws up edges that pass close by, but there the distance wanders and the run
goes nowhere. So the test is on the run, not on the pair:

  * long -- at least --min-run edges and --min-len of boundary;
  * steady -- the gap's spread within --gap-var of its own mean;
  * narrow -- the whole thing inside --close, which is well under any opening a
    building has, so a doorway is never mistaken for a slot.

Runs that pass are filled with a quad strip. Runs that fail are left open: an
open edge in the right place is better than surface that is not there.
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
                    help="m: the widest slot it will close")
    ap.add_argument("--min-gap", type=float, default=0.004,
                    help="m: below this the seam is a weld, not a slot")
    ap.add_argument("--min-run", type=int, default=6, help="edges")
    ap.add_argument("--min-len", type=float, default=0.25,
                    help="m: total boundary in a run before it counts")
    ap.add_argument("--gap-var", type=float, default=0.40,
                    help="the gap's standard deviation over its mean; above "
                         "this the two sides are not two sides of one slot")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    m.merge_vertices()
    V = np.array(m.vertices, float)
    F = np.array(m.faces)
    be = open_edges(F)
    print(f"in: {len(F):,} tris, {m.area:.0f} m2, {len(be):,} open edges")

    bv = np.unique(be)
    # which way each boundary vertex's own boundary runs: a bridge crosses a
    # slot, it does not run along the edge it starts from
    along = np.zeros((len(V), 3))
    for x, y in be:
        d = V[y] - V[x]
        n = np.linalg.norm(d)
        if n > 1e-12:
            along[x] += d / n
            along[y] += d / n
    n = np.linalg.norm(along, axis=1)
    along[n > 1e-12] /= n[n > 1e-12][:, None]

    tree = cKDTree(V[bv])
    partner, best = {}, defaultdict(lambda: np.inf)
    for i, j in tree.query_pairs(a.close, output_type="ndarray"):
        vi, vj = int(bv[i]), int(bv[j])
        d = float(np.linalg.norm(V[vi] - V[vj]))
        if d < a.min_gap:
            continue
        u = (V[vj] - V[vi]) / d
        if abs(u @ along[vi]) > 0.7 or abs(u @ along[vj]) > 0.7:
            continue                      # along the same boundary, not across
        if d < best[vi]:
            best[vi], partner[vi] = d, vj
        if d < best[vj]:
            best[vj], partner[vj] = d, vi

    cand = [k for k, (x, y) in enumerate(be)
            if partner.get(int(x)) is not None and partner.get(int(y)) is not None
            and partner[int(x)] != partner[int(y)]]
    at = defaultdict(list)
    for k in cand:
        at[int(be[k][0])].append(k)
        at[int(be[k][1])].append(k)

    seen, runs = set(), []
    for k0 in cand:
        if k0 in seen:
            continue
        run, stack = [], [k0]
        seen.add(k0)
        while stack:
            k = stack.pop()
            run.append(k)
            for v in be[k]:
                for k2 in at[int(v)]:
                    if k2 not in seen:
                        seen.add(k2)
                        stack.append(k2)
        runs.append(run)

    keep, why = [], defaultdict(int)
    for r in runs:
        if len(r) < a.min_run:
            why["too few edges"] += 1
            continue
        ln = float(sum(np.linalg.norm(V[be[k][0]] - V[be[k][1]]) for k in r))
        if ln < a.min_len:
            why["too short"] += 1
            continue
        gs = np.array([best[int(be[k][0])] for k in r])
        if gs.mean() <= 0 or gs.std() / gs.mean() > a.gap_var:
            why["gap wanders"] += 1
            continue
        keep.extend(r)
    print(f"  {len(cand):,} edges face something across a gap, in {len(runs):,} runs; "
          f"kept {len(keep):,} (" +
          ", ".join(f"{v} {k}" for k, v in sorted(why.items())) + ")")

    add, done = [], set()
    for k in keep:
        x, y = int(be[k][0]), int(be[k][1])
        px, py = partner[x], partner[y]
        key = tuple(sorted((x, y, px, py)))
        if key in done:
            continue
        done.add(key)
        add.append([x, y, py])
        add.append([x, py, px])

    if not add:
        print("  nothing to bridge")
    else:
        F = np.vstack([F, np.array(add)])
        print(f"  closed {len(done):,} slots with {len(add):,} triangles")

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} tris, {out.area:.0f} m2, "
          f"{len(open_edges(np.array(out.faces))):,} open edges -> {a.out}")


if __name__ == "__main__":
    main()
