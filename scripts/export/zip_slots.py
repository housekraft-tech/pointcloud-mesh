"""Zip the two sides of a slot together, as two polylines rather than as points.

Three attempts at closing these gaps by matching PAIRS of boundary vertices all
made the model worse -- more pieces, more open edge, more surface than the scan
has. The reason is the same each time: a per-vertex nearest-neighbour match is
not coherent. Where the two sides are triangulated differently the partner
flips back and forth, so the quads emitted along one slot do not share their
crossing edges, and every flip leaves two new open edges behind. The pass adds
triangles and subtracts connectivity.

A slot has two SIDES, and a side is a polyline. So this walks the boundary into
ordered chains, finds where a chain runs alongside another chain at a steady
distance, and zips those two chains the way a triangle strip between two
polylines is built: at each step advance whichever side has the shorter
diagonal. The result is a strip that is manifold by construction, shares every
internal edge, and joins the two pieces into one.

A run is only zipped if it is long, steady, and narrow -- the same three tests
as before, because they are the right tests; it was the stitching underneath
them that was wrong.
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


def chains(be):
    """Boundary vertices in order. Junctions of three or more end a chain."""
    nb = defaultdict(list)
    for x, y in be:
        nb[int(x)].append(int(y))
        nb[int(y)].append(int(x))
    used = set()
    out = []
    starts = [v for v, k in nb.items() if len(k) != 2] or list(nb)
    for s in starts + list(nb):
        for n0 in nb[s]:
            e = (s, n0) if s < n0 else (n0, s)
            if e in used:
                continue
            ch, prev, cur = [s], s, n0
            used.add(e)
            while True:
                ch.append(cur)
                nxt = [w for w in nb[cur] if w != prev]
                if len(nb[cur]) != 2 or not nxt:
                    break
                w = nxt[0]
                e2 = (cur, w) if cur < w else (w, cur)
                if e2 in used:
                    break
                used.add(e2)
                prev, cur = cur, w
            if len(ch) > 1:
                out.append(ch)
    return out


def zip_chains(V, A, B):
    """A triangle strip between two polylines: always take the shorter diagonal."""
    tris = []
    i = j = 0
    while i < len(A) - 1 or j < len(B) - 1:
        if j >= len(B) - 1:
            tris.append([A[i], A[i + 1], B[j]])
            i += 1
        elif i >= len(A) - 1:
            tris.append([A[i], B[j + 1], B[j]])
            j += 1
        else:
            da = np.linalg.norm(V[A[i + 1]] - V[B[j]])
            db = np.linalg.norm(V[B[j + 1]] - V[A[i]])
            if da <= db:
                tris.append([A[i], A[i + 1], B[j]])
                i += 1
            else:
                tris.append([A[i], B[j + 1], B[j]])
                j += 1
    return tris


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--close", type=float, default=0.22, help="m: widest slot")
    ap.add_argument("--min-gap", type=float, default=0.004, help="m")
    ap.add_argument("--min-run", type=int, default=5, help="vertices in a run")
    ap.add_argument("--min-len", type=float, default=0.20, help="m")
    ap.add_argument("--gap-var", type=float, default=0.40,
                    help="the gap's spread over its mean, along the run")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    m.merge_vertices()
    V = np.array(m.vertices, float)
    F = np.array(m.faces)
    be = open_edges(F)
    print(f"in: {len(F):,} tris, {m.area:.0f} m2, {len(be):,} open edges")

    chs = chains(be)
    where = {}                       # vertex -> (chain index, position)
    for ci, ch in enumerate(chs):
        for pi, v in enumerate(ch):
            where.setdefault(int(v), (ci, pi))
    print(f"  {len(chs):,} boundary chains")

    bv = np.array(sorted(where))
    tree = cKDTree(V[bv])
    d, idx = tree.query(V[bv], k=12)
    partner, gapof = {}, {}
    for r, vi in enumerate(bv):
        ci = where[int(vi)][0]
        for dd, kk in zip(d[r], idx[r]):
            vj = int(bv[kk])
            if vj == vi or dd < a.min_gap or dd > a.close:
                continue
            if where[vj][0] == ci and abs(where[vj][1] - where[int(vi)][1]) < 6:
                continue                  # its own neighbour along the chain
            partner[int(vi)], gapof[int(vi)] = vj, float(dd)
            break

    add, used, why = [], set(), defaultdict(int)
    for ci, ch in enumerate(chs):
        i = 0
        while i < len(ch):
            if ch[i] not in partner:
                i += 1
                continue
            cj = where[partner[ch[i]]][0]
            k = i
            while (k < len(ch) and ch[k] in partner
                   and where[partner[ch[k]]][0] == cj):
                k += 1
            run = ch[i:k]
            i = k
            if len(run) < a.min_run:
                why["too few"] += 1
                continue
            ln = float(np.linalg.norm(np.diff(V[run], axis=0), axis=1).sum())
            if ln < a.min_len:
                why["too short"] += 1
                continue
            gs = np.array([gapof[v] for v in run])
            if gs.std() / gs.mean() > a.gap_var:
                why["gap wanders"] += 1
                continue
            other = [partner[v] for v in run]
            pos = [where[v][1] for v in other]
            desc = pos[-1] < pos[0]
            if desc:
                pos = pos[::-1]
            if not all(x <= y for x, y in zip(pos, pos[1:])):
                why["not monotone"] += 1
                continue
            # the WHOLE span of the far chain, not only the vertices that
            # happened to match. Skipping the ones in between means the strip's
            # far edges are not mesh edges, so every one of them is a fresh
            # open edge and the zip splits the model instead of joining it.
            seq = chs[cj][min(pos):max(pos) + 1]
            if where[run[0]][1] > where[run[-1]][1]:
                pass
            if len(seq) < 2:
                why["far side collapses"] += 1
                continue
            key = (min(run[0], seq[0]), max(run[0], seq[0]), len(run))
            if key in used:
                continue
            used.add(key)
            add.extend(zip_chains(V, run, seq[::-1] if desc else seq))

    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(why.items())))
    if add:
        F = np.vstack([F, np.array(add)])
        print(f"  zipped {len(used):,} slots with {len(add):,} triangles")
    else:
        print("  nothing to zip")

    out = trimesh.Trimesh(V, F, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} tris, {out.area:.0f} m2, "
          f"{len(open_edges(np.array(out.faces))):,} open edges -> {a.out}")


if __name__ == "__main__":
    main()
