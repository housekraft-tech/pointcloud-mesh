"""Pair the two faces of every wall, and read the building's thicknesses off them.

A wall is scanned twice -- once from each room -- so it arrives as two parallel
planes with opposing normals. Nothing needs to be constructed to get its
thickness: it is the distance between the pair, and the plane fits are already
good to about 3 mm, so the thickness is too.

That makes this the first brick of the volumetric model. The cell complex wants
to label the slab BETWEEN a pair as solid, and it wants a prior for how thick a
wall in this building is allowed to be. Real construction has two to four
thickness modes -- a brick size plus its finishes -- and the histogram of
confidently-paired walls is where they come from. Once the modes are known they
adjudicate the doubtful pairs, and they impute the missing face of a wall that
was only ever seen from one side.

A pair has to survive four tests, in order of how much they are trusted:

  * opposing normals, within --ang;
  * a gap in a plausible range for a wall;
  * an overlapping footprint -- two walls either side of a shaft are parallel
    and opposed and are not one wall, and only the overlap says so;
  * and the gap has to be consistent along that overlap, because two surfaces
    that converge are a wall and a soffit, not a wall.

What comes out is the pairing, the histogram, and -- deliberately separate --
the list of faces that could not be paired, which are the ones the model will
have to infer and flag.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import trimesh

Z = np.array([0.0, 0.0, 1.0])


def frames(n):
    """Along the wall, and up it."""
    t = np.cross(Z, n)
    L = np.linalg.norm(t)
    if L < 1e-9:
        return None
    return t / L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None, help="json")
    ap.add_argument("--solids", default=None,
                    help="write the paired walls as solids, to look at")
    ap.add_argument("--min-area", type=float, default=0.5, help="m2")
    ap.add_argument("--ang", type=float, default=3.0,
                    help="deg: how far from opposed two faces may be")
    ap.add_argument("--tmin", type=float, default=0.05, help="m")
    ap.add_argument("--tmax", type=float, default=0.60, help="m")
    ap.add_argument("--coplanar", type=float, default=0.015,
                    help="m: fragments of one face are within this offset")
    ap.add_argument("--drift", type=float, default=0.02,
                    help="m: how much the thickness may vary along the wall")
    ap.add_argument("--overlap", type=float, default=0.35,
                    help="fraction of the smaller face that must be shared")
    a = ap.parse_args()

    z = np.load(a.labels)
    V, F, flab, PN, PD = z["V"], z["F"], z["flab"], z["plane_n"], z["plane_d"]
    A = trimesh.Trimesh(V, F, process=False).area_faces

    area = np.zeros(len(PN))
    for p in range(len(PN)):
        area[p] = A[flab == p].sum()
    vert = (np.abs(PN[:, 2]) < 0.15) & (area >= a.min_area)
    walls = np.flatnonzero(vert)
    print(f"{len(walls):,} wall faces over {a.min_area} m2 "
          f"({area[walls].sum():.0f} m2 of the scan)")

    # One wall face usually arrives as several coplanar planes -- segmentation
    # splits a surface at a doorway, at a pipe, wherever the growing stalled.
    # Pairing before merging them wastes the wall on whichever fragment matched
    # first and strands the rest: 25 pairs out of 149 faces, with more area left
    # one-sided than paired. So coplanar fragments become one face first.
    cosm = np.cos(np.radians(a.ang))
    group, gid = [], {}
    for p in sorted(walls, key=lambda k: -area[k]):
        hit = None
        for g, (n_g, d_g, w_g) in enumerate(group):
            if PN[p] @ n_g >= cosm and abs(PD[p] - d_g) <= a.coplanar:
                hit = g
                break
        if hit is None:
            gid[int(p)] = len(group)
            group.append([PN[p].copy(), float(PD[p]), float(area[p])])
        else:
            n_g, d_g, w_g = group[hit]
            w = float(area[p])
            nn = n_g * w_g + PN[p] * w
            group[hit] = [nn / max(np.linalg.norm(nn), 1e-12),
                          (d_g * w_g + PD[p] * w) / (w_g + w), w_g + w]
            gid[int(p)] = hit
    members = defaultdict(list)
    for p, g in gid.items():
        members[g].append(p)
    PN = np.array([g[0] for g in group])
    PD = np.array([g[1] for g in group])
    area = np.array([g[2] for g in group])
    remap = np.full(int(flab.max()) + 2, -1)
    for p, g in gid.items():
        remap[p] = g
    flab = np.where(flab >= 0, remap[np.clip(flab, 0, None)], -1)
    walls = np.arange(len(group))
    print(f"coplanar fragments merged into {len(group):,} wall faces")

    # each face's extent along its own wall, and its height range
    ext = {}
    for p in walls:
        t = frames(PN[p])
        if t is None:
            continue
        pts = V[np.unique(F[np.flatnonzero(flab == p)])]
        ext[int(p)] = (float((pts @ t).min()), float((pts @ t).max()),
                       float(pts[:, 2].min()), float(pts[:, 2].max()), t, pts)
    walls = [p for p in walls if int(p) in ext]

    cos = np.cos(np.radians(a.ang))
    cands = []
    rej = defaultdict(int)
    # How far the BEST candidate for each face got. Recording the last
    # candidate instead makes the commonest reason whichever plane happened to
    # be examined last -- for a wall that is the far side of the room, whose
    # gap is the room width and is meant to be rejected.
    STAGE = ["no opposed face", "gap out of range", "no footprint overlap",
             "overlap too small", "too few points in overlap",
             "thickness drifts", "a partner was available"]
    reach = defaultdict(int)

    def got(p, q, k):
        reach[int(p)] = max(reach[int(p)], k)
        reach[int(q)] = max(reach[int(q)], k)
    for i, p in enumerate(walls):
        for q in walls[i+1:]:
            if PN[p] @ PN[q] > -cos:
                continue
            got(p, q, 1)
            gap = abs(float(PD[p] + PD[q]))
            if not (a.tmin <= gap <= a.tmax):
                rej["gap out of range"] += 1
                continue
            got(p, q, 2)
            u0, u1, z0, z1, t, pp = ext[int(p)]
            v0, v1, w0, w1, s, qq = ext[int(q)]
            # both extents in ONE frame, or the overlap is meaningless
            b0, b1 = float((qq @ t).min()), float((qq @ t).max())
            ou = min(u1, b1) - max(u0, b0)
            oz = min(z1, w1) - max(z0, w0)
            if ou <= 0 or oz <= 0:
                rej["no footprint overlap"] += 1
                continue
            got(p, q, 3)
            share = (ou * oz) / max(min((u1-u0)*(z1-z0), (b1-b0)*(w1-w0)), 1e-9)
            if share < a.overlap:
                rej["overlap too small"] += 1
                continue
            got(p, q, 4)
            # Consistency: the thickness must not drift along the overlap.
            # Measured properly -- the distance from face p's OWN points to
            # plane q -- because two surfaces that converge are a wall and a
            # soffit, and a single number computed from the two plane offsets
            # cannot tell the difference.
            sel = pp[(pp @ t >= max(u0, b0)) & (pp @ t <= min(u1, b1))]
            if len(sel) < 20:
                rej["too few points in overlap"] += 1
                continue
            got(p, q, 5)
            th_pt = np.abs(sel @ PN[q] - PD[q])
            drift = float(th_pt.std())
            if drift > a.drift:
                rej["thickness drifts"] += 1
                continue
            got(p, q, 6)
            cands.append((share * min(area[p], area[q]), int(p), int(q),
                          gap, share, drift))

    # Greedy pairing strands faces whose only partner was taken by a better
    # candidate: 67 faces had a valid partner and greedy realised 46 of them.
    # Pairing is a matching problem, so it gets a matching algorithm.
    import networkx as nx
    G = nx.Graph()
    meta = {}
    for w, p, q, gap, share, drift in cands:
        G.add_edge(p, q, weight=float(w))
        meta[(min(p, q), max(p, q))] = (gap, share, drift)
    match = nx.max_weight_matching(G, maxcardinality=False)
    used, pairs = set(), []
    for p, q in sorted(match):
        gap, share, drift = meta[(min(p, q), max(p, q))]
        used.add(p)
        used.add(q)
        pairs.append(dict(a=int(p), b=int(q), thickness=round(gap, 4),
                          overlap=round(share, 3), drift=round(drift, 4),
                          area=round(float(min(area[p], area[q])), 2)))
    lone = [int(p) for p in walls if int(p) not in used]
    pa = float(sum(min(area[d["a"]], area[d["b"]]) * 2 for d in pairs))
    print(f"{len(pairs):,} walls paired ({pa:.0f} m2 of face), "
          f"{len(lone):,} faces left one-sided "
          f"({area[lone].sum():.0f} m2)")

    stuck = defaultdict(float)
    for q in lone:
        stuck[STAGE[reach[int(q)]]] += float(area[q])
    print("\nwhy the one-sided faces are one-sided (m2):")
    for k, v in sorted(stuck.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6.0f}  {k}")

    th = np.array([d["thickness"] for d in pairs])
    if len(th):
        print("\nthickness histogram (mm):")
        hist, edges = np.histogram(th, bins=np.arange(0.04, 0.62, 0.01))
        for c, e0 in zip(hist, edges):
            if c:
                print(f"  {e0*1000:4.0f}-{(e0+0.01)*1000:4.0f}  {c:3d}  "
                      + "#" * c)
        # the modes: bins with real weight, refined to the mean of their members
        modes = []
        for k in np.flatnonzero(hist >= max(2, int(0.05*len(th)))):
            m = th[(th >= edges[k]) & (th < edges[k+1])]
            modes.append((float(m.mean()), len(m)))
        modes.sort(key=lambda x: -x[1])
        print("\nconstruction modes:")
        for mm, n in modes:
            sel = th[np.abs(th - mm) < 0.012]
            print(f"  {mm*1000:6.1f} mm   {n:3d} walls   "
                  f"spread {sel.std()*1000:.1f} mm")

    if a.solids:
        # Each pair, as the slab of space between its two planes -- the wall,
        # not a picture of the wall. Extent along the wall and in height is the
        # UNION of the two faces, because a face seen only through a doorway
        # still belongs to the same wall as the one seen whole.
        boxes = []
        for d in pairs:
            p, q = d["a"], d["b"]
            u0, u1, z0, z1, t, pp = ext[p]
            v0, v1, w0, w1, s2, qq = ext[q]
            b0, b1 = float((qq @ t).min()), float((qq @ t).max())
            lo_u, hi_u = min(u0, b0), max(u1, b1)
            lo_z, hi_z = min(z0, w0), max(z1, w1)
            n = PN[p]
            base = PD[p]
            far = -PD[q]                      # plane q, in face p's sense
            corners = []
            for uu in (lo_u, hi_u):
                for zz in (lo_z, hi_z):
                    for dd in (base, far):
                        corners.append(t*uu + Z*zz + n*dd
                                       - Z*(t*uu + Z*zz + n*dd)[2] + Z*zz)
            boxes.append(trimesh.PointCloud(np.array(corners)).convex_hull)
        if boxes:
            trimesh.util.concatenate(boxes).export(a.solids)
            print(f"{len(boxes)} wall solids -> {a.solids}")

    if a.out:
        json.dump(dict(pairs=pairs, one_sided=lone), open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
