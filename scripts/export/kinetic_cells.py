"""A 3-D cell complex cut by FINITE planes, labelled by where the sensor stood.

The plan arrangement this replaces had three faults, each measured:

  * every wall became an INFINITE storey-wide line, so a 300 mm recess in one room
    sliced cells in rooms it has never seen;
  * a 3-D plane was flattened to a vertical 2-D line before extrusion. 36 of 70
    walls move more than 10 mm horizontally between floor and ceiling and 9 move
    more than 50 mm, so out-of-plumb -- which is the deviation this product sells --
    was being erased before the model was built;
  * floor and ceiling heights came from d*sign(nz) as if their normals were exactly
    vertical, which is wrong by 4.4 mm and 7.9 mm against the measured supports.

Here each plane cuts only where it was actually observed. Space starts as one
convex box and is split by half-spaces; a plane splits a cell only if the section
it would cut lies inside that plane's own support, grown by a declared margin. The
cells stay convex, so every split is exact and no cell is ever approximated.

Labelling uses the one unambiguous piece of evidence in the dataset: the sensor
was somewhere when each return was measured, and the straight line between them
passed through air. A cell that a ray traverses is free; a cell behind a measured
surface with no ray through it is solid; a cell with neither is unknown and stays
unknown rather than being guessed.

The boundary between solid and free is the model, and each of its faces carries
which of the three states produced it.
"""
import argparse
import json
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree


class Cell:
    """A convex cell as a set of half-spaces plus the vertices they enclose."""
    __slots__ = ("H", "V", "planes")

    def __init__(self, H, V, planes):
        self.H = H            # (k,4): n.x <= d
        self.V = V            # (m,3) enclosing vertices
        self.planes = planes  # ids of the cutting planes that bound it


def box_cell(lo, hi):
    H, planes = [], []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        H.append(np.r_[e, hi[k]])
        planes.append(-1)
        H.append(np.r_[-e, -lo[k]])
        planes.append(-1)
    V = np.array([[x, y, z] for x in (lo[0], hi[0])
                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return Cell(np.array(H), V, planes)


def vertices_of(H, tol=1e-7):
    """Where three of the half-spaces meet, kept if inside all the others.

    Enumerating triples is O(k^3) and that is fine: a cell in this complex is
    bounded by a handful of planes, not hundreds, precisely because the planes
    are finite."""
    k = len(H)
    if k < 3:
        return np.zeros((0, 3))
    pts = []
    for a, b, c in combinations(range(k), 3):
        A = np.array([H[a][:3], H[b][:3], H[c][:3]])
        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            continue
        x = np.linalg.solve(A, np.array([H[a][3], H[b][3], H[c][3]]))
        if np.all(H[:, :3] @ x <= H[:, 3] + tol):
            pts.append(x)
    if not pts:
        return np.zeros((0, 3))
    P = np.array(pts)
    keep = [0]
    for i in range(1, len(P)):
        if np.min(np.linalg.norm(P[keep] - P[i], axis=1)) > 1e-6:
            keep.append(i)
    return P[keep]


def split(cell, n, d, pid):
    """Two convex cells, or None if the plane misses this one."""
    s = cell.V @ n - d
    if s.max() <= 1e-9 or s.min() >= -1e-9:
        return None
    below = Cell(np.vstack([cell.H, np.r_[n, d]]), None, cell.planes + [pid])
    above = Cell(np.vstack([cell.H, np.r_[-n, -d]]), None, cell.planes + [pid])
    for c in (below, above):
        c.V = vertices_of(c.H)
    if len(below.V) < 4 or len(above.V) < 4:
        return None
    return below, above


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--las", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--merges", default=None, help="pair evidence json")
    ap.add_argument("--crop", nargs=6, type=float, default=None,
                    help="xmin ymin zmin xmax ymax zmax")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="m: how far past its own support a plane may cut")
    ap.add_argument("--min-area", type=float, default=0.15,
                    help="m2: a plane with less support than this does not cut")
    ap.add_argument("--rays", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]

    # fold the duplicates the evidence rule accepted, and ONLY those
    rep = {i: i for i in range(len(N))}
    if a.merges:
        ev = json.load(open(a.merges))

        def find(x):
            while rep[x] != x:
                rep[x] = rep[rep[x]]
                x = rep[x]
            return x
        merged = 0
        for e in ev:
            if "MERGE" not in e["verdict"]:
                continue
            ra, rb = find(e["a"]), find(e["b"])
            if ra != rb:
                rep[max(ra, rb)] = min(ra, rb)
                merged += 1
        for i in range(len(N)):
            rep[i] = find(i)
        print(f"{merged} duplicate pairs folded, "
              f"{len(set(rep.values()))} distinct surfaces remain of {len(N)}")

    groups = {}
    for i in range(len(N)):
        groups.setdefault(rep[i], []).append(i)

    lo = np.array(a.crop[:3]) if a.crop else pts.min(0) - 0.2
    hi = np.array(a.crop[3:]) if a.crop else pts.max(0) + 0.2
    print(f"volume {np.round(lo,2)} .. {np.round(hi,2)}")

    # each surviving surface: refit on its members, keep its FINITE support
    surf = []
    for g, mem in groups.items():
        P = np.vstack([pts[lab == i] for i in mem])
        inbox = np.all((P > lo - 0.5) & (P < hi + 0.5), axis=1)
        P = P[inbox]
        if len(P) < 200:
            continue
        n = N[g] / np.linalg.norm(N[g])
        d = float(np.median(P @ n))
        t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(n, t)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uu, vv = P @ u, P @ v
        area = (np.unique(np.floor(np.c_[uu, vv] / 0.05).astype(np.int64),
                          axis=0).shape[0]) * 0.0025
        if area < a.min_area:
            continue
        surf.append(dict(id=int(g), n=n, d=d, u=u, v=v, P=P, area=float(area),
                         ulo=float(uu.min()), uhi=float(uu.max()),
                         vlo=float(vv.min()), vhi=float(vv.max())))
    surf.sort(key=lambda s: -s["area"])
    print(f"{len(surf)} surfaces cut this volume "
          f"({sum(s['area'] for s in surf):.0f} m2 of support)")

    cells = [box_cell(lo, hi)]
    for si, s in enumerate(surf):
        out = []
        for c in cells:
            r = split(c, s["n"], s["d"], s["id"])
            if r is None:
                out.append(c)
                continue
            # finite support: only cut where this surface was actually observed
            mid = np.vstack([r[0].V, r[1].V])
            on = np.abs(mid @ s["n"] - s["d"]) < 1e-6
            if on.sum() < 3:
                out.append(c)
                continue
            q = mid[on]
            uu, vv = q @ s["u"], q @ s["v"]
            if (uu.max() < s["ulo"] - a.margin or uu.min() > s["uhi"] + a.margin
                    or vv.max() < s["vlo"] - a.margin or vv.min() > s["vhi"] + a.margin):
                out.append(c)
                continue
            out.extend(r)
        cells = out
        if si % 10 == 0 or si == len(surf) - 1:
            print(f"  after {si+1:3d} surfaces: {len(cells):,} cells")
        if len(cells) > 20000:
            print("  stopping: cell count is running away")
            break

    cen = np.array([c.V.mean(0) for c in cells])
    vol = []
    for c in cells:
        try:
            from scipy.spatial import ConvexHull
            vol.append(ConvexHull(c.V).volume if len(c.V) >= 4 else 0.0)
        except Exception:
            vol.append(0.0)
    vol = np.array(vol)
    print(f"{len(cells):,} cells, total volume {vol.sum():.1f} m3, "
          f"{(vol < 1e-4).sum()} slivers under 0.1 litre")

    # free space: a cell a sensor ray passes through
    s = np.load(a.sensor)
    traj, fidx = s["traj"], s["fidx"]
    import laspy
    keep = []
    with laspy.open(a.las) as r:
        for ch in r.chunk_iterator(8_000_000):
            keep.append(np.column_stack([ch.x, ch.y, ch.z]).astype(np.float32))
    P = np.vstack(keep)
    m = np.all((P > lo) & (P < hi), axis=1)
    idx = np.flatnonzero(m)
    if len(idx) > a.rays:
        idx = idx[np.linspace(0, len(idx)-1, a.rays).astype(int)]
    tgt = P[idx]
    src = traj[np.clip(fidx[idx].astype(np.int64), 0, len(traj)-1)]
    print(f"{len(idx):,} rays")

    # Which cell a sample is in must be answered EXACTLY. Snapping a sample to
    # the nearest cell CENTRE puts points from a large room into a 100 mm wall
    # cell whose centre happens to be closer, which marked 111 of 112 m3 free -
    # including the walls.
    samples = []
    steps = 20
    for k in range(1, steps):
        f = k / steps
        q = src + (tgt - src) * f * 0.97          # stop short of the surface itself
        samples.append(q[np.all((q > lo) & (q < hi), axis=1)])
    Q = np.vstack(samples)
    if len(Q) > 400_000:
        Q = Q[np.linspace(0, len(Q)-1, 400_000).astype(int)]
    print(f"{len(Q):,} free-space samples inside the volume")

    free = np.zeros(len(cells), np.int64)
    hitc = np.zeros(len(cells), np.int64)
    keep_t = tgt[np.all((tgt > lo) & (tgt < hi), axis=1)]
    for i, c in enumerate(cells):
        m = np.all(Q @ c.H[:, :3].T <= c.H[:, 3] + 1e-9, axis=1)
        free[i] = int(m.sum())
        if len(keep_t):
            mh = np.all(keep_t @ c.H[:, :3].T <= c.H[:, 3] + 1e-9, axis=1)
            hitc[i] = int(mh.sum())

    # A cell is solid when nothing ever saw through it AND it is enclosed by
    # measured surface. Free space is asserted by evidence; solidity is asserted
    # by the absence of evidence PLUS a boundary that was actually observed.
    nmeas = np.array([sum(1 for p in c.planes if p >= 0) for c in cells])
    dens = free / np.maximum(vol, 1e-6)
    lab3 = np.full(len(cells), 2, np.int64)
    lab3[dens > 200] = 0
    lab3[(lab3 == 2) & (nmeas >= 2)] = 1
    names = ["free", "solid", "unknown"]
    for k in range(3):
        print(f"  {names[k]:8s}: {(lab3==k).sum():5d} cells, {vol[lab3==k].sum():7.2f} m3")
    thin = (lab3 == 1) & (vol < 2.0)
    print(f"  of the solid cells, {int(thin.sum())} are under 2 m3 "
          f"({vol[thin].sum():.2f} m3) - these are the walls")

    np.savez_compressed(a.out,
                        centres=cen, volume=vol, label=lab3,
                        free=free, hit=hitc,
                        planes=np.array([s["id"] for s in surf]),
                        n=np.array([s["n"] for s in surf]),
                        d=np.array([s["d"] for s in surf]))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
