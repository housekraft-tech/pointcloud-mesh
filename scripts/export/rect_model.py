"""A model made of rectangles: no triangles, and every junction square.

The clipped-plane rebuild is accurate to about 3 mm but it is a triangle mesh,
and a triangle mesh is the wrong thing to hand a designer. What a wall wants to
be is a rectangle, what a floor wants to be is a rectangle, and where they meet
they should stop on the same coordinate rather than near it.

The building allows this: 85% of the wall area lies within 2 degrees of one
90-degree grid, and that grid is already aligned to X and Y. So

  * every plane whose normal is within --snap-ang of an axis is SNAPPED to that
    axis, and its offset re-measured as the median of its own points -- the
    median, not the fit, because a fitted plane tilts to follow the noise;
  * planes on the same axis within --merge of each other become one. This is
    what removes the false steps: a wall segmented into surfaces at -5.85 and
    -5.80 is one wall, and the 50 mm slot between them was never real;
  * each plane's covered area is rasterised at --grid and decomposed into
    maximal rectangles by merging identical runs between rows. Flat walls come
    out as one or two rectangles; an opening stays an opening, because a hole
    in the mask is a hole in the decomposition;
  * every rectangle edge is then SNAPPED to the offsets of the perpendicular
    planes when it lands within --close of one. That is what closes the
    junctions: both surfaces end on the same number, so there is nothing left
    between them to see through.

Output is quads. STL cannot hold a quad, so the deliverable is OBJ (and DAE),
which is what SketchUp reads anyway.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh

AXES = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


def runs(mask):
    """Rectangles covering a boolean mask: identical row runs merged downward."""
    h, w = mask.shape
    out = []
    open_runs = {}                       # (x0, x1) -> y0
    for y in range(h + 1):
        row = mask[y] if y < h else np.zeros(w, bool)
        cur = {}
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                cur[(x0, x)] = True
            else:
                x += 1
        for k in list(open_runs):
            if k not in cur:
                out.append((open_runs.pop(k), y, k[0], k[1]))
        for k in cur:
            open_runs.setdefault(k, y)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help=".obj")
    ap.add_argument("--snap-ang", type=float, default=12.0, help="deg")
    ap.add_argument("--merge", type=float, default=0.02,
                    help="m: planes on one axis closer than this are one plane")
    ap.add_argument("--grid", type=float, default=0.03, help="m")
    ap.add_argument("--close", type=float, default=0.16,
                    help="m: how far a rectangle edge may be pulled to land on "
                         "a perpendicular plane")
    ap.add_argument("--min-area", type=float, default=0.02, help="m2")
    a = ap.parse_args()

    z = np.load(a.labels)
    V, F, flab = z["V"], z["F"], z["flab"]
    PN, PD = z["plane_n"], z["plane_d"]
    tm = trimesh.Trimesh(V, F, process=False)
    A = tm.area_faces
    total = float(A.sum())

    # 1. snap each plane to an axis, and drop the ones that will not go
    cos = np.cos(np.radians(a.snap_ang))
    axis = np.full(len(PN), -1)
    for p in range(len(PN)):
        d = np.abs(PN[p] @ AXES.T)
        k = int(d.argmax())
        if d[k] >= cos:
            axis[p] = k
    kept = np.isin(flab, np.flatnonzero(axis >= 0)) & (flab >= 0)
    print(f"{100*A[kept].sum()/total:.1f}% of the scan lies on planes that snap "
          f"to an axis within {a.snap_ang:.0f} deg")
    # The rest are rectangles too, just not axis-aligned ones: a plane at 30
    # degrees still cuts into rectangles in its OWN frame. Snapping it to an
    # axis instead would tilt it by 30 degrees, and a wall 3 m long tilted by
    # even 2 degrees has moved 100 mm at the end of it. Accuracy first.
    oblique = [int(p) for p in range(len(PN))
               if axis[p] < 0 and (flab == p).any()]

    # 2. one offset per plane, from its own points, then merge along each axis
    off = {}
    for p in np.flatnonzero(axis >= 0):
        sel = np.flatnonzero(flab == p)
        if not len(sel):
            continue
        off[p] = float(np.median(V[np.unique(F[sel])][:, axis[p]]))
    groups = defaultdict(list)           # (axis, group id) -> [planes]
    gid = {}
    for k in range(3):
        ps = sorted((p for p in off if axis[p] == k), key=lambda p: off[p])
        g, last = -1, None
        for p in ps:
            if last is None or off[p] - last > a.merge:
                g += 1
            gid[p] = (k, g)
            groups[(k, g)].append(p)
            last = off[p]
    pos = {g: float(np.mean([off[p] for p in ps])) for g, ps in groups.items()}
    print(f"{len(off):,} axis planes merge into {len(groups):,} surfaces "
          f"({sum(1 for g in groups if g[0]==0)} X, "
          f"{sum(1 for g in groups if g[0]==1)} Y, "
          f"{sum(1 for g in groups if g[0]==2)} Z)")

    # the coordinates a rectangle edge is allowed to end on
    lines = {k: np.array(sorted(v for g, v in pos.items() if g[0] == k))
             for k in range(3)}

    def snap(vals, k):
        L = lines[k]
        if not len(L):
            return vals
        i = np.clip(np.searchsorted(L, vals), 1, len(L) - 1)
        near = np.where(np.abs(L[i] - vals) < np.abs(L[i-1] - vals), L[i], L[i-1])
        return np.where(np.abs(near - vals) <= a.close, near, vals)

    # 3. rasterise each surface and cut it into rectangles
    quads, verts, index = [], [], {}

    def vid(p):
        key = (round(p[0], 5), round(p[1], 5), round(p[2], 5))
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    def frame_of(g):
        """(origin, u, v, n, offset) for an axis surface or an oblique plane."""
        if isinstance(g, tuple):
            k = g[0]
            i1, i2 = [x for x in range(3) if x != k]
            u, v, n = AXES[i1], AXES[i2], AXES[k]
            return u, v, n, pos[g], (i1, i2, k)
        n = PN[g] / np.linalg.norm(PN[g])
        t = AXES[2] if abs(n[2]) < 0.9 else AXES[0]
        u = np.cross(n, t)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return u, v, n, float(PD[g]), None

    nrect = 0
    for g in list(groups) + oblique:
        u, v, nvec, dval, ax = frame_of(g)
        sel = (np.flatnonzero(np.isin(flab, groups[g])) if ax
               else np.flatnonzero(flab == g))
        if not len(sel):
            continue
        P = V[F[sel]].reshape(-1, 3)
        pu, pv = P @ u, P @ v
        lo = np.array([pu.min(), pv.min()]) - a.grid
        hi = np.array([pu.max(), pv.max()]) + a.grid
        ncell = np.maximum(np.ceil((hi - lo) / a.grid).astype(int), 1)
        if ncell[0] * ncell[1] > 4_000_000:
            continue
        mask = np.zeros((ncell[1], ncell[0]), bool)
        i1, i2 = (ax[0], ax[1]) if ax else (None, None)
        # a triangle covers the cells its own samples fall in: dense enough at
        # this grid, and far cheaper than a scan conversion
        for f in sel:
            t = V[F[f]]
            m = max(2, int(np.ceil(np.linalg.norm(t[1]-t[0]) / a.grid) + 1))
            for w1 in np.linspace(0, 1, m):
                w2 = np.linspace(0, 1 - w1, max(2, int(m * (1 - w1)) + 1))
                pts = (t[0] + np.outer(w1, t[1] - t[0])
                       + np.outer(w2, t[2] - t[0]))
                ij = np.floor((np.c_[pts @ u, pts @ v] - lo) / a.grid).astype(int)
                ij[:, 0] = np.clip(ij[:, 0], 0, ncell[0]-1)
                ij[:, 1] = np.clip(ij[:, 1], 0, ncell[1]-1)
                mask[ij[:, 1], ij[:, 0]] = True
        for y0, y1, x0, x1 in runs(mask):
            u0, u1 = lo[0] + x0*a.grid, lo[0] + x1*a.grid
            v0, v1 = lo[1] + y0*a.grid, lo[1] + y1*a.grid
            if (u1-u0) * (v1-v0) < a.min_area:
                continue
            if ax:
                u0, u1 = snap(np.array([u0, u1]), i1)
                v0, v1 = snap(np.array([v0, v1]), i2)
            if u1 <= u0 or v1 <= v0:
                continue
            q = []
            for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                q.append(vid(u*uu + v*vv + nvec*dval))
            quads.append(q)
            nrect += 1

    print(f"{nrect:,} rectangles, {len(verts):,} corners")
    with open(a.out, "w") as fh:
        fh.write("# rectangles only -- no triangles\n")
        for x, y, zz in verts:
            fh.write(f"v {x:.4f} {y:.4f} {zz:.4f}\n")
        for q in quads:
            fh.write("f " + " ".join(str(i + 1) for i in q) + "\n")
    print(f"out -> {a.out}")


if __name__ == "__main__":
    main()
