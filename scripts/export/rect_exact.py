"""Rectangles with no grid: cut the plane's own polygon, not a picture of it.

The first rectangle model rasterised each plane at 40 mm and decomposed the
mask. Both of its defects came from that one decision -- the outlines
stair-stepped at the cell size, and accuracy fell from 3.2 mm to 4.9 mm -- and
neither had anything to do with rectangles being the wrong idea.

edge_rebuild already has the exact polygon for every plane. So take it:

  * the polygon's boundary is snapped to the coordinates that MATTER, which are
    the offsets of the perpendicular planes. A wall edge that lands within
    --close of the floor's height becomes that height exactly. This is what
    squares a junction: both surfaces end on the same number;
  * anything left diagonal is turned into a step, so the boundary is
    rectilinear by construction;
  * the rectilinear polygon is then cut EXACTLY, by sweeping the distinct
    boundary coordinates into slabs and merging neighbouring slabs that have
    the same cross-section. No cell size, so no quantisation and no stairs
    except the ones the building has.

Planes too oblique to snap are still cut into rectangles, in their own frame.
Tilting a 3 m wall by 2 degrees moves its far end 100 mm, which is a worse
defect than the one being fixed.

Output is quads, in OBJ. STL cannot hold a quad.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

AXES = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
EPS = 1e-6


def rectilinear(coords, eps):
    """Make a ring axis-parallel: every diagonal becomes a step."""
    out = []
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        out.append((x0, y0))
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx > eps and dy > eps:
            out.append((x1, y0) if dx >= dy else (x0, y1))
    # drop repeats and points in the middle of a straight run
    ring = []
    for p in out:
        if not ring or abs(p[0]-ring[-1][0]) > eps or abs(p[1]-ring[-1][1]) > eps:
            ring.append(p)
    if len(ring) > 1 and abs(ring[0][0]-ring[-1][0]) < eps and abs(ring[0][1]-ring[-1][1]) < eps:
        ring.pop()
    return ring


def slabs(poly, eps):
    """Cut a rectilinear polygon into rectangles, exactly.

    Every boundary x is a slab edge, so inside a slab the polygon is a stack of
    full-width rectangles. Neighbouring slabs with the same stack are one
    rectangle, which is what keeps a plain wall a single face.
    """
    xs = set()
    for ring in [poly.exterior] + list(poly.interiors):
        for x, _ in ring.coords:
            xs.add(round(x, 6))
    xs = sorted(xs)
    if len(xs) < 2:
        return []
    ymin, ymax = poly.bounds[1] - 1.0, poly.bounds[3] + 1.0
    cols = []
    for x0, x1 in zip(xs, xs[1:]):
        if x1 - x0 < eps:
            continue
        piece = poly.intersection(box(x0, ymin, x1, ymax))
        spans = []
        for g in getattr(piece, "geoms", [piece]):
            if g.is_empty or g.area < eps:
                continue
            b = g.bounds
            spans.append((round(b[1], 6), round(b[3], 6)))
        cols.append((x0, x1, tuple(sorted(spans))))
    out = []
    i = 0
    while i < len(cols):
        j = i
        while (j + 1 < len(cols) and cols[j + 1][2] == cols[i][2]
               and abs(cols[j + 1][0] - cols[j][1]) < eps):
            j += 1
        x0, x1 = cols[i][0], cols[j][1]
        for y0, y1 in cols[i][2]:
            out.append((x0, x1, y0, y1))
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help=".obj")
    ap.add_argument("--snap-ang", type=float, default=3.0, help="deg")
    ap.add_argument("--merge", type=float, default=0.012,
                    help="m: planes on one axis closer than this are one plane")
    ap.add_argument("--close", type=float, default=0.10,
                    help="m: how far a boundary may be pulled onto a "
                         "perpendicular plane")
    ap.add_argument("--simplify", type=float, default=0.02,
                    help="m: boundary detail below this is noise, not shape")
    ap.add_argument("--min-area", type=float, default=0.004, help="m2")
    a = ap.parse_args()

    z = np.load(a.labels)
    V, F, flab = z["V"], z["F"], z["flab"]
    PN, PD = z["plane_n"], z["plane_d"]
    A = trimesh.Trimesh(V, F, process=False).area_faces
    total = float(A.sum())

    cos = np.cos(np.radians(a.snap_ang))
    axis = np.full(len(PN), -1)
    for p in range(len(PN)):
        d = np.abs(PN[p] @ AXES.T)
        k = int(d.argmax())
        if d[k] >= cos:
            axis[p] = k
    onax = np.isin(flab, np.flatnonzero(axis >= 0)) & (flab >= 0)
    print(f"{100*A[onax].sum()/total:.1f}% of the scan snaps to an axis "
          f"within {a.snap_ang:.0f} deg")

    off = {}
    for p in np.flatnonzero(axis >= 0):
        sel = np.flatnonzero(flab == p)
        if len(sel):
            off[p] = float(np.median(V[np.unique(F[sel])][:, axis[p]]))
    groups = defaultdict(list)
    for k in range(3):
        ps = sorted((p for p in off if axis[p] == k), key=lambda p: off[p])
        g, last = -1, None
        for p in ps:
            if last is None or off[p] - last > a.merge:
                g += 1
            groups[(k, g)].append(p)
            last = off[p]
    pos = {g: float(np.mean([off[p] for p in ps])) for g, ps in groups.items()}
    lines = {k: np.array(sorted(v for g, v in pos.items() if g[0] == k))
             for k in range(3)}
    print(f"{len(off):,} axis planes merge into {len(groups):,} surfaces")

    def snap(val, k):
        L = lines[k]
        if not len(L):
            return val
        i = int(np.argmin(np.abs(L - val)))
        return float(L[i]) if abs(L[i] - val) <= a.close else val

    oblique = [int(p) for p in range(len(PN))
               if axis[p] < 0 and (flab == p).any()
               and A[flab == p].sum() >= a.min_area]

    quads, verts, index = [], [], {}

    def vid(p):
        key = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    nrect = 0
    for g in list(groups) + oblique:
        ax = isinstance(g, tuple)
        if ax:
            k = g[0]
            i1, i2 = [x for x in range(3) if x != k]
            u, v, nvec, dval = AXES[i1], AXES[i2], AXES[k], pos[g]
            sel = np.flatnonzero(np.isin(flab, groups[g]))
        else:
            nvec = PN[g] / np.linalg.norm(PN[g])
            t = AXES[2] if abs(nvec[2]) < 0.9 else AXES[0]
            u = np.cross(nvec, t)
            u /= np.linalg.norm(u)
            v = np.cross(nvec, u)
            dval = float(PD[g])
            sel = np.flatnonzero(flab == g)
        if not len(sel):
            continue
        tris = []
        for f in sel:
            t3 = V[F[f]]
            tris.append(Polygon([(t3[i] @ u, t3[i] @ v) for i in range(3)]))
        tris = [q for q in tris if q.is_valid and q.area > 1e-9]
        if not tris:
            continue
        patch = unary_union(tris).buffer(a.simplify).buffer(-a.simplify)
        if patch.is_empty:
            continue
        for geom in getattr(patch, "geoms", [patch]):
            geom = geom.simplify(a.simplify)
            if geom.is_empty or geom.area < a.min_area:
                continue
            rings = []
            for ring in [geom.exterior] + list(geom.interiors):
                pts = list(ring.coords)[:-1]
                if ax:
                    pts = [(snap(x, i1), snap(y, i2)) for x, y in pts]
                pts = rectilinear(pts, EPS)
                if len(pts) >= 4:
                    rings.append(pts)
            if not rings:
                continue
            try:
                poly = Polygon(rings[0], rings[1:])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            except Exception:
                continue
            for pg in getattr(poly, "geoms", [poly]):
                if pg.is_empty or pg.area < a.min_area:
                    continue
                for x0, x1, y0, y1 in slabs(pg, 1e-5):
                    if (x1 - x0) * (y1 - y0) < a.min_area:
                        continue
                    q = [vid(u*xx + v*yy + nvec*dval) for xx, yy in
                         ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
                    quads.append(q)
                    nrect += 1

    print(f"{nrect:,} rectangles, {len(verts):,} corners")
    with open(a.out, "w") as fh:
        fh.write("# rectangles only, cut from the plane polygons -- no grid\n")
        for x, y, zz in verts:
            fh.write(f"v {x:.4f} {y:.4f} {zz:.4f}\n")
        for q in quads:
            fh.write("f " + " ".join(str(i + 1) for i in q) + "\n")
    print(f"out -> {a.out}")


if __name__ == "__main__":
    main()
