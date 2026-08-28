"""Carry each wall down to its floor and up to its ceiling, and say that you did.

29 of the 97 wall planes in this storey have no point below 460 to 610 mm --
median 510 mm -- because the bottom of the wall was occluded by whatever stood
against it. Nothing is wrong with the fitted plane; the wall simply has no data
down there, so every model built by tracing support stops half a metre short of
the floor and leaves the gap you can see along the skirting line.

A wall that stops in mid-air is worse than a wall that is extended, provided
the extension is honest. So each vertical surface is swept down to the floor
plane and up to the ceiling plane -- the planes it would meet, at their own
measured heights -- and the swept part is written to a SEPARATE file. Measured
surface and inferred surface never share a mesh, because the product reports
deviations and a deviation against invented geometry is a lie.

The sweep is bounded: --max-extend, beyond which the wall is not "occluded at
the bottom", it is a different surface that happens to be coplanar.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.affinity import affine_transform, translate

Z = np.array([0.0, 0.0, 1.0])


def support(P, u, v, cell, close, open_):
    uu, vv = P @ u, P @ v
    lo = np.array([uu.min(), vv.min()]) - 3*cell
    hi = np.array([uu.max(), vv.max()]) + 3*cell
    w, h = np.maximum(np.ceil((hi-lo)/cell).astype(int), 2)
    if w*h > 8_000_000:
        return None
    m = np.zeros((h, w), np.uint8)
    ij = np.floor((np.c_[uu, vv]-lo)/cell).astype(int)
    m[np.clip(ij[:, 1], 0, h-1), np.clip(ij[:, 0], 0, w-1)] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close*2+1,)*2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_*2+1,)*2, np.uint8))
    cont, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return None
    hier = hier[0]
    ps = []
    for ci, c in enumerate(cont):
        if hier[ci][3] != -1 or len(c) < 3:
            continue
        holes = [cont[k].reshape(-1, 2) for k in range(len(cont))
                 if hier[k][3] == ci and len(cont[k]) >= 3]
        try:
            g = Polygon(c.reshape(-1, 2), holes)
            g = g if g.is_valid else g.buffer(0)
        except Exception:
            continue
        if not g.is_empty:
            ps.append(g)
    if not ps:
        return None
    return affine_transform(unary_union(ps), [cell, 0, 0, cell, lo[0], lo[1]])


def tri(g, u, v, n, d, V, F):
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or p.area < 1e-4:
            continue
        p = p.simplify(0.004)
        try:
            vv, ff = trimesh.creation.triangulate_polygon(p, engine="earcut")
        except Exception:
            continue
        b = len(V)
        for x, y in vv:
            V.append(u*x + v*y + n*d)
        F.extend((np.array(ff)+b).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True, help="folder")
    ap.add_argument("--cell", type=float, default=0.03)
    ap.add_argument("--gap", type=float, default=0.03,
                    help="m: a wall this far short of the slab is short")
    ap.add_argument("--max-extend", type=float, default=1.20,
                    help="m: beyond this it is not an occluded wall bottom")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]
    hor = [i for i in range(len(N)) if abs(N[i, 2]) > 0.9 and (lab == i).sum() > 200]
    hz = np.array([float(np.median(pts[lab == i][:, 2])) for i in hor])
    vert = [i for i in range(len(N)) if abs(N[i, 2]) < 0.2 and (lab == i).sum() > 200]
    # The nearest slab below a wall is often a sill or a ledge 85 mm down, not
    # the floor -- so reaching for it closes almost nothing. The storey's own
    # floor and ceiling are the ones a wall runs between, and they are the
    # biggest horizontal surfaces low and high.
    hcnt = np.array([(lab == i).sum() for i in hor])
    mid = 0.5*(hz.min() + hz.max())
    lowh = hz < mid
    zf = float(hz[lowh][np.argmax(hcnt[lowh])]) if lowh.any() else float(hz.min())
    zc = float(hz[~lowh][np.argmax(hcnt[~lowh])]) if (~lowh).any() else float(hz.max())
    print(f"{len(vert)} wall surfaces, {len(hor)} slabs; storey floor z={zf:.3f}, "
          f"ceiling z={zc:.3f}")

    Vm, Fm, Vi, Fi = [], [], [], []
    nlow = nhigh = 0
    grew = []
    for i in vert:
        P = pts[lab == i]
        n = N[i]
        u = np.cross(Z, n)
        u /= np.linalg.norm(u)
        v = Z.copy()
        g = support(P, u, v, a.cell, 3, 2)
        if g is None:
            continue
        tri(g, u, v, n, D[i], Vm, Fm)
        z0, z1 = g.bounds[1], g.bounds[3]
        # the slab below and the slab above, at their own measured heights
        below = hz[hz < z0 - a.gap]
        above = hz[hz > z1 + a.gap]
        # prefer the storey slab when it is within reach, otherwise the nearest
        # surface in that direction
        tdn = zf if (a.gap < z0 - zf <= a.max_extend) else             (below.max() if len(below) else None)
        tup = zc if (a.gap < zc - z1 <= a.max_extend) else             (above.min() if len(above) else None)
        for target, lo_hi in ((tdn, "down"), (tup, "up")):
            if target is None:
                continue
            drop = (z0 - target) if lo_hi == "down" else (target - z1)
            if drop <= a.gap or drop > a.max_extend:
                continue
            steps = max(2, int(np.ceil(drop / (a.cell*2))))
            sweep = unary_union([translate(g, 0, -k*drop/steps if lo_hi == "down"
                                           else k*drop/steps)
                                 for k in range(steps+1)])
            band = box(g.bounds[0]-1, target, g.bounds[2]+1, z0) if lo_hi == "down" \
                else box(g.bounds[0]-1, z1, g.bounds[2]+1, target)
            add = sweep.intersection(band)
            if add.is_empty or add.area < 0.02:
                continue
            tri(add, u, v, n, D[i], Vi, Fi)
            grew.append(drop)
            if lo_hi == "down":
                nlow += 1
            else:
                nhigh += 1

    grew = np.array(grew) if grew else np.zeros(1)
    for nm, Vx, Fx in (("measured", Vm, Fm), ("inferred", Vi, Fi)):
        if not Fx:
            print(f"  no {nm} geometry")
            continue
        m = trimesh.Trimesh(np.array(Vx), np.array(Fx), process=False)
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        m.export(out / f"{nm}.stl")
        print(f"  {nm}: {len(m.faces):,} tris, {m.area:.0f} m2 -> {out/(nm+'.stl')}")
    print(f"{nlow} walls carried down to a floor, {nhigh} up to a ceiling; "
          f"median extension {np.median(grew)*1000:.0f} mm, max {grew.max()*1000:.0f} mm")


if __name__ == "__main__":
    main()
