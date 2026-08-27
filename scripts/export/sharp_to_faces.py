"""The sharpened scan, as faces SketchUp can hold.

Three quarters of the sharpened surface now lies exactly on an axis plane, so
those triangles merge into a handful of polygons per plane -- a whole wall
becomes one face with its openings as holes. The quarter that is genuinely not
flat (curved reveals, arch soffits, the ragged edges of the scan) stays as
triangles, because that is what it is.
"""
import argparse, collections, json
import numpy as np, trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=0.0008)
ap.add_argument("--min-area", type=float, default=0.004, help="m2")
a = ap.parse_args()

m = trimesh.load(a.inp, force="mesh")
n = m.face_normals
ax = np.argmax(np.abs(n), axis=1)
flat = np.max(np.abs(n), axis=1) > 0.999
tri = m.triangles
print(f"{len(m.faces):,} triangles; {flat.sum():,} lie on an axis plane "
      f"({100*m.area_faces[flat].sum()/m.area:.0f}% of the area)")

planes = collections.defaultdict(list)
for i in np.flatnonzero(flat):
    k = int(ax[i])
    planes[(k, round(float(tri[i][0][k]), 4))].append(tri[i])
faces, holes_n = [], 0
for (k, off), tris in planes.items():
    u, v = [c for c in (0, 1, 2) if c != k]
    polys = [Polygon([(t[j][u], t[j][v]) for j in range(3)]) for t in tris]
    polys = [p for p in polys if p.is_valid and p.area > 1e-8]
    if not polys:
        continue
    merged = unary_union(polys).buffer(a.tol).buffer(-a.tol)
    for g in getattr(merged, "geoms", [merged]):
        if g.is_empty or g.area < a.min_area:
            continue
        g = g.simplify(a.tol)
        faces.append(dict(axis=k, offset=off, sign=1,
                          outer=[[round(float(x), 5), round(float(y), 5)]
                                 for x, y in g.exterior.coords[:-1]],
                          holes=[[[round(float(x), 5), round(float(y), 5)] for x, y in r.coords[:-1]]
                                 for r in g.interiors if Polygon(r).area > a.min_area]))
        holes_n += len(faces[-1]["holes"])
rest = np.flatnonzero(~flat)
print(f"{len(faces):,} merged faces ({holes_n} holes) + {len(rest):,} loose triangles")
json.dump(dict(faces=faces, boxes=[],
               tris=[[[round(float(c), 5) for c in p] for p in tri[i]] for i in rest]),
          open(a.out, "w"))
print(f"-> {a.out}")
