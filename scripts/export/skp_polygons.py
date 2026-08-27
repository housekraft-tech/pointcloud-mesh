"""Do the solid modelling in Python; let SketchUp draw the faces.

SketchUp's Solid Tools cannot survive 117 booleans on one model: a failed call
still deletes its operands, so the network vanishes part-way through and every
later cut fails against nothing. Two builds died that way.

manifold3d does the same job without drama -- it is what boxify already uses --
so the union of the runs and the pilasters, minus the niches and the openings,
is computed here. What SketchUp then receives is not a mesh but the FACES of
that solid: every triangle grouped onto its plane and merged, so a wall side
comes back as one rectangle with the doorway as a hole in it. SketchUp draws
those as native faces, which is what a designer can push/pull.
"""
import argparse, json, collections
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

def box(lo, hi):
    m = trimesh.creation.box(extents=np.maximum(np.array(hi)-np.array(lo), 1e-5))
    m.apply_translation((np.array(lo)+np.array(hi))/2)
    return m


def tidy(S, snap, min_depth, min_area):
    """Take out the jogs and ledges that are noise, and say what went.

    The model measures a face wherever it can, so a wall's plane wanders by a
    few millimetres along its length and the box model turns each wander into a
    step. 147 of them on this flat, 41% under 30 mm -- and on screen that reads
    as undulation the Poisson mesh does not have. The same goes for a 50 mm
    ledge half a metre square: the detector is right that the surface moved,
    but it is not a feature of the building.
    """
    import collections
    runs = [r for r in S if r["op"] == "run"]
    coords = collections.defaultdict(list)
    for r in runs:
        e = np.array(r["hi"]) - np.array(r["lo"])
        ax = int(np.argmin(e[:2]))
        coords[ax] += [r["lo"][ax], r["hi"][ax]]
    snapmap = {}
    for ax, vals in coords.items():
        order = sorted(set(vals)); grp = [order[0]]
        for v in order[1:]:
            if v - grp[-1] <= snap:
                grp.append(v)
            else:
                c = float(np.mean(grp))
                for g in grp:
                    snapmap[(ax, g)] = c
                grp = [v]
        c = float(np.mean(grp))
        for g in grp:
            snapmap[(ax, g)] = c
    moved = []
    for r in runs:
        e = np.array(r["hi"]) - np.array(r["lo"])
        ax = int(np.argmin(e[:2]))
        for side in ("lo", "hi"):
            old = r[side][ax]
            new = snapmap.get((ax, old), old)
            moved.append(abs(new-old))
            r[side][ax] = new
    out, dropped = [], collections.Counter()
    for r in S:
        if r["op"] in ("relief", "niche"):
            e = np.array(r["hi"]) - np.array(r["lo"])
            ax = int(np.argmin(e[:2]))
            depth = e[ax]; area = e[1-ax]*e[2]
            if depth < min_depth or area < min_area:
                dropped[r["op"]] += 1
                continue
        out.append(r)
    m = np.array(moved)*1000
    print(f"tidy: faces snapped within {snap*1000:.0f} mm -- median move "
          f"{np.median(m):.1f} mm, worst {m.max():.1f}; dropped "
          + ", ".join(f"{v} {k}" for k, v in dropped.items()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.0005)
    ap.add_argument("--tidy", action="store_true",
                    help="flatten sub-threshold jogs and drop ledges that are noise")
    ap.add_argument("--snap", type=float, default=0.020, help="m")
    ap.add_argument("--min-depth", type=float, default=0.060, help="m")
    ap.add_argument("--min-area", type=float, default=0.25, help="m2")
    ap.add_argument("--manifest", default=None,
                    help="the modular manifest, for the floor and ceiling heights")
    a = ap.parse_args()
    S = json.load(open(a.schedule))["parts"]
    if a.tidy:
        S = tidy(S, a.snap, a.min_depth, a.min_area)
    grab = lambda op: [r for r in S if r["op"] == op]

    adds = [box(r["lo"], r["hi"]) for r in grab("run") + grab("relief")]
    cuts = [box(r["lo"], r["hi"]) for r in grab("niche") + grab("opening")]
    print(f"{len(adds)} solids to add, {len(cuts)} to cut")
    net = trimesh.boolean.union(adds, engine="manifold")
    if cuts:
        net = trimesh.boolean.difference([net] + cuts, engine="manifold")
    print(f"network: {len(net.faces):,} triangles, watertight={net.is_watertight}, "
          f"volume={net.volume:.2f} m3")

    # group triangles onto planes, then merge each plane's triangles into
    # polygons -- one face per wall side, with its openings as holes
    n = net.face_normals
    keep = np.max(np.abs(n), axis=1) > 0.999          # axis-aligned faces only
    ax = np.argmax(np.abs(n), axis=1)
    tri = net.triangles
    planes = collections.defaultdict(list)
    for i in np.flatnonzero(keep):
        k = int(ax[i])
        off = round(float(tri[i][0][k]), 4)
        sgn = 1 if n[i][k] > 0 else -1
        planes[(k, off, sgn)].append(tri[i])
    print(f"{len(planes)} planes carry {int(keep.sum()):,} of {len(net.faces):,} triangles")

    out = []
    for (k, off, sgn), tris in planes.items():
        u, v = [c for c in (0, 1, 2) if c != k]
        polys = [Polygon([(t[j][u], t[j][v]) for j in range(3)]) for t in tris]
        polys = [p for p in polys if p.is_valid and p.area > 1e-7]
        if not polys:
            continue
        merged = unary_union(polys).buffer(a.tol).buffer(-a.tol)
        geoms = list(getattr(merged, "geoms", [merged]))
        for g in geoms:
            if g.is_empty or g.area < 1e-4:
                continue
            g = g.simplify(a.tol)
            outer = [[round(float(x), 5), round(float(y), 5)] for x, y in g.exterior.coords[:-1]]
            holes = [[[round(float(x), 5), round(float(y), 5)] for x, y in r.coords[:-1]]
                     for r in g.interiors if Polygon(r).area > 1e-4]
            out.append(dict(axis=k, offset=off, sign=sgn, outer=outer, holes=holes))
    nh = sum(len(f["holes"]) for f in out)
    print(f"{len(out)} faces, {nh} holes in them")
    # Name what each overhead surface IS, so it can be switched off.
    #
    # The measurement calls anything horizontal and high a "dropped ceiling",
    # which on this site put 85 m2 of structural soffit in the same bucket as a
    # scaffold deck. They are told apart by where they sit and how big they
    # are: the storey's soffit is the large area at the modal ceiling height;
    # a false ceiling is a large area well below it; a bulkhead is a small area
    # below it over a room; and a small patch at an odd height on a site under
    # construction is scaffolding, which is tagged rather than deleted -- it is
    # not my place to decide what was temporary.
    man = json.load(open(a.manifest)) if a.manifest else {}
    zf = man.get("floor_z", 0.0)
    zc = zf + man.get("modal_ceiling_height_mm", 2700)/1000.0

    def classify(r):
        e = np.array(r["hi"]) - np.array(r["lo"])
        area = float(e[0]*e[1])
        top = r["hi"][2]; bot = r["lo"][2]
        if r["op"] == "column":
            return "Columns"
        if bot <= zf + 0.25:
            return "Floors"
        drop = zc - bot
        if abs(drop) <= 0.12:
            return "Ceiling (structural)"
        if drop > 0.12 and area >= 8.0:
            return "Ceiling (false)"
        if drop > 0.12 and area >= 1.0 and bot > zf + 1.9:
            return "Ceiling (dropped)"
        if e[2] > 0.25 and min(e[0], e[1]) < 0.6:
            return "Beams"
        return "Site clutter / scaffolding"

    boxes = []
    tally = collections.Counter()
    for r in grab("slab") + grab("column"):
        t = classify(r)
        tally[t] += 1
        boxes.append(dict(name=r["name"], lo=r["lo"], hi=r["hi"], tag=t))
    print("tagged: " + ", ".join(f"{v} {k}" for k, v in tally.most_common()))
    json.dump(dict(faces=out, boxes=boxes), open(a.out, "w"))
    print(f"-> {a.out} ({Path(a.out).stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
