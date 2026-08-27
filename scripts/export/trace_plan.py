"""Trace the scan the way a person does: section, outline, push up.

Everything so far built walls by pairing faces, and that is not how anyone
traces a survey. A person slices the model at about waist height, looks at the
band of masonry the section cuts, draws round it with the line tool, and pulls
the whole plan up to the ceiling. Doors and windows are cut afterwards from the
elevations.

Done that way the fragmentation disappears by construction: a section through a
wall is ONE closed band whatever the wall does above and below it, so a flat
comes out with as many polygons as it has pieces of masonry -- not one per
thickness step.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def orthogonalise(poly, step=0.005, ang=8.0):
    """Pull nearly-axis-aligned edges onto the axes, as a tracer's inference does."""
    def fix(ring):
        p = np.array(ring)
        for i in range(len(p)):
            a, b = p[i], p[(i+1) % len(p)]
            d = b - a
            if abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9:
                continue
            th = np.degrees(np.arctan2(abs(d[1]), abs(d[0])))
            if th < ang:                      # horizontal
                y = round((a[1]+b[1])/2/step)*step
                p[i][1] = y; p[(i+1) % len(p)][1] = y
            elif th > 90-ang:                 # vertical
                x = round((a[0]+b[0])/2/step)*step
                p[i][0] = x; p[(i+1) % len(p)][0] = x
        return p
    ext = fix(list(poly.exterior.coords)[:-1])
    ints = [fix(list(r.coords)[:-1]) for r in poly.interiors]
    q = Polygon(ext, ints)
    return q if q.is_valid else poly.buffer(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--verts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solid", default=None,
                    help="a boxes_schedule.json to section instead of the scan shell")
    ap.add_argument("--at", default="1.2",
                    help="m above the floor; comma-separated for several sections")
    ap.add_argument("--close", type=float, default=0.035,
                    help="m: gaps up to this are closed, as a tracer's snap would")
    ap.add_argument("--simplify", type=float, default=0.015, help="m")
    ap.add_argument("--min-area", type=float, default=0.05, help="m2")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    zf = man["floor_z"]; zc = zf + man["modal_ceiling_height_mm"]/1000.0
    if a.solid:
        # Section the SOLID, not the scan shell.
        #
        # A tracer sections a solid model and gets closed bands. Sectioning the
        # Poisson surface gives 45 open fragments totalling 1.8 m2 -- because a
        # shell has no inside, so there is nothing for the plane to cut through.
        # The box network is a real solid and 5 mm from the scan, so its plan
        # is the plan, and it comes out as closed polygons by construction.
        import sys as _s
        _s.path.insert(0, str(Path(__file__).parent))
        from skp_polygons import box as _box
        S = json.load(open(a.solid))["parts"]
        g = lambda op: [r for r in S if r["op"] == op]
        mesh = trimesh.boolean.union([_box(r["lo"], r["hi"])
                                      for r in g("run") + g("relief")], engine="manifold")
        cuts = [_box(r["lo"], r["hi"]) for r in g("niche") + g("opening")]
        if cuts:
            mesh = trimesh.boolean.difference([mesh] + cuts, engine="manifold")
    else:
        T = np.load(a.cache)["T"].astype(np.int64)
        V = np.load(a.verts).astype(np.float64)
        mesh = trimesh.Trimesh(V, T, process=False)
    print(f"scan: {len(mesh.faces):,} triangles; slicing at {a.at} m over the floor")

    # the section, as a tracer would take it
    # One section only sees what exists at that height: trace at 1.2 m alone
    # and every low wall, parapet and high bulkhead is missing -- the tail went
    # to 1.6 m. A tracer checks a couple of heights for exactly this reason.
    heights = [float(x) for x in str(a.at).split(",")]
    secs = []
    for h in heights:
        sc_ = mesh.section(plane_origin=[0, 0, zf + h], plane_normal=[0, 0, 1])
        if sc_ is not None:
            secs.append(sc_)
    if not secs:
        raise SystemExit("every section is empty")
    sec = secs[0]
    extra = secs[1:]
    # to_planar gives coordinates in the SECTION PLANE's own frame and hands
    # back the transform to world. Using the 2D numbers directly put the whole
    # plan somewhere else -- 956 mm from the scan. Map them back.
    planar, to_3D = sec.to_planar()
    def to_world(e):
        p2 = np.asarray(e, float)
        p4 = np.column_stack([p2, np.zeros(len(p2)), np.ones(len(p2))])
        w = (to_3D @ p4.T).T
        return w[:, :2]
    polys = [Polygon(to_world(e)) for e in planar.discrete if len(e) > 3]
    for sc_ in extra:
        pl, t3 = sc_.to_planar()
        def tw(e, T=t3):
            p2 = np.asarray(e, float)
            p4 = np.column_stack([p2, np.zeros(len(p2)), np.ones(len(p2))])
            return ((T @ p4.T).T)[:, :2]
        polys += [Polygon(tw(e)) for e in pl.discrete if len(e) > 3]
    polys = [p.buffer(0) for p in polys if p.is_valid or True]
    print(f"section: {len(polys)} loops")

    # A section through a SHELL gives nested loops: the outside of the
    # building, then the rooms inside it, then anything standing in a room.
    # Filling them all and unioning turns the whole floor into solid -- which
    # is why the first attempt reported 1.6 m2 of masonry instead of 16. The
    # rule is even-odd: a loop inside an odd number of others is a void.
    polys = [p for p in polys if p.area > 1e-6]
    polys.sort(key=lambda p: -p.area)
    depth = []
    for i, p in enumerate(polys):
        c = p.representative_point()
        depth.append(sum(1 for j, q in enumerate(polys) if j != i and q.contains(c)))
    solid = unary_union([p for p, d in zip(polys, depth) if d % 2 == 0])
    void = unary_union([p for p, d in zip(polys, depth) if d % 2 == 1])
    merged = solid.difference(void) if not void.is_empty else solid
    print(f"   {sum(1 for d in depth if d % 2 == 0)} solid loops, "
          f"{sum(1 for d in depth if d % 2 == 1)} voids -> {merged.area:.1f} m2 of masonry")
    merged = merged.buffer(a.close).buffer(-a.close)      # close the door-jamb gaps
    geoms = list(getattr(merged, "geoms", [merged]))
    out = []
    for gm in geoms:
        if gm.area < a.min_area:
            continue
        gm = gm.simplify(a.simplify, preserve_topology=True)
        gm = orthogonalise(gm)
        if gm.is_empty or gm.area < a.min_area:
            continue
        out.append(dict(outer=[[round(x, 4), round(y, 4)] for x, y in gm.exterior.coords[:-1]],
                        holes=[[[round(x, 4), round(y, 4)] for x, y in r.coords[:-1]]
                               for r in gm.interiors if Polygon(r).area > a.min_area]))
    tot = sum(Polygon(o["outer"]).area - sum(Polygon(h).area for h in o["holes"]) for o in out)
    verts = sum(len(o["outer"]) + sum(len(h) for h in o["holes"]) for o in out)
    print(f"traced plan: {len(out)} polygons, {verts} corners, {tot:.1f} m2 of masonry in section")
    json.dump(dict(z0=round(zf-0.02, 4), z1=round(zc, 4), plan=out),
              open(a.out, "w"))
    print(f"-> {a.out}   (extrude {zf-0.02:.3f} to {zc:.3f} m)")


if __name__ == "__main__":
    main()
