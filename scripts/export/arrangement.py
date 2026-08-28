"""Stop tracing surfaces and start bounding them: the plan arrangement.

Everything before this traced a surface's outline from where its points
happened to stop, and every one of those attempts failed the same way -- the
scan's ragged fringe became the model's ragged edge, and rectilinearising it
made staircases. The outline was never the building's; it was the scan's.

Here a surface has no outline of its own. Every wall face becomes an infinite
LINE in plan, the lines cut the storey into cells, and a wall's extent is
whatever the neighbouring lines leave it. A wall bounded by the floor, the
ceiling and the two walls it runs into IS a rectangle -- with no grid, no
snapping and no tolerance, because nothing was ever fitted to the fringe.

Cells are then labelled from evidence rather than shape:

  * ROOM: the scanner physically stood there. The trajectory is the one piece
    of unambiguous free-space evidence in the whole dataset -- a pose inside a
    cell proves the cell is air. Rooms grow from those seeds through cells that
    also have floor beneath them.
  * WALL: a cell with no floor support, bounded by wall faces, narrow enough
    to be construction.
  * OUTSIDE: everything else.

Walls and rooms are then extruded between the floor and ceiling planes, which
gives solids with measured thickness and rooms with measured volume.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union


def storey_levels(N, D, pts, lab):
    """The floor and the ceiling: the biggest horizontal surface low and high."""
    hor = np.flatnonzero(np.abs(N[:, 2]) > 0.9)
    if not len(hor):
        raise SystemExit("no horizontal surface")
    z = np.array([D[i] * np.sign(N[i, 2]) for i in hor])
    cnt = np.array([(lab == i).sum() for i in hor])
    lo_half = z < np.median(z)
    fl = hor[lo_half][np.argmax(cnt[lo_half])] if lo_half.any() else hor[np.argmin(z)]
    hi_half = ~lo_half
    ce = hor[hi_half][np.argmax(cnt[hi_half])] if hi_half.any() else hor[np.argmax(z)]
    return float(z[hor == fl][0] if False else D[fl]*np.sign(N[fl, 2])), \
        float(D[ce]*np.sign(N[ce, 2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--odo", default="data/Soulace/odometerdata.txt")
    ap.add_argument("--out", required=True, help="folder")
    ap.add_argument("--min-wall", type=float, default=1.0,
                    help="m2: a wall face smaller than this does not cut the plan")
    ap.add_argument("--extend", type=float, default=0.35,
                    help="m: how far a wall line runs past its own points, so "
                         "two walls that meet actually cross")
    ap.add_argument("--max-thick", type=float, default=0.60,
                    help="m: a cell thicker than this is not a wall")
    ap.add_argument("--min-built", type=float, default=1.0,
                    help="m: total boundary of a wall cell that must have wall "
                         "points along it -- a wall has two faces")
    ap.add_argument("--min-room", type=float, default=1.0, help="m2")
    a = ap.parse_args()

    from pathlib import Path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]
    cnt = np.array([(lab == i).sum() for i in range(len(N))])
    zf, zc = storey_levels(N, D, pts, lab)
    print(f"floor at z={zf:.3f} m, ceiling at z={zc:.3f} m "
          f"-> storey height {zc-zf:.3f} m")

    vert = np.flatnonzero(np.abs(N[:, 2]) < 0.2)
    segs = []
    for i in vert:
        P = pts[lab == i]
        if len(P) < 50:
            continue
        n2 = N[i, :2] / np.linalg.norm(N[i, :2])
        t2 = np.array([-n2[1], n2[0]])
        # area in plan terms: length x height of its own points
        s = P[:, :2] @ t2
        h = P[:, 2]
        if (s.max()-s.min()) * (h.max()-h.min()) < a.min_wall:
            continue
        c = float(np.median(P[:, :2] @ n2))
        # Full length, not the extent of its own points. A line that stops
        # where the scan stopped does not cross its neighbours, so the plan is
        # never partitioned: 70 lines produced only 153 cells, one of which
        # spanned several rooms and swallowed the whole storey the moment a
        # pose landed in it. Partitioning is the arrangement's job; deciding
        # which parts are real is the labelling's.
        segs.append((n2, c, float(s.min()), float(s.max())))
    print(f"{len(segs):,} wall lines from {len(vert):,} vertical surfaces")

    allp = pts[:, :2]
    lo = allp.min(0) - 0.5
    hi = allp.max(0) + 0.5
    fr = box(lo[0], lo[1], hi[0], hi[1])
    R = float(np.linalg.norm(hi - lo))
    lines = []
    for n2, c, s0, s1 in segs:
        t2 = np.array([-n2[1], n2[0]])
        g = LineString([n2*c + t2*(-R), n2*c + t2*R]).intersection(fr)
        if g.geom_type == "LineString" and g.length > 0.1:
            lines.append(g)
    cells = list(polygonize(unary_union(lines + [fr.boundary])))
    print(f"the arrangement has {len(cells):,} cells")

    # floor support: where does the floor plane actually have points
    fl_pts = None
    hor = np.flatnonzero(np.abs(N[:, 2]) > 0.9)
    near_floor = [i for i in hor if abs(D[i]*np.sign(N[i, 2]) - zf) < 0.25]
    if near_floor:
        fl_pts = np.vstack([pts[lab == i][:, :2] for i in near_floor])
    from scipy.spatial import cKDTree
    ftree = cKDTree(fl_pts) if fl_pts is not None else None

    odo = np.loadtxt(a.odo)
    poses = odo[:, 2:5]
    # The trajectory lives in the scanner's frame. If the planes have been
    # rotated onto the axes, the poses have to come with them -- otherwise the
    # seeds land in the wrong cells and the storey loses three quarters of its
    # floor, which is exactly what happened: 86 seeded cells became 16.
    if "R" in z.files:
        poses = poses @ np.array(z["R"]).T - np.array(z["origin"])
        print(f"trajectory rotated into the model frame "
              f"({np.degrees(float(z['yaw'])):+.2f} deg)")
    inside = (poses[:, 2] > zf - 0.3) & (poses[:, 2] < zc + 0.3)
    poses = poses[inside]
    print(f"{len(poses):,} trajectory poses stand on this storey")

    seeded, floored, areas = [], [], []
    for c in cells:
        ar = c.area
        areas.append(ar)
        rp = c.representative_point()
        seeded.append(any(c.contains(Point(p[0], p[1])) for p in poses)
                      if ar > 0.2 else False)
        if ftree is not None and ar > 0.05:
            k = ftree.query_ball_point([rp.x, rp.y], 0.25)
            floored.append(len(k) > 20)
        else:
            floored.append(False)
    seeded = np.array(seeded)
    floored = np.array(floored)
    areas = np.array(areas)
    print(f"{seeded.sum()} cells contain a pose, {floored.sum()} have floor under them")

    # An arrangement line runs past the wall that generated it -- that is what
    # makes two walls meet -- so most cell boundaries are not walls at all, just
    # the extension of one. Growing a room across every shared edge therefore
    # floods the whole storey: the first run labelled 356 m2 of "floor" inside a
    # 21 x 17 m frame, i.e. everything. A boundary only blocks if there is wall
    # THERE: points on it, at height, along its length.
    wall_xy = np.vstack([pts[lab == i][:, :2] for i in vert
                         if (lab == i).sum() > 50]) if len(vert) else np.zeros((0, 2))
    wtree = cKDTree(wall_xy) if len(wall_xy) else None

    def blocked(g):
        """Is this shared boundary actually built?"""
        if wtree is None or g.length < 1e-6:
            return False
        k = max(3, int(g.length / 0.15))
        hits = 0
        for t in np.linspace(0, g.length, k):
            q = g.interpolate(t)
            if len(wtree.query_ball_point([q.x, q.y], 0.06)) >= 3:
                hits += 1
        return hits / k > 0.6

    from shapely.strtree import STRtree
    tree_c = STRtree(cells)
    nbr = defaultdict(list)
    for i in range(len(cells)):
        for j in tree_c.query(cells[i]):
            j = int(j)
            if j <= i or not cells[i].intersects(cells[j]):
                continue
            g = cells[i].intersection(cells[j])
            if g.geom_type not in ("LineString", "MultiLineString") or g.length < 0.05:
                continue
            if not blocked(g):
                nbr[i].append(j)
                nbr[j].append(i)
    open_edges = sum(len(v) for v in nbr.values()) // 2
    print(f"{open_edges} cell boundaries are not built (a room may cross them)")

    room = seeded.copy()
    stack = list(np.flatnonzero(seeded))
    while stack:
        i = stack.pop()
        for j in nbr[i]:
            if not room[j] and floored[j]:
                room[j] = True
                stack.append(j)
    print(f"{room.sum()} room cells, {areas[room].sum():.1f} m2 of floor")

    # A wall cell is not merely "narrow and next to a room" -- that test passed
    # every full-frame sliver the arrangement makes, which is why the walls came
    # out as bars shooting off the plan in all directions. A wall has two faces,
    # so a wall cell has TWO long sides with wall actually built along them, and
    # it lies between rooms rather than out in the field.
    room_hull = unary_union([cells[i] for i in np.flatnonzero(room)])
    if not room_hull.is_empty:
        room_hull = room_hull.buffer(0.8)
    wall = np.zeros(len(cells), bool)
    why = defaultdict(int)
    for i, c in enumerate(cells):
        if room[i] or areas[i] < 0.01:
            continue
        if not room_hull.is_empty and not room_hull.contains(c.representative_point()):
            why["outside the rooms"] += 1
            continue
        w = 4*c.area/max(c.length, 1e-9)
        if w > a.max_thick:
            why["too thick"] += 1
            continue
        built = 0.0
        for j in tree_c.query(c):
            j = int(j)
            if j == i or not c.intersects(cells[j]):
                continue
            g = c.intersection(cells[j])
            if g.geom_type not in ("LineString", "MultiLineString") or g.length < 0.05:
                continue
            if blocked(g):
                built += g.length
        if built < a.min_built:
            why["not built"] += 1
            continue
        wall[i] = True
    print(f"{wall.sum()} wall cells, {areas[wall].sum():.1f} m2 in plan  (" +
          ", ".join(f"{v} {k}" for k, v in sorted(why.items())) + ")")

    def extrude(sel, z0, z1, path):
        ms = []
        for i in np.flatnonzero(sel):
            try:
                ms.append(trimesh.creation.extrude_polygon(cells[i], z1 - z0))
            except Exception:
                continue
        if not ms:
            print(f"  nothing to write for {path}")
            return
        m = trimesh.util.concatenate(ms)
        m.apply_translation([0, 0, z0])
        m.export(path)
        print(f"  {len(ms)} solids, {len(m.faces):,} tris -> {path}")

    extrude(wall, zf, zc, str(out / "walls.stl"))
    if wall.any():
        fused = unary_union([cells[i] for i in np.flatnonzero(wall)])
        ms = []
        for g in getattr(fused, "geoms", [fused]):
            if g.is_empty or g.area < 0.005:
                continue
            try:
                ms.append(trimesh.creation.extrude_polygon(g, zc - zf))
            except Exception:
                continue
        if ms:
            m = trimesh.util.concatenate(ms)
            m.apply_translation([0, 0, zf])
            m.export(str(out / "walls_fused.stl"))
            print(f"  fused into {len(ms)} solids, {len(m.faces):,} tris, "
                  f"watertight={m.is_watertight}, volume={m.volume:.1f} m3 "
                  f"-> {out/'walls_fused.stl'}")
    extrude(room, zf, zc, str(out / "rooms.stl"))
    np.savez_compressed(out / "cells.npz",
                        room=room, wall=wall, area=areas, zf=zf, zc=zc)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
