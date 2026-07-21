"""room_areas.py
------------
Per-room floor area, measured from the LiDAR and tagged by the drawing.

The old number came from summing the ceiling plateaus assigned to each room,
and it was wrong for a structural reason, not a tuning one: a plateau is a
HEIGHT LEVEL, not a room. The kitchen and the living room share one flat slab,
so region growing merged them into a single 21 m2 plateau; whichever room won
that plateau took all of it and the other was left with scraps. The kitchen
read 3.7 m2 against the drawing's 11.4.

Rooms are separated by WALLS, not by ceiling height, so walls are what has to
do the cutting:

  footprint   ceiling cells UNION floor cells. Neither alone is enough -- the
              floor is occluded by furniture (108 m2), the ceiling is shadowed
              at the wall line, and each fills the other's holes.
  barrier     the wall point cloud rasterised. Measured at 100 mm median band
              against 107 mm of fitted wall thickness, so it is the wall, not
              a fattened proxy for it.
  seeds       the drawing's room boxes, eroded so a slightly misregistered box
              cannot seed the room next door.
  growth      geodesic propagation through free space only.

No background marker is used. That is not a detail: a background seed touches
every room's perimeter, wins the whole ring of cells against the walls, and
takes about 30% off every room.

Leftovers are split by shape, because the two kinds mean opposite things. A
thin strip stranded behind a wall band is a rasterising artefact and is given
to the room whose drawing footprint contains it. A room-shaped blob is a real
space the detector did not name -- here the foyer (1.70 x 1.50 m against the
drawing's 1650 x 1300) and the bedroom corridor -- and is reported as UNTAGGED
rather than folded into a neighbour, which would both inflate that neighbour
and hide a whole space.

Reading the numbers: this measures CLEAR INTERNAL area, wall face to wall face.
The architect's room dimensions are to centre lines -- they run a median 185 mm
larger per axis, which is one wall thickness -- so they are not the right thing
to check against. The drawing's own carpet-area panel is, and against it the
total lands 105.2 m2 against 106.2 (-0.9%).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\room_areas.py \\
      <annotated_model.obj> <rooms_with_quads.json> [out.json]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.viz_deviation_plan import parse_groups

CELL = 0.05        # m: grid pitch, half the thinnest wall
ERODE_M = 0.30     # m: seed inset from the drawing box
THIN_M = 0.45      # m: thinner than this is a strip, not a space
MIN_SPACE = 0.50   # m2: report untagged spaces at least this big


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _grid(pts, lo, nx, ny):
    g = np.zeros((ny, nx), np.uint8)
    ij = ((pts - lo) / CELL).astype(np.int32)
    ok = ((ij[:, 0] >= 0) & (ij[:, 0] < nx) &
          (ij[:, 1] >= 0) & (ij[:, 1] < ny))
    ij = ij[ok]
    g[ij[:, 1], ij[:, 0]] = 255
    return g


def room_regions(obj_path, quads):
    """-> (areas m2 per quad, untagged spaces, label grid, origin).

    `quads` is a list of model-frame room polygons (drawing boxes pushed
    through the registration transform)."""
    from scipy.ndimage import grey_dilation

    ceil = parse_groups(str(obj_path), "ceiling")
    floor = parse_groups(str(obj_path), "floor")
    wall = parse_groups(str(obj_path), "wall")
    if not ceil and not floor:
        raise SystemExit("no ceiling or floor objects -- cannot bound rooms")

    F = np.vstack(list(ceil.values()) + list(floor.values()))[:, :2]
    W = np.vstack(list(wall.values()))[:, :2]
    lo = np.minimum(F.min(0), W.min(0)) - 0.3
    hi = np.maximum(F.max(0), W.max(0)) + 0.3
    nx = int((hi[0] - lo[0]) / CELL) + 1
    ny = int((hi[1] - lo[1]) / CELL) + 1

    fm = _grid(F, lo, nx, ny)
    wm = _grid(W, lo, nx, ny)
    fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                          iterations=2)
    free = ((fm > 0) & (wm == 0)).astype(np.uint8)
    log(f"grid {nx}x{ny} @ {CELL} m: footprint {int((fm>0).sum())} cells, "
        f"walls {int((wm>0).sum())}, free {free.sum()*CELL**2:.1f} m2")

    er = max(1, int(ERODE_M / CELL))
    lab = np.zeros((ny, nx), np.int32)
    for i, q in enumerate(quads):
        pix = np.round((np.asarray(q)[:, :2] - lo) / CELL).astype(np.int32)
        m = np.zeros((ny, nx), np.uint8)
        cv2.fillPoly(m, [pix], 1)
        me = cv2.erode(m, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
        sel = (me > 0) & (free > 0) & (lab == 0)
        if sel.sum() < 4:            # box too small or barely on the model
            sel = (m > 0) & (free > 0) & (lab == 0)
        lab[sel] = i + 1

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    freeb = free > 0
    for _ in range(4 * max(nx, ny)):
        unl = freeb & (lab == 0)
        if not unl.any():
            break
        lab = np.where(unl, grey_dilation(lab, footprint=cross), lab)

    # leftovers: strips go back to their room, spaces get reported
    un = (freeb & (lab == 0)).astype(np.uint8)
    ncc, cc, stats, cent = cv2.connectedComponentsWithStats(un, 8)
    untagged = []
    for c in range(1, ncc):
        comp = cc == c
        a = float(comp.sum()) * CELL ** 2
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        if 2 * float(dt.max()) * CELL <= THIN_M:
            # stranded behind a wall band, so it touches no room and neighbour
            # voting finds nothing -- ask which room's footprint holds it
            ys, xs = np.where(comp)
            pts = np.c_[lo[0] + xs * CELL, lo[1] + ys * CELL][::3]
            best, bi = 0, None
            for i, q in enumerate(quads):
                poly = np.asarray(q)[:, :2].astype(np.float32)
                k = sum(1 for p in pts
                        if cv2.pointPolygonTest(
                            poly, (float(p[0]), float(p[1])), False) >= 0)
                if k > best:
                    best, bi = k, i
            if bi is not None and best >= 0.5 * len(pts):
                lab[comp] = bi + 1
                continue
            # outside every room: soffit overhanging the envelope, not floor
        untagged.append(dict(area_m2=round(a, 2),
                             centre_xy=[round(float(lo[0] + cent[c][0] * CELL), 2),
                                        round(float(lo[1] + cent[c][1] * CELL), 2)]))

    areas = [round(float((lab == i + 1).sum()) * CELL ** 2, 2)
             for i in range(len(quads))]

    # outline of each room region, so a plan can draw and dimension what was
    # actually measured instead of falling back to the plateau extents
    outlines = []
    for i in range(len(quads)):
        m = (lab == i + 1).astype(np.uint8)
        if not m.any():
            outlines.append([])
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.02 / CELL, True).reshape(-1, 2)
        outlines.append([[round(float(lo[0] + (x + 0.5) * CELL), 3),
                          round(float(lo[1] + (y + 0.5) * CELL), 3)]
                         for x, y in c])
    untagged = [u for u in untagged if u["area_m2"] >= MIN_SPACE]
    untagged.sort(key=lambda u: -u["area_m2"])
    return areas, untagged, outlines, lo


def name(r):
    """Room records come from the detector ("name") or from a fused file
    ("room"); accept either."""
    return r.get("name") or r.get("room") or "?"


def main(obj_path, rooms_json, out_json=None):
    rooms = json.load(open(rooms_json))
    if isinstance(rooms, dict):
        rooms = rooms["rooms"]
    quads = [r["quad"] for r in rooms]
    areas, untagged, outlines, lo = room_regions(obj_path, quads)

    log("")
    log("PER-ROOM CLEAR FLOOR AREA (wall face to wall face):")
    for r, a in zip(rooms, areas):
        log(f"  {name(r):16} {a:6.1f} m2")
    log(f"  {'TOTAL':16} {sum(areas):6.1f} m2")
    if untagged:
        log("")
        log("SPACES THE LIDAR SEES BUT THE DRAWING DETECTOR DID NOT NAME:")
        for u in untagged:
            log(f"  {u['area_m2']:6.1f} m2 at "
                f"({u['centre_xy'][0]:6.2f},{u['centre_xy'][1]:6.2f})")
        log(f"  {sum(u['area_m2'] for u in untagged):6.1f} m2 total")

    if out_json:
        json.dump(dict(cell_m=CELL,
                       rooms=[dict(name=name(r), area_m2=a)
                              for r, a in zip(rooms, areas)],
                       untagged=untagged),
                  open(out_json, "w"), indent=1)
        log(f"wrote {out_json}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
