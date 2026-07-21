"""viz_floorplan_clean.py
--------------------
A presentation floorplan that looks drafted, not scanned.

This is the DELIBERATELY IDEALISED sheet. The measured truth lives on
viz_deviation_plan.py; here every wall is straightened to the building's
orthogonal grid, snapped to shared gridlines, given one uniform thickness, and
rooms are closed into solid-poche walls with door swings. As-built deviation is
intentionally NOT shown -- do not read millimetres off this drawing, read them
off the deviation plan.

How it is built:
  rooms      each room's measured wall-bounded outline (from room_areas.py),
             rotated onto the grid and snapped so shared walls between rooms
             land on one line
  walls      the classic architect trick: fill the whole envelope solid, then
             cut each room's interior back by half a wall thickness. What is
             left between rooms is a uniform-thickness wall, and every room is
             closed by construction -- no floating stubs, no open corners
  openings   doors as a gap in the poche with a 90 deg swing arc, windows as a
             glazing line, balcony doors as a wide gap with a swing -- positions
             from the LiDAR openings, snapped onto their wall

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\viz_floorplan_clean.py \\
      <annotated_model.obj> <fused_detections.json> <modular_manifest.json> \\
      <out_png>
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Arc, Rectangle
from shapely.geometry import Polygon, box
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.viz_deviation_plan import parse_groups, wall_grid_angle
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of

WALL_T = 0.11         # m: one uniform interior wall thickness (poche)
EXT_T = 0.20          # m: exterior wall a touch heavier, as on a real plan
SNAP = 0.22           # m: vertices within this of a shared gridline snap to it
QUANT = 0.02          # m: vertex quantisation before rectifying
WALL_DARK = "#2b2b2b"
ROOM_FILL = "#ffffff"
LINE = "#2b2b2b"


def cluster(vals, tol):
    """1-D gridlines: sorted value -> representative of its merged cluster."""
    vals = sorted(vals)
    reps = {}
    group = [vals[0]]
    for v in vals[1:]:
        if v - group[-1] <= tol:
            group.append(v)
        else:
            m = float(np.mean(group))
            for g in group:
                reps[round(g, 4)] = m
            group = [v]
    m = float(np.mean(group))
    for g in group:
        reps[round(g, 4)] = m
    return reps


MIN_EDGE = 0.28       # m: drop wall jogs shorter than this (raster stair noise)


def snap1(v, reps):
    return reps.get(round(v, 4), v)


def rectify(poly_xy, xrep, yrep):
    """Snap a room outline to shared gridlines, force edges orthogonal, and
    drop the tiny jogs that make a scan outline look ragged."""
    P = [(snap1(x, xrep), snap1(y, yrep)) for x, y in poly_xy]
    # force each edge pure H or V (collapse the smaller delta to zero)
    out = [P[0]]
    for x1, y1 in P[1:]:
        x0, y0 = out[-1]
        if abs(x1 - x0) < abs(y1 - y0):
            x1 = x0
        else:
            y1 = y0
        if (x1, y1) != (x0, y0):
            out.append((x1, y1))
    # The ring closes from out[-1] back to out[0], and that edge never went
    # through the loop above -- left alone it stays diagonal and puts a bevelled
    # wall in the corner. Insert a corner vertex so the closure is orthogonal.
    if len(out) >= 2:
        x0, y0 = out[0]
        xe, ye = out[-1]
        if abs(xe - x0) > 1e-9 and abs(ye - y0) > 1e-9:
            xp, yp = out[-2]
            out.append((xe, y0) if abs(xe - xp) < 1e-9 else (x0, ye))

    # drop micro-edges: a run shorter than MIN_EDGE is stair noise, not a real
    # jog. Remove its interior vertex so the two neighbours join on one line.
    changed = True
    while changed and len(out) > 4:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            if np.hypot(b[0] - a[0], b[1] - a[1]) < MIN_EDGE:
                # snap b onto a's line so a->c stays orthogonal
                nb = (a[0], c[1]) if abs(c[0] - a[0]) < abs(c[1] - a[1]) \
                    else (c[0], a[1])
                out[i] = nb
                # merge duplicates
                out = [out[j] for j in range(len(out))
                       if out[j] != out[j - 1]]
                changed = True
                break
    return out


def main(obj_path, fused_path, manifest_path, out_png):
    fused = json.load(open(fused_path))
    man = json.load(open(manifest_path))
    cinfo = {o["name"]: o for o in man["objects"]
             if o["name"].startswith("ceiling")}

    walls = parse_groups(obj_path, "wall")
    cols = parse_groups(obj_path, "column")
    boxes = parse_named_boxes(obj_path, ("door", "balcony_door", "window",
                                         "archway", "opening"))

    # ---- rotate everything onto the building grid so walls are H / V
    theta = wall_grid_angle(walls)
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    allw = np.vstack(list(walls.values()))[:, :2]
    pivot = allw.mean(0)

    def rot(P):
        P = np.asarray(P, float)
        flat = P.ndim == 1
        Q = np.atleast_2d(P).copy()
        Q[:, :2] = (Q[:, :2] - pivot) @ R.T + pivot
        return Q[0] if flat else Q

    # ---- snap room outlines onto shared gridlines
    room_polys = []
    outlines = []
    # Simplify BEFORE rectifying. The outline comes off a 50 mm raster, so it
    # carries hundreds of stair-step vertices; snapping those to gridlines just
    # produces neat-looking noise. Douglas-Peucker first leaves the real
    # corners, and the gridlines are then derived from corners that mean
    # something.
    for r in fused["rooms"]:
        o = r.get("outline")
        if not o or len(o) < 4:
            continue
        p = Polygon(rot(np.asarray(o, float))[:, :2]).buffer(0)
        if p.is_empty:
            continue
        if hasattr(p, "geoms"):
            p = max(p.geoms, key=lambda g: g.area)
        p = p.simplify(0.16, preserve_topology=True)
        outlines.append((r, np.asarray(p.exterior.coords)[:-1]))

    allx = [round(x, 4) for _, o in outlines for x, y in o]
    ally = [round(y, 4) for _, o in outlines for x, y in o]
    xrep = cluster(allx, SNAP)
    yrep = cluster(ally, SNAP)

    for r, o in outlines:
        rc = rectify(o, xrep, yrep)
        if len(rc) < 4:
            # a narrow room (the 0.72 x 2.66 m balcony) can lose an edge to the
            # micro-edge pass. On an idealised plan its snapped bounding
            # rectangle IS the honest shape, so fall back to that rather than
            # dropping the room off the drawing entirely.
            ax0, ay0 = o.min(0); ax1, ay1 = o.max(0)
            sx0, sx1 = snap1(round(ax0, 4), xrep), snap1(round(ax1, 4), xrep)
            sy0, sy1 = snap1(round(ay0, 4), yrep), snap1(round(ay1, 4), yrep)
            # gridline clustering chains, so a narrow room can have both its
            # faces pulled onto ONE line, collapsing it. Keep the measured
            # extent whenever snapping would flatten the room.
            if abs(sx1 - sx0) < 0.30:
                sx0, sx1 = ax0, ax1
            if abs(sy1 - sy0) < 0.30:
                sy0, sy1 = ay0, ay1
            rc = [(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)]
        poly = Polygon(rc).buffer(0)
        if hasattr(poly, "geoms"):
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty or poly.area < 0.3:
            print(f"  dropped {r['room']} ({r.get('area_m2')} m2): "
                  f"outline did not rectify to a usable polygon")
            continue
        room_polys.append((r, poly))

    # ---- poche: solid envelope minus each room shrunk by half a wall
    # mitre_limit keeps sharp corners square -- the default bevels them, which
    # put a diagonal wall in the bottom-left corner that is not in the building
    half = WALL_T / 2.0
    interiors = []
    for r, p in room_polys:
        ins = p.buffer(-half, join_style=2, mitre_limit=8.0)
        if not ins.is_empty:
            interiors.append(ins)
    inner_u = unary_union(interiors)
    outer = unary_union([p for _, p in room_polys]).buffer(
        half + (EXT_T - WALL_T), join_style=2, mitre_limit=8.0)
    poche = outer.difference(inner_u)

    # ---- openings: cut gaps, collect swings to draw after the poche
    from shapely.geometry import Point
    swings, glazings = [], []
    seen = []
    for name, b in boxes:
        b = rot(b)
        c, n, along, up, w, h = frame_of(b)
        c = np.asarray(c[:2]); u = np.asarray(along[:2]); nn = np.asarray(n[:2])
        u = u / (np.linalg.norm(u) + 1e-9); nn = nn / (np.linalg.norm(nn) + 1e-9)
        if name.startswith("balcony"):
            kind, w = "balcony_door", float(np.clip(w, 0.8, 1.8))
        elif name.startswith("window"):
            kind, w = "window", float(np.clip(w, 0.6, 2.0))
        elif name.startswith(("opening", "archway")):
            kind, w = "opening", float(np.clip(w, 0.7, 2.2))
        else:
            # A door leaf is 700-1000 mm here (NBC). Anything wider is a
            # merged detection, and swinging it drew a 2.4 m arc across the
            # living room.
            kind, w = "door", float(np.clip(w, 0.7, 1.05))
        # one swing per doorway: near-coincident boxes are the same opening
        if any(np.hypot(*(c - s)) < 0.55 for s in seen):
            continue
        # an opening must sit ON the building. Ones that do not are detections
        # the idealised outline no longer has a wall for, and drawing them put
        # swings and glazing floating in space outside the plan.
        if not outer.buffer(0.10).contains(Point(*c)):
            continue
        seen.append(c)

        ang = np.degrees(np.arctan2(u[1], u[0]))
        gap = box(-w / 2, -EXT_T, w / 2, EXT_T)
        gap = shp_rotate(gap, ang, origin=(0, 0), use_radians=False)
        gap = shp_translate(gap, c[0], c[1])
        if kind == "window":
            glazings.append((c, u, nn, w))
            continue                     # keep the wall, just draw glazing
        poche = poche.difference(gap)
        if kind == "opening":
            continue                     # cased opening: gap, but no leaf
        hinge = c - u * (w / 2)
        side = nn if inner_u.contains(Point(*(c + nn * 0.25))) else -nn
        swings.append((hinge, u, side, w, kind))

    # ---- draw
    fig, ax = plt.subplots(figsize=(17, 16))
    ax.set_facecolor("white")

    def draw_shp(geom, **kw):
        geoms = getattr(geom, "geoms", [geom])
        for g in geoms:
            if g.is_empty:
                continue
            ax.add_patch(MplPoly(np.asarray(g.exterior.coords), **kw))
            for ring in g.interiors:
                ax.add_patch(MplPoly(np.asarray(ring.coords), fc="white",
                                     ec="none", zorder=kw.get("zorder", 1) + 0.1))

    draw_shp(inner_u, fc=ROOM_FILL, ec="none", zorder=1)
    draw_shp(poche, fc=WALL_DARK, ec=WALL_DARK, lw=0, zorder=3)

    # door swings
    for hinge, u, side, w, kind in swings:
        leaf = hinge + side * w
        ax.plot([hinge[0], leaf[0]], [hinge[1], leaf[1]], color=LINE,
                lw=1.1, zorder=5, solid_capstyle="round")
        a0 = np.degrees(np.arctan2(u[1], u[0]))
        a1 = np.degrees(np.arctan2(side[1], side[0]))
        lo, hi = sorted([a0, a1])
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        ax.add_patch(Arc(hinge, 2 * w, 2 * w, angle=0, theta1=lo, theta2=hi,
                         color=LINE, lw=0.8, zorder=5))

    # window glazing: a thin line across the wall with a slot around it
    for c, u, nn, w in glazings:
        p0 = c - u * w / 2; p1 = c + u * w / 2
        for s in (-1, 1):
            q0 = p0 + nn * s * EXT_T / 2.4; q1 = p1 + nn * s * EXT_T / 2.4
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=LINE, lw=0.6, zorder=5)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#5c6bc0", lw=1.0, zorder=5)

    # columns as solid squares
    for n, P in cols.items():
        P = rot(P)
        cx, cy = P[:, :2].mean(0)
        s = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 0.15)
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc=WALL_DARK,
                               ec=WALL_DARK, hatch="////", zorder=4))

    # room labels
    for r, p in room_polys:
        cx, cy = p.representative_point().coords[0]
        parts = [q for q in r["ceiling_parts"] if q in cinfo]
        h = cinfo[max(parts, key=lambda q: cinfo[q]["area_m2"])]["height_mm"] \
            if parts else None
        txt = r["room"].upper()
        if r.get("area_m2"):
            txt += f"\n{r['area_m2']:.1f} m²"
        if h:
            txt += f"   h {h:.0f}"
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9,
                weight="bold", color="#1a1a1a", zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    xmin, ymin, xmax, ymax = outer.bounds
    m = 0.5
    ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(ymin - m, ymax + m)
    ax.set_title("AS-BUILT FLOOR PLAN  ·  koushik flat\n"
                 "idealised presentation drawing — geometry from LiDAR, "
                 "straightened to the building grid; read deviations off the "
                 "deviation plan", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}: {len(room_polys)} rooms, "
          f"{len(swings)} door swings, {len(glazings)} windows")


if __name__ == "__main__":
    main(*sys.argv[1:5])
