"""house_entrance_path.py
--------------------
STAGE 2 of the house rebuild: the entrance and the walked circulation route.

The scan's own trajectory answers a question no static geometry can: which
way a person actually moves through this flat. The walk starts OUTSIDE the
apartment (it reaches x=11.0 where the flat ends at x=8.1), so the moment it
first crosses the envelope is the main entrance -- found from motion, not from
guessing which door is biggest.

The drawing is then used only to NAME it: the elements model puts a Door and a
FOYER on the plan, and whichever drawing door lies nearest the measured entry
supplies the label. Geometry from the LiDAR, semantics from the drawing, which
is the same split the rest of the pipeline uses.

Outputs the entry point, the approach direction, and the route split into the
part walked outside and the part walked inside, ready for Stage 4 to carry into
the model.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\house_entrance_path.py \\
      <scan.las> <annotated_model.obj> <fused_detections.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.walk_path_openings import walk_path

SMOOTH = 3          # poses: light smoothing, the path is already a median pose
DOOR_NEAR = 1.60    # m: a drawing door this close to the entry names it
ENVELOPE_PAD = 0.12  # m: rooms are clear-internal, so grow to the wall centre


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def envelope_of(obj_path):
    """The flat's true footprint, from the FLOOR slab.

    Not from the room outlines: the foyer and the bedroom corridor are
    untagged (the drawing detector gave them no room box), so a room-union
    envelope has a hole exactly where someone walks in. The first "entry" then
    landed in the middle of the living room. The floor slab has no such gap.
    """
    import cv2
    from scripts.experiments.viz_deviation_plan import parse_groups
    CELL = 0.05
    groups = {}
    for pref in ("floor", "ceiling"):
        groups.update(parse_groups(str(obj_path), pref))
    F = np.vstack(list(groups.values()))[:, :2]
    lo = F.min(0) - 0.4
    nx = int((F[:, 0].max() - lo[0] + 0.4) / CELL) + 1
    ny = int((F[:, 1].max() - lo[1] + 0.4) / CELL) + 1
    g = np.zeros((ny, nx), np.uint8)
    ij = ((F - lo) / CELL).astype(np.int32)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    ij = ij[ok]
    g[ij[:, 1], ij[:, 0]] = 255
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(g, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    env = Polygon([(lo[0] + x * CELL, lo[1] + y * CELL) for x, y in c]).buffer(0)
    if hasattr(env, "geoms"):
        env = max(env.geoms, key=lambda g_: g_.area)
    return env.simplify(0.05)


def main(las_path, obj_path, fused_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fused = json.load(open(fused_path))
    env = envelope_of(obj_path)
    log(f"building envelope {env.area:.1f} m2, bounds "
        f"{[round(v,2) for v in env.bounds]}")

    P = walk_path(las_path)
    # The trajectory is already in the model frame (verified against the model
    # bounds); only smooth it, do not transform it.
    if SMOOTH > 1:
        k = np.ones(SMOOTH) / SMOOTH
        P = np.c_[np.convolve(P[:, 0], k, "same"),
                  np.convolve(P[:, 1], k, "same"),
                  np.convolve(P[:, 2], k, "same")]
        P[:SMOOTH], P[-SMOOTH:] = P[SMOOTH], P[-SMOOTH - 1]

    inside = np.array([env.contains(Point(x, y)) for x, y, _ in P])
    log(f"{len(P)} poses: {int(inside.sum())} inside the flat, "
        f"{int((~inside).sum())} outside")

    # transitions outside -> inside
    entries = []
    for i in range(1, len(P)):
        if inside[i] and not inside[i - 1]:
            seg = LineString([P[i - 1, :2], P[i, :2]])
            hit = seg.intersection(env.exterior)
            if hit.is_empty:
                pt = P[i, :2]
            else:
                pt = np.asarray((hit.geoms[0] if hasattr(hit, "geoms") else hit)
                                .coords[0])
            head = P[i, :2] - P[i - 1, :2]
            entries.append(dict(i=i, xy=[round(float(v), 3) for v in pt],
                                heading=[round(float(v), 3) for v in head]))
    log(f"{len(entries)} outside->inside transitions")
    if not entries:
        raise SystemExit("the walk never crosses the envelope -- cannot locate "
                         "an entrance from motion")

    # The scan STARTED inside the flat (pose 0 sits in the living room), so
    # "the first time the walker went in" is meaningless -- it fired on a
    # balcony step. What identifies the front door is the DEEPEST excursion:
    # stepping onto a balcony leaves the slab by ~0.3 m, while stepping into
    # the common corridor here goes 2.9 m out. Take the crossing that belongs
    # to the excursion reaching furthest outside.
    excursions = []
    i = 0
    while i < len(P):
        if inside[i]:
            i += 1
            continue
        j = i
        while j < len(P) and not inside[j]:
            j += 1
        depth = max(env.distance(Point(*P[k, :2])) for k in range(i, j))
        excursions.append((depth, i, j))
        i = j
    excursions.sort(reverse=True)
    for depth, a, b in excursions:
        log(f"  excursion poses {a}-{b-1}: reaches {depth:.2f} m outside")
    depth, a, b = excursions[0]
    door = P[max(a - 1, 0), :2] if a > 0 else P[b % len(P), :2]
    seg = LineString([P[max(a - 1, 0), :2], P[min(a, len(P) - 1), :2]])
    hit = seg.intersection(env.exterior)
    if not hit.is_empty:
        door = np.asarray((hit.geoms[0] if hasattr(hit, "geoms") else hit)
                          .coords[0])
    main_entry = dict(i=int(a), xy=[round(float(v), 3) for v in door],
                      excursion_depth_m=round(float(depth), 2),
                      heading=[round(float(v), 3)
                               for v in (P[min(b, len(P) - 1), :2] -
                                         P[max(a - 1, 0), :2])])
    log(f"MAIN ENTRANCE (deepest excursion, {depth:.2f} m out at pose {a}): "
        f"{main_entry['xy']}")

    # name it from the drawing
    # fused_detections carries no drawing "Door" rows for this scan (the
    # elements model's doors did not survive fusion), so fall back to the
    # openings the LiDAR itself cut, which are named in the annotated model.
    from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of
    label, best = None, 1e9
    for name, b in parse_named_boxes(str(obj_path), ("door", "opening", "archway")):
        c = frame_of(b)[0][:2]
        d = float(np.hypot(*(np.asarray(c) - np.asarray(main_entry["xy"]))))
        if d < best:
            best, label = d, name
    if label is not None and best <= DOOR_NEAR:
        log(f"nearest modelled opening: {label} at {best*1000:.0f} mm")
        main_entry["opening"] = label
        main_entry["named_by_drawing"] = True
    else:
        log(f"no drawing door within {DOOR_NEAR} m "
            f"(nearest {best*1000:.0f} mm) -- entrance stands on the walk alone")
        main_entry["named_by_drawing"] = False

    approach = P[:main_entry["i"]]
    interior = P[main_entry["i"]:]
    log(f"approach outside: {len(approach)} poses, "
        f"interior route: {len(interior)} poses, "
        f"{np.linalg.norm(np.diff(interior[:, :2], axis=0), axis=1).sum():.1f} m walked inside")

    json.dump(dict(entrance=main_entry,
                   all_entries=entries,
                   sensor_z_median=round(float(np.median(P[:, 2])), 3),
                   path=[[round(float(v), 3) for v in q] for q in P],
                   inside_flags=[bool(b) for b in inside]),
              open(out / "entrance_path.json", "w"), indent=1)
    log(f"wrote {out/'entrance_path.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
