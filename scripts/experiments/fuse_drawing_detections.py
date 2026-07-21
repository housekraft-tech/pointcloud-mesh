"""fuse_drawing_detections.py
--------------------------
Run the drawing-trained detectors on the architect plan, push every detection
through the align_drawing.py transform, and let the LiDAR measure what the
drawing has named. Drawing supplies semantics; LiDAR supplies millimetres.

This is the fusion the two sources were always pointing at. Each half fails
alone in a way the other does not:

  the LiDAR cannot see glass, so it found 2 windows; the walls/windows model
  finds 5 on the drawing, and cannot measure any of them
  the LiDAR cannot name a room, so plateaus had to be paired to drawing rooms
  by AREA RANK, which silently swapped the living room with a bedroom; the
  elements model labels every room directly
  width alone called 6 doors "bathroom" against the plan's 3

Outputs, per detection: the drawing's class, the model-frame position, and --
where the LiDAR can see it -- the measured width, head and the difference.
Where the LiDAR cannot measure, that is said instead of substituting the
drawing's own number and calling it a measurement.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\fuse_drawing_detections.py \\
      <floorplan.png> <drawing_transform.json> <annotated_model.obj> \\
      <detailed_modular.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE,
    ELEMENTS_CLASS_THRESHOLDS, WALLS_WINDOWS_CLASSES,
    WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
from scripts.experiments.transfer_drawing_marks import px_to_model
from scripts.experiments.wall_elevations import parse_obj, merge_coplanar, grids
from scripts.experiments import walk_path_openings as W
from scripts.experiments.openings_from_drawing import measure_bounded

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}
OPENING_CLASSES = {"window", "balcony door", "Door"}
PLANE_TOL = 0.50      # m
DEFAULT_CAP = {"window": 2.0, "balcony door": 3.0, "Door": 1.6}
GLAZED_FILL = 0.45    # occupancy across the drawing's span above this = glazed


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def detect(img_path):
    bgr = cv2.imread(str(img_path))
    e = run_model(load("elements"), bgr, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE,
                  ELEMENTS_CLASS_THRESHOLDS)
    w = run_model(load("walls"), bgr, WALLS_WINDOWS_CLASSES,
                  WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    for d in e:
        d["src"] = "elements"
    for d in w:
        d["src"] = "walls_windows"
    log(f"detections: {len(e)} elements + {len(w)} walls/windows")
    return e + w, bgr


def box_to_model(box, tf):
    """Box corners -> model-frame quad (the transform includes a rotation, so an
    axis-aligned pixel box is NOT axis-aligned in the model)."""
    x0, y0, x1, y1 = box
    return np.array([px_to_model((x, y), tf)
                     for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def ceiling_parts(obj_path):
    V = []; g = []
    for ln in open(obj_path):
        if ln.startswith("o "):
            g.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            if g:
                g[-1][2] = len(V)
    V = np.asarray(V)
    return {n: V[a:b] for n, a, b in g if n.startswith("ceiling") and b > a}


def span_fill(occ, a0, z0, ac, width_m, z_floor):
    """Occupancy fraction across the drawing's stated span, at waist height."""
    from scripts.experiments.wall_elevations import CELL
    nz, na = occ.shape
    zs = z0 + (np.arange(nz) + 0.5) * CELL
    band = (zs > z_floor + 0.30) & (zs < z_floor + 1.60)
    if not band.any():
        return 0.0
    c0 = int(round((ac - a0) / CELL))
    half = max(2, int((width_m / 2) / CELL))
    lo, hi = max(0, c0 - half), min(na, c0 + half + 1)
    if lo >= hi:
        return 0.0
    return float(occ[np.ix_(band, np.arange(lo, hi))].mean())


def main(img_path, tf_path, ann_obj, mod_obj, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = json.load(open(tf_path))
    d = json.load(open(mj))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))

    dets, bgr = detect(img_path)
    rooms = [x for x in dets if x["name"] in ROOM_CLASSES]
    opens = [x for x in dets if x["name"] in OPENING_CLASSES]

    # ---- rooms: name the ceiling plateaus
    parts = ceiling_parts(ann_obj)
    cents = {n: P[:, :2].mean(0) for n, P in parts.items()}
    areas = {n: len(P) for n, P in parts.items()}
    named = []
    for i, r in enumerate(rooms):
        quad = box_to_model(r["box"], tf)
        poly = quad.astype(np.float32)
        inside = []
        for n, c in cents.items():
            if cv2.pointPolygonTest(poly, (float(c[0]), float(c[1],)), False) >= 0:
                inside.append(n)
        inside.sort(key=lambda n: -areas[n])
        named.append(dict(room=r["name"], conf=round(float(r["score"]), 3),
                          centre_xy=[round(float(v), 3) for v in quad.mean(0)],
                          ceiling_parts=inside[:4], n_parts=len(inside)))
    log("")
    log("ROOMS named from the drawing:")
    for r in named:
        log(f"  {r['room']:12} conf {r['conf']:.2f}  -> "
            f"{r['n_parts']} ceiling part(s) {r['ceiling_parts'][:2]}")

    # ---- openings: measure with the LiDAR
    G = merge_coplanar({n: P for n, P in parse_obj(mod_obj).items()
                        if n.startswith("wall")})
    planes = {}
    for name, P in G.items():
        if len(P) < 2000:
            continue
        c, n, dv = W.plane_of(P)
        a = (P[:, :2] - c) @ dv
        planes[name] = (c, n, dv, float(a.min()), float(a.max()), P)

    rows = []
    for o in opens:
        quad = box_to_model(o["box"], tf)
        p = quad.mean(0)
        best = None
        for name, (c, n, dv, amin, amax, P) in planes.items():
            perp = abs(float((p - c) @ n))
            along = float((p - c) @ dv)
            if perp > PLANE_TOL or not (amin - 0.3 < along < amax + 0.3):
                continue
            if best is None or perp < best[0]:
                best = (perp, name, along)
        # drawing's own width = the long side of the transformed box
        e0 = np.linalg.norm(quad[1] - quad[0]); e1 = np.linalg.norm(quad[3] - quad[0])
        dw = max(e0, e1)
        rec = dict(cls=o["name"], conf=round(float(o["score"]), 3),
                   model_xy=[round(float(v), 3) for v in p],
                   drawing_width_mm=round(dw * 1000))
        if best is None:
            rec["status"] = "no wall plane in the model here"
            rows.append(rec); continue
        perp, name, along = best
        c, n, dv, amin, amax, P = planes[name]
        _, occ, a0, z0, na, nz = grids(P)
        cap = min(dw * 1.5 + 0.25, DEFAULT_CAP.get(o["name"], 2.0))
        m = measure_bounded(occ, a0, z0, along, z_floor, z_ceil, cap)
        rec.update(wall=name, perp_mm=round(perp * 1000, 1))

        # Is the drawing's span actually a VOID in the LiDAR? Often not. Glass
        # returns off the pane, and a CLOSED door returns off the leaf -- fill
        # alone cannot separate those two, so the label claims neither. What it
        # does establish is that the void method has nothing to measure here,
        # which is how a 2312 mm slider came out at 950 mm. The width-from-void
        # numbers are only trustworthy for openings that stood open.
        fill = span_fill(occ, a0, z0, along, dw, z_floor)
        rec["span_fill"] = round(fill, 2)
        if fill > GLAZED_FILL:
            rec.update(status=f"not a void at scan time (fill {fill:.2f}) - "
                              "glazing or a closed leaf; drawing width stands")
            rows.append(rec); continue

        if m is None:
            rec["status"] = "LiDAR cannot measure here (occluded / no void)"
        elif m["hit_cap"]:
            rec.update(status="unbounded (merges into open space)",
                       lidar_width_mm=m["width_mm"])
        else:
            rec.update(status="measured", lidar_width_mm=m["width_mm"],
                       lidar_head_mm=m["head_mm"], lidar_sill_mm=m["sill_mm"],
                       width_diff_mm=round(m["width_mm"] - dw * 1000))
        rows.append(rec)

    log("")
    log(f"{'class':14} {'conf':>5} {'draw_mm':>8} {'lidar_mm':>9} {'diff':>7} "
        f"{'sill':>6} {'head':>6}  status")
    for r in sorted(rows, key=lambda r: (r["cls"], -r["drawing_width_mm"])):
        lw = r.get("lidar_width_mm", "-")
        df = f"{r['width_diff_mm']:+.0f}" if "width_diff_mm" in r else "-"
        sl = f"{r['lidar_sill_mm']:.0f}" if "lidar_sill_mm" in r else "-"
        hd = f"{r['lidar_head_mm']:.0f}" if "lidar_head_mm" in r else "-"
        log(f"{r['cls']:14} {r['conf']:5.2f} {r['drawing_width_mm']:8} "
            f"{str(lw):>9} {df:>7} {sl:>6} {hd:>6}  {r['status']}")

    ok = [r for r in rows if r["status"] == "measured"]
    if ok:
        dif = np.array([r["width_diff_mm"] for r in ok])
        log("")
        log(f"{len(ok)}/{len(rows)} drawing openings measured in the LiDAR; "
            f"width diff mean {dif.mean():+.0f} mm, median {np.median(dif):+.0f} mm")
    json.dump(dict(rooms=named, openings=rows),
              open(out / "fused_detections.json", "w"), indent=1)
    log(f"wrote {out/'fused_detections.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
