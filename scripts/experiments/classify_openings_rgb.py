"""classify_openings_rgb.py
------------------------
Decide what each opening actually IS, using two independent signals that
geometry alone cannot provide.

RGB  koushik carries the model but exported RGB as all zeros, so colour is
     sampled from the mujammel scan through the rigid transform from
     register_scans.py. For each opening we look at what is visible THROUGH it
     on each side. An opening onto a balcony sees daylight -- the camera
     overexposes outdoors, so those points are far brighter and less saturated
     than any interior surface. An interior doorway sees another room, at
     interior brightness on both sides. The leaf itself is sampled too: a wooden
     door reads warm (red > blue).

WIDTH Indian practice (NBC 2016) fixes door widths by role -- 1000 mm main,
     900 mm bedroom, 800 mm kitchen/utility, 700 mm bathroom/WC, all at a
     2100 mm height. Snapping a measured width to the nearest standard names
     the door's ROLE and, more usefully, prints the deviation: a door 60 mm off
     its standard is a real as-built deviation, which is the product.

Openings are read straight out of annotated_model.obj -- each is an 8-corner
slab, so its plane, width and height come back exactly as they were written.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\classify_openings_rgb.py \\
      <annotated_model.obj> <rgb_scan.las> <scan_transform.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np

# NBC 2016 / common Indian residential practice, leaf width in mm
STANDARD_DOORS = [
    (1000, "main entrance"),
    (900,  "bedroom"),
    (800,  "kitchen / utility"),
    (700,  "bathroom / WC"),
]
STD_HEIGHT_MM = 2100          # universal in this practice; user confirmed 7 ft
SNAP_TOL_MM = 150             # beyond this the width matches no standard

NEAR_LO, NEAR_HI = 0.25, 1.80  # m: sample this far through the opening
IN_PLANE_PAD = 0.05            # m: shrink the aperture so reveals do not leak
LEAF_HALF = 0.10               # m: slab around the plane = the leaf itself
MIN_PTS = 60                   # below this the colour verdict is unsupported
DAYLIGHT_RATIO = 1.30          # brighter than this x the interior baseline...
DAYLIGHT_SAT = 0.16            # ...and this washed out = blown-out daylight
UNROOFED = 0.35                # ceiling cover below this beyond an opening = open sky
ROOFED_INTERIOR = 0.85         # a genuine interior doorway is roofed at least this much


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_named_boxes(path, prefixes):
    V = []; groups = []
    for ln in open(path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            if groups:
                groups[-1][2] = len(V)
    V = np.asarray(V)
    return [(n, V[a:b]) for n, a, b in groups
            if b - a == 8 and n.startswith(prefixes)]


def frame_of(box):
    """Recover (centre, normal, in-plane axes, extents) from the 8 corners.
    The thinnest principal axis is the wall normal."""
    c = box.mean(0)
    Q = box - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    ext = np.array([np.ptp(Q @ Vt[i]) for i in range(3)])
    k = int(np.argmin(ext))                 # thickness axis = wall normal
    n = Vt[k] / np.linalg.norm(Vt[k])
    others = [i for i in range(3) if i != k]
    # in-plane: the more vertical of the two is the height axis
    a1, a2 = Vt[others[0]], Vt[others[1]]
    if abs(a1[2]) > abs(a2[2]):
        up, along = a1, a2
        e_up, e_al = ext[others[0]], ext[others[1]]
    else:
        up, along = a2, a1
        e_up, e_al = ext[others[1]], ext[others[0]]
    return c, n, along / np.linalg.norm(along), up / np.linalg.norm(up), e_al, e_up


def load_rgb(las_path, T):
    import laspy
    las = laspy.read(las_path)
    P = np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)].astype(float)
    C = np.c_[np.asarray(las.red), np.asarray(las.green),
              np.asarray(las.blue)].astype(float) / 65535.0
    T = np.asarray(T)
    P = P @ T[:3, :3].T + T[:3, 3]
    log(f"{len(P):,} RGB points transformed into the model frame")
    return P, C


def sample(P, C, c, n, along, up, half_w, half_h, lo, hi):
    """Points inside the aperture prism, between lo and hi along +normal."""
    rel = P - c
    s = rel @ n
    m = (s > lo) & (s < hi)
    if not m.any():
        return np.empty((0, 3))
    rel = rel[m]
    a = np.abs(rel @ along); u = np.abs(rel @ up)
    m2 = (a < half_w) & (u < half_h)
    return C[m][m2]


def lum(C):
    return float(np.median(0.2126 * C[:, 0] + 0.7152 * C[:, 1] + 0.0722 * C[:, 2]))


def snap_width(w_mm):
    best = min(STANDARD_DOORS, key=lambda s: abs(s[0] - w_mm))
    delta = w_mm - best[0]
    if abs(delta) > SNAP_TOL_MM:
        return None, None, None
    return best[0], best[1], round(delta, 1)


def ceiling_grid(obj_path, cell=0.12):
    """XY occupancy of the ceiling slabs.

    This is the reliable exterior test, and it needs no colour: the flat is
    roofed, a balcony is not. If the space beyond an opening has no ceiling
    over it, the opening leads outside. (mujammel's RGB turns out to be ~76%
    greyscale -- chroma < 0.05, channels correlated 0.92-0.96 -- so brightness
    is the only colour signal available and it is confounded by range and
    incidence angle. Geometry carries this one.)"""
    V = []; groups = []
    for ln in open(obj_path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            if groups:
                groups[-1][2] = len(V)
    V = np.asarray(V)
    P = np.vstack([V[a:b] for n, a, b in groups
                   if n.startswith("ceiling") and b > a])
    ij = np.floor(P[:, :2] / cell).astype(int)
    return set(map(tuple, ij)), cell


def roofed(cells, cell, c, n, along, half_w, lo, hi, sign):
    """Fraction of the footprint beyond the opening that has ceiling over it."""
    ts = np.arange(lo, hi, cell / 2)
    us = np.arange(-half_w, half_w, cell / 2)
    pts = (c[None, None, :2] + sign * n[None, None, :] * ts[:, None, None]
           + along[None, None, :] * us[None, :, None]).reshape(-1, 2)
    ij = np.floor(pts / cell).astype(int)
    hit = sum(1 for k in map(tuple, ij) if k in cells)
    return hit / max(len(ij), 1)


def main(obj_path, las_path, tf_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = json.load(open(tf_path))
    if not tf.get("usable"):
        raise SystemExit(f"registration not usable (fitness {tf['fitness']:.2f}, "
                         f"rmse {tf['rmse_mm']:.0f} mm) -- refusing to sample colour")
    log(f"using transform: fitness {tf['fitness']:.3f}, rmse {tf['rmse_mm']:.1f} mm")

    boxes = parse_named_boxes(obj_path, ("door", "balcony_door", "window",
                                         "archway", "opening"))
    log(f"{len(boxes)} openings in the model")
    cells, ccell = ceiling_grid(obj_path)
    log(f"ceiling footprint: {len(cells):,} cells of {ccell} m")
    P, C = load_rgb(las_path, tf["transform"])
    scene_lum = lum(C[::37])
    log(f"scene median luminance {scene_lum:.3f}")

    rows = []
    for name, box in boxes:
        c, n, along, up, w, h = frame_of(box)
        hw = max(w / 2 - IN_PLANE_PAD, 0.10)
        hh = max(h / 2 - IN_PLANE_PAD, 0.10)

        front = sample(P, C, c, n, along, up, hw, hh, NEAR_LO, NEAR_HI)
        back = sample(P, C, c, n, along, up, hw, hh, -NEAR_HI, -NEAR_LO)
        leaf = sample(P, C, c, n, along, up, hw, hh, -LEAF_HALF, LEAF_HALF)

        rec = dict(name=name, width_mm=round(w * 1000), height_mm=round(h * 1000),
                   n_front=len(front), n_back=len(back), n_leaf=len(leaf))

        # ---- width -> role (only meaningful for door-like openings)
        if name.startswith(("door", "opening")):
            std, role, delta = snap_width(w * 1000)
            rec.update(std_width_mm=std, inferred_role=role, width_dev_mm=delta)
        if not name.startswith(("archway", "window")):
            rec["height_dev_vs_2100_mm"] = round(h * 1000 - STD_HEIGHT_MM, 1)

        # ---- RGB -> interior or exterior
        # Do NOT compare front against back. The scanner stood on one side, so
        # it sees the far room through the aperture (tens of thousands of
        # points) while the near side is empty air (zero) -- the asymmetry is
        # occlusion, not brightness. Take whichever side actually has the
        # through-view and compare it against the interior baseline.
        through = front if len(front) >= len(back) else back
        side = "front" if len(front) >= len(back) else "back"
        if len(through) < MIN_PTS:
            rec.update(rgb_verdict="unsupported (nothing visible through it)")
        else:
            lt = lum(through)
            sat = float(np.median(through.max(1) - through.min(1)))
            rec.update(through_side=side, n_through=len(through),
                       lum_through=round(lt, 3),
                       lum_vs_scene=round(lt / scene_lum, 2),
                       saturation=round(sat, 3))
            # daylight blows out the camera: bright AND washed out
            if lt > scene_lum * DAYLIGHT_RATIO and sat < DAYLIGHT_SAT:
                rec["rgb_verdict"] = "EXTERIOR (daylight through it)"
            elif lt > scene_lum * DAYLIGHT_RATIO:
                rec["rgb_verdict"] = "bright but saturated (lit interior?)"
            else:
                rec["rgb_verdict"] = "interior"
        # ---- geometry -> interior or exterior (the reliable one)
        rf = roofed(cells, ccell, c, n[:2] / np.linalg.norm(n[:2]),
                    along[:2] / np.linalg.norm(along[:2]), hw, 0.40, 2.20, +1)
        rb = roofed(cells, ccell, c, n[:2] / np.linalg.norm(n[:2]),
                    along[:2] / np.linalg.norm(along[:2]), hw, 0.40, 2.20, -1)
        rec.update(roofed_front=round(rf, 2), roofed_back=round(rb, 2))
        # Graded, because a COVERED balcony is still roofed. Interior doorways
        # sit at ~0.95-1.00 on both sides; anything materially below that has
        # open sky or a projecting slab beyond it.
        m = min(rf, rb)
        if m < UNROOFED:
            rec["leads"] = "EXTERIOR (open sky beyond)"
        elif m < ROOFED_INTERIOR:
            # NOT a reliable exterior call: an occlusion hole in the ceiling
            # gives the same partial coverage as a covered balcony, and this
            # tier does flag bathroom doors. Reported as uncertain, not decided.
            rec["leads"] = "uncertain (partial roof: covered balcony OR ceiling gap)"
        else:
            rec["leads"] = "interior (roofed both sides)"

        if len(leaf) >= MIN_PTS:
            m = np.median(leaf, axis=0)
            rec["leaf_rgb"] = [round(float(x), 3) for x in m]
            rec["leaf_warm"] = bool(m[0] > m[2] * 1.12)      # wood reads red>blue
        rows.append(rec)

    json.dump(rows, open(out / "openings_classified.json", "w"), indent=1)

    log("")
    log(f"{'opening':30} {'w_mm':>5} {'std':>5} {'dev':>6} {'role':<18} "
        f"{'roof f/b':>9}  {'leads'}")
    for r in sorted(rows, key=lambda r: -r["width_mm"]):
        std = r.get("std_width_mm") or "-"
        dev = f"{r['width_dev_mm']:+.0f}" if r.get("width_dev_mm") is not None else "-"
        role = r.get("inferred_role") or "-"
        v = r.get("rgb_verdict", "-")
        warm = " warm-leaf" if r.get("leaf_warm") else ""
        rr = f"{r.get('roofed_front',0):.2f}/{r.get('roofed_back',0):.2f}"
        log(f"{r['name']:30} {r['width_mm']:5} {std:>5} {dev:>6} {role:<18} "
            f"{rr:>9}  {r.get('leads','-')}{warm}")
    ext = [r for r in rows if "EXTERIOR" in r.get("rgb_verdict", "")]
    log(f"\n{len(ext)} openings see daylight -> exterior (balcony/entrance): "
        + ", ".join(r["name"] for r in ext))
    log(f"wrote {out/'openings_classified.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
