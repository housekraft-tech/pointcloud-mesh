"""openings_from_drawing.py
------------------------
Drawing says WHERE an opening is. LiDAR says HOW BIG it actually is.

Detecting openings from the scan alone was the weak link: blind hole detection
needs a fully enclosed void, the walk path only finds openings someone walked
through, and width alone assigned roles that came out 6 bathrooms against the
drawing's 3. The drawing has no such trouble -- every door and window is a clean
gap in a clean wall. So take POSITION from the drawing and DIMENSION from the
LiDAR, which is the half each source is good at.

How the drawing's openings are found: close the wall mask with a kernel wider
than any door, subtract the original mask, and what remains is exactly the set
of gaps that were bridged -- the doors and windows. Each gap is transferred
through the align_drawing.py transform, matched to the wall plane it lies on,
and then measured against the LiDAR occupancy at that spot.

Reported per opening: drawing width, LiDAR width, and the difference. Where the
LiDAR cannot measure (furniture occluding the reveal) that is said, rather than
falling back on the drawing and presenting it as a measurement.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\openings_from_drawing.py \\
      <floorplan.png> <drawing_transform.json> <detailed_modular.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.align_drawing import drawing_wall_points
from scripts.experiments.wall_elevations import parse_obj, merge_coplanar, grids, CELL
from scripts.experiments import walk_path_openings as W
from scripts.experiments.transfer_drawing_marks import px_to_model

OPEN_MIN_MM = 550     # narrower than any door in this practice
OPEN_MAX_MM = 2600    # wider is a room mouth, not an opening
STRIP_MIN_PX = 30     # ink per column/row for it to count as a wall run
DEDUP_PX = 40         # same doorway seen in two overlapping wall strips
PLANE_TOL = 0.45      # m: how close a gap must lie to a wall plane
WIDTH_CAP_FACTOR = 1.5  # growth cap = drawing width x this ...
WIDTH_CAP_PAD = 0.25    # ... plus this, to allow a genuinely wider as-built
STD = [(1000, "main entrance"), (900, "bedroom"),
       (800, "kitchen / utility"), (700, "bathroom / WC")]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def drawing_gaps(img_path, tf):
    """Find openings as gaps ALONG each wall run.

    A global morphological close does not work: the kernel has to be wider than
    a door (~63 px here), but a toilet is only ~108 px across, so the close
    swallows whole small rooms and the subtraction returns room interiors
    instead of doorways.

    The plan is orthogonal, so walls are runs of columns (or rows) with high
    ink. Take each run, look along it, and the stretches with no wall pixel are
    the openings -- the same gap-along-a-plane logic used on the LiDAR."""
    pts, box, bgr = drawing_wall_points(img_path)
    xs = pts[:, 0].astype(int); ys = (-pts[:, 1]).astype(int)
    H, W_ = bgr.shape[:2]
    mask = np.zeros((H, W_), np.uint8)
    mask[ys, xs] = 1

    mm_px = tf["scale_m_per_px"] * 1000
    lo_px = OPEN_MIN_MM / mm_px
    hi_px = OPEN_MAX_MM / mm_px
    out = []
    gapimg = np.zeros_like(mask)

    for axis in (0, 1):                      # 0 = vertical walls, 1 = horizontal
        prof = mask.sum(axis=axis)           # ink per column (or per row)
        strong = prof > STRIP_MIN_PX
        lab, n = ndimage.label(strong)
        for k in range(1, n + 1):
            idx = np.flatnonzero(lab == k)
            if len(idx) < 2:
                continue
            band = mask[:, idx] if axis == 0 else mask[idx, :]
            occ = band.any(axis=1 if axis == 0 else 0)
            on = np.flatnonzero(occ)
            if on.size < 20:
                continue
            seg = occ[on.min():on.max() + 1]
            holes, m = ndimage.label(~seg)
            for j in range(1, m + 1):
                w = int((holes == j).sum())
                if not (lo_px <= w <= hi_px):
                    continue
                pos = np.flatnonzero(holes == j)
                centre = on.min() + (pos.min() + pos.max()) / 2
                along = float(idx.mean())
                cx, cy = (along, centre) if axis == 0 else (centre, along)
                out.append(dict(px=(cx, cy), span_px=w,
                                width_mm=round(w * mm_px),
                                orient="vertical" if axis == 0 else "horizontal"))
                if axis == 0:
                    gapimg[on.min() + pos.min():on.min() + pos.max() + 1,
                           idx.min():idx.max() + 1] = 1
                else:
                    gapimg[idx.min():idx.max() + 1,
                           on.min() + pos.min():on.min() + pos.max() + 1] = 1

    # a doorway seen in two overlapping wall strips appears twice
    out.sort(key=lambda g: (-g["span_px"]))
    dedup = []
    for g in out:
        if any((abs(g["px"][0] - h["px"][0]) < DEDUP_PX and
                abs(g["px"][1] - h["px"][1]) < DEDUP_PX) for h in dedup):
            continue
        dedup.append(g)
    log(f"drawing: {len(dedup)} openings as gaps along wall runs "
        f"({len(out)} before dedup)")
    return dedup, mask, gapimg


def measure_bounded(occ, a0, z0, ac, z_floor, z_ceil, cap_m):
    """Measure the opening at `ac`, refusing to grow further than cap_m.

    Returns hit_cap=True when the clear span reaches the cap on both sides,
    which means no reveal was found and the number is not a door width."""
    nz, na = occ.shape
    zs = z0 + (np.arange(nz) + 0.5) * CELL
    walk = (zs > z_floor + 0.30) & (zs < z_floor + 1.60)
    if not walk.any():
        return None
    clear = occ[walk, :].mean(axis=0) < 0.35
    c0 = int(round((ac - a0) / CELL))
    if not (0 <= c0 < na):
        return None
    if not clear[c0]:
        near = [c for c in range(max(0, c0 - 8), min(na, c0 + 9)) if clear[c]]
        if not near:
            return None
        c0 = min(near, key=lambda c: abs(c - c0))
    # cap_m is the TOTAL allowed width, so each side gets half of it
    lim = max(1, int((cap_m / 2) / CELL))
    cl = c0
    while cl - 1 >= 0 and clear[cl - 1] and (c0 - cl) < lim:
        cl -= 1
    cr = c0
    while cr + 1 < na and clear[cr + 1] and (cr - c0) < lim:
        cr += 1
    hit = ((c0 - cl) >= lim) and ((cr - c0) >= lim)
    above = np.flatnonzero(zs > z_floor + 1.60)
    heads = [zs[above[occ[above, c]].min()] for c in range(cl, cr + 1)
             if occ[above, c].any()]
    head = float(np.median(heads)) - z_floor if heads else float(z_ceil - z_floor)
    below = np.flatnonzero(zs < z_floor + 0.30)
    sills = [zs[below[occ[below, c]].max()] for c in range(cl, cr + 1)
             if occ[below, c].any()]
    sill = float(np.median(sills)) - z_floor if sills else 0.0
    return dict(width_mm=round((cr - cl + 1) * CELL * 1000),
                head_mm=round(head * 1000, 1), sill_mm=round(sill * 1000, 1),
                hit_cap=bool(hit))


def role_of(w_mm):
    b = min(STD, key=lambda s: abs(s[0] - w_mm))
    return b[1] if abs(b[0] - w_mm) <= 150 else None


def main(img_path, tf_path, obj_path, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = json.load(open(tf_path))
    d = json.load(open(mj))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))

    gaps, mask, gapimg = drawing_gaps(img_path, tf)
    G = merge_coplanar({n: P for n, P in parse_obj(obj_path).items()
                        if n.startswith("wall")})
    planes = {}
    for name, P in G.items():
        if len(P) < 2000:
            continue
        c, n, dv = W.plane_of(P)
        a = (P[:, :2] - c) @ dv
        planes[name] = (c, n, dv, float(a.min()), float(a.max()), P)
    log(f"{len(planes)} model wall planes")

    rows = []
    for g in gaps:
        p = px_to_model(g["px"], tf)
        best = None
        for name, (c, n, dv, amin, amax, P) in planes.items():
            perp = abs(float((p - c) @ n))
            along = float((p - c) @ dv)
            if perp > PLANE_TOL or not (amin - 0.2 < along < amax + 0.2):
                continue
            if best is None or perp < best[0]:
                best = (perp, name, along)
        rec = dict(drawing_px=[round(v, 1) for v in g["px"]],
                   model_xy=[round(float(p[0]), 3), round(float(p[1]), 3)],
                   drawing_width_mm=g["width_mm"],
                   role_from_drawing_width=role_of(g["width_mm"]))
        if best is None:
            rec.update(matched_wall=None,
                       status="no wall plane in the model at this position")
            rows.append(rec); continue

        perp, name, along = best
        c, n, dv, amin, amax, P = planes[name]
        _, occ, a0, z0, na, nz = grids(P)
        # Bound the growth to the neighbourhood the drawing points at. Unbounded,
        # the clear span runs straight past the doorway into whatever open space
        # adjoins it -- that is how a 601 mm door measured 1450 mm.
        cap = g["width_mm"] / 1000.0 * WIDTH_CAP_FACTOR + WIDTH_CAP_PAD
        m = measure_bounded(occ, a0, z0, along, z_floor, z_ceil, cap)
        rec.update(matched_wall=name, perp_offset_mm=round(perp * 1000, 1),
                   search_cap_mm=round(cap * 1000))
        if m is None:
            rec.update(status="LiDAR could not measure here (occluded reveal)")
        elif m["hit_cap"]:
            rec.update(status="unbounded: no reveal found within the cap "
                              "(opening merges into open space)",
                       lidar_width_mm=m["width_mm"])
        elif m["width_mm"] < 0.5 * g["drawing_width_mm"] if False else \
                m["width_mm"] < 0.5 * g["width_mm"]:
            rec.update(status="partially blocked (furniture in the opening)",
                       lidar_width_mm=m["width_mm"],
                       lidar_head_mm=m["head_mm"])
        else:
            rec.update(status="measured",
                       lidar_width_mm=m["width_mm"],
                       lidar_head_mm=m["head_mm"],
                       lidar_sill_mm=m["sill_mm"],
                       width_diff_mm=round(m["width_mm"] - g["width_mm"], 1))
        rows.append(rec)

    json.dump(rows, open(out / "openings_from_drawing.json", "w"), indent=1)

    ok = [r for r in rows if r.get("status") == "measured"]
    log("")
    log(f"{'#':>3} {'drawing_mm':>10} {'lidar_mm':>9} {'diff':>7} {'head':>6} "
        f"{'role (drawing width)':22} {'wall'}")
    for i, r in enumerate(rows):
        if r.get("status") != "measured":
            log(f"{i:>3} {r['drawing_width_mm']:>10} {'-':>9} {'-':>7} {'-':>6} "
                f"{str(r['role_from_drawing_width']):22} {r['status']}")
            continue
        log(f"{i:>3} {r['drawing_width_mm']:>10} {r['lidar_width_mm']:>9} "
            f"{r['width_diff_mm']:>+7.0f} {r['lidar_head_mm']:>6.0f} "
            f"{str(r['role_from_drawing_width']):22} {r['matched_wall']}")

    if ok:
        dif = np.array([r["width_diff_mm"] for r in ok])
        log("")
        log(f"{len(ok)}/{len(rows)} drawing openings measured in the LiDAR")
        log(f"width difference: mean {dif.mean():+.0f} mm, median "
            f"{np.median(dif):+.0f} mm, spread {dif.min():+.0f}..{dif.max():+.0f} mm")
        roles = {}
        for r in rows:
            k = r["role_from_drawing_width"] or "(non-standard)"
            roles[k] = roles.get(k, 0) + 1
        log("roles implied by DRAWING widths: "
            + ", ".join(f"{k}={v}" for k, v in sorted(roles.items())))
    log(f"wrote {out/'openings_from_drawing.json'}")

    vis = np.zeros(mask.shape + (3,), np.uint8)
    vis[mask > 0] = (80, 80, 80)
    vis[gapimg > 0] = (0, 200, 255)
    for i, r in enumerate(rows):
        x, y = int(r["drawing_px"][0]), int(r["drawing_px"][1])
        c = (0, 230, 90) if r.get("status") == "measured" else (0, 80, 255)
        cv2.circle(vis, (x, y), 13, c, 2)
        cv2.putText(vis, str(i), (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, c, 1)
    cv2.imwrite(str(out / "drawing_openings.png"), vis)
    log(f"wrote {out/'drawing_openings.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
