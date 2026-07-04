"""fuse_floorplan.py
-----------------
Reconstruct an accurate, good-looking floorplan by FUSING:
  * LiDAR geometry   -> accurate metric walls + room clear-dimensions
  * RF-DETR (walls_windows model on the architect drawing) -> windows + balcony
    doors, placed onto the LiDAR walls via a robust Manhattan registration.

Registration: both plans are axis-aligned. Vectorise the drawing walls, then
map the drawing wall raster into the LiDAR frame under each of 8 orientations
(4 rotations x mirror) using the wall bounding boxes for scale+translation, and
keep the orientation whose interior walls best OVERLAP the LiDAR walls. The
window / balcony-door detections are carried through the same transform and
snapped to the nearest LiDAR wall.

Stage 1 here writes a registration overlay so we can confirm alignment before
styling. Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\fuse_floorplan.py <drawing.png> <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.rfdetr_infer import (load, run_model, WALLS_WINDOWS_CLASSES,
                                              WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL, measure_axis, STRIP


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def drawing_walls(drawing_bgr, panel_cut=0.62):
    """Vectorise the architect drawing's walls (Hough on the dark linework in the
    plan region, panel/legend cropped). Returns (segments[(x0,y0,x1,y1)], bbox)."""
    H, W = drawing_bgr.shape[:2]
    g = cv2.cvtColor(drawing_bgr, cv2.COLOR_BGR2GRAY)
    g[:, int(panel_cut * W):] = 255                       # drop unit-stack/legend panel
    dark = (g < 120).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lines = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=60,
                            minLineLength=45, maxLineGap=12)
    segs = []
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:      # keep near-axis (Manhattan) only
                segs.append((int(x0), int(y0), int(x1), int(y1)))
    pts = np.array([[s[0], s[1]] for s in segs] + [[s[2], s[3]] for s in segs])
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    return segs, bbox


def raster_segs(segs, size, thick=3):
    m = np.zeros(size, np.uint8)
    for x0, y0, x1, y1 in segs:
        cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, thick)
    return m


def orient(nx, ny, o):
    """8 orientations on unit-box coords."""
    if o & 1:                       # mirror x
        nx = 1 - nx
    r = o >> 1
    if r == 0:   return nx, ny
    if r == 1:   return ny, 1 - nx
    if r == 2:   return 1 - nx, 1 - ny
    return 1 - ny, nx


def map_pt(p, dbox, lbox, o):
    dx0, dy0, dx1, dy1 = dbox; lx0, ly0, lx1, ly1 = lbox
    nx = (p[0] - dx0) / max(dx1 - dx0, 1); ny = (p[1] - dy0) / max(dy1 - dy0, 1)
    ox, oy = orient(nx, ny, o)
    return (lx0 + ox * (lx1 - lx0), ly0 + oy * (ly1 - ly0))


def regularize(lsegs):
    """Turn loose Manhattan wall segments into a clean wall GRAPH the way a
    floorplan is drawn: snap every wall onto a shared set of gridlines, then
    extend each endpoint onto the crossing gridline so walls actually MEET at
    L/T junctions. Rendering thick strokes of the result unions cleanly at every
    corner (no gaps, no notches)."""
    Hs, Vs = [], []
    for x0, y0, x1, y1 in lsegs:
        if abs(x1 - x0) >= abs(y1 - y0):
            Hs.append([min(x0, x1), max(x0, x1), (y0 + y1) / 2.0])   # xa, xb, y
        else:
            Vs.append([min(y0, y1), max(y0, y1), (x0 + x1) / 2.0])   # ya, yb, x

    def cluster(vals, tol):
        if not vals:
            return []
        s = sorted(vals); groups = [[s[0]]]
        for v in s[1:]:
            if v - groups[-1][-1] <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups]

    tol_c = 0.30 / CELL           # merge wall lines within 30cm onto one gridline
    tol_s = 0.60 / CELL           # extend an endpoint up to 60cm onto a crossing line
    ys = cluster([h[2] for h in Hs], tol_c)
    xs = cluster([v[2] for v in Vs], tol_c)

    def near(v, arr):
        return min(arr, key=lambda a: abs(a - v)) if arr else v
    for h in Hs:
        h[2] = near(h[2], ys)
        if abs(near(h[0], xs) - h[0]) <= tol_s: h[0] = near(h[0], xs)
        if abs(near(h[1], xs) - h[1]) <= tol_s: h[1] = near(h[1], xs)
    for v in Vs:
        v[2] = near(v[2], xs)
        if abs(near(v[0], ys) - v[0]) <= tol_s: v[0] = near(v[0], ys)
        if abs(near(v[1], ys) - v[1]) <= tol_s: v[1] = near(v[1], ys)

    # merge collinear overlapping/abutting runs on the same gridline
    def merge(lines):
        lines = sorted(lines, key=lambda s: (s[2], s[0])); out = []
        for a0, a1, c in lines:
            if out and out[-1][2] == c and a0 <= out[-1][1] + tol_c:
                out[-1][1] = max(out[-1][1], a1)
            else:
                out.append([a0, a1, c])
        return out
    Hs, Vs = merge(Hs), merge(Vs)
    return [(a0, c, a1, c) for a0, a1, c in Hs] + [(c, a0, c, a1) for a0, a1, c in Vs]


def _seg_dist(c, s):
    """distance from point c to segment s=(x0,y0,x1,y1), and the projected point."""
    a = np.array([s[0], s[1]], float); b = np.array([s[2], s[3]], float); p = np.array(c, float)
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), proj


def snap(c, span, lsegs):
    """snap detection centre c to the nearest LiDAR wall; return (proj, axis, halfspan_px)."""
    best = None
    for s in lsegs:
        d, proj = _seg_dist(c, s)
        if best is None or d < best[0]:
            horiz = abs(s[2] - s[0]) >= abs(s[3] - s[1])
            best = (d, proj, "h" if horiz else "v")
    _, proj, axis = best
    return proj, axis, max(span / 2, 0.4 / CELL)


def room_dims(R):
    """clear width/depth per LiDAR room (validated method), for labelling."""
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    wb = (z >= zf + 0.2) & (z <= zc - 0.1)
    xw, yw, zw = x[wb], y[wb], z[wb]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    out = []
    for Lr in R["room_labels"]:
        m = (R["mk"] == Lr)
        if m.sum() < (1.0 / (CELL * CELL)):
            continue
        rd = R["distm"] * m
        cy, cx = np.unravel_index(int(np.argmax(rd)), rd.shape)
        rcx = xmin + cx * CELL; rcy = ymax - cy * CELL
        row = m[cy, :]; c0 = cx; c1 = cx
        while c0 > 0 and row[c0 - 1]: c0 -= 1
        while c1 < W - 1 and row[c1 + 1]: c1 += 1
        col = m[:, cx]; r0 = cy; r1 = cy
        while r0 > 0 and col[r0 - 1]: r0 -= 1
        while r1 < H - 1 and col[r1 + 1]: r1 += 1
        mx = np.abs(yw - rcy) <= STRIP; my = np.abs(xw - rcx) <= STRIP
        if mx.sum() < 30 or my.sum() < 30:
            continue
        rw = measure_axis(xw[mx], zw[mx], rcx, xmin + c0 * CELL, xmin + c1 * CELL)
        rh = measure_axis(yw[my], zw[my], rcy, ymax - r1 * CELL, ymax - r0 * CELL)
        opn = rw["open_l"] or rw["open_r"] or rh["open_l"] or rh["open_r"]
        out.append((cx, cy, rw["span"], rh["span"], opn))
    return out


def render_fused(R, lsegs, wins, bdoors, dbox, lbox, o, out_dir):
    H, W = R["H"], R["W"]
    TOP = 34                                              # margin band for the legend
    plan = np.full((H + TOP, W, 3), 255, np.uint8)
    tpx = max(6, int(round(0.16 / CELL)))

    def P(pt):
        return (int(round(pt[0])), int(round(pt[1])) + TOP)

    # 1. floor fill (clean light gray over reachable interior)
    floor = (R["free"] > 0)
    plan[TOP:][floor] = (245, 244, 242)

    # 2. CLEAN poche from the regularized wall GRAPH: grid-snapped centerlines
    #    that meet at L/T junctions, drawn as thick strokes -> union fills every
    #    corner seamlessly (the architectural way -- no gaps, no notches).
    rsegs = regularize(lsegs)
    wm = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in rsegs:
        cv2.line(wm, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 255, tpx, cv2.LINE_8)
    plan[TOP:][wm > 0] = (40, 40, 40)

    # 3. openings: carve the poche, then draw the architectural symbol.
    def opening(det, half_m):
        b = det["box"]
        c = map_pt(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), dbox, lbox, o)
        e0 = map_pt((b[0], b[1]), dbox, lbox, o); e1 = map_pt((b[2], b[3]), dbox, lbox, o)
        span = np.hypot(e1[0] - e0[0], e1[1] - e0[1])
        proj, axis, _ = snap(c, span, rsegs)
        half = int(np.clip(span / 2, half_m / CELL, 1.6 / CELL))
        t = tpx // 2 + 1
        cx, cy = int(proj[0]), int(proj[1])
        if axis == "h":
            box = (cx - half, cy - t, cx + half, cy + t); along = "h"
        else:
            box = (cx - t, cy - half, cx + t, cy + half); along = "v"
        return box, along, (cx, cy)

    def rect(box, color, fill=True):
        cv2.rectangle(plan, P((box[0], box[1])), P((box[2], box[3])), color, -1 if fill else 2, cv2.LINE_AA)

    for d in bdoors:                                     # balcony door = sliding panels (orange)
        box, along, ctr = opening(d, 0.7)
        rect(box, (255, 255, 255))                       # clear wall
        if along == "h":
            mid = (box[1] + box[3]) // 2
            cv2.line(plan, P((box[0], mid - 3)), P(((box[0] + box[2]) // 2, mid - 3)), (200, 120, 0), 3, cv2.LINE_AA)
            cv2.line(plan, P(((box[0] + box[2]) // 2, mid + 3)), P((box[2], mid + 3)), (200, 120, 0), 3, cv2.LINE_AA)
        else:
            mid = (box[0] + box[2]) // 2
            cv2.line(plan, P((mid - 3, box[1])), P((mid - 3, (box[1] + box[3]) // 2)), (200, 120, 0), 3, cv2.LINE_AA)
            cv2.line(plan, P((mid + 3, (box[1] + box[3]) // 2)), P((mid + 3, box[3])), (200, 120, 0), 3, cv2.LINE_AA)

    for d in wins:                                       # window = glass bar + frame (blue)
        box, along, ctr = opening(d, 0.5)
        rect(box, (255, 255, 255))                       # clear wall
        rect(box, (245, 224, 200))                       # light glass fill
        rect(box, (210, 140, 30), fill=False)            # frame
        if along == "h":
            cv2.line(plan, P((box[0], ctr[1])), P((box[2], ctr[1])), (210, 140, 30), 1, cv2.LINE_AA)
        else:
            cv2.line(plan, P((ctr[0], box[1])), P((ctr[0], box[3])), (210, 140, 30), 1, cv2.LINE_AA)

    # 4. validated room clear-dimensions
    FT = cv2.FONT_HERSHEY_SIMPLEX
    for cx, cy, wdt, hgt, opn in room_dims(R):
        op = "~" if opn else ""
        t = f"{op}{wdt*1000:.0f}x{hgt*1000:.0f}"
        (tw, th), _ = cv2.getTextSize(t, FT, 0.4, 1)
        tx = int(np.clip(cx - tw // 2, 2, W - tw - 2)); ty = int(np.clip(cy, TOP + th + 2, H + TOP - 2))
        cv2.rectangle(plan, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 3), (255, 255, 255), -1)
        cv2.putText(plan, t, (tx, ty), FT, 0.4, (150, 0, 160), 1, cv2.LINE_AA)

    # 5. legend in the top margin
    cv2.rectangle(plan, (0, 0), (W, TOP), (255, 255, 255), -1)
    cv2.putText(plan, "FUSED FLOORPLAN   LiDAR walls + clear dims (mm)   |   window (blue)   balcony door (orange)",
                (8, 22), FT, 0.44, (0, 0, 0), 1, cv2.LINE_AA)
    p = out_dir / "fused_floorplan.png"
    cv2.imwrite(str(p), plan)
    log(f"wrote {p}  ({len(wins)} windows, {len(bdoors)} balcony doors placed)")


def main(drawing, las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- detections on the drawing (walls + windows + balcony doors) ---
    sess = load("walls")
    bgr = cv2.imread(str(drawing))
    dets = run_model(sess, bgr, WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    wins = [d for d in dets if d["name"] == "window"]
    bdoors = [d for d in dets if d["name"] == "balcony door"]
    log(f"drawing dets: {len(wins)} windows, {len(bdoors)} balcony doors")

    # --- drawing wall vectors + LiDAR walls ---
    dsegs, dbox = drawing_walls(bgr)
    log(f"drawing walls vectorised: {len(dsegs)} segments  bbox={tuple(int(v) for v in dbox)}")
    R = reconstruct(las)
    H, W, xmin, ymax = R["H"], R["W"], R["xmin"], R["ymax"]

    def m2px(px, py):
        return (px - xmin) / CELL, (ymax - py) / CELL
    lsegs = []
    for p0, p1 in R["walls"]:
        a = m2px(*p0); b = m2px(*p1)
        lsegs.append((a[0], a[1], b[0], b[1]))
    lpts = np.array([[s[0], s[1]] for s in lsegs] + [[s[2], s[3]] for s in lsegs])
    lbox = (lpts[:, 0].min(), lpts[:, 1].min(), lpts[:, 0].max(), lpts[:, 1].max())
    lmask = raster_segs(lsegs, (H, W), thick=5)
    lmask_d = cv2.dilate(lmask, np.ones((7, 7), np.uint8))

    # --- pick the orientation whose mapped drawing walls best overlap LiDAR ---
    best = None
    for o in range(8):
        mapped = [(*map_pt((x0, y0), dbox, lbox, o), *map_pt((x1, y1), dbox, lbox, o))
                  for x0, y0, x1, y1 in dsegs]
        mm = raster_segs(mapped, (H, W), thick=5)
        overlap = int((cv2.dilate(mm, np.ones((7, 7), np.uint8)) & lmask_d > 0).sum())
        denom = int((mm > 0).sum()) + 1
        score = overlap / denom
        log(f"   orient {o}: overlap score {score:.3f}")
        if best is None or score > best[0]:
            best = (score, o, mapped)
    score, o, mapped = best
    log(f"best orientation = {o}  (score {score:.3f})")

    # --- registration overlay: LiDAR walls (black) + mapped drawing walls (red)
    #     + windows (green) + balcony doors (blue) ---
    ov = np.full((H, W, 3), 255, np.uint8)
    for x0, y0, x1, y1 in lsegs:
        cv2.line(ov, (int(x0), int(y0)), (int(x1), int(y1)), (30, 30, 30), 4)
    for x0, y0, x1, y1 in mapped:
        cv2.line(ov, (int(x0), int(y0)), (int(x1), int(y1)), (60, 60, 235), 1, cv2.LINE_AA)
    for d in wins:
        b = d["box"]; c = map_pt(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), dbox, lbox, o)
        cv2.circle(ov, (int(c[0]), int(c[1])), 7, (0, 180, 0), -1)
    for d in bdoors:
        b = d["box"]; c = map_pt(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), dbox, lbox, o)
        cv2.circle(ov, (int(c[0]), int(c[1])), 7, (220, 120, 0), -1)
    cv2.putText(ov, f"REGISTRATION  LiDAR walls(blk) + drawing walls(red) orient {o} score {score:.2f}"
                    f"  green=window blue=balconyDoor", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    p = out_dir / "registration_overlay.png"
    cv2.imwrite(str(p), ov)
    log(f"wrote {p}")

    # --- Stage 2: the accurate, good-looking fused floorplan ---
    render_fused(R, lsegs, wins, bdoors, dbox, lbox, o, out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
