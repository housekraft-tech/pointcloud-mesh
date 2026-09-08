"""register_drawing.py
----------------------
Register an architect floorplan image onto the LiDAR reconstruction frame by
FOOTPRINT alignment (no ML-JSON needed). Steps:
  1. Run rfdetr elements -> use the room boxes to auto-crop the plan region
     (excludes the legend / floor-selector on the drawing).
  2. Threshold the drawing's dark wall pixels -> drawing wall + footprint mask.
  3. reconstruct() the LiDAR -> LiDAR footprint + wall mask (metric, CELL m/px).
  4. Search rotation(0/90/180/270) x mirror; scale from footprint areas; align
     centroids; refine translation by hill-climb -> maximise footprint IoU.
  5. Write registration_overlay.png (drawing walls on LiDAR walls) + transform.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\register_drawing.py \
      <floorplan.png> <lidar.las> <out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS,
    WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}


def log(m): print(m, flush=True)


def drawing_masks(img_path):
    """Return (wall_mask, footprint_mask) for the plan region of the drawing."""
    bgr = cv2.imread(str(img_path)); H, W = bgr.shape[:2]
    dets = run_model(load("elements"), bgr, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    rooms = [d for d in dets if d["name"] in ROOM_CLASSES]
    xs = [v for d in rooms for v in (d["box"][0], d["box"][2])]
    ys = [v for d in rooms for v in (d["box"][1], d["box"][3])]
    m = 40
    x0, y0 = max(0, int(min(xs)) - m), max(0, int(min(ys)) - m)
    x1, y1 = min(W, int(max(xs)) + m), min(H, int(max(ys)) + m)
    plan = bgr[y0:y1, x0:x1]
    ph, pw = plan.shape[:2]
    gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
    # footprint = union of detected room boxes (covers the whole apartment
    # incl. balconies); robust to broken outer wall loops at door/balcony gaps.
    foot = np.zeros((ph, pw), np.uint8)
    for d in rooms:
        bx0, by0, bx1, by1 = d["box"]
        foot[max(0, int(by0) - y0):int(by1) - y0, max(0, int(bx0) - x0):int(bx1) - x0] = 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    foot = ndimage.binary_fill_holes(foot).astype(np.uint8)
    # walls are the dark thick strokes within the footprint
    wall = (gray < 110).astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & foot
    # strip room-label TEXT: keep only large connected components (real walls);
    # text glyphs are small blobs.
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    clean = np.zeros_like(wall)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 120:
            clean[lbl == i] = 1
    wall = clean
    # room seeds in crop coords: label + box center
    seeds = [dict(name=d["name"],
                  cx=(d["box"][0] + d["box"][2]) / 2 - x0,
                  cy=(d["box"][1] + d["box"][3]) / 2 - y0) for d in rooms]
    # CLEAN wall segments from the walls model: each wall box -> a line along its
    # long axis at the short-axis centre (in crop coords).
    wdets = run_model(load("walls"), bgr, WALLS_WINDOWS_CLASSES,
                      WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    wall_segs = []
    for d in wdets:
        if d["name"] != "wall":
            continue
        bx0, by0, bx1, by1 = d["box"]
        w, h = bx1 - bx0, by1 - by0
        if max(w, h) < 12:
            continue
        if w >= h:                       # horizontal wall
            p0 = (bx0 - x0, (by0 + by1) / 2 - y0); p1 = (bx1 - x0, (by0 + by1) / 2 - y0)
        else:                            # vertical wall
            p0 = ((bx0 + bx1) / 2 - x0, by0 - y0); p1 = ((bx0 + bx1) / 2 - x0, by1 - y0)
        wall_segs.append((np.array(p0, float), np.array(p1, float)))
    return wall, foot, seeds, wall_segs


def lidar_masks(las):
    R = reconstruct(las)
    free, occ = R["free"], R["occ"]
    foot = ndimage.binary_fill_holes((free > 0) | (occ > 0)).astype(np.uint8)
    wall = (foot & (free == 0)).astype(np.uint8)
    return wall, foot, R


def to_poly_pts(mask):
    ys, xs = np.where(mask > 0)
    return np.column_stack([xs, ys]).astype(np.float64)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni else 0.0


def main(img_path, las, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dw, dfoot, seeds, wall_segs = drawing_masks(img_path)
    lw, lfoot, R = lidar_masks(las)
    LH, LW = lfoot.shape
    log(f"drawing plan {dfoot.shape}  footprint px {int(dfoot.sum())}")
    log(f"lidar frame {lfoot.shape}  footprint px {int(lfoot.sum())}")

    # scale so drawing footprint area matches lidar footprint area
    area_scale = np.sqrt(lfoot.sum() / max(1, dfoot.sum()))
    dpts = to_poly_pts(dfoot)
    dwall = to_poly_pts(dw)
    dc = dpts.mean(0)
    lc = to_poly_pts(lfoot).mean(0)

    best = None
    for ang in (0, 90, 180, 270):
        for mir in (False, True):
            th = np.radians(ang); cth, sth = np.cos(th), np.sin(th)
            Rm = np.array([[cth, -sth], [sth, cth]])
            def xf(pts, tx=0, ty=0):
                p = pts - dc
                if mir:
                    p = p * [-1, 1]
                p = (p * area_scale) @ Rm.T
                return p + lc + [tx, ty]
            # rasterize transformed footprint, hill-climb translation for IoU
            def raster(pts):
                q = np.round(pts).astype(int)
                m = np.zeros((LH, LW), np.uint8)
                ok = (q[:, 0] >= 0) & (q[:, 0] < LW) & (q[:, 1] >= 0) & (q[:, 1] < LH)
                m[q[ok, 1], q[ok, 0]] = 1
                return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            tx = ty = 0.0
            cur = iou(raster(xf(dpts)), lfoot)
            for step in (16, 8, 4, 2):
                improved = True
                while improved:
                    improved = False
                    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        v = iou(raster(xf(dpts, tx + dx, ty + dy)), lfoot)
                        if v > cur:
                            cur, tx, ty, improved = v, tx + dx, ty + dy, True
            if best is None or cur > best["iou"]:
                best = dict(iou=cur, ang=ang, mir=mir, tx=tx, ty=ty, scale=area_scale)
    log(f"COARSE (footprint): IoU={best['iou']:.3f}  angle={best['ang']} mirror={best['mir']} "
        f"scale={best['scale']*CELL*1000:.1f} mm/px_drawing")

    # ---- GLOBAL wall refine: footprint's area-based scale is a few % off; hill-
    # climb (scale, fine angle, tx, ty) to bring drawing walls close to LiDAR walls
    # so per-wall snapping then has a good starting point.
    lw_dt = ndimage.distance_transform_edt(lw == 0)
    dwpx = dwall

    def gxf(scale, ang, tx, ty, pts):
        th = np.radians(ang); c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        p = np.asarray(pts, float) - dc
        if best["mir"]:
            p = p * [-1, 1]
        return (p * scale) @ Rm.T + lc + [tx, ty]

    def gscore(scale, ang, tx, ty, tol=5.0):
        q = np.round(gxf(scale, ang, tx, ty, dwpx)).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < LW) & (q[:, 1] >= 0) & (q[:, 1] < LH)
        return float((lw_dt[q[ok, 1], q[ok, 0]] <= tol).mean()) if ok.sum() else 0.0

    sc, an, tx, ty = best["scale"], float(best["ang"]), best["tx"], best["ty"]
    cur = gscore(sc, an, tx, ty)
    for ss, aa, tt in [(0.03, 2.0, 14), (0.015, 1.0, 7), (0.007, 0.5, 3), (0.003, 0.25, 1)]:
        improved = True
        while improved:
            improved = False
            for dsc in (ss, -ss, 0):
                for da in (aa, -aa, 0):
                    for dx in (tt, -tt, 0):
                        for dy in (tt, -tt, 0):
                            v = gscore(sc * (1 + dsc), an + da, tx + dx, ty + dy)
                            if v > cur + 1e-6:
                                cur, sc, an, tx, ty = v, sc * (1 + dsc), an + da, tx + dx, ty + dy
                                improved = True
    best.update(scale=sc, ang=an, tx=tx, ty=ty)
    log(f"GLOBAL refine: {cur*100:.0f}% walls within {5*CELL*1000:.0f} mm  (scale {sc*CELL*1000:.1f} mm/px, angle {an:.2f})")

    # base transform (footprint + global wall refine)
    th = np.radians(best["ang"]); cth, sth = np.cos(th), np.sin(th)
    Rm = np.array([[cth, -sth], [sth, cth]])
    def xf(pts):
        p = np.asarray(pts, float) - dc
        if best["mir"]:
            p = p * [-1, 1]
        p = (p * best["scale"]) @ Rm.T
        return p + lc + [best["tx"], best["ty"]]

    # ---- PER-WALL SNAPPING: footprint fixes global pose; now shift each detected
    # drawing wall perpendicular onto the nearest parallel LiDAR wall (absorbs the
    # residual scale/rotation drift locally). lw_dt = distance to nearest LiDAR wall.
    lw_dt = ndimage.distance_transform_edt(lw == 0)

    def sample(p0, p1, step=1.0):
        L = np.hypot(*(p1 - p0))
        n = max(2, int(L / step))
        t = np.linspace(0, 1, n)[:, None]
        return p0 + (p1 - p0) * t

    snapped = []
    MAXSHIFT = 42            # px (~84 cm search) to absorb edge scale drift
    n_snapped = 0
    for p0, p1 in wall_segs:
        q0, q1 = xf(p0), xf(p1)
        d = q1 - q0; L = np.hypot(*d)
        if L < 3:
            continue
        d /= L; nrm = np.array([-d[1], d[0]])
        pts = sample(q0, q1)
        best_sh, best_cost = 0, np.inf
        for sh in range(-MAXSHIFT, MAXSHIFT + 1):
            g = np.round(pts + sh * nrm).astype(int)
            ok = (g[:, 0] >= 0) & (g[:, 0] < LW) & (g[:, 1] >= 0) & (g[:, 1] < LH)
            if ok.sum() < 3:
                continue
            cost = lw_dt[g[ok, 1], g[ok, 0]].mean()
            if cost < best_cost:
                best_cost, best_sh = cost, sh
        if best_cost <= 3.5:            # a real LiDAR wall was found near this drawing wall
            q0 = q0 + best_sh * nrm; q1 = q1 + best_sh * nrm
            n_snapped += 1
        snapped.append((q0, q1))
    log(f"snapped {n_snapped}/{len(snapped)} detected drawing walls onto LiDAR walls")

    # overlay: LiDAR walls (grey) + snapped drawing walls (red)
    ov = np.full((LH, LW, 3), 255, np.uint8)
    ov[lw > 0] = (175, 175, 175)
    for q0, q1 in snapped:
        cv2.line(ov, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), (0, 0, 220), 2)
    cv2.imwrite(str(out / "registration_overlay.png"), ov)
    best["n_snapped"] = n_snapped
    (out / "transform.json").write_text(json.dumps(best, indent=2))

    # ---- 1:1 SEGMENTATION: seed a watershed on the LiDAR free-space with the
    # drawing's room centers (topology from drawing, boundaries from LiDAR walls).
    free = R["free"].copy()
    # close LiDAR wall gaps (undetected/occluded partitions) with the SNAPPED
    # drawing walls -> stops small rooms leaking during the seeded watershed.
    dwall_reg = np.zeros((LH, LW), np.uint8)
    for q0, q1 in snapped:
        cv2.line(dwall_reg, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), 1, 2)
    free[dwall_reg > 0] = 0                   # snapped drawing walls become barriers too
    seed_pts = xf(np.array([[s["cx"], s["cy"]] for s in seeds]))
    mk = np.zeros((LH, LW), np.int32)
    labels = []
    ncnt = {}
    for s in seeds:                          # unique names: Bedroom-1/-2/-3 ...
        ncnt[s["name"]] = ncnt.get(s["name"], 0) + 1
    seen = {}
    for s in seeds:
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        s["uname"] = f"{s['name']}-{seen[s['name']]}" if ncnt[s["name"]] > 1 else s["name"]
    for i, (s, (px, py)) in enumerate(zip(seeds, seed_pts), start=2):
        px, py = int(round(px)), int(round(py))
        # snap seed into free space if it landed on/in a wall
        if not (0 <= px < LW and 0 <= py < LH and free[py, px]):
            fy, fx = np.where(free > 0)
            if len(fx):
                j = np.argmin((fx - px) ** 2 + (fy - py) ** 2)
                px, py = int(fx[j]), int(fy[j])
        cv2.circle(mk, (px, py), 3, i, -1)
        labels.append(s["uname"])
    mk[free == 0] = 1                       # walls = barrier
    cv2.watershed(cv2.merge([free * 200] * 3).astype(np.uint8), mk)

    # colour each room + place its label
    rng = np.random.default_rng(0)
    seg = np.full((LH, LW, 3), 255, np.uint8)
    seg[lw > 0] = (60, 60, 60)
    areas = {}
    for i, name in enumerate(labels, start=2):
        m = (mk == i)
        a = m.sum() * CELL * CELL
        if a < 0.5:
            continue
        col = tuple(int(c) for c in rng.integers(60, 230, 3))
        seg[m] = col
        areas[name] = areas.get(name, 0) + a
    seg[lw > 0] = (40, 40, 40)
    for i, name in enumerate(labels, start=2):
        ys, xs = np.where(mk == i)
        if len(xs) and (mk == i).sum() * CELL * CELL >= 0.5:
            cv2.putText(seg, name, (int(xs.mean()) - 20, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "segmentation.png"), seg)

    # ---- DIMENSIONED PLAN: measure each room's clear W x L (mask metric bbox)
    # and ceiling height (peak of z-histogram of its points), annotate on plan.
    x, y, z = R["x"], R["y"], R["z"]
    xmin, ymax = R["xmin"], R["ymax"]
    pc = np.clip(((x - xmin) / CELL).astype(int), 0, LW - 1)
    prr = np.clip(((ymax - y) / CELL).astype(int), 0, LH - 1)
    plab = mk[prr, pc]

    def peak_h(zz):
        if len(zz) < 300:
            return None
        e = np.arange(zz.min() - 0.01, zz.max() + 0.02, 0.01)
        h, _ = np.histogram(zz, e); ce = 0.5 * (e[:-1] + e[1:])
        zf = ce[np.argmax(np.where(ce < zz.min() + 0.4, h, 0))]
        zc = ce[np.argmax(np.where(ce > zz.max() - 0.4, h, 0))]
        return (zc - zf) * 1000

    dim_img = np.full((LH, LW, 3), 255, np.uint8)
    dim_img[lw > 0] = (150, 150, 150)
    meas = []
    for i, name in enumerate(labels, start=2):
        m = (mk == i)
        if m.sum() * CELL * CELL < 0.5:
            continue
        ys, xs = np.where(m)
        wmm = (xs.max() - xs.min()) * CELL * 1000
        lmm = (ys.max() - ys.min()) * CELL * 1000
        h = peak_h(z[plab == i])
        w, l = int(min(wmm, lmm)), int(max(wmm, lmm))
        cxp, cyp = int(xs.mean()), int(ys.mean())
        col = tuple(int(c) for c in rng.integers(40, 200, 3))
        dim_img[m] = tuple(min(255, c + 90) for c in col)
        txt = [name, f"{w} x {l}", (f"H {h:.0f}" if h else "H ?")]
        for j, tt in enumerate(txt):
            fs = 0.42 if j == 0 else 0.4
            cv2.putText(dim_img, tt, (cxp - 30, cyp - 8 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        meas.append(dict(room=name, w_mm=w, l_mm=l, height_mm=round(h) if h else None,
                         area_m2=round(m.sum() * CELL * CELL, 1)))
    dim_img[lw > 0] = (110, 110, 110)
    cv2.imwrite(str(out / "measurements.png"), dim_img)
    (out / "room_measurements.json").write_text(json.dumps(meas, indent=2))

    log(f"BEST registration: IoU={best['iou']:.3f}  angle={best['ang']} mirror={best['mir']}")
    log("per-room measured clear W x L (mm) + ceiling height:")
    for r in sorted(meas, key=lambda r: -r["area_m2"]):
        log(f"    {r['room']:12} {r['w_mm']:>5} x {r['l_mm']:<5}  H {r['height_mm']}  ({r['area_m2']} m2)")
    log(f"segmented {len([a for a in areas.values() if a>=0.5])} rooms from drawing seeds:")
    for name, a in sorted(areas.items(), key=lambda kv: -kv[1]):
        log(f"    {name:14} {a:5.1f} m2")
    log(f"-> {out}/registration_overlay.png  segmentation.png  transform.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
