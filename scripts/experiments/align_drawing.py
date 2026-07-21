"""align_drawing.py
----------------
Register the architect drawing onto the LiDAR model with a full similarity
transform -- continuous rotation, scale and translation.

register_drawing.py searches only 0/90/180/270 and takes scale from a footprint
area ratio, which leaves the plan a few degrees and a few percent out. That is
fine for a sanity overlay and useless for transferring room identity, because a
2 deg error walks a label a whole room's width across a 12 m plan.

Method: chamfer matching. Rasterise the model's wall footprint, take its
distance transform, and minimise the mean distance from transformed drawing
wall pixels to the nearest model wall. Distance transforms give a smooth basin
(unlike IoU, which is flat until the masks touch), so a local optimiser can
actually refine. Multi-start over rotation and mirror, then Powell.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\align_drawing.py \\
      <floorplan.png> <detailed_modular.obj> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage, optimize

PX = 0.02            # m per pixel of the model raster
PAD = 1.0            # m padding around the model
MAX_D = 0.60         # m: clip chamfer distance (robust to drawing-only detail)
N_SAMPLE = 4000      # drawing wall pixels used per evaluation
ISOLATED_PX = 260    # a component this far from any other is a plan symbol
N_STARTS = 12        # coarse starts refined with Powell


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def drawing_wall_points(img_path):
    """Thick, dark, unsaturated strokes = walls.

    Unsaturated matters: the floor-selector panel on the right is dark BLUE and
    would otherwise be taken for structure. Opening removes dimension text,
    which is dark and grey too but thin."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"cannot read {img_path}")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = ((gray < 130) & (hsv[:, :, 1] < 70)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Do NOT pick a single connected component. A plan's walls are broken into
    # many pieces by door openings and glazing -- here 11 of them, the largest
    # holding under a fifth of the wall pixels -- so "largest component" returns
    # a fragment. Keep every substantial component and drop plan SYMBOLS
    # instead: the north arrow is a solid near-circular disc, which no wall run
    # is.
    lab, n = ndimage.label(mask)
    if n == 0:
        raise SystemExit("no wall pixels found in the drawing")
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    keep_ids = []
    for i, (sz, sl) in enumerate(zip(sizes, objs), start=1):
        if sz < 800:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        fill = sz / max(h * w, 1)
        squarish = abs(w - h) / max(w, h) < 0.30
        if squarish and fill > 0.55 and sz < 4000:
            log(f"  dropping symbol component {i} ({int(sz)} px, {w}x{h}, "
                f"fill {fill:.2f}) -- north arrow, not a wall")
            continue
        keep_ids.append(i)
    # Drop spatially isolated leftovers (the north arrow's chevron survives the
    # circularity test). Wall runs always have another wall run nearby; a plan
    # symbol sits alone in white space.
    cents = {i: np.array([(objs[i-1][1].start + objs[i-1][1].stop) / 2,
                          (objs[i-1][0].start + objs[i-1][0].stop) / 2])
             for i in keep_ids}
    isolated = []
    for i in keep_ids:
        near = min((np.linalg.norm(cents[i] - cents[j])
                    for j in keep_ids if j != i), default=1e9)
        if near > ISOLATED_PX:
            isolated.append(i)
            log(f"  dropping isolated component {i} ({int(sizes[i-1])} px, "
                f"nearest neighbour {near:.0f} px away) -- plan symbol")
    keep_ids = [i for i in keep_ids if i not in isolated]
    final = np.isin(lab, keep_ids)
    ys, xs = np.nonzero(final)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ys, xs = np.nonzero(final)
    log(f"drawing: {len(xs):,} wall px, plan box x {x0}..{x1} y {y0}..{y1}")
    # image y grows downward; flip so the drawing shares the model's handedness
    return np.c_[xs.astype(float), -ys.astype(float)], (x0, x1, y0, y1), bgr


def model_wall_raster(obj_path):
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
    P = np.vstack([V[a:b] for n, a, b in groups if n.startswith("wall") and b > a])
    lo = P[:, :2].min(0) - PAD
    hi = P[:, :2].max(0) + PAD
    nx, ny = np.ceil((hi - lo) / PX).astype(int)
    img = np.zeros((ny, nx), np.uint8)
    ij = np.floor((P[:, :2] - lo) / PX).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)]
    img[ij[:, 1], ij[:, 0]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    log(f"model raster {nx}x{ny} px @ {PX} m, {int(img.sum()):,} wall px")
    return img, lo, (nx, ny)


def main(img_path, obj_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    D_pts, box, bgr = drawing_wall_points(img_path)
    img, lo, (nx, ny) = model_wall_raster(obj_path)

    # distance transform of the model walls, in metres, clipped
    dist_px = ndimage.distance_transform_edt(img == 0)
    dist = np.clip(dist_px * PX, 0, MAX_D)

    # centre both, and take an initial scale from the bounding boxes
    dc = D_pts.mean(0)
    D0 = D_pts - dc
    ys, xs = np.nonzero(img)
    M_pts = np.c_[xs, ys] * PX + lo
    mc = M_pts.mean(0)
    s0 = (np.ptp(M_pts, axis=0).max()) / max(np.ptp(D0, axis=0).max(), 1e-9)
    log(f"initial scale {s0*1000:.3f} mm per drawing px")

    rng = np.random.default_rng(0)
    sub = D0[rng.choice(len(D0), min(N_SAMPLE, len(D0)), replace=False)]

    def cost(p, pts):
        s, th, tx, ty = p
        c, sn = np.cos(th), np.sin(th)
        R = np.array([[c, -sn], [sn, c]])
        Q = (pts * s) @ R.T + np.array([tx, ty])
        ij = np.floor((Q - lo) / PX).astype(int)
        ok = ((ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny))
        if ok.sum() < 0.35 * len(pts):
            return MAX_D * 2                       # mostly off-raster
        d = np.full(len(pts), MAX_D)
        d[ok] = dist[ij[ok, 1], ij[ok, 0]]
        return float(d.mean())

    # Coarse search over rotation AND scale, then refine the best few. A single
    # start is not enough: the chamfer surface is multi-modal (a plan can lock
    # onto the wrong parallel wall), and scale is only roughly known from the
    # bounding boxes, which the balconies distort.
    cands = []
    for mirror in (1, -1):
        pts_m = sub * np.array([mirror, 1.0])
        for deg in range(0, 360, 3):
            for fs in (0.90, 0.95, 1.00, 1.05, 1.10):
                p0 = np.array([s0 * fs, np.radians(deg), mc[0], mc[1]])
                cands.append((cost(p0, pts_m), mirror, p0))
    cands.sort(key=lambda t: t[0])
    log(f"coarse best {cands[0][0]*1000:.0f} mm over {len(cands)} starts")

    best = (1e9, None, None)
    for c0, mirror, p0 in cands[:N_STARTS]:
        pts_m = sub * np.array([mirror, 1.0])
        r = optimize.minimize(cost, p0, args=(pts_m,), method="Powell",
                              options=dict(xtol=1e-6, ftol=1e-8, maxiter=20000))
        if r.fun < best[0]:
            best = (r.fun, mirror, r.x)
    _, mirror, x0p = best
    pts_full = D0 * np.array([mirror, 1.0])
    res2 = optimize.minimize(cost, x0p, args=(pts_full,), method="Powell",
                             options=dict(xtol=1e-7, ftol=1e-9, maxiter=20000))
    s, th, tx, ty = res2.x
    mean_err = res2.fun
    log(f"ALIGNED: scale {s*1000:.4f} mm/px  rot {np.degrees(th)%360:.3f} deg  "
        f"mirror {mirror}  mean chamfer {mean_err*1000:.0f} mm")

    # how much of the drawing lands ON a wall
    c_, sn = np.cos(th), np.sin(th)
    R = np.array([[c_, -sn], [sn, c_]])
    Q = (pts_full * s) @ R.T + np.array([tx, ty])
    ij = np.floor((Q - lo) / PX).astype(int)
    ok = ((ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny))
    dd = np.full(len(Q), MAX_D); dd[ok] = dist[ij[ok, 1], ij[ok, 0]]
    within = {f"{t*1000:.0f}mm": round(float((dd < t).mean()), 3)
              for t in (0.05, 0.10, 0.20, 0.40)}
    log(f"drawing wall px within: {within}")

    json.dump(dict(scale_m_per_px=float(s), rotation_deg=float(np.degrees(th) % 360),
                   mirror=int(mirror), tx=float(tx), ty=float(ty),
                   drawing_centre_px=[float(dc[0]), float(dc[1])],
                   mean_chamfer_mm=float(mean_err * 1000),
                   frac_within=within,
                   note="drawing px (x, -y) -> centre -> mirror x -> scale -> "
                        "rotate -> translate = model metres"),
              open(out / "drawing_transform.json", "w"), indent=1)

    # ---- overlay
    canvas = np.zeros((ny, nx, 3), np.uint8)
    canvas[img > 0] = (70, 70, 70)
    keep = ok & (dd < MAX_D)
    col = np.where((dd[ok] < 0.10)[:, None], np.array([0, 230, 90]),
                   np.array([0, 120, 255]))
    canvas[ij[ok, 1], ij[ok, 0]] = col
    canvas = cv2.flip(canvas, 0)
    cv2.putText(canvas, f"chamfer {mean_err*1000:.0f} mm | "
                f"{within['100mm']*100:.0f}% within 100mm | rot "
                f"{np.degrees(th)%360:.2f} deg | {s*1000:.3f} mm/px",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(out / "drawing_alignment.png"), canvas)
    log(f"wrote {out/'drawing_alignment.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
