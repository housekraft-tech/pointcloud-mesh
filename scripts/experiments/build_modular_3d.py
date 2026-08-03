"""build_modular_3d.py
-------------------
Recreate a sharp MODULAR 3D model directly from the CUT/SLICE data.

Instead of extruding a flat box and guessing details, each wall is built from
its real vertical profile: for every wall line we form the (along-wall u x
height z) occupancy from the points -- exactly the vertical-section "cut" --
and extrude THAT silhouette by the measured thickness. Everything then falls
out of the data automatically:
   * doors / windows  -> gaps in the (u,z) silhouette (floor-touching vs sill)
   * railings         -> the silhouette only reaches waist height
   * beams / soffits  -> material only near the top
   * GROOVES / L-cuts -> height bands where the face recedes are cut as
     channels of the measured depth
Plus floor + ceiling slabs. Exports a modular GLB (named parts).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_modular_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.recon import clean, frame
from scripts.recon.metrology import clear_between, detect_wall_faces
from scripts.recon.regularize import merge_collinear_runs
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime
from scripts.experiments.hough_vectorize import snap_and_merge
from scripts.experiments.make_floorplan import close_junctions, merge_parallels

PPM = 50
CELL = 1.0 / PPM
UZ_RES = 0.04            # (u,z) silhouette cell size
WALL_HALF_BAND = 0.18
GROOVE_MIN_DEPTH = 0.03     # min face setback to count as a groove (sensitive -> visible reveals)
MIN_WALL_LEN = 0.35     # keep short walls (utility/duct/wardrobe returns) -- don't drop them


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slab(cx, cy, cz, sx, sy, sz):
    b = trimesh.creation.box(extents=(sx, sy, sz))
    b.apply_translation((cx, cy, cz))
    return b


def silhouette_polygons(mask, ures, zres, z_base):
    """(u,z) binary mask -> list of shapely polygons (with holes = openings),
    coords in metric (u, z_world)."""
    H, W = mask.shape
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    polys = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:            # this is a hole; handled with its parent
            continue
        if cv2.contourArea(cnt) < (0.05 / (ures * zres)):
            continue
        ext = [(c * ures, z_base + (H - r) * zres) for c, r in cnt[:, 0, :]]
        holes = []
        child = hier[i][2]
        while child != -1:
            hc = contours[child]
            if cv2.contourArea(hc) >= (0.05 / (ures * zres)):
                holes.append([(c * ures, z_base + (H - r) * zres) for c, r in hc[:, 0, :]])
            child = hier[child][0]
        try:
            p = Polygon(ext, holes)
            if p.is_valid and p.area > 0.05:
                polys.append(p)
        except Exception:
            pass
    return polys


def footprint_polygons(ws, xmin, ymax, eps_m=0.03):
    """The carved wall material (ws mask) -> metric shapely polygons (with room
    holes), Manhattan-simplified so real 90-degree JOGS / pilasters / niches are
    KEPT while pixel noise is smoothed. Extruding these gives thick walls that
    follow the true footprint, not straightened boxes."""
    cnts, hier = cv2.findContours(ws, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    eps = eps_m / CELL

    def m(c, r):
        return (xmin + c * CELL, ymax - r * CELL)
    out = []
    for i, cnt in enumerate(cnts):
        if hier[i][3] != -1:                       # a hole; handled with its parent
            continue
        ap = cv2.approxPolyDP(cnt, eps, True)
        if len(ap) < 3:
            continue
        ext = [m(c, r) for c, r in ap[:, 0, :]]
        holes = []
        child = hier[i][2]
        while child != -1:
            hc = cv2.approxPolyDP(cnts[child], eps, True)
            if len(hc) >= 3:
                holes.append([m(c, r) for c, r in hc[:, 0, :]])
            child = hier[child][0]
        try:
            pg = Polygon(ext, holes)
            if not pg.is_valid:
                pg = pg.buffer(0)
            if pg.area > 0.05:
                out.append(pg)
        except Exception:
            pass
    return out


def extrude_wall(polys, p0, d, n, thick):
    """Extrude (u,z) polygons by thickness along n, place at wall p0/d/n."""
    geom = unary_union(polys) if len(polys) > 1 else polys[0]
    prism = trimesh.creation.extrude_polygon(geom, height=thick)
    # local: X=u(along d), Y=z(world up), Z=thickness(along n)
    M = np.array([[d[0], 0, n[0], p0[0] - n[0] * thick / 2],
                  [d[1], 0, n[1], p0[1] - n[1] * thick / 2],
                  [0,    1, 0,    0],
                  [0,    0, 0,    1]], float)
    prism.apply_transform(M)
    return prism


def recover_tiny_walls(walls, walls_solid, xmin, ymax, min_len=0.12, max_len=0.55):
    """Recover SHORT wall nibs (foyer jambs, small returns) that Hough drops.
    Take the carved wall material NOT already covered by a detected wall; keep
    the small elongated pieces that TOUCH an existing wall (a real nib, not
    furniture). Line detectors miss these; the material is right there."""
    H, W = walls_solid.shape
    tpx = max(6, int(round(0.16 / CELL)))
    covered = np.zeros((H, W), np.uint8)
    for p0, p1 in walls:
        a = (int((p0[0] - xmin) / CELL), int((ymax - p0[1]) / CELL))
        b = (int((p1[0] - xmin) / CELL), int((ymax - p1[1]) / CELL))
        cv2.line(covered, a, b, 255, tpx + 4)
    resid = ((walls_solid > 0) & (covered == 0)).astype(np.uint8) * 255
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near_wall = cv2.dilate(covered, np.ones((tpx, tpx), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(resid, 8)
    add = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < (0.015 / (CELL * CELL)):
            continue
        comp = lbl == i
        if not (comp & (near_wall > 0)).any():          # must attach to a real wall
            continue
        L = max(w, h) * CELL
        if L < min_len or L > max_len:                  # tiny only
            continue
        if w >= h:
            yc = y + h / 2.0
            add.append((np.array([xmin + x * CELL, ymax - yc * CELL]),
                        np.array([xmin + (x + w) * CELL, ymax - yc * CELL])))
        else:
            xc = x + w / 2.0
            add.append((np.array([xmin + xc * CELL, ymax - y * CELL]),
                        np.array([xmin + xc * CELL, ymax - (y + h) * CELL])))
    return add


def regularize_metric(walls, occ, xmin, ymax, tol_c=0.25, tol_s=0.60):
    """Grid-snap the wall graph and bridge Hough-miss gaps so rooms CLOSE and no
    wall is left missing (the same fix as the 2D fusion plan). Manhattan walls
    are snapped onto shared gridlines, endpoints extended onto crossing lines,
    and collinear runs merged across a gap only when that gap is filled with wall
    material (occ) -- a real doorway (empty gap) stays open."""
    Hi, Wi = occ.shape
    Hs, Vs = [], []
    for p0, p1 in walls:
        dd = p1 - p0
        if abs(dd[0]) >= abs(dd[1]):
            Hs.append([min(p0[0], p1[0]), max(p0[0], p1[0]), (p0[1] + p1[1]) / 2])   # xa,xb,y
        else:
            Vs.append([min(p0[1], p1[1]), max(p0[1], p1[1]), (p0[0] + p1[0]) / 2])   # ya,yb,x

    def cluster(vals, tol):
        if not vals:
            return []
        s = sorted(vals); g = [[s[0]]]
        for v in s[1:]:
            if v - g[-1][-1] <= tol:
                g[-1].append(v)
            else:
                g.append([v])
        return [float(np.mean(x)) for x in g]
    ys = cluster([h[2] for h in Hs], tol_c); xs = cluster([v[2] for v in Vs], tol_c)

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

    def occupied(axis, c, g0, g1):
        if g1 - g0 < 0.02:
            return False
        if axis == "h":
            r = int((ymax - c) / CELL); a = int((g0 - xmin) / CELL); b = int((g1 - xmin) / CELL)
            band = occ[max(0, r - 4):r + 5, max(0, min(a, b)):min(Wi, max(a, b))]
            prof = band.max(axis=0) if band.size else np.array([0])
        else:
            col = int((c - xmin) / CELL); a = int((ymax - g1) / CELL); b = int((ymax - g0) / CELL)
            band = occ[max(0, min(a, b)):min(Hi, max(a, b)), max(0, col - 4):col + 5]
            prof = band.max(axis=1) if band.size else np.array([0])
        return prof.size > 0 and (prof > 0).mean() >= 0.6

    def merge(lines, axis):
        lines = sorted(lines, key=lambda s: (s[2], s[0])); out = []
        for a0, a1, c in lines:
            if out and abs(out[-1][2] - c) < 1e-6 and (a0 <= out[-1][1] + tol_c or occupied(axis, c, out[-1][1], a0)):
                out[-1][1] = max(out[-1][1], a1)
            else:
                out.append([a0, a1, c])
        return out
    Hs, Vs = merge(Hs, "h"), merge(Vs, "v")
    res = [(np.array([a0, y]), np.array([a1, y])) for a0, a1, y in Hs]
    res += [(np.array([x, a0]), np.array([x, a1])) for a0, a1, x in Vs]
    return res


def main(las_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log(f"loading + isolating {las_path} ...")
    scan = load_scan(str(las_path), max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (sensor_trajectory_from_gpstime(scan.xyz, scan.gps_time)
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, _ = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                           max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    # How much does the single Manhattan grid actually cost? An angular
    # residual is not an error until multiplied by a lever arm, so report it
    # in mm over a 10m span -- directly comparable with the rest of the budget.
    fres = frame.axis_residuals(normals)
    frame_note = "frame residual: not measurable (no wall normals)"
    if fres is not None:
        frame_note = (f"frame residual: p90 {fres.p90_dev_deg:.2f}deg "
                      f"= {frame.deviation_mm(fres.p90_dev_deg, 10.0):.0f}mm over 10m"
                      f"  |  grid {fres.theta_deg:.3f}+/-{fres.theta_stderr_deg:.3f}deg")
        log(frame_note)
        log(f"   dispersion {fres.dispersion_deg:.2f}deg | median dev "
            f"{fres.p50_dev_deg:.2f}deg | max {fres.max_dev_deg:.2f}deg | n={fres.n:,}")
        if fres.frac_off_grid > 0.10:
            log(f"   WARNING {fres.frac_off_grid:.0%} of walls are >5deg off the grid -- "
                f"this building is not Manhattan, so snapping to it is WRONG, "
                f"not merely imprecise, and the grid angle itself is unreliable")
    scan = frame.axis_align(scan, R)
    xyz = scan.xyz
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    z_floor = float(np.percentile(z, cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(z, cfg["z_ceiling_pct"]))
    storey = z_ceiling - z_floor
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    W = int((xmax - xmin) / CELL) + 1; H = int((ymax - ymin) / CELL) + 1
    log(f"aligned {scan.n:,} pts | storey {storey:.2f}m")

    from scipy import ndimage
    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
    rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
    occ = np.zeros((H, W), np.uint8); occ[rr, cc] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # ---- FREE-SPACE CARVED walls (connected, clean -- includes the real
    # perimeter AND interior partitions), the same accurate source as the
    # good floorplan. Skeletonize -> Hough -> Manhattan -> close junctions. ----
    empty = (occ == 0).astype(np.uint8)
    free = np.zeros_like(empty)
    if traj.shape[0]:
        seed = np.zeros_like(empty)
        for px, py in traj[:, :2]:
            cx = int((px - xmin) / CELL); cy = int((ymax - py) / CELL)
            if 0 <= cy < H and 0 <= cx < W and empty[cy, cx]:
                seed[cy, cx] = 1
        lbl0, _ = ndimage.label(empty, structure=np.ones((3, 3)))
        keep = set(lbl0[seed == 1].tolist()) - {0}
        free = np.isin(lbl0, list(keep)).astype(np.uint8)
    footprint = ndimage.binary_fill_holes((free | (occ > 0)))
    walls_solid = (footprint & (free == 0)).astype(np.uint8)
    walls_solid = ndimage.binary_fill_holes(walls_solid).astype(np.uint8)
    walls_solid = cv2.morphologyEx(walls_solid, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # thin to centerlines if cv2 thinning is available, else Hough the solid
    # directly (merge_parallels collapses the doubled faces either way).
    try:
        thin = cv2.ximgproc.thinning(walls_solid)
    except Exception:
        thin = walls_solid
    lines = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=28,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.35 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.3 * PPM), coord_tol_px=int(0.12 * PPM))
    hsegs, vsegs = merge_parallels(hsegs), merge_parallels(vsegs)
    hsegs, vsegs = close_junctions(hsegs, vsegs, snap=int(0.5 * PPM))
    hsegs = [s for s in hsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]
    vsegs = [s for s in vsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]

    def px2m(col, row):
        return xmin + col * CELL, ymax - row * CELL
    interior = [(np.array(px2m(a0, yr)), np.array(px2m(a1, yr))) for a0, a1, yr in hsegs] + \
               [(np.array(px2m(xc, a0)), np.array(px2m(xc, a1))) for a0, a1, xc in vsegs]
    exterior = []   # perimeter already included in the carved walls above

    # ---- dedup (merge near-identical parallel overlapping segments) ----
    def dedup(walls, tol=0.16):
        Hs, Vs = [], []
        for p0, p1 in walls:
            dd = p1 - p0
            if abs(dd[0]) >= abs(dd[1]):
                Hs.append([min(p0[0], p1[0]), max(p0[0], p1[0]), (p0[1] + p1[1]) / 2])
            else:
                Vs.append([min(p0[1], p1[1]), max(p0[1], p1[1]), (p0[0] + p1[0]) / 2])

        def m(segs):
            return merge_collinear_runs(segs, tol=tol)
        res = []
        for a0, a1, y in m(Hs):
            res.append((np.array([a0, y]), np.array([a1, y])))
        for a0, a1, x in m(Vs):
            res.append((np.array([x, a0]), np.array([x, a1])))
        return res

    n_raw = len(interior) + len(exterior)
    walls = dedup(interior + exterior)
    # close the wall GRAPH (grid-snap + occupancy-aware gap fill) so the 3D has
    # no missing/gappy walls -- the same fix the 2D fusion plan uses.
    walls = regularize_metric(walls, occ, xmin, ymax)
    tiny = recover_tiny_walls(walls, walls_solid, xmin, ymax)   # foyer jambs / small returns
    walls = walls + tiny
    log(f"{len(interior)} interior + {len(exterior)} exterior = {n_raw} -> "
        f"{len(walls)} walls ({len(tiny)} tiny nibs recovered)")

    # ---- 2D plan visualization of exactly the walls the 3D is built from ----
    plan = np.full((H, W, 3), 255, np.uint8)
    def m2px(p):
        return int((p[0] - xmin) / CELL), int((ymax - p[1]) / CELL)
    for p0, p1 in walls:
        dd = p1 - p0
        col = (0, 150, 0) if abs(dd[0]) >= abs(dd[1]) else (200, 60, 0)  # H green, V blue
        cv2.line(plan, m2px(p0), m2px(p1), col, 4, cv2.LINE_AA)
    cv2.putText(plan, f"2D plan of the 3D model: {len(walls)} walls "
                      f"(green=horizontal, blue=vertical, deduped)",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "wall_plan_2d.png"), plan)

    # ---- CAD-clean architectural plan (POCHE): the deduped vector walls drawn
    # at real thickness on white, corners squared so junctions read solid. This
    # is the domain an RF-DETR trained on 2D drawings understands -- the carved
    # rasters (filled/centerline) are too noisy for the detector. ----
    cad = np.full((H, W, 3), 255, np.uint8)
    tpx = max(5, int(round(0.14 / CELL)))            # ~140mm poche wall thickness
    EXT = 0.14                                       # extend ends to seal corner/T gaps
    for p0, p1 in walls:
        d = p1 - p0
        n = np.linalg.norm(d)
        u = d / n if n > 0 else d
        q0, q1 = p0 - u * EXT, p1 + u * EXT
        cv2.line(cad, m2px(q0), m2px(q1), (0, 0, 0), tpx, cv2.LINE_8)
    cv2.imwrite(str(out_dir / "floorplan_cad.png"), cad)

    # ---- dimensioned plan: internal wall-to-wall distances in mm + ft ----
    def cluster(vals, tol=0.15):
        cl = []
        for v in sorted(vals):
            if cl and v - cl[-1][-1] <= tol:
                cl[-1].append(v)
            else:
                cl.append([v])
        return [float(np.mean(c)) for c in cl]

    vx = cluster([(p0[0] + p1[0]) / 2 for p0, p1 in walls if abs((p1 - p0)[1]) > abs((p1 - p0)[0])])
    hy = cluster([(p0[1] + p1[1]) / 2 for p0, p1 in walls if abs((p1 - p0)[0]) >= abs((p1 - p0)[1])])

    # padded canvas: dimension strings live in clean margins OUTSIDE the plan
    PT, PL, PR, PB = 150, 210, 60, 60
    DH, DW = H + PT + PB, W + PL + PR
    dim = np.full((DH, DW, 3), 255, np.uint8)
    MAG = (170, 0, 170)

    def dpx(p):
        c, r = m2px(p)
        return c + PL, r + PT
    for p0, p1 in walls:
        dd = p1 - p0
        col = (0, 150, 0) if abs(dd[0]) >= abs(dd[1]) else (200, 60, 0)
        cv2.line(dim, dpx(p0), dpx(p1), col, 4, cv2.LINE_AA)

    # INTERNAL CLEAR dimensions, measured inner face to inner face directly
    # from the points. The old path took centerline distance minus two
    # percentile-spread "thicknesses" -- a z-blind window that also swallowed
    # floor, ceiling and furniture, and could not measure a perimeter wall
    # whose outer face was never scanned at all. Measuring the two inner faces
    # needs no thickness estimate: they are surfaces the scanner actually saw.
    # Full storey height so a candidate face must span the storey (furniture won't).
    _cb = (z >= z_floor + 0.2) & (z <= z_ceiling - 0.1)
    _cx, _cy, _cz = x[_cb], y[_cb], z[_cb]

    def gl_thick(coord, axis):
        """Fallback thickness, used only when a face cannot be measured."""
        sel = (np.abs(x - coord) <= 0.30) if axis == "v" else (np.abs(y - coord) <= 0.30)
        vals = (x[sel] if axis == "v" else y[sel])
        if vals.size < 100:
            return DEFAULT_THICK
        return float(np.clip(np.percentile(vals, 90) - np.percentile(vals, 10), 0.06, 0.35))
    vx_t = [gl_thick(c, "v") for c in vx]
    hy_t = [gl_thick(c, "h") for c in hy]

    n_est = 0

    def lab_clear(a, b, ta, tb, axis):
        """Clear dimension label. Measured where possible; '*' marks a fallback
        to the estimated-thickness path so a derived number is never printed as
        if it were measured."""
        nonlocal n_est
        got = clear_between(_cx if axis == "v" else _cy, _cz, a, b)
        if got is not None:
            clear, err = got
            return f"{clear*1000:.0f}mm +/-{1.96*err*1000:.1f} ({clear*3.28084:.1f}ft)"
        n_est += 1
        clear = max(abs(b - a) - ta / 2 - tb / 2, 0.0)     # inner face to inner face
        return f"*{clear*1000:.0f}mm ({clear*3.28084:.1f}ft)"
    MIN_GAP = 0.35   # skip sub-doorway gaps (wall thicknesses) to cut clutter
    # horizontal dimensions (room widths) -- stacked in the top margin, 2 rows
    x_top = min(dpx((0, hy[0]))[1] if hy else PT, PT) - 8
    row = 0
    for i in range(len(vx) - 1):
        if vx[i + 1] - vx[i] < MIN_GAP:
            continue
        x0 = dpx((vx[i], 0))[0]; x1 = dpx((vx[i + 1], 0))[0]
        # faint guide lines the FULL height of the plan so each tick visibly
        # lines up with the vertical wall it belongs to.
        cv2.line(dim, (x0, PT - 100), (x0, DH - PB), (215, 195, 215), 1)
        cv2.line(dim, (x1, PT - 100), (x1, DH - PB), (215, 195, 215), 1)
        yl = PT - 95 + (row % 2) * 34
        cv2.arrowedLine(dim, (x0, yl), (x1, yl), MAG, 1, cv2.LINE_AA, tipLength=0.03)
        cv2.arrowedLine(dim, (x1, yl), (x0, yl), MAG, 1, cv2.LINE_AA, tipLength=0.03)
        t = lab_clear(vx[i], vx[i + 1], vx_t[i], vx_t[i + 1], "v"); (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(dim, ((x0 + x1) // 2 - tw // 2 - 2, yl - 18), ((x0 + x1) // 2 + tw // 2 + 2, yl - 5), (255, 255, 255), -1)
        cv2.putText(dim, t, ((x0 + x1) // 2 - tw // 2, yl - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MAG, 1, cv2.LINE_AA)
        row += 1
    # vertical dimensions (room depths) -- stacked in the left margin, 2 cols
    col = 0
    for i in range(len(hy) - 1):
        if hy[i + 1] - hy[i] < MIN_GAP:
            continue
        y0 = dpx((0, hy[i]))[1]; y1 = dpx((0, hy[i + 1]))[1]
        # faint guide lines the FULL width so each tick lines up with its wall
        cv2.line(dim, (PL - 155, y0), (DW - PR, y0), (215, 195, 215), 1)
        cv2.line(dim, (PL - 155, y1), (DW - PR, y1), (215, 195, 215), 1)
        xl = PL - 150 + (col % 2) * 96
        cv2.arrowedLine(dim, (xl, y0), (xl, y1), MAG, 1, cv2.LINE_AA, tipLength=0.03)
        cv2.arrowedLine(dim, (xl, y1), (xl, y0), MAG, 1, cv2.LINE_AA, tipLength=0.03)
        t = lab_clear(hy[i], hy[i + 1], hy_t[i], hy_t[i + 1], "h"); (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(dim, (xl + 2, (y0 + y1) // 2 - 9), (xl + 6 + tw, (y0 + y1) // 2 + 4), (255, 255, 255), -1)
        cv2.putText(dim, t, (xl + 4, (y0 + y1) // 2 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAG, 1, cv2.LINE_AA)
        col += 1
    cv2.putText(dim, "Internal CLEAR room dimensions (inner face to inner face)  -  mm (ft)",
               (PL, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(dim, "+/- = 95% interval on the measured faces    "
                     "* = inner face unseen, derived from estimated wall thickness",
               (PL, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(dim, frame_note, (PL, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
               (110, 110, 110), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "wall_plan_dimensioned.png"), dim)

    # ---- COMPLETE floorplan: every internal wall of every room, rendered as
    # SOLID filled walls straight from the free-space carved mask (nothing
    # dropped/fragmented like the vectorized lines). This is the true wall
    # structure of all rooms. ----
    ws = walls_solid.copy()
    lblw, nw = ndimage.label(ws, structure=np.ones((3, 3)))
    szs = ndimage.sum(np.ones_like(lblw), lblw, index=np.arange(1, nw + 1))
    ws[np.isin(lblw, [k for k, s in enumerate(szs, 1) if s < (0.06 / (CELL * CELL))])] = 0
    comp = np.full((H, W, 3), 255, np.uint8)
    comp[ws > 0] = (35, 35, 35)
    cv2.putText(comp, "Complete floorplan - all internal walls of every room (filled)",
               (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "floorplan_complete.png"), comp)

    # ---- proper 2D floorplan with per-ROOM internal clear dimensions INSIDE
    # each room. Rooms are segmented from free-space by watershed (splitting at
    # the doorway necks); each room's free-space bounding box IS its clear
    # internal size, because free space is bounded by the wall inner faces. ----
    distm = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * CELL
    cores = (distm > 0.6).astype(np.uint8)          # room cores (beyond doorway reach)
    ncore, cmark = cv2.connectedComponents(cores)
    mk = np.zeros((H, W), np.int32)
    mk[cores > 0] = cmark[cores > 0] + 1            # room seeds: 2 ..
    mk[(free > 0) & (cores == 0)] = 0               # doorway/near-wall = unknown -> flooded
    mk[free == 0] = 1                               # walls / outside = background
    cv2.watershed(cv2.merge([free * 200] * 3).astype(np.uint8), mk)

    # ---- MERGE regions split across OPEN space. Watershed cuts a big open-plan
    # area at the distance-transform ridge even where there is NO partition,
    # inventing two "rooms" out of one Living/Dining. Two adjacent regions whose
    # shared interface runs through WIDE free space (not a wall, not a ~0.9m
    # doorway neck) are really one open room -> union them. Real rooms stay split
    # because their interface is a narrow doorway (low distm) or a wall. ----
    from collections import defaultdict
    parent = list(range(ncore + 2))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # watershed separates regions by a -1 ridge line. The width of that ridge =
    # the opening between the two regions. A doorway is <=~1.2m; furniture in an
    # open room fragments the free-space distance transform so distm can't tell
    # open from doorway -- but the RIDGE LENGTH can: a shared interface several
    # metres wide is one open room split in two (there are no 5m doorways).
    iface = defaultdict(int)
    ys, xs = np.where(mk == -1)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, yy - 1), min(H, yy + 2)
        x0, x1 = max(0, xx - 1), min(W, xx + 2)
        labs = sorted(int(v) for v in np.unique(mk[y0:y1, x0:x1]) if v > 1)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                iface[(labs[i], labs[j])] += 1
    OPEN_RIDGE_M = 1.6     # interface wider than this => open connection, not a doorway
    nmerge = 0
    for (a, b), n in sorted(iface.items()):
        if n * CELL >= OPEN_RIDGE_M:                   # open connection, not a doorway
            union(a, b); nmerge += 1
            log(f"   merged open regions across {n*CELL:.1f}m opening (no partition)")
    for r in range(2, ncore + 1):                     # collapse each region to its root
        mk[mk == r] = find(r)
    room_labels = sorted(set(int(v) for v in np.unique(mk) if v > 1))

    fr = np.full((H, W, 3), 255, np.uint8)
    fr[ws > 0] = (40, 40, 40)                       # filled walls (complete)
    seg = np.full((H, W, 3), 255, np.uint8)         # DEBUG: colored room regions
    seg[ws > 0] = (40, 40, 40)
    # ---- DIRECT-FROM-LIDAR clear-span measurement (not from the 2D raster).
    # At a room's center, scan a thin strip of the actual wall-height points
    # along each axis; a wall is a bin whose points span the full storey height
    # (>1.5m) -- furniture doesn't. Clear span = distance between the nearest
    # full-height wall faces on each side, from the real point coordinates. ----
    # use the FULL storey height so a real wall bin spans >1.5m (furniture won't)
    wb = (z >= z_floor + 0.2) & (z <= z_ceiling - 0.1)
    xw, yw, zw = x[wb], y[wb], z[wb]

    OPEN_MARGIN = 0.30    # a wall this far BEYOND the room's free-space edge means
                          # that side is open -> cap at the partition, don't overshoot

    def clear_span(axis, along_c, perp_c, fs_lo, fs_hi):
        """Clear span between the room's bounding faces, direct from LiDAR.

        On an ENCLOSED side a full-height wall sits right at the room's
        free-space edge (fs_lo/fs_hi) -> use its exact point-measured position
        (this is the 0-2%-accurate case). On an OPEN side (open-plan living,
        one-open-side room) the nearest wall is far past the segmented edge, so
        the ray would overshoot to a distant wall -> instead cap the span at the
        watershed partition (fs_lo/fs_hi) and flag it open. Returns (span, open)."""
        if axis == "x":
            m2 = np.abs(yw - perp_c) <= 0.12
            co, zc = xw[m2], zw[m2]
        else:
            m2 = np.abs(xw - perp_c) <= 0.12
            co, zc = yw[m2], zw[m2]
        if co.size < 30:
            return None
        # Coarse bins only LOCATE the faces; every returned position is
        # re-measured from the raw coordinates, so nothing is grid-quantized.
        faces = detect_wall_faces(co, zc, bin_m=0.05, min_points=6, min_span_m=1.5)
        if not faces:
            return None
        wp = np.array([f.value for f in faces])
        lft = wp[wp < along_c]; rgt = wp[wp > along_c]
        # left face: nearest wall if it sits at the room edge, else the partition.
        # A capped (open) side has no measured face, so it carries no stderr --
        # its uncertainty is the segmentation's, not the LiDAR's.
        if lft.size and lft.max() >= fs_lo - OPEN_MARGIN:
            left, open_l = float(lft.max()), False
            se_l = faces[int(np.argmin(np.abs(wp - left)))].stderr
        else:
            left, open_l, se_l = fs_lo, True, CELL
        # right face: same logic mirrored
        if rgt.size and rgt.min() <= fs_hi + OPEN_MARGIN:
            right, open_r = float(rgt.min()), False
            se_r = faces[int(np.argmin(np.abs(wp - right)))].stderr
        else:
            right, open_r, se_r = fs_hi, True, CELL
        span = right - left
        if span < 0.5:
            return None
        return span, (open_l or open_r), float(np.hypot(se_l, se_r))

    nroom = 0
    FT = cv2.FONT_HERSHEY_SIMPLEX
    for Lr in room_labels:
        m = (mk == Lr)
        if m.sum() < (1.0 / (CELL * CELL)):         # ignore < 1 m^2
            continue
        rd = distm * m
        cy, cx = np.unravel_index(int(np.argmax(rd)), rd.shape)
        rcx = xmin + cx * CELL; rcy = ymax - cy * CELL   # room center in METRIC
        dbg_col = cv2.applyColorMap(np.uint8([[(Lr * 47) % 256]]), cv2.COLORMAP_HSV)[0, 0].tolist()
        seg[m] = dbg_col                                 # DEBUG: paint this region
        cv2.circle(seg, (cx, cy), 4, (0, 0, 0), -1)
        cv2.putText(seg, str(nroom + 1), (cx + 5, cy), FT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        # room free-space extent through the center (contiguous run of this
        # room's mask): the watershed edge that bounds it on each side, used to
        # cap open sides so the LiDAR ray can't overshoot to a distant wall.
        row = m[cy, :]
        c0 = cx
        while c0 > 0 and row[c0 - 1]:
            c0 -= 1
        c1 = cx
        while c1 < W - 1 and row[c1 + 1]:
            c1 += 1
        fs_x_lo, fs_x_hi = xmin + c0 * CELL, xmin + c1 * CELL
        col = m[:, cx]
        r0 = cy
        while r0 > 0 and col[r0 - 1]:
            r0 -= 1
        r1 = cy
        while r1 < H - 1 and col[r1 + 1]:
            r1 += 1
        fs_y_lo, fs_y_hi = ymax - r1 * CELL, ymax - r0 * CELL   # row down = -y
        rw = clear_span("x", rcx, rcy, fs_x_lo, fs_x_hi)        # direct from LiDAR
        rh = clear_span("y", rcy, rcx, fs_y_lo, fs_y_hi)
        if rw is None or rh is None:
            continue
        wdt, ow, sew = rw; hgt, oh, seh = rh
        nroom += 1
        op = "~" if (ow or oh) else ""                          # ~ = open-side span
        # 95% interval on the worse of the two axes; a mm-level number is only
        # a claim if it carries its error bar.
        pm = 1.96 * max(sew, seh) * 1000
        for t, dy, sc in [(f"{op}{wdt*1000:.0f} x {hgt*1000:.0f} mm", -6, 0.5),
                          (f"+/-{pm:.1f}mm  ({wdt*3.28084:.1f} x {hgt*3.28084:.1f} ft)", 12, 0.42)]:
            (tw, th), _ = cv2.getTextSize(t, FT, sc, 1)
            tx = int(np.clip(cx - tw // 2, 2, W - tw - 2))       # keep label on-canvas
            cv2.rectangle(fr, (tx - 2, cy + dy - th - 1), (tx + tw + 2, cy + dy + 3), (255, 255, 255), -1)
            cv2.putText(fr, t, (tx, cy + dy), FT, sc, (150, 0, 160), 1, cv2.LINE_AA)
    cv2.putText(fr, f"2D floorplan - {nroom} rooms, internal CLEAR size (mm / ft)",
               (10, 20), FT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(fr, "~ = open side, span capped at partition",
               (10, 40), FT, 0.44, (150, 0, 160), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "floorplan_rooms_dimensioned.png"), fr)
    cv2.putText(seg, f"DEBUG regions: {ncore-1} watershed seeds -> {len(room_labels)} merged -> {nroom} rooms",
               (10, 22), FT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "debug_room_regions.png"), seg)
    log(f"wrote floorplan_rooms_dimensioned.png ({nroom} rooms) + debug_room_regions.png")
    log(f"wrote wall_plan_2d + wall_plan_dimensioned + floorplan_complete "
        f"({len(vx)} V-gridlines, {len(hy)} H-gridlines)")

    z_base = z_floor - 0.10
    nz = int((z_ceiling + 0.15 - z_base) / UZ_RES) + 1
    parts = []
    n_open = n_groove = 0
    all_cutters = []
    for wi, (p0, p1) in enumerate(walls):
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < MIN_WALL_LEN:
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        rel = xyz[:, :2] - p0
        u = rel @ d; perp = rel @ n
        near = (np.abs(perp) <= WALL_HALF_BAND) & (u >= 0) & (u <= L)
        mid = (p0 + p1) / 2
        if near.sum() < 120:            # keep sparser (small) walls
            continue
        uu = u[near]; zz = z[near]; pp = perp[near]
        thick = float(np.clip(np.percentile(pp, 92) - np.percentile(pp, 8), 0.06, 0.35))

        # ---- phantom-wall filter: a real wall has material along most of its
        # length. Drop segments whose u-coverage is sparse (furniture / a
        # free-space carving artifact -- e.g. false walls inside a bathroom). ----
        ubins = max(int(L / 0.15) + 1, 2)
        cov = len(np.unique(np.clip((uu / 0.15).astype(int), 0, ubins - 1))) / ubins
        if cov < 0.35:                  # more lenient so short real walls survive
            continue

        # The wall SOLID comes from the extruded footprint (jogs preserved); here
        # we only compute the opening + reveal cutters per segment. Rz orients the
        # cutter boxes along this wall.
        Rz = trimesh.transformations.rotation_matrix(np.arctan2(d[1], d[0]), [0, 0, 1])

        # ---- grooves from the EXACT 3D recess silhouette (top-view slices x side
        # view). Build a (height z x along-wall u) grid of the room-side surface
        # depth; the recess = main face - surface. Cut only the connected recessed
        # POCKETS (their real u-range), room-side only, above the wall's own noise
        # and coherent -- so small walls and non-recessed spans are never carved
        # away, and grooves land exactly where the 3D space actually steps back. ----
        cutters = []
        room_set = set(room_labels)
        UB = 0.10
        for side in (+1.0, -1.0):
            sel = (pp * side) > 0 if pp.size else np.zeros(0, bool)
            if sel.sum() < 200:
                continue
            # only cut a face that looks toward a ROOM (top-view room map)
            hits = tot = 0
            for tt in np.linspace(0.15, 0.85, 6):
                mp = p0 + d * (tt * L) + n * side * 0.25
                cxp = int((mp[0] - xmin) / CELL); cyp = int((ymax - mp[1]) / CELL)
                if 0 <= cyp < H and 0 <= cxp < W:
                    tot += 1; hits += int(mk[cyp, cxp] in room_set)
            if tot == 0 or hits / tot < 0.5:
                continue
            sp = pp[sel] * side; sz = zz[sel]; su = uu[sel]
            nu = max(int(L / UB), 1)
            zb = np.arange(z_floor + 0.15, z_ceiling - 0.15, 0.1)
            surf = np.full((len(zb), nu), np.nan)      # (z, u) room-side surface depth
            for iz, zl in enumerate(zb):
                zm = (sz >= zl) & (sz < zl + 0.1)
                if zm.sum() < 8:
                    continue
                ui = np.clip((su[zm] / UB).astype(int), 0, nu - 1); spz = sp[zm]
                for iu in np.unique(ui):
                    cm = ui == iu
                    if cm.sum() >= 3:
                        surf[iz, iu] = np.median(spz[cm])
            valid = ~np.isnan(surf)
            if valid.sum() < 6:
                continue
            face = float(np.nanmedian(surf))
            noise = 1.4826 * float(np.nanmedian(np.abs(surf[valid] - face))) + 1e-6
            thr = max(0.025, 3.0 * noise)              # above this wall's own scatter
            recess = np.where(valid, face - surf, 0.0)
            groove = (recess >= thr) & valid           # exact (z,u) groove silhouette
            lbl, ncomp = ndimage.label(groove, structure=np.ones((3, 3)))
            for cc in range(1, ncomp + 1):
                ys, xs = np.where(lbl == cc)
                if ys.size < 3:
                    continue
                bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
                if ys.size / bb < 0.5:                 # coherent, not a scatter of specks
                    continue
                z0 = zb[ys.min()]; z1 = zb[ys.max()] + 0.1
                u0 = xs.min() * UB; u1 = (xs.max() + 1) * UB
                # SHALLOW surface reveal (a deviation), not a hole: remove only the
                # outer measured-depth shell off this face; wall stays solid behind.
                depth = float(np.clip(np.median(recess[ys, xs]), 0.005, min(0.035, thick * 0.4)))
                uw = max(u1 - u0, UB)
                box_th = depth + 0.02                       # slight overshoot for a clean cut
                cn = thick / 2 - depth + box_th / 2         # inner face at thick/2-depth
                ch = trimesh.creation.box(extents=(uw * 0.98, box_th, z1 - z0))
                ch.apply_transform(Rz)
                off = (p0 + d * (u0 + u1) / 2) + n * side * cn
                ch.apply_translation((off[0], off[1], (z0 + z1) / 2))
                cutters.append(ch); n_groove += 1

        # ---- openings as their REAL (u,z) SILHOUETTE, not rectangles. Build a
        # fine (height z x along-wall u) material mask; an opening is a wide EMPTY
        # region inside the wall body, flanked by material. Cut its exact contour
        # so an ARCHED head (material curving down over the opening) stays arched,
        # and door/window profiles match what was actually built. ----
        UR = 0.02
        z0w = z_floor - 0.05; z1w = z_ceiling + 0.05
        nu2 = max(int(L / UR) + 1, 4); nz2 = max(int((z1w - z0w) / UR) + 1, 4)
        mat = np.zeros((nz2, nu2), np.uint8)
        mat[np.clip(((zz - z0w) / UR).astype(int), 0, nz2 - 1),
            np.clip((uu / UR).astype(int), 0, nu2 - 1)] = 1
        mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # fill scan gaps
        core = np.zeros_like(mat)
        core[int((z_floor + 0.08 - z0w) / UR):int((z_ceiling - 0.12 - z0w) / UR), :] = 1
        empty = cv2.morphologyEx(((core > 0) & (mat == 0)).astype(np.uint8),
                                 cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ncE, lblE, statsE, _ = cv2.connectedComponentsWithStats(empty, 8)
        for i in range(1, ncE):
            x, y, w, h, _ = statsE[i]
            if not (0.5 <= w * UR <= 3.0) or h * UR < 0.6:      # opening size gate
                continue
            if x <= 1 or x + w >= nu2 - 1:                      # must be flanked by wall (not wall end)
                continue
            cnts, _ = cv2.findContours((lblE == i).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            cnt = cv2.approxPolyDP(cnt, 0.02 / UR, True)        # smooth pixel steps, keep the arch
            if len(cnt) < 3:
                continue
            poly = Polygon([(c * UR, z0w + r * UR) for c, r in cnt[:, 0, :]])
            if not poly.is_valid or poly.area < 0.3:
                continue
            try:
                cutter = trimesh.creation.extrude_polygon(poly, height=thick * 3)
                cutter.apply_transform(np.array(
                    [[d[0], 0, n[0], p0[0] - n[0] * thick * 1.5],
                     [d[1], 0, n[1], p0[1] - n[1] * thick * 1.5],
                     [0, 1, 0, 0], [0, 0, 0, 1]], float))
                cutters.append(cutter); n_open += 1
            except Exception:
                pass

        all_cutters.extend(cutters)

    # ---- WALL SOLID from the extruded carved footprint (jogs / pilasters /
    # niches / true thickness preserved), then subtract every opening + reveal
    # cutter. This replaces the straight-box walls so the small 90-degree jogs
    # the user flagged are kept. ----
    fp_polys = footprint_polygons(ws, xmin, ymax)
    wall_solid = None
    for pg in fp_polys:
        try:
            geoms = pg.geoms if hasattr(pg, "geoms") else [pg]
            for g in geoms:
                pr = trimesh.creation.extrude_polygon(g, height=storey)
                pr.apply_translation((0, 0, z_floor))
                wall_solid = pr if wall_solid is None else trimesh.util.concatenate([wall_solid, pr])
        except Exception:
            pass
    for ch in all_cutters:
        try:
            wall_solid = wall_solid.difference(ch)
        except Exception:
            pass
    parts.append(("walls", wall_solid))

    # floor slab only (no ceiling, per request -- keeps the interior visible)
    fx, fy = (xmax - xmin), (ymax - ymin)
    parts.append(("floor", slab((xmin + xmax) / 2, (ymin + ymax) / 2, z_floor - 0.05, fx, fy, 0.10)))

    scene = trimesh.Scene()
    for name, m in parts:
        if m is None or getattr(m, "is_empty", True):
            continue
        m.visual.face_colors = ([150, 130, 110, 255] if name == "floor"
                                else [205, 205, 210, 255])
        scene.add_geometry(m, geom_name=name)
    scene.export(str(out_dir / "modular_model.glb"))
    scene.export(str(out_dir / "modular_model.obj"))
    log(f"exported modular_model.glb/.obj: {len(parts)} parts, "
        f"{n_open} openings (from silhouette gaps), {n_groove} groove cuts")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
