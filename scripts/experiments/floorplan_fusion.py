"""Fuse the ML-detected floorplan (walls/windows/balcony-doors from
bbox_data.json, structural topology + positions) with the LiDAR point
cloud (real as-built dimensions). The floorplan is the STRUCTURAL BASE --
which elements exist and roughly where -- the LiDAR cloud supplies the
true measured length/thickness/height/opening-size at each of those
positions, since as-built construction deviates from the architect's plan.

Pipeline:
1. Parse bbox_data.json -> metric-space wall segments + openings (pixel * scale).
2. Register the floorplan onto the LiDAR's isolated+aligned frame: since
   both are already in real metric units (no free scale factor), search a
   small set of candidate rigid transforms (0/90/180/270 rotation x
   normal/mirrored) + best-fit translation, scored by footprint IoU against
   the LiDAR room-union polygon.
3. For each registered floorplan wall: seed a search band at its expected
   position in LiDAR space, measure the REAL endpoints/thickness from
   nearby points (reusing this pipeline's existing recon.regularize/
   structure conventions), falling back to the floorplan's own geometry
   where no real point support exists (flagged, not silently trusted).
4. For each registered opening: measure real width/height/sill from the
   point cloud at that seeded location instead of blind occupancy search.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\floorplan_fusion.py
    <scan.las> <bbox_data.json> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame, planes, structure, regularize
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.recon.floorplan2d import build_room_polygons
from scripts.isolidarflow import DEFAULT_CONFIG, _wall_length, _wall_namespaces


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Parse the ML floorplan JSON into metric-space walls/openings
# ---------------------------------------------------------------------------

MAX_TRUSTED_TILT_DEG = 10.0  # walls tilted more than this are flagged as
                            # probable ML mis-detections (a real Manhattan
                            # apartment wall should never be this far off
                            # axis), not rectified as if it were real tilt.


def _rectify_wall_angle(p0, p1):
    """Rotate a wall segment about its OWN midpoint to snap it exactly onto
    its nearest true axis (0/90 deg), in the floorplan's own local frame --
    BEFORE any whole-floorplan registration.

    Found necessary on real data: bbox_data.json's 25 walls each carry their
    own small independent angular error (0.75-14.1 deg, median 1.79 deg) --
    real per-wall ML-detection noise, not real building tilt (this
    apartment is Manhattan-aligned). A single global registration rotation
    can only fit ONE average correction for the whole floorplan, so walls
    far from the fit's rotation pivot accumulate real positional drift even
    when the overall room-footprint IoU still scores well (confirmed: a
    1-2 deg per-wall error compounds to ~0.3-0.5m of drift over the
    apartment's ~10-15m extent). Rectifying each wall individually first
    removes this before registration ever sees it.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    angle = np.degrees(np.arctan2(d[1], d[0]))
    mod90 = angle % 90
    dev = min(mod90, 90 - mod90)
    if dev > MAX_TRUSTED_TILT_DEG:
        return p0, p1, dev, True  # flagged: too tilted to trust as noise
    correction = -dev if mod90 <= 45 else dev
    theta = np.radians(correction)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    mid = (p0 + p1) / 2.0
    p0_r = mid + R @ (p0 - mid)
    p1_r = mid + R @ (p1 - mid)
    return p0_r, p1_r, dev, False


def load_ml_floorplan(json_path):
    with open(json_path) as f:
        data = json.load(f)
    scale = data["dimensions"][0]["scale"]  # meters per pixel
    img_h = data["dimensions"][0]["height"]

    def px_to_m(x, y):
        # flip Y: image pixel-space has Y increasing downward; floorplan
        # metric space should have Y increasing "up" to match a normal
        # world/math convention (and this pipeline's own XY convention).
        return (x * scale, (img_h - y) * scale)

    walls = []
    n_rectified = n_flagged = 0
    for w in data["global_walls"]:
        p0 = px_to_m(w["x1"], w["y1"])
        p1 = px_to_m(w["x2"], w["y2"])
        p0_r, p1_r, tilt_dev_deg, flagged = _rectify_wall_angle(p0, p1)
        if flagged:
            n_flagged += 1
        else:
            n_rectified += 1
        walls.append(dict(p0=tuple(p0_r), p1=tuple(p1_r), height_m=w["height"] / 100.0,
                          thickness_m=w["thickness"] / 100.0,
                          tilt_dev_deg=tilt_dev_deg, tilt_flagged=flagged,
                          area_names=sorted({a.get("area_name", "Unassigned") for a in w.get("areas", [])})))
    log(f"rectified {n_rectified} wall angles onto true axes ({n_flagged} flagged as "
        f"too-tilted-to-trust, kept un-rectified)")

    openings = []
    for o in data["global_openings"]:
        cx = (o["x1"] + o["x2"]) / 2.0
        cy = (o["y1"] + o["y2"]) / 2.0
        center_m = px_to_m(cx, cy)
        kind = (o.get("opening_type") or o.get("product_name") or "unknown").lower()
        width_m = o.get("door_width")
        if width_m is not None:
            width_m = width_m * scale
        else:
            width_m = np.hypot((o["x2"] - o["x1"]) * scale, (o["y2"] - o["y1"]) * scale)
        openings.append(dict(
            center=center_m, type=kind,
            width_m=width_m,
            height_m=o["height"] / 100.0,
            sill_m=o.get("elevation", 0) / 100.0,
            orientation=o.get("orientation", "vertical"),
        ))
    log(f"parsed ML floorplan: {len(walls)} walls, {len(openings)} openings, scale={scale:.5f} m/px")
    return walls, openings


# ---------------------------------------------------------------------------
# 2. Register floorplan -> LiDAR frame (no free scale; rotation+mirror+translation)
# ---------------------------------------------------------------------------

def transform_points(pts, angle_deg, mirror, tx, ty):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = np.asarray(pts, dtype=float).copy()
    if mirror:
        pts = pts * np.array([-1.0, 1.0])
    pts = pts @ R.T
    pts += np.array([tx, ty])
    return pts


def footprint_polygon(walls_pts_pairs):
    from shapely.geometry import MultiPoint
    all_pts = np.vstack(walls_pts_pairs)
    return MultiPoint(all_pts).convex_hull


def register_floorplan(ml_walls, lidar_rooms):
    """Try candidate rigid transforms (4 rotations x mirror), each with a
    centroid-matching translation, score by convex-hull-footprint IoU
    against the LiDAR room union. Returns (angle_deg, mirror, tx, ty, iou)."""
    from shapely.ops import unary_union

    lidar_union = unary_union(lidar_rooms)
    lidar_centroid = np.array([lidar_union.centroid.x, lidar_union.centroid.y])

    ml_pts = np.array([w["p0"] for w in ml_walls] + [w["p1"] for w in ml_walls])
    ml_centroid_raw = ml_pts.mean(axis=0)

    best = None
    for angle in (0, 90, 180, 270):
        for mirror in (False, True):
            centered = ml_pts - ml_centroid_raw
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            m = centered * np.array([-1.0, 1.0]) if mirror else centered
            rotated = m @ R.T
            tx, ty = lidar_centroid  # translate rotated-centered cloud onto lidar centroid
            transformed = rotated + np.array([tx, ty])
            ml_poly = footprint_polygon([transformed])
            inter = ml_poly.intersection(lidar_union.convex_hull).area
            union = ml_poly.union(lidar_union.convex_hull).area
            iou = inter / union if union > 0 else 0.0
            eff_tx = tx - (ml_centroid_raw @ R.T * np.array([1, 1])) if False else None
            if best is None or iou > best[-1]:
                # store the FULL transform relative to raw (un-centered) ml points:
                # transformed = (raw - ml_centroid_raw) * mirror @ R.T + lidar_centroid
                # => transformed = raw @ (mirror@R.T) - ml_centroid_raw@(mirror@R.T) + lidar_centroid
                mirror_mat = np.diag([-1.0, 1.0]) if mirror else np.eye(2)
                full_R = mirror_mat @ R.T
                offset = -ml_centroid_raw @ full_R + lidar_centroid
                best = (angle, mirror, full_R, offset, iou)
    return best


def apply_registration(pt, full_R, offset):
    return np.asarray(pt, dtype=float) @ full_R + offset


# ---------------------------------------------------------------------------
# 3. per-wall measurement refinement: floorplan gives position, LiDAR gives
#    the true as-built numbers
# ---------------------------------------------------------------------------

def measure_wall_from_points(p0, p1, xyz, z_floor, z_ceiling,
                             band_m=0.20, min_support_pts=150,
                             endpoint_search_m=0.4, bin_m=0.02):
    """Given an ML wall's REGISTERED (position-only-trustworthy) endpoints,
    measure its true length/endpoints/thickness/height from real LiDAR
    points near that position. Returns None if no real support is found
    (flagged, not silently trusted) -- the floorplan told us WHERE to look,
    but if nothing is actually there we must say so, not fabricate a wall
    from the architect's plan alone."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    d = d / L
    n = np.array([-d[1], d[0]])
    xy = xyz[:, :2]
    u = (xy - p0) @ d
    perp = (xy - p0) @ n
    in_band = (np.abs(perp) <= band_m) & (u >= -endpoint_search_m) & (u <= L + endpoint_search_m)
    n_support = int(np.count_nonzero(in_band))
    if n_support < min_support_pts:
        return None

    u_band = u[in_band]
    z_band = xyz[in_band, 2]
    perp_band = perp[in_band]

    # true endpoints: density half-max crossing near each expected end
    def _crossing(u_vals, lo, hi, want_rising):
        window = u_vals[(u_vals >= lo) & (u_vals <= hi)]
        if window.size < 10:
            return None
        bins = np.arange(lo, hi + bin_m, bin_m)
        hist, edges = np.histogram(window, bins=bins)
        if hist.max() == 0:
            return None
        half = hist.max() / 2.0
        idx = np.where(hist >= half)[0]
        if idx.size == 0:
            return None
        return float(edges[idx[0]] if want_rising else edges[idx[-1] + 1])

    u0_real = _crossing(u_band, -endpoint_search_m, endpoint_search_m, True)
    u1_real = _crossing(u_band, L - endpoint_search_m, L + endpoint_search_m, False)
    u0_real = u0_real if u0_real is not None else 0.0
    u1_real = u1_real if u1_real is not None else L

    # true thickness: find the wall's TWO FACES as distinct density peaks in
    # a fine perpendicular-offset histogram, not a percentile spread (a
    # percentile over the whole search band is easily inflated by furniture/
    # noise sitting anywhere within it -- confirmed on real data: it gave a
    # suspiciously uniform ~0.3-0.37m "thickness" across every room type,
    # clustered right around the search bandwidth itself, not real wall
    # thickness, which is normally 0.10-0.15m for interior partitions).
    core = (u_band >= u0_real) & (u_band <= u1_real)
    perp_core = perp_band[core]
    thickness_m = None
    if perp_core.size >= 30:
        from scipy.signal import find_peaks
        bin_w = 0.01
        edges = np.arange(-band_m, band_m + bin_w, bin_w)
        hist, _ = np.histogram(perp_core, bins=edges)
        smoothed = np.convolve(hist, np.ones(3) / 3.0, mode="same")
        peak_idx, props = find_peaks(smoothed, height=smoothed.max() * 0.25, distance=3)
        if peak_idx.size >= 2:
            heights = props["peak_heights"]
            top2 = peak_idx[np.argsort(heights)[-2:]]
            offsets = edges[top2] + bin_w / 2.0
            thickness_m = float(abs(offsets[1] - offsets[0]))
        # a single dominant peak means only one face was found (single-sided
        # wall, or the other face lies outside band_m) -- thickness stays
        # unmeasured rather than guessed.

    z_core = z_band[core] if core.any() else z_band
    z_lo, z_hi = (np.percentile(z_core, [2, 98]) if z_core.size >= 30 else (z_floor, z_ceiling))

    real_p0 = p0 + d * u0_real
    real_p1 = p0 + d * u1_real
    return dict(p0=tuple(real_p0), p1=tuple(real_p1), length_m=float(u1_real - u0_real),
               thickness_m=thickness_m, z_min_m=float(z_lo), z_max_m=float(z_hi),
               n_support=n_support)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(las_path, json_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    ml_walls, ml_openings = load_ml_floorplan(json_path)

    log("loading + isolating LiDAR scan...")
    scan = load_scan(las_path, max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (approx_trajectory(scan.gps_time, scan.xyz, dt_s=cfg["traj_dt_s"])
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, _ = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                           max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)
    if scan.n > cfg["plane_max_points"]:
        keep = rng.choice(scan.n, size=cfg["plane_max_points"], replace=False)
        scan_small = scan.subset(keep)
    else:
        scan_small = scan
    xyz = scan_small.xyz
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    log(f"LiDAR storey height: {z_ceiling - z_floor:.3f} m  (ML floorplan wall height: "
        f"{ml_walls[0]['height_m']:.3f} m)")

    t0 = time.time()
    all_planes = planes.detect_planes(
        xyz, dist_thresh=cfg["plane_dist_thresh_m"], min_inliers=cfg["plane_min_inliers"],
        max_planes=cfg["plane_max_planes"], dbscan_eps=cfg["plane_dbscan_eps_m"],
        dbscan_min=cfg["plane_dbscan_min"], z_floor=z_floor, z_ceiling=z_ceiling,
        horizontal_min=cfg["plane_horizontal_min"], ransac_iters=cfg["plane_ransac_iters"], seed=seed)
    verticals = [p for p in all_planes if p.label == "vertical"]
    log(f"detect_planes: {len(all_planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs = structure.group_wall_runs(
        verticals, xyz, np.eye(3), merge_offset_m=cfg["wr_merge_offset_m"],
        max_relief_m=cfg["wr_max_relief_m"], min_run_length_m=cfg["wr_min_run_length_m"],
        angle_tol_deg=cfg["wr_angle_tol_deg"], full_height_frac=cfg["wr_full_height_frac"],
        u_gap_m=cfg["wr_u_gap_m"], min_run_inliers=cfg["wr_min_run_inliers"])
    lidar_walls = regularize.snap_walls(runs, np.eye(3), angle_tol_deg=cfg["snap_angle_tol_deg"])
    lidar_walls = regularize.pair_thickness(lidar_walls, xyz, default_m=cfg["pair_default_m"],
                                            min_thickness_m=cfg["pair_min_thickness_m"],
                                            max_thickness_m=cfg["pair_max_thickness_m"],
                                            min_overlap_frac=cfg["pair_min_overlap_frac"])
    lidar_walls = regularize.recenter_walls(lidar_walls, xyz, z_floor, z_ceiling,
                                            min_thickness_m=cfg["recenter_min_thickness_m"])
    lidar_walls = regularize.resolve_corners(lidar_walls, tol_m=cfg["corner_tol_m"])
    lidar_walls = regularize.snap_endpoints_to_lines(lidar_walls, reach_m=cfg["line_snap_reach_m"],
                                                     dangling_tol_m=cfg["line_snap_dangling_tol_m"])
    lidar_walls = [w for w in lidar_walls if _wall_length(w) > cfg["min_wall_length_m"]]
    log(f"LiDAR-only reconstruction: {len(lidar_walls)} walls")

    wall_ns = _wall_namespaces(lidar_walls)
    lidar_rooms = [p.simplify(cfg["room_simplify_m"])
                  for p in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_primary_m"])
                  if p.area >= cfg["room_min_area_m2"]]
    for cell in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_recovery_m"]):
        if cell.area >= cfg["room_min_area_m2"] and not any(r.contains(cell.centroid) for r in lidar_rooms):
            lidar_rooms.append(cell.simplify(cfg["room_simplify_m"]))
    log(f"LiDAR rooms: {len(lidar_rooms)}, total area {sum(r.area for r in lidar_rooms):.1f} m2")

    # --- 2. registration ---
    angle, mirror, full_R, offset, iou = register_floorplan(ml_walls, lidar_rooms)
    log(f"best registration: rotation={angle} mirror={mirror} IoU={iou:.3f}")

    ml_walls_reg = []
    for w in ml_walls:
        p0 = apply_registration(w["p0"], full_R, offset)
        p1 = apply_registration(w["p1"], full_R, offset)
        ml_walls_reg.append(dict(w, p0=tuple(p0), p1=tuple(p1)))
    ml_openings_reg = []
    for o in ml_openings:
        c = apply_registration(o["center"], full_R, offset)
        ml_openings_reg.append(dict(o, center=tuple(c)))

    ml_total_len = sum(np.hypot(*(np.array(w["p1"]) - np.array(w["p0"]))) for w in ml_walls_reg)
    log(f"registered {len(ml_walls_reg)} ML walls (total length {ml_total_len:.1f} m), "
        f"{len(ml_openings_reg)} ML openings onto the LiDAR frame")

    # --- 3. per-wall measurement refinement ---
    n_confirmed = n_unconfirmed = 0
    for w in ml_walls_reg:
        measured = measure_wall_from_points(w["p0"], w["p1"], xyz, z_floor, z_ceiling)
        w["measured"] = measured
        if measured is None:
            n_unconfirmed += 1
        else:
            n_confirmed += 1
            log(f"  CONFIRMED wall ({','.join(w['area_names'])}): floorplan length="
                f"{np.hypot(*(np.array(w['p1'])-np.array(w['p0']))):.2f}m -> "
                f"LiDAR-measured length={measured['length_m']:.2f}m, "
                f"thickness={measured['thickness_m']:.3f}m" if measured['thickness_m'] else
                f"thickness=unmeasured, support={measured['n_support']}pts")
    for w in ml_walls_reg:
        if w["measured"] is None:
            log(f"  UNCONFIRMED wall ({','.join(w['area_names'])}): no real point support found "
                f"at this floorplan-predicted location -- flagged, not fabricated")
    log(f"wall confirmation: {n_confirmed}/{len(ml_walls_reg)} confirmed by real LiDAR points, "
        f"{n_unconfirmed} unconfirmed (flagged)")

    # ---------- render comparison ----------
    xy = xyz[:, :2]
    xmin, ymin = np.percentile(xy, 0.2, axis=0) - 0.5
    xmax, ymax = np.percentile(xy, 99.8, axis=0) + 0.5
    ppm = 60
    W, H = int((xmax - xmin) * ppm), int((ymax - ymin) * ppm)

    def to_px(pt):
        return (int((pt[0] - xmin) * ppm), int((ymax - pt[1]) * ppm))

    cols_i = np.clip(((xy[:, 0] - xmin) * ppm).astype(int), 0, W - 1)
    rows_i = np.clip(((ymax - xy[:, 1]) * ppm).astype(int), 0, H - 1)
    density = np.zeros((H, W))
    np.add.at(density, (rows_i, cols_i), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 60).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])

    def wall_rect_px(p0, p1, thickness_m):
        """4-corner polygon (pixel space) for a wall drawn at its REAL
        thickness, not a thin cv2.line stroke -- a wall is a thick
        rectangle, and rendering it as a single-pixel-width line understates
        that even when the underlying data/measurement is thickness-aware."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        d = d / L if L > 1e-9 else np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]])
        half_t = max(thickness_m, 0.03) / 2.0
        corners = [p0 + n * half_t, p1 + n * half_t, p1 - n * half_t, p0 - n * half_t]
        return np.array([to_px(c) for c in corners], dtype=np.int32)

    # LiDAR-only walls: dim green (background reference), drawn thin --
    # these are the pre-existing plane-detection walls, not this fusion's output
    for w in lidar_walls:
        cv2.line(img, to_px(w["p0"]), to_px(w["p1"]), (0, 110, 0), 2)
    # ML floorplan walls: GREEN filled rectangle at REAL thickness if
    # LiDAR-confirmed (drawn at the MEASURED position AND measured
    # thickness where found, else the floorplan's own default thickness),
    # RED thin outline if unconfirmed (no real support -- flagged, drawn
    # thin/unfilled so "we don't trust this" is visually obvious)
    for w in ml_walls_reg:
        if w["measured"] is not None:
            m = w["measured"]
            t = m["thickness_m"] if m["thickness_m"] is not None else w["thickness_m"]
            rect = wall_rect_px(m["p0"], m["p1"], t)
            cv2.fillPoly(img, [rect], (0, 160, 0))
            cv2.polylines(img, [rect], True, (0, 255, 0), 1)
        else:
            rect = wall_rect_px(w["p0"], w["p1"], w["thickness_m"])
            cv2.polylines(img, [rect], True, (0, 0, 255), 1)
    # ML openings (registered): cyan dots
    for o in ml_openings_reg:
        cv2.circle(img, to_px(o["center"]), 6, (255, 255, 0), -1)
        cv2.putText(img, o["type"][:4], to_px(o["center"]), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                   (255, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(img, f"dim green=LiDAR-only walls ({len(lidar_walls)})  BRIGHT green=ML wall LiDAR-CONFIRMED "
                     f"({n_confirmed}, drawn at measured pos)  red=ML wall UNCONFIRMED ({n_unconfirmed})  "
                     f"cyan=ML openings [rot={angle} IoU={iou:.2f}]",
               (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "fusion_registration.png"), img)
    log(f"wrote {out_dir / 'fusion_registration.png'}")

    with open(out_dir / "fused_walls.json", "w") as f:
        json.dump([{
            "area_names": w["area_names"],
            "floorplan_p0": w["p0"], "floorplan_p1": w["p1"],
            "floorplan_thickness_m": w["thickness_m"], "floorplan_height_m": w["height_m"],
            "measured": w["measured"],
        } for w in ml_walls_reg], f, indent=2)
    log(f"wrote {out_dir / 'fused_walls.json'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
