"""2D-density line-rescue pass (prototype).

Motivation: the balcony railing (and similar thin, offset-scattered
structures) fragments into many small vertical planes in 3D RANSAC that fail
group_wall_runs's length/merge thresholds -- but reads as one obviously
continuous bright line in the top-down point-density image (see
inspect_gap_region.py's gap_density_full.png).

This is NOT a return to the rejected top-down-projection floorplan pipeline
(that flattened Z entirely and hallucinated walls from aligned clutter). Here
the 2D image only PROPOSES candidate wall lines (via Hough transform); every
candidate is then independently verified against the REAL 3D point cloud --
it must have real point support in a band around the line AND must span a
large fraction of the floor-to-ceiling height AND must touch the floor
(rejects a horizontal furniture edge that happens to be long and straight).
Only candidates passing all three become synthetic wall RUNS, in the exact
same dict shape group_wall_runs produces, so they flow through the existing,
unmodified regularize/openings/model pipeline unchanged.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\rescue_2d_lines.py <scan.las> <out_dir>
"""
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
from scripts.recon.trajectory import approx_trajectory, wall_crossings
from scripts.recon.openings import detect_openings
from scripts.recon.floorplan2d import build_room_polygons
from scripts.recon.schema import WallStep
from scripts.isolidarflow import DEFAULT_CONFIG, _wall_length, _wall_namespaces, _write_debug_png

PPM = 80
HOUGH_THRESH = 60          # votes needed to accept a Hough line
HOUGH_MIN_LEN_PX = 60      # ~0.75 m at 80 ppm
HOUGH_MAX_GAP_PX = 15      # ~0.19 m -- bridges railing baluster gaps
ANGLE_TOL_DEG = 8.0        # must be within this of world X or Y (already axis-aligned)
CLUSTER_OFFSET_TOL_M = 0.20
EXISTING_WALL_REACH_M = 0.25   # skip candidates already explained by an accepted wall
BAND_M = 0.15
MIN_SUPPORT_PTS = 150
MIN_HEIGHT_FRAC = 0.55          # tier 1: solid wall (near-full floor-to-ceiling)
FLOOR_TOUCH_TOL_M = 0.35
RAILING_MIN_HEIGHT_M = 0.70     # tier 2: railing/half-wall -- floor-touching but
RAILING_MAX_HEIGHT_M = 1.30     # only waist-high (0.7-1.3m), NOT near ceiling.
RAILING_MIN_SUPPORT_PTS = 400   # stricter point-count bar than a full wall (3x),
                               # since a shorter candidate needs stronger real
                               # evidence to rule out furniture (e.g. a sofa
                               # back) rather than a genuine balcony railing.


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_density_image(xy, ppm):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    W, H = int((xmax - xmin) * ppm) + 1, int((ymax - ymin) * ppm) + 1
    cols = np.clip(((xy[:, 0] - xmin) * ppm).astype(int), 0, W - 1)
    rows = np.clip(((ymax - xy[:, 1]) * ppm).astype(int), 0, H - 1)
    density = np.zeros((H, W), dtype=np.float64)
    np.add.at(density, (rows, cols), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 255).astype(np.uint8)
    return gray, xmin, ymin, xmax, ymax, W, H


def px_to_world(x_px, y_px, xmin, ymax, ppm):
    return (xmin + x_px / ppm, ymax - y_px / ppm)


def detect_candidate_lines(gray):
    """Binarize (plain Otsu, non-inverted) + close small gaps + Hough.

    Four other attempts were tried and rejected by visual inspection:
    adaptiveThreshold (both C signs), Otsu inverted, and Canny edge
    detection (which picks up per-pixel shot noise across the whole sparse
    interior instead of just the walls). Plain non-inverted Otsu -- despite
    putting walls on the black/background side rather than white/foreground
    -- is the one that empirically traced a clean, complete, fully-connected
    floorplan outline: cv2.HoughLinesP still finds the right lines here
    because it traces the boundary of the large white floor-interior blobs,
    and that boundary is geometrically coincident with the wall centerlines
    either way -- which side is nominally "foreground" turned out not to
    matter for line position, only for how much noise survives, and this
    combination has the least.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN_PX, maxLineGap=HOUGH_MAX_GAP_PX)
    return closed, (lines if lines is not None else np.zeros((0, 1, 4), dtype=int))


def classify_and_cluster(lines_px, xmin, ymax, ppm):
    """Convert Hough segments to world coords, keep only near-axis-aligned
    ones, and cluster by (direction, offset) into candidate wall lines with
    a merged u-range (same style as structure._greedy_chain)."""
    items = []
    lines_flat = lines_px.reshape(-1, 4) if lines_px.size else lines_px.reshape(0, 4)
    for seg in lines_flat:
        x1, y1, x2, y2 = seg
        wx1, wy1 = px_to_world(x1, y1, xmin, ymax, ppm)
        wx2, wy2 = px_to_world(x2, y2, xmin, ymax, ppm)
        dx, dy = wx2 - wx1, wy2 - wy1
        length = np.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 0=horizontal, 90=vertical
        if angle <= ANGLE_TOL_DEG:
            direction, offset = "y", (wy1 + wy2) / 2.0   # runs along X -> normal along Y
            u_lo, u_hi = sorted((wx1, wx2))
        elif angle >= 90 - ANGLE_TOL_DEG:
            direction, offset = "x", (wx1 + wx2) / 2.0   # runs along Y -> normal along X
            u_lo, u_hi = sorted((wy1, wy2))
        else:
            continue  # not axis-aligned enough -- discard (diagonal noise)
        items.append(dict(direction=direction, offset=offset, u_min=u_lo, u_max=u_hi))

    clusters = []
    for axis_name in ("x", "y"):
        group = sorted([it for it in items if it["direction"] == axis_name], key=lambda s: s["offset"])
        cur = []
        for it in group:
            if cur and abs(it["offset"] - np.mean([c["offset"] for c in cur])) > CLUSTER_OFFSET_TOL_M:
                clusters.append(cur)
                cur = []
            cur.append(it)
        if cur:
            clusters.append(cur)

    merged = []
    for members in clusters:
        direction = members[0]["direction"]
        offset = float(np.mean([m["offset"] for m in members]))
        u_min = min(m["u_min"] for m in members)
        u_max = max(m["u_max"] for m in members)
        merged.append(dict(direction=direction, offset=offset, u_min=u_min, u_max=u_max))
    return merged


def already_explained(cand, existing_walls):
    for w in existing_walls:
        if w["direction"] != cand["direction"]:
            continue
        if abs(w["offset_m"] - cand["offset"]) > EXISTING_WALL_REACH_M:
            continue
        u_i = 1 if w["direction"] == "x" else 0
        p0, p1 = np.asarray(w["p0"]), np.asarray(w["p1"])
        w_lo, w_hi = sorted((p0[u_i], p1[u_i]))
        overlap = min(w_hi, cand["u_max"]) - max(w_lo, cand["u_min"])
        if overlap > 0.3 * (cand["u_max"] - cand["u_min"]):
            return True
    return False


def verify_candidate(cand, xyz, z_floor, z_ceiling):
    """3D verification, two tiers: a full solid wall (near-full height) or a
    railing/half-wall (floor-touching, waist-high only, stricter point-count
    bar). Returns None if neither tier is satisfied -- a candidate that is
    floor-touching but SHORT (e.g. a sofa back) and lacks the railing tier's
    higher support count is correctly rejected as furniture, not a wall."""
    axis_i = 0 if cand["direction"] == "y" else 1   # perpendicular axis index
    u_i = 1 - axis_i
    perp = xyz[:, axis_i]
    u = xyz[:, u_i]
    z = xyz[:, 2]
    in_band = (np.abs(perp - cand["offset"]) <= BAND_M) & (u >= cand["u_min"]) & (u <= cand["u_max"])
    n_support = int(np.count_nonzero(in_band))
    if n_support < MIN_SUPPORT_PTS:
        return None
    band_z = z[in_band]
    z_lo, z_hi = np.percentile(band_z, [2, 98])
    height = z_hi - z_lo
    height_frac = height / max(z_ceiling - z_floor, 1e-6)
    touches_floor = (z_lo - z_floor) <= FLOOR_TOUCH_TOL_M
    if not touches_floor:
        return None

    if height_frac >= MIN_HEIGHT_FRAC:
        tier = "wall"
    elif (RAILING_MIN_HEIGHT_M <= height <= RAILING_MAX_HEIGHT_M
          and n_support >= RAILING_MIN_SUPPORT_PTS):
        tier = "railing"
    else:
        return None
    return dict(n_support=n_support, z_min=float(z_lo), z_max=float(z_hi),
               height_frac=float(height_frac), tier=tier)


def build_rescue_run(cand, verified):
    direction = cand["direction"]
    axis_vec = np.array([1.0, 0.0, 0.0]) if direction == "y" else np.array([0.0, 1.0, 0.0])
    u_vec = np.array([0.0, 1.0, 0.0]) if direction == "y" else np.array([1.0, 0.0, 0.0])
    offset = cand["offset"]
    p0 = axis_vec[:2] * offset + u_vec[:2] * cand["u_min"]
    p1 = axis_vec[:2] * offset + u_vec[:2] * cand["u_max"]
    step = WallStep(0.0, cand["u_min"], cand["u_max"], verified["z_min"], verified["z_max"])
    return dict(direction=direction, normal=tuple(axis_vec), offset_m=float(offset),
               p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])),
               steps=[step], members=[(float(cand["u_min"]), float(cand["u_max"]))],
               members_z=[dict(u_min=float(cand["u_min"]), u_max=float(cand["u_max"]),
                              z_min=verified["z_min"], z_max=verified["z_max"],
                              offset=0.0, n=verified["n_support"])],
               rescued_2d=True, tier=verified["tier"])


def main(in_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass

    log("loading + isolating...")
    scan = load_scan(in_path, max_points=cfg["max_points"], rng_seed=seed)
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
    if traj.shape[0]:
        traj = traj @ np.asarray(R, dtype=float).T

    if scan.n > cfg["plane_max_points"]:
        keep = rng.choice(scan.n, size=cfg["plane_max_points"], replace=False)
        scan = scan.subset(keep)
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(xyz[:, 2], cfg["z_ceiling_pct"]))
    log(f"working cloud: {scan.n:,} pts, z-band ({z_floor:.2f}, {z_ceiling:.2f})")

    t0 = time.time()
    all_planes = planes.detect_planes(
        xyz, dist_thresh=cfg["plane_dist_thresh_m"], min_inliers=cfg["plane_min_inliers"],
        max_planes=cfg["plane_max_planes"], dbscan_eps=cfg["plane_dbscan_eps_m"],
        dbscan_min=cfg["plane_dbscan_min"], z_floor=z_floor, z_ceiling=z_ceiling,
        horizontal_min=cfg["plane_horizontal_min"], ransac_iters=cfg["plane_ransac_iters"], seed=seed)
    verticals = [p for p in all_planes if p.label == "vertical"]
    log(f"detect_planes: {len(all_planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, used_w = structure.group_wall_runs(
        verticals, xyz, np.eye(3), merge_offset_m=cfg["wr_merge_offset_m"],
        max_relief_m=cfg["wr_max_relief_m"], min_run_length_m=cfg["wr_min_run_length_m"],
        angle_tol_deg=cfg["wr_angle_tol_deg"], full_height_frac=cfg["wr_full_height_frac"],
        u_gap_m=cfg["wr_u_gap_m"], min_run_inliers=cfg["wr_min_run_inliers"], return_used_indices=True)
    columns, beams, used_cb = structure.extract_columns_beams(
        verticals, xyz, z_floor, z_ceiling, min_size_m=cfg["cb_min_size_m"], max_size_m=cfg["cb_max_size_m"],
        height_tol_m=cfg["cb_height_tol_m"], beam_elevation_m=cfg["cb_beam_elevation_m"],
        beam_ceiling_tol_m=cfg["cb_beam_ceiling_tol_m"], beam_offset_gap_m=cfg["cb_beam_offset_gap_m"],
        return_used_indices=True)
    log(f"baseline: runs={len(runs)} columns={len(columns)} beams={len(beams)}")

    # --- 2D rescue pass ---
    xy = xyz[:, :2]
    gray, xmin, ymin, xmax, ymax, W, H = build_density_image(xy, PPM)
    binary, lines_px = detect_candidate_lines(gray)
    log(f"Hough proposed {len(lines_px)} raw line segments")
    candidates = classify_and_cluster(lines_px, xmin, ymax, PPM)
    log(f"clustered into {len(candidates)} axis-aligned candidate lines")

    rescued_runs = []
    rejected_no3d = 0
    already_have = 0
    for cand in candidates:
        if cand["u_max"] - cand["u_min"] < cfg["wr_min_run_length_m"]:
            continue
        if already_explained(cand, runs):
            already_have += 1
            continue
        verified = verify_candidate(cand, xyz, z_floor, z_ceiling)
        if verified is None:
            rejected_no3d += 1
            continue
        rescued_runs.append(build_rescue_run(cand, verified))

    log(f"rescue result: {len(rescued_runs)} NEW walls accepted "
        f"({already_have} already explained, {rejected_no3d} failed 3D verification)")
    for r in rescued_runs:
        log(f"  rescued [{r.get('tier','?')}] {r['direction']}-wall at offset={r['offset_m']:.2f} "
            f"u=[{r['members'][0][0]:.2f},{r['members'][0][1]:.2f}] "
            f"z=[{r['steps'][0].z_min_m:.2f},{r['steps'][0].z_max_m:.2f}] "
            f"support={r['members_z'][0]['n']}")

    # --- rebuild rooms/openings with baseline vs baseline+rescued ---
    def finish(all_runs, label):
        walls = regularize.snap_walls(all_runs, np.eye(3), angle_tol_deg=cfg["snap_angle_tol_deg"])
        walls = regularize.pair_thickness(walls, xyz, default_m=cfg["pair_default_m"],
                                          min_thickness_m=cfg["pair_min_thickness_m"],
                                          max_thickness_m=cfg["pair_max_thickness_m"],
                                          min_overlap_frac=cfg["pair_min_overlap_frac"])
        walls = regularize.recenter_walls(walls, xyz, z_floor, z_ceiling,
                                          min_thickness_m=cfg["recenter_min_thickness_m"])
        walls = regularize.resolve_corners(walls, tol_m=cfg["corner_tol_m"])
        walls = regularize.snap_endpoints_to_lines(walls, reach_m=cfg["line_snap_reach_m"],
                                                   dangling_tol_m=cfg["line_snap_dangling_tol_m"])
        walls = [w for w in walls if _wall_length(w) > cfg["min_wall_length_m"]]
        for w in walls:
            w["floor_z_m"], w["ceiling_z_m"] = z_floor, z_ceiling
        crossings = wall_crossings(traj, walls, end_margin_m=cfg["crossing_end_margin_m"])
        openings_by_wall = detect_openings(walls, xyz, traj, crossings, z_floor, z_ceiling, cfg["priors"])
        n_openings = sum(len(v) for v in openings_by_wall.values())
        wall_ns = _wall_namespaces(walls)
        rooms = [p.simplify(cfg["room_simplify_m"])
                 for p in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_primary_m"])
                 if p.area >= cfg["room_min_area_m2"]]
        for cell in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_recovery_m"]):
            if cell.area >= cfg["room_min_area_m2"] and not any(r.contains(cell.centroid) for r in rooms):
                rooms.append(cell.simplify(cfg["room_simplify_m"]))
        log(f"[{label}] walls={len(walls)} openings={n_openings} rooms={len(rooms)} "
            f"areas={sorted(round(r.area,1) for r in rooms)}")
        try:
            _write_debug_png(str(out_dir), xyz, walls, rooms, columns, openings_by_wall, z_floor, z_ceiling, scan)
            (out_dir / "floorplan_debug.png").replace(out_dir / f"debug_{label}.png")
        except Exception as exc:
            log(f"  debug PNG failed: {exc}")
        return walls, rooms, n_openings

    finish(list(runs), "baseline")
    finish(list(runs) + rescued_runs, "rescued")

    # side-by-side overlay: baseline green, rescued additions magenta
    density_bgr = cv2.merge([gray, gray, gray])
    for r in runs:
        p0 = (int((r["p0"][0] - xmin) * PPM), int((ymax - r["p0"][1]) * PPM))
        p1 = (int((r["p1"][0] - xmin) * PPM), int((ymax - r["p1"][1]) * PPM))
        cv2.line(density_bgr, p0, p1, (0, 220, 0), 3)
    for r in rescued_runs:
        p0 = (int((r["p0"][0] - xmin) * PPM), int((ymax - r["p0"][1]) * PPM))
        p1 = (int((r["p1"][0] - xmin) * PPM), int((ymax - r["p1"][1]) * PPM))
        cv2.line(density_bgr, p0, p1, (255, 0, 255), 4)
    cv2.putText(density_bgr, f"green=baseline({len(runs)}) magenta=2D-rescued-NEW({len(rescued_runs)})",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "rescue_overlay.png"), density_bgr)
    cv2.imwrite(str(out_dir / "hough_binary_input.png"), binary)
    log("wrote rescue_overlay.png, hough_binary_input.png, debug_baseline.png, debug_rescued.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
