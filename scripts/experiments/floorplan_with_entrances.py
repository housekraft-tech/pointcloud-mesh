"""2D floorplan rebuild with real floorplan-structure logic, not just
per-wall geometric thresholds:

1. EXTERIOR/INTERIOR wall classification (real, geometric -- a wall bordering
   only one room polygon is exterior; bordering two is interior). This
   finally makes `detect_openings`'s `exterior_flags` parameter meaningful --
   every prior run in this project passed it as None/always-False because
   nothing computed it.
2. ROOM-ADJACENCY GRAPH built from detected openings: which rooms connect to
   which. A room with ZERO connections is almost certainly hiding a missed
   entrance (real apartments have no fully sealed rooms except a balcony
   dead-end) -- flagged explicitly, never silently accepted.
3. ENTRANCE-FIRST classification: type may be uncertain, but "is this a real
   walkable opening" is answered independently and never downgraded by type
   uncertainty -- any void that passes the trajectory visibility gate,
   touches the floor, and isn't flagged oversized (likely missing wall, not
   an opening) counts as an entrance regardless of whether it confidently
   resolves to door/balcony_door/unknown_opening.
4. BALCONY PREDICTION from floorplan topology: a room is flagged
   likely_balcony if it is a graph LEAF (degree 1), its one connection is
   through an EXTERIOR wall, and either the opening is wide or the room
   itself is small relative to its neighbour -- matching how a real balcony
   sits (a small dead-end space off a main room, reached through the
   building's outer wall), rather than relying on opening width alone.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\floorplan_with_entrances.py <scan.las> <out_dir>
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
from scripts.isolidarflow import DEFAULT_CONFIG, _wall_length, _wall_namespaces

BALCONY_MAX_AREA_M2 = 6.0
BALCONY_MIN_WIDTH_M = 1.0
WALL_SIDE_PROBE_DISTANCES_M = (0.15, 0.25, 0.35, 0.5, 0.7)  # nearest hit wins


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def room_at_point(rooms, pt):
    from shapely.geometry import Point
    p = Point(pt[0], pt[1])
    for i, poly in enumerate(rooms):
        if poly.contains(p):
            return i
    return None


def _nearest_room_on_side(rooms, mid, n_dir):
    """Try increasing probe distances along n_dir from mid; return the room
    index of the FIRST (nearest) hit, or None if no distance finds one.
    More robust than one fixed distance: a fixed offset can overshoot into
    the SAME room again near a corner/non-convex boundary (confirmed on
    real data: a fixed 0.35 m probe wrongly returned the same room on both
    sides of a wall near a room corner), or undershoot past a wall's own
    thickness for an unusually thick wall."""
    for dist in WALL_SIDE_PROBE_DISTANCES_M:
        hit = room_at_point(rooms, mid + n_dir * dist)
        if hit is not None:
            return hit
    return None


def classify_wall_sides(walls, rooms):
    """For each wall, sample its two perpendicular sides at several u
    positions and see which room polygon (if any) each side falls in.
    Returns per-wall: exterior (bool), room_a, room_b (room indices or None
    on each side, majority vote across sample points along the wall)."""
    from collections import Counter
    result = []
    for w in walls:
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            result.append(dict(exterior=True, room_a=None, room_b=None))
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        u_samples = np.linspace(0.15, max(L - 0.15, 0.15), max(3, int(L / 0.5)))
        side_a_votes, side_b_votes = Counter(), Counter()
        for u in u_samples:
            mid = p0 + d * u
            side_a_votes[_nearest_room_on_side(rooms, mid, n)] += 1
            side_b_votes[_nearest_room_on_side(rooms, mid, -n)] += 1
        room_a = side_a_votes.most_common(1)[0][0] if side_a_votes else None
        room_b = side_b_votes.most_common(1)[0][0] if side_b_votes else None
        # exterior iff at most one side maps to a real room (the other side
        # is "outside" the reconstructed unit entirely, or unassigned).
        exterior = (room_a is None) or (room_b is None)
        result.append(dict(exterior=exterior, room_a=room_a, room_b=room_b))
    return result


def build_adjacency_graph(walls, wall_sides, openings_by_wall, rooms):
    """room_idx -> set(connected room_idx), plus a list of (room_idx, wall_idx,
    opening) for every entrance touching an exterior wall (candidate exits
    to the outside -- balconies, front doors)."""
    graph = {i: set() for i in range(len(rooms))}
    exterior_entrances = []  # (room_idx, wall_idx, opening_dict)
    for wi, ops in (openings_by_wall or {}).items():
        sides = wall_sides[wi]
        for op in ops:
            if not op.get("is_entrance", False):
                continue
            ra, rb = sides["room_a"], sides["room_b"]
            if ra is not None and rb is not None and ra != rb:
                graph[ra].add(rb)
                graph[rb].add(ra)
            elif ra is not None and rb is None:
                exterior_entrances.append((ra, wi, op))
            elif rb is not None and ra is None:
                exterior_entrances.append((rb, wi, op))
    return graph, exterior_entrances


def annotate_entrances(openings_by_wall, max_plausible_width_m):
    """Add is_entrance: floor-touching, gated (already true if it survived
    detect_openings), and not flagged oversized -- true regardless of
    whether `type` confidently resolved. This is what lets "I can't tell if
    it's a door" still count as "this is walkable", per the brief: type
    uncertainty must never demote a real, gated opening to a non-entrance.
    """
    for wi, ops in (openings_by_wall or {}).items():
        for op in ops:
            floor_touching = op["sill_m"] <= 0.15
            op["is_entrance"] = floor_touching and not op.get("oversized", False)
    return openings_by_wall


def predict_balconies(graph, exterior_entrances, rooms, openings_by_wall, wall_sides):
    """A room is likely_balcony if it is SMALL (mandatory -- a real balcony
    is never the size of a living room, no matter how wide its door is),
    is a graph leaf (<=1 interior-room connection -- a balcony is reached
    off exactly one main room, never a through-route), and has an
    exterior-wall entrance -- with a wide/walked entrance only ADDING
    confidence, never substituting for the size requirement.

    Earlier version used (small_room OR wide_opening) as the second
    condition -- on real data this flagged the 45.2 m2 main living room as
    "likely_balcony" purely because one of ITS exterior doors happened to be
    1.44 m wide, which any front door or double-door easily is. Room size is
    what actually distinguishes a balcony from a big room with a wide door.
    """
    predictions = {}
    for room_idx, wi, op in exterior_entrances:
        area = rooms[room_idx].area
        if area > BALCONY_MAX_AREA_M2:
            continue
        degree = len(graph.get(room_idx, ()))
        if degree > 1:
            continue
        width = op["u1"] - op["u0"]
        confidence = "high" if (width >= BALCONY_MIN_WIDTH_M or op.get("walked")) else "medium"
        predictions.setdefault(room_idx, []).append(
            dict(wall_idx=wi, width_m=width, room_area_m2=area, confidence=confidence,
                reason=f"small room ({area:.1f}m2), degree={degree}, "
                       f"{'wide/walked' if confidence=='high' else 'narrow, unwalked'} exterior opening"))
    return predictions


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
    walls = regularize.snap_walls(runs, np.eye(3), angle_tol_deg=cfg["snap_angle_tol_deg"])
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

    wall_ns = _wall_namespaces(walls)
    rooms = [p.simplify(cfg["room_simplify_m"])
             for p in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_primary_m"])
             if p.area >= cfg["room_min_area_m2"]]
    for cell in build_room_polygons(wall_ns, epsilon_m=cfg["room_eps_recovery_m"]):
        if cell.area >= cfg["room_min_area_m2"] and not any(r.contains(cell.centroid) for r in rooms):
            rooms.append(cell.simplify(cfg["room_simplify_m"]))
    log(f"{len(walls)} walls, {len(rooms)} rooms")

    # --- STEP 1: real exterior/interior wall classification ---
    wall_sides = classify_wall_sides(walls, rooms)
    n_ext = sum(1 for s in wall_sides if s["exterior"])
    log(f"wall sides: {n_ext}/{len(walls)} exterior, {len(walls)-n_ext} interior")
    for wi, s in enumerate(wall_sides):
        log(f"  wall_{wi}: exterior={s['exterior']} room_a={s['room_a']} room_b={s['room_b']}")
    exterior_flags = {i: s["exterior"] for i, s in enumerate(wall_sides)}

    # --- detect openings WITH real exterior_flags this time ---
    crossings = wall_crossings(traj, walls, end_margin_m=cfg["crossing_end_margin_m"])
    openings_by_wall = detect_openings(walls, xyz, traj, crossings, z_floor, z_ceiling,
                                       cfg["priors"], exterior_flags=exterior_flags)
    n_openings = sum(len(v) for v in openings_by_wall.values())
    log(f"{n_openings} openings detected (with real exterior_flags)")
    for wi in range(len(walls)):
        ops = openings_by_wall.get(wi, [])
        if not ops:
            log(f"  wall_{wi}: NO openings detected at all")
        else:
            for op in ops:
                log(f"  wall_{wi}: type={op['type']} u=[{op['u0']:.2f},{op['u1']:.2f}] "
                    f"sill={op['sill_m']:.2f} walked={op['walked']}")

    # --- STEP 3: entrance-first annotation ---
    openings_by_wall = annotate_entrances(openings_by_wall, cfg["priors"].get("max_opening_width_m", 4.0))

    # --- STEP 2: room-adjacency graph ---
    graph, exterior_entrances = build_adjacency_graph(walls, wall_sides, openings_by_wall, rooms)
    isolated_rooms = [i for i in range(len(rooms)) if not graph.get(i) and
                      not any(r == i for r, _, _ in exterior_entrances)]
    log(f"room graph: {sum(len(v) for v in graph.values())//2} interior connections, "
        f"{len(exterior_entrances)} exterior entrances, {len(isolated_rooms)} ISOLATED rooms "
        f"(no detected entrance at all -- likely a missed opening)")
    for i in isolated_rooms:
        log(f"  ISOLATED room_{i} ({rooms[i].area:.1f} m2) -- no entrance found on any of its walls")

    # --- STEP 4: balcony prediction from topology ---
    balcony_predictions = predict_balconies(graph, exterior_entrances, rooms, openings_by_wall, wall_sides)
    for room_idx, preds in balcony_predictions.items():
        for p in preds:
            log(f"  PREDICTED BALCONY: room_{room_idx} ({p['room_area_m2']:.1f} m2) via wall_{p['wall_idx']} "
                f"(opening {p['width_m']:.2f} m wide) -- {p['reason']}")

    # ---------- render ----------
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

    overlay = img.copy()
    for i, poly in enumerate(rooms):
        color = (30, 90, 160) if i in balcony_predictions else (
            (10, 60, 120) if i in isolated_rooms else (120, 60, 10))
        pts = np.array([to_px(c) for c in poly.exterior.coords], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    def wall_rect(w, u0=None, u1=None, extra=0.0):
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        d = d / L if L else np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]])
        ht = w.get("thickness_m", 0.1) / 2 + extra
        a = p0 + d * (0.0 if u0 is None else u0)
        b = p0 + d * (L if u1 is None else u1)
        return np.array([to_px(a + n * ht), to_px(b + n * ht),
                         to_px(b - n * ht), to_px(a - n * ht)], dtype=np.int32)

    for w, side in zip(walls, wall_sides):
        color = (235, 235, 235) if side["exterior"] else (170, 170, 100)
        cv2.fillPoly(img, [wall_rect(w)], color)

    type_color = {"door": (0, 150, 255), "balcony_door": (0, 220, 255),
                  "window": (255, 180, 0), "unknown_opening": (160, 130, 110)}
    for wi, ops in (openings_by_wall or {}).items():
        w = walls[wi]
        p0 = np.asarray(w["p0"], float)
        d = np.asarray(w["p1"], float) - p0
        L = float(np.linalg.norm(d))
        if L == 0:
            continue
        d = d / L
        u_i = 1 if w["direction"] == "x" else 0
        if abs(d[u_i]) < 1e-6:
            continue
        for op in ops:
            t0 = (op["u0"] - p0[u_i]) / d[u_i]
            t1 = (op["u1"] - p0[u_i]) / d[u_i]
            lo, hi = sorted((t0, t1))
            lo, hi = max(lo, 0.0), min(hi, L)
            if hi - lo < 0.02:
                continue
            color = type_color.get(op["type"], (200, 200, 200))
            if op["is_entrance"]:
                cv2.fillPoly(img, [wall_rect(w, u0=lo, u1=hi, extra=0.03)], color)
            else:
                cv2.fillPoly(img, [wall_rect(w, u0=lo, u1=hi, extra=0.01)], (60, 60, 200))  # flagged: NOT an entrance

    for w, side in zip(walls, wall_sides):
        cv2.polylines(img, [wall_rect(w)], True, (90, 90, 90), 1)
    for i, poly in enumerate(rooms):
        cen = poly.centroid
        tag = " [BALCONY?]" if i in balcony_predictions else (" [ISOLATED]" if i in isolated_rooms else "")
        cv2.putText(img, f"R{i}:{poly.area:.1f}m2{tag}", to_px((cen.x, cen.y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 160), 1, cv2.LINE_AA)

    banner = (f"walls={len(walls)} ({n_ext} ext/{len(walls)-n_ext} int) rooms={len(rooms)} "
              f"openings={n_openings} isolated_rooms={len(isolated_rooms)} "
              f"predicted_balconies={len(balcony_predictions)}")
    cv2.putText(img, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "floorplan_with_entrances.png"), img)
    log(f"wrote {out_dir / 'floorplan_with_entrances.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
