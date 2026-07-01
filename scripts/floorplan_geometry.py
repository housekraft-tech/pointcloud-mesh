"""Pure numpy+cv2 floor plan reconstruction primitives.

No open3d import anywhere in this file -- this is what makes it fully
unit-testable on a 32-bit Python install that cannot install open3d at all.
Any function here that needs open3d-loaded data takes plain numpy arrays;
the open3d-facing wrapper lives in mesh_common.py.
"""
import numpy as np
import cv2
from collections import defaultdict, deque


# ---------- plane / frame math ----------

def plane_normal(plane_model):
    a, b, c, _d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    return n / np.linalg.norm(n)


def signed_plane_distance(points, plane_model):
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ n + float(d)


def refine_plane_model(points, plane_model):
    """SVD least-squares plane refit on the given points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return list(plane_model)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    n0 = plane_normal(plane_model)
    if float(np.dot(normal, n0)) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return [float(normal[0]), float(normal[1]), float(normal[2]), d]


def wall_uv_basis(normal):
    """U along the wall (perpendicular to normal, in the horizontal plane),
    V = world-up projected onto the wall plane."""
    normal = normal / np.linalg.norm(normal)
    world_up = np.array([0.0, 0.0, 1.0])
    v = world_up - normal * np.dot(world_up, normal)
    if np.linalg.norm(v) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v
    v = v / np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    return u, v


def project_to_plane(points, plane_model):
    signed = signed_plane_distance(points, plane_model)
    n = plane_normal(plane_model)
    pts = np.asarray(points, dtype=np.float64)
    return pts - signed[:, None] * n


def points_to_wall_uv(points, plane_model, origin_xyz, u_axis, v_axis=None):
    """Project 3D points onto a wall plane, express as (u=along wall, v=height)."""
    if v_axis is None:
        v_axis = np.array([0.0, 0.0, 1.0])
    projected = project_to_plane(points, plane_model)
    rel = projected - np.asarray(origin_xyz, dtype=np.float64)
    u = rel @ np.asarray(u_axis, dtype=np.float64)
    v = rel @ np.asarray(v_axis, dtype=np.float64)
    return np.column_stack([u, v])


# ---------- Phase 0: bounding-box auto-crop ----------

def crop_to_percentile_bounds(xyz, low_pct=1.0, high_pct=99.0, margin_m=0.5):
    """Robust bounding box from per-axis percentiles + margin, dropping the
    sparse SLAM-drift/ghost-point tail that inflates a raw min/max bbox.

    Confirmed on the real koushikexport.las/mujammelexport.las scans: 99% of
    points sit in a ~11x12x3.3m room while raw bbox balloons to 30-85m due to
    a sparse stray tail; this recovers the tight room bounds.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("crop_to_percentile_bounds: empty point array")
    lo = np.percentile(xyz, low_pct, axis=0) - margin_m
    hi = np.percentile(xyz, high_pct, axis=0) + margin_m
    keep_mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    stats = {
        "input_points": int(xyz.shape[0]),
        "kept_points": int(keep_mask.sum()),
        "dropped_points": int((~keep_mask).sum()),
        "dropped_fraction": float((~keep_mask).sum() / xyz.shape[0]),
        "raw_bounds_min": xyz.min(axis=0).tolist(),
        "raw_bounds_max": xyz.max(axis=0).tolist(),
        "cropped_bounds_min": lo.tolist(),
        "cropped_bounds_max": hi.tolist(),
    }
    return lo, hi, keep_mask, stats


# ---------- Phase 1: density image ----------

def points_to_density_image(xy, cell_size_m, bounds_min, bounds_max):
    xy = np.asarray(xy, dtype=np.float64)
    width = int(np.ceil((bounds_max[0] - bounds_min[0]) / cell_size_m)) + 1
    height = int(np.ceil((bounds_max[1] - bounds_min[1]) / cell_size_m)) + 1
    ix = np.floor((xy[:, 0] - bounds_min[0]) / cell_size_m).astype(np.int64)
    iy = np.floor((xy[:, 1] - bounds_min[1]) / cell_size_m).astype(np.int64)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    image = np.zeros((height, width), dtype=np.uint16)
    np.add.at(image, (iy[valid], ix[valid]), 1)
    origin = np.array([bounds_min[0], bounds_min[1]], dtype=np.float64)
    return image, origin


def threshold_density_image(image, min_count=2, morph_kernel=3):
    binary = (image >= min_count).astype(np.uint8) * 255
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ---------- Phase 2: wall segment extraction ----------

def extract_wall_segments(binary_image, origin, cell_size_m, epsilon_cells=2.0, min_span_cells=3.0):
    """min_span_cells filters by bounding-box span, NOT area: a wall face
    seen from only one side (no opposing face within threshold, nothing to
    'fill' between them) produces a genuinely thin, near-zero-area contour
    that a naive area filter would incorrectly discard as noise -- confirmed
    during design validation this silently drops every single-sided wall."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        _x, _y, w, h = cv2.boundingRect(contour)
        if max(w, h) < min_span_cells:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_cells, closed=True)
        pts_px = approx.reshape(-1, 2).astype(np.float64)
        pts_world = origin + pts_px * cell_size_m
        n = len(pts_world)
        if n < 2:
            continue
        for i in range(n):
            p0 = pts_world[i]
            p1 = pts_world[(i + 1) % n]
            length = float(np.linalg.norm(p1 - p0))
            if length < cell_size_m:
                continue
            segments.append({"p0": p0, "p1": p1, "length": length})
    return segments


# ---------- wall pairing (mutual nearest-neighbor with thickness/angle/overlap constraints) ----------

def _segment_dir(seg):
    v = seg["p1"] - seg["p0"]
    return v / np.linalg.norm(v)


def _segment_normal(seg):
    d = _segment_dir(seg)
    return np.array([-d[1], d[0]])


def _overlap_fraction(seg_a, seg_b, direction):
    a0 = np.dot(seg_a["p0"], direction)
    a1 = np.dot(seg_a["p1"], direction)
    b0 = np.dot(seg_b["p0"], direction)
    b1 = np.dot(seg_b["p1"], direction)
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    overlap = max(0.0, hi - lo)
    # raw segments (Phase 2) use "length"; paired walls (Phase 3+) use
    # "length_m" -- support both so this helper works on either shape.
    len_a = seg_a["length"] if "length" in seg_a else seg_a["length_m"]
    len_b = seg_b["length"] if "length" in seg_b else seg_b["length_m"]
    shorter = min(len_a, len_b)
    return overlap / shorter if shorter > 1e-9 else 0.0


def _project_point_onto_segment(point, seg):
    d = _segment_dir(seg)
    t = np.dot(point - seg["p0"], d)
    return seg["p0"] + t * d


def _build_wall_from_pair(seg_a, seg_b, thickness_m):
    centerline_p0 = (seg_a["p0"] + _project_point_onto_segment(seg_a["p0"], seg_b)) / 2
    centerline_p1 = (seg_a["p1"] + _project_point_onto_segment(seg_a["p1"], seg_b)) / 2
    return {
        "p0": centerline_p0,
        "p1": centerline_p1,
        "thickness_m": thickness_m,
        "thickness_source": "measured",
    }


def _build_wall_from_single(seg):
    return {
        "p0": seg["p0"],
        "p1": seg["p1"],
        "thickness_m": None,
        "thickness_source": "assumed",
    }


def pair_wall_surfaces(segments, min_thickness_m=0.06, max_thickness_m=0.35,
                        max_angle_deg=5.0, min_overlap_frac=0.5):
    """Mutual-nearest-neighbor pairing: each segment's candidate partners are
    filtered by near-parallel direction, sufficient overlap, and a plausible
    wall-thickness gap (60-350mm default) -- this envelope is what rejects a
    T-junction pairing a segment across to an unrelated wall/room-width gap.
    A pair is only accepted if each segment's closest-gap candidate is the
    other (mutual best match), not just a one-sided nearest match."""
    n = len(segments)
    candidates = {i: [] for i in range(n)}
    for i in range(n):
        d_i = _segment_dir(segments[i])
        n_i = _segment_normal(segments[i])
        for j in range(i + 1, n):
            d_j = _segment_dir(segments[j])
            cos_angle = abs(float(np.dot(d_i, d_j)))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            if angle_deg > max_angle_deg:
                continue
            overlap = _overlap_fraction(segments[i], segments[j], d_i)
            if overlap < min_overlap_frac:
                continue
            mid_i = (segments[i]["p0"] + segments[i]["p1"]) / 2
            mid_j = (segments[j]["p0"] + segments[j]["p1"]) / 2
            gap = abs(float(np.dot(mid_j - mid_i, n_i)))
            if gap < min_thickness_m or gap > max_thickness_m:
                continue
            candidates[i].append((j, gap, overlap))
            candidates[j].append((i, gap, overlap))

    best = {}
    for i, opts in candidates.items():
        if opts:
            opts.sort(key=lambda t: t[1])
            best[i] = opts[0][0]

    walls = []
    used = set()
    for i, j in best.items():
        if i in used or j in used:
            continue
        if best.get(j) == i:
            gap = next(g for (jj, g, _o) in candidates[i] if jj == j)
            walls.append(_build_wall_from_pair(segments[i], segments[j], gap))
            used.add(i)
            used.add(j)

    for i in range(n):
        if i in used:
            continue
        walls.append(_build_wall_from_single(segments[i]))
        used.add(i)

    return walls


def apply_modal_thickness_fallback(walls, default_thickness_m=0.1):
    """Never silently substitute a hardcoded default: derive the fallback
    from the modal MEASURED thickness across the structure, so an assumption
    is at least grounded in this building's actual wall construction."""
    measured = [w["thickness_m"] for w in walls if w["thickness_source"] == "measured"]
    modal = float(np.median(measured)) if measured else default_thickness_m
    for w in walls:
        if w["thickness_source"] == "assumed":
            w["thickness_m"] = modal
    return walls


# ---------- endpoint snapping ----------

def snap_wall_endpoints(walls, tolerance_m=0.05):
    """Cluster nearby wall centerline endpoints (across all walls) within
    tolerance so corners meet cleanly; replaces each endpoint with its
    cluster centroid."""
    endpoints = []
    for wi, w in enumerate(walls):
        endpoints.append((wi, "p0", w["p0"]))
        endpoints.append((wi, "p1", w["p1"]))

    clusters = []
    for (wi, key, pt) in endpoints:
        found = None
        for ci, members in enumerate(clusters):
            # Never join a cluster that already contains an endpoint from
            # this same wall: doing so can collapse a short wall's own p0
            # and p1 onto a single point (length_m -> 0.0), directly or
            # transitively via a third-party bridging endpoint.
            if any(m[0] == wi for m in members):
                continue
            rep_pt = members[0][2]
            if np.linalg.norm(pt - rep_pt) <= tolerance_m:
                found = ci
                break
        if found is None:
            clusters.append([(wi, key, pt)])
        else:
            clusters[found].append((wi, key, pt))

    for members in clusters:
        centroid = np.mean([m[2] for m in members], axis=0)
        for (wi, key, _pt) in members:
            walls[wi][key] = centroid

    for w in walls:
        w["length_m"] = float(np.linalg.norm(w["p1"] - w["p0"]))

    return walls, clusters


def drop_short_walls(walls, min_length_m=0.3):
    """Drop T-junction/corner pixel-noise stubs: confirmed via design
    validation that mutual-NN pairing can match two short (<150mm) segments
    near a T-junction into a plausible-looking but spurious 'wall'. If a real
    building has a legitimate short partition stub, lower this threshold and
    add a corresponding case to validate_measurements.py's ground truth."""
    return [w for w in walls if w["length_m"] >= min_length_m]


# ---------- wall dedup/merge ----------

def _walls_are_duplicates(a, b, angle_tol_deg=3.0, offset_tol_m=0.35, u_gap_tol_m=0.1):
    d_a = _segment_dir(a)
    d_b = _segment_dir(b)
    cos_angle = abs(float(np.dot(d_a, d_b)))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle_deg > angle_tol_deg:
        return False
    n_a = _segment_normal(a)
    mid_a = (a["p0"] + a["p1"]) / 2
    mid_b = (b["p0"] + b["p1"]) / 2
    perp_offset = abs(float(np.dot(mid_b - mid_a, n_a)))
    if perp_offset > offset_tol_m:
        return False
    # require overlapping or near-adjacent u-ranges along the shared direction
    overlap = _overlap_fraction(a, b, d_a)
    if overlap > 0.0:
        return True
    a0, a1 = np.dot(a["p0"], d_a), np.dot(a["p1"], d_a)
    b0, b1 = np.dot(b["p0"], d_a), np.dot(b["p1"], d_a)
    gap = max(min(b0, b1) - max(a0, a1), min(a0, a1) - max(b0, b1))
    return gap <= u_gap_tol_m


def merge_duplicate_walls(walls, angle_tol_deg=3.0, offset_tol_m=0.35, u_gap_tol_m=0.1):
    """Collapse near-collinear, overlapping-or-adjacent wall entries that
    describe the same physical wall run. Confirmed necessary during design
    validation: a connected wall network traced through multiple findContours
    loops (exterior boundary + one void per room) produces several duplicate
    entries per physical wall, since mutual-NN pairing is one-to-one and
    doesn't itself deduplicate across contours.

    offset_tol_m defaults to pair_wall_surfaces's max_thickness_m (0.35m):
    an "assumed" single-sided wall's centerline sits at its detected face
    (uncentered), so vs. a "measured" duplicate of the same wall it can be
    off by up to a full wall thickness."""
    n = len(walls)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _walls_are_duplicates(walls[i], walls[j], angle_tol_deg, offset_tol_m, u_gap_tol_m):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(walls[i])

    merged = []
    for group in groups.values():
        measured = [w for w in group if w["thickness_source"] == "measured"]
        chosen_pool = measured if measured else group
        best = max(chosen_pool, key=lambda w: w["length_m"])
        merged.append(best)
    return merged
