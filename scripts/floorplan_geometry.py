"""Pure numpy+cv2 floor plan reconstruction primitives.

No open3d import anywhere in this file -- this is what makes it fully
unit-testable on a 32-bit Python install that cannot install open3d at all.
Any function here that needs open3d-loaded data takes plain numpy arrays;
the open3d-facing wrapper lives in mesh_common.py.
"""
import numpy as np
import cv2
import warnings
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


def find_dense_z_band(z_values, bin_m=0.1, min_bin_points=5):
    """Find the primary dense Z-band (e.g. one story's wall-height range) by
    histogramming point density along Z, marking bins with at least
    min_bin_points as 'occupied' (a small fixed absolute floor, not relative
    to the tallest bin), grouping occupied bins into contiguous runs, and
    returning the run with the most TOTAL points.

    This replaces an earlier peak-relative-expansion version that failed on
    real scan data: a flat floor or ceiling slab is scanned near-perpendicular
    over a huge area and collects a massive density spike in one or two bins
    -- confirmed on a real house scan, one 100mm bin held ~1.7M points versus
    ~70k-290k for ordinary wall/room-volume bins, 6-13x denser. Thresholding
    relative to that spike (density_ratio_threshold * peak) set the bar far
    too high, so legitimate but much-less-dense room-volume bins between
    floor and ceiling failed to qualify and the detected band collapsed to
    just the spike itself (confirmed: 0 walls on real data, both automatically
    and even when manually pre-filtered to a known-good range).

    A fixed low absolute threshold sidesteps this: floor/ceiling spikes and
    ordinary room-volume bins both clear it easily and merge into one
    contiguous run (since there's no genuine empty gap between them within a
    real story), while scattered noise/stray points mostly fail to clear it
    and any noise run that does is dwarfed in total point count by real
    structure. A genuine secondary structure (another floor, stairwell) still
    gets correctly excluded when it's separated from the primary story by an
    actual near-empty Z-gap (real physical separation, not just lower
    density) -- confirmed against both a floor/ceiling-spike scenario and a
    genuinely sparser secondary-structure scenario.

    The histogram is only used to LOCATE the band (coarse, bin_m resolution).
    The returned (z_min, z_max) are refined against the actual point extrema
    within that band, never the raw bin edges -- returning bin edges directly
    would quantize floor/ceiling height to bin_m (e.g. 100mm), a real precision
    regression on a value that flows straight into manufacturing-relevant wall
    height with no downstream refit to correct it.
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    if z_values.size == 0:
        raise ValueError("find_dense_z_band: empty z_values array")
    n_bins = max(int(np.ceil((z_values.max() - z_values.min()) / bin_m)), 4)
    counts, edges = np.histogram(z_values, bins=n_bins)

    occupied = counts >= min_bin_points
    runs = []
    start = None
    for i, occ in enumerate(occupied):
        if occ and start is None:
            start = i
        elif not occ and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(occupied) - 1))
    if not runs:
        # No bin cleared min_bin_points at all -- e.g. a globally very sparse
        # point cloud. Falling back to raw min/max silently reintroduces the
        # exact unbounded-stray-outlier problem this function exists to solve
        # (confirmed as a real, reachable path during design review), so this
        # is surfaced loudly rather than silently trusted.
        warnings.warn(
            "find_dense_z_band: no bin met min_bin_points threshold; falling back to "
            "raw point extrema, which may include unbounded stray outliers. Consider "
            "z_band_override for this scan.",
            RuntimeWarning, stacklevel=2,
        )
        return float(z_values.min()), float(z_values.max())

    run_totals = [(int(counts[a:b + 1].sum()), a, b) for a, b in runs]
    run_totals.sort(key=lambda t: -t[0])
    _best_pts, lo, hi = run_totals[0]

    band_lo, band_hi = edges[lo], edges[hi + 1]
    in_band = z_values[(z_values >= band_lo) & (z_values <= band_hi)]
    if in_band.size == 0:
        return float(band_lo), float(band_hi)
    return float(in_band.min()), float(in_band.max())


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


# ---------- corner-aware point selection + two-pass wall plane refit ----------

def refine_wall_plane_two_pass(points, plane_model, coarse_band_m=0.025, fine_band_m=0.008, trim_frac=0.02):
    """Two-pass least-squares plane refit on ORIGINAL (non-downsampled)
    points: pass 1 uses a wide (25mm) band to absorb coarse-detection
    positional slop, pass 2 re-selects within a tight (8mm, ~0.5x the 16mm
    median scan spacing) band around the pass-1 result and refits again,
    trimming the highest-residual 2% of points first (corner contamination)."""
    pts = np.asarray(points, dtype=np.float64)
    dist = np.abs(signed_plane_distance(pts, plane_model))
    coarse_pts = pts[dist <= coarse_band_m]
    if len(coarse_pts) < 3:
        return list(plane_model)
    plane1 = refine_plane_model(coarse_pts, plane_model)

    dist2 = np.abs(signed_plane_distance(pts, plane1))
    fine_pts = pts[dist2 <= fine_band_m]
    if len(fine_pts) < 3:
        return plane1

    resid = np.abs(signed_plane_distance(fine_pts, plane1))
    if trim_frac > 0 and len(fine_pts) > 10:
        keep_n = int(len(fine_pts) * (1 - trim_frac))
        keep_idx = np.argsort(resid)[:keep_n]
        fine_pts = fine_pts[keep_idx]

    return refine_plane_model(fine_pts, plane1)


def select_wall_band_points(points, wall, corner_margin_m=0.5, band_m=0.06):
    """Points belonging to ONE wall's own run: within band_m perpendicular
    distance of the wall's centerline direction AND within the wall's own
    U-range minus corner_margin_m on each end.

    Confirmed via design validation that perpendicular-distance filtering
    ALONE is insufficient near a T-junction: a perpendicular wall's face can
    be coincidentally near-coplanar with this wall right at the junction
    (observed residual mean ~23mm, max ~75mm without this). Excluding a
    margin near the wall's own detected corners removes the contamination
    (observed residual mean ~1.6mm, refined thickness error dropped from
    ~30mm to ~0.3mm on a 100mm true partition)."""
    pts = np.asarray(points, dtype=np.float64)
    d = _segment_dir(wall)
    normal = _segment_normal(wall)
    dist_perp = np.abs(np.dot(pts[:, :2] - wall["p0"], normal))
    u = np.dot(pts[:, :2] - wall["p0"], d)
    length = wall["length_m"]
    in_band = dist_perp <= band_m
    in_u_range = (u >= corner_margin_m) & (u <= length - corner_margin_m)
    return pts[in_band & in_u_range]


# ---------- opening detection (ported void-flood-fill + rectangle merge) ----------

def merge_grid_cells(occupied):
    """Merge occupied (iu, iv) cells into axis-aligned rectangles (U-runs then V-runs)."""
    if not occupied:
        return []
    bars = []
    for iv in sorted({iv for _iu, iv in occupied}):
        row = sorted(iu for iu, jv in occupied if jv == iv)
        start = prev = row[0]
        for iu in row[1:]:
            if iu == prev + 1:
                prev = iu
            else:
                bars.append((start, prev, iv))
                start = prev = iu
        bars.append((start, prev, iv))

    by_span = defaultdict(list)
    for iu0, iu1, iv in bars:
        by_span[(iu0, iu1)].append(iv)

    rects = []
    for (iu0, iu1), ivs in by_span.items():
        ivs = sorted(set(ivs))
        start = prev = ivs[0]
        for iv in ivs[1:]:
            if iv == prev + 1:
                prev = iv
            else:
                rects.append((iu0, iu1, start, prev))
                start = prev = iv
        rects.append((iu0, iu1, start, prev))
    return rects


def _interior_void_cells(occupied, iu0, iu1, iv0, iv1, floor_is_boundary=True):
    """Empty cells not reachable from the grid border (door/window holes).

    The bottom row (iv0, floor level) is NOT seeded as an open border when
    floor_is_boundary=True: a floor-to-ceiling door's void touches iv0 by
    construction (there's no wall below floor level to begin with), but the
    floor itself is a real physical boundary, not open space -- confirmed
    this bug makes every full-height door silently undetectable without
    the fix (flood-fill marks the whole void as 'exterior')."""
    exterior = set()
    queue = deque()

    def seed(iu, iv):
        if (iu, iv) in occupied or (iu, iv) in exterior:
            return
        exterior.add((iu, iv))
        queue.append((iu, iv))

    for iu in range(iu0, iu1 + 1):
        if not floor_is_boundary:
            seed(iu, iv0)
        seed(iu, iv1)
    for iv in range(iv0, iv1 + 1):
        seed(iu0, iv)
        seed(iu1, iv)

    while queue:
        iu, iv = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = iu + di, iv + dj
            if not (iu0 <= ni <= iu1 and iv0 <= nj <= iv1):
                continue
            if (ni, nj) in occupied or (ni, nj) in exterior:
                continue
            exterior.add((ni, nj))
            queue.append((ni, nj))

    voids = set()
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            if (iu, iv) not in occupied and (iu, iv) not in exterior:
                voids.add((iu, iv))
    return voids


def classify_opening(width_m, height_m, sill_m):
    if width_m >= 1.0 and height_m >= 2.0 and sill_m < 0.25:
        return "balcony_door"
    if 0.65 <= width_m <= 1.4 and 1.75 <= height_m <= 2.5 and sill_m < 0.25:
        return "door"
    if sill_m >= 0.4 and width_m >= 0.45 and height_m >= 0.45:
        return "window"
    return "unknown_opening"


def detect_openings_on_wall_face(uv_points, wall_length_m, cell_m=0.05,
                                  min_points_per_cell=3, min_opening_w=0.45,
                                  min_opening_h=0.45, edge_margin_cells=1):
    """uv_points: (N,2) array of (u, v=height) points on ONE wall face."""
    uv = np.asarray(uv_points, dtype=np.float64)
    if len(uv) == 0:
        return []
    # + 1e-9 guards against float division landing a hair under an exact
    # cell-boundary multiple of cell_m (e.g. 2.15 / 0.05 == 42.99999999999999),
    # which would silently floor into the wrong cell and leave a spurious
    # empty row/column that bridges an interior void to the grid border.
    iu = np.floor(uv[:, 0] / cell_m + 1e-9).astype(np.int64)
    iv = np.floor(uv[:, 1] / cell_m + 1e-9).astype(np.int64)
    counts = defaultdict(int)
    for k in range(len(uv)):
        counts[(int(iu[k]), int(iv[k]))] += 1
    occupied = {key for key, c in counts.items() if c >= min_points_per_cell}
    if not occupied:
        return []

    iu0, iu1 = min(c[0] for c in occupied), max(c[0] for c in occupied)
    iv0, iv1 = min(c[1] for c in occupied), max(c[1] for c in occupied)
    max_iu = int(np.floor(wall_length_m / cell_m))

    voids = _interior_void_cells(occupied, iu0, iu1, iv0, iv1, floor_is_boundary=True)
    if not voids:
        return []
    rects = merge_grid_cells(voids)

    openings = []
    for a0, a1, b0, b1 in rects:
        u_min, u_max = a0 * cell_m, (a1 + 1) * cell_m
        v_min, v_max = b0 * cell_m, (b1 + 1) * cell_m
        width, height = u_max - u_min, v_max - v_min
        if width < min_opening_w or height < min_opening_h:
            continue
        if a0 <= edge_margin_cells or a1 >= max_iu - edge_margin_cells:
            continue  # touches the wall's snapped end -- termination, not an opening
        sill = v_min
        openings.append({
            "u_min": u_min, "u_max": u_max, "v_min": v_min, "v_max": v_max,
            "width_m": width, "height_m": height, "sill_m": sill,
            "type": classify_opening(width, height, sill),
        })
    return openings


def cross_check_opening_both_faces(opening, other_face_uv_points, cell_m=0.05, min_points_per_cell=3):
    """A true through-wall opening must ALSO be void on the wall's other
    face; a gap present on only one face is furniture occlusion, not a hole
    through the wall. Returns True if the opening survives (other face is
    also mostly empty in that u,v range)."""
    uv = np.asarray(other_face_uv_points, dtype=np.float64)
    if len(uv) == 0:
        return True
    in_range = (
        (uv[:, 0] >= opening["u_min"]) & (uv[:, 0] <= opening["u_max"]) &
        (uv[:, 1] >= opening["v_min"]) & (uv[:, 1] <= opening["v_max"])
    )
    pts_in_rect = uv[in_range]
    if len(pts_in_rect) == 0:
        return True
    iu = np.floor((pts_in_rect[:, 0] - opening["u_min"]) / cell_m + 1e-9).astype(np.int64)
    iv = np.floor((pts_in_rect[:, 1] - opening["v_min"]) / cell_m + 1e-9).astype(np.int64)
    counts = defaultdict(int)
    for k in range(len(pts_in_rect)):
        counts[(int(iu[k]), int(iv[k]))] += 1
    occupied_cells = sum(1 for c in counts.values() if c >= min_points_per_cell)
    total_cells = max(1, int((opening["u_max"] - opening["u_min"]) / cell_m) *
                      int((opening["v_max"] - opening["v_min"]) / cell_m))
    return (occupied_cells / total_cells) < 0.3


# ---------- opening edge refinement (density half-max crossing on original points) ----------

def _half_max_crossing_1d(coords_1d, bin_m, solid_is_low, smooth_k=5):
    """Locate the sub-bin position where 1D point density drops to half the
    'solid wall' plateau density, via a smoothed histogram + linear
    interpolation between the straddling bins.

    smooth_k applies a small moving-average box filter to the raw bin counts
    before thresholding: on real scan data (irregular point placement) this is
    unnecessary, but on any near-regular point grid (e.g. this repo's
    synthetic test fixtures, which sample at a fixed 16mm spacing with only a
    couple mm of jitter) raw per-bin counts alias into a strong on/off
    striping pattern at fine bin widths, which a raw half-max crossing on
    unsmoothed counts detects as a false edge tens of mm from the truth --
    confirmed via direct testing (150-190mm error unsmoothed vs 1-8mm smoothed
    on synthetic door/window fixtures with known sub-cell true edges).

    Returns None if there isn't enough data to determine a reliable crossing.
    """
    coords_1d = np.asarray(coords_1d, dtype=np.float64)
    if len(coords_1d) < 20:
        return None
    lo, hi = coords_1d.min(), coords_1d.max()
    n_bins = max(int(np.ceil((hi - lo) / bin_m)), 4)
    counts, edges = np.histogram(coords_1d, bins=n_bins, range=(lo, hi))
    k = min(smooth_k, n_bins)
    counts = np.convolve(counts.astype(np.float64), np.ones(k) / k, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0

    if not solid_is_low:
        counts = counts[::-1]
        centers = centers[::-1]
    # counts now starts on the solid-wall side and drops toward the void side

    plateau_n = max(1, n_bins // 4)
    plateau = float(np.median(counts[:plateau_n]))
    if plateau <= 0:
        return None
    half = plateau / 2.0

    below = None
    for i in range(len(counts)):
        if counts[i] < half:
            below = i
            break
    if below is None or below == 0:
        return None

    c0, c1 = counts[below - 1], counts[below]
    x0, x1 = centers[below - 1], centers[below]
    if c1 == c0:
        return float(x0)
    t = (half - c0) / (c1 - c0)
    return float(x0 + t * (x1 - x0))


def refine_opening_edges(opening, uv_points, search_band_m=0.15, bin_m=0.004):
    """Refine an opening's u_min/u_max/v_max edges (and v_min/sill only when
    it isn't floor-level) from the coarse 50mm grid-detected rectangle to
    sub-cell accuracy, using a density half-max crossing on the ORIGINAL
    (non-downsampled) points on this wall face -- never the coarse grid
    detection itself, matching this pipeline's accuracy strategy for every
    other final numeric output.

    A floor-level v_min (sill <= 0.05m, i.e. a floor-to-ceiling door) is left
    at its coarse value: there's no wall material below a real floor, so
    there's no density transition to refine against -- the floor itself
    already defines that edge exactly.

    Falls back to the coarse value for any edge where refinement can't find a
    reliable crossing (too few points near that edge)."""
    uv = np.asarray(uv_points, dtype=np.float64)
    refined = dict(opening)
    any_refined = False

    window = uv[(uv[:, 0] >= opening["u_min"] - search_band_m) & (uv[:, 0] <= opening["u_min"] + search_band_m)
                & (uv[:, 1] >= opening["v_min"]) & (uv[:, 1] <= opening["v_max"])]
    crossing = _half_max_crossing_1d(window[:, 0], bin_m, solid_is_low=True)
    if crossing is not None:
        refined["u_min"] = crossing
        any_refined = True

    window = uv[(uv[:, 0] >= opening["u_max"] - search_band_m) & (uv[:, 0] <= opening["u_max"] + search_band_m)
                & (uv[:, 1] >= opening["v_min"]) & (uv[:, 1] <= opening["v_max"])]
    crossing = _half_max_crossing_1d(window[:, 0], bin_m, solid_is_low=False)
    if crossing is not None:
        refined["u_max"] = crossing
        any_refined = True

    window = uv[(uv[:, 1] >= opening["v_max"] - search_band_m) & (uv[:, 1] <= opening["v_max"] + search_band_m)
                & (uv[:, 0] >= opening["u_min"]) & (uv[:, 0] <= opening["u_max"])]
    crossing = _half_max_crossing_1d(window[:, 1], bin_m, solid_is_low=False)
    if crossing is not None:
        refined["v_max"] = crossing
        any_refined = True

    if opening["v_min"] > 0.05:
        window = uv[(uv[:, 1] >= opening["v_min"] - search_band_m) & (uv[:, 1] <= opening["v_min"] + search_band_m)
                    & (uv[:, 0] >= opening["u_min"]) & (uv[:, 0] <= opening["u_max"])]
        crossing = _half_max_crossing_1d(window[:, 1], bin_m, solid_is_low=True)
        if crossing is not None:
            refined["v_min"] = crossing
            any_refined = True

    refined["width_m"] = refined["u_max"] - refined["u_min"]
    refined["height_m"] = refined["v_max"] - refined["v_min"]
    refined["sill_m"] = refined["v_min"]
    refined["edge_method"] = "density_half_max" if any_refined else "grid_coarse"
    return refined


# ---------- floor plan image rendering ----------

def render_floorplan_image(walls, openings_by_wall_id, output_path, px_per_meter=100):
    """Top-down floor plan render: walls as thick lines with length/thickness
    labels, openings marked with their type. Writes a blank placeholder image
    if there are no walls, rather than crashing -- the zero-walls case is a
    real outcome (e.g. a misconfigured crop/ceiling-band on a real scan), not
    a case that should produce a numpy stack-trace instead of a usable file."""
    if not walls:
        blank = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "No walls detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.imwrite(output_path, blank)
        return

    all_pts = np.vstack([np.vstack([w["p0"], w["p1"]]) for w in walls])
    margin_m = 0.5
    min_xy = all_pts.min(axis=0) - margin_m
    max_xy = all_pts.max(axis=0) + margin_m
    width_px = int((max_xy[0] - min_xy[0]) * px_per_meter) + 1
    height_px = int((max_xy[1] - min_xy[1]) * px_per_meter) + 1
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    def to_px(pt):
        x = int((pt[0] - min_xy[0]) * px_per_meter)
        y = int((max_xy[1] - pt[1]) * px_per_meter)  # flip Y for image coords
        return (x, y)

    for wi, w in enumerate(walls):
        p0_px, p1_px = to_px(w["p0"]), to_px(w["p1"])
        thickness_px = max(1, int(w["thickness_m"] * px_per_meter))
        cv2.line(img, p0_px, p1_px, (40, 40, 40), thickness=thickness_px)
        mid_px = ((p0_px[0] + p1_px[0]) // 2, (p0_px[1] + p1_px[1]) // 2)
        label = f"{w['length_m']:.2f}m/{w['thickness_m']*1000:.0f}mm"
        cv2.putText(img, label, mid_px, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1, cv2.LINE_AA)
        for op in openings_by_wall_id.get(wi, []):
            d = (w["p1"] - w["p0"]) / w["length_m"]
            op_p0 = w["p0"] + d * op["u_min_m"]
            op_p1 = w["p0"] + d * op["u_max_m"]
            cv2.line(img, to_px(op_p0), to_px(op_p1), (0, 150, 0), thickness=max(2, thickness_px))
            cv2.putText(img, op["type"], to_px(op_p0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
