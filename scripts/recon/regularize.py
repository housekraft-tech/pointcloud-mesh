"""Regularize: axis snap, watertight corner intersection, wall thickness.

Operates on the list[dict] wall-run shape produced by
structure.group_wall_runs (see that module's docstring for the exact
contract): each dict has keys direction ("x"|"y"), normal (unit 3-tuple),
p0/p1 (centerline endpoints, world XY), offset_m (absolute perpendicular
offset of the main step), steps (list[WallStep], main step at relative
offset_m == 0.0). All three functions here consume and return that same
list[dict] shape -- they add/update keys (p0/p1 after snap_walls/
resolve_corners; thickness_m/thickness_source after pair_thickness) rather
than converting to a different dataclass. Task 17 (final manifest wiring)
is where these get converted to floorplan_schema.Wall.

None of these functions mutate the input list or its dicts in place --
each returns a new list of shallow-copied dicts, so callers holding a
reference to the original `walls` list see it unchanged.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# snap_walls
# ---------------------------------------------------------------------------

def _axis_candidates(axes):
    """Unit 2-D (XY) direction vectors for axes[:, 0] and axes[:, 1] --
    the two dominant Manhattan directions, per structure.group_wall_runs's
    convention. Degenerate (near-zero) columns are dropped.
    """
    axes = np.asarray(axes, dtype=float)
    out = []
    for col in (axes[:, 0], axes[:, 1]):
        v = np.asarray(col[:2], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-9:
            out.append(v / n)
    return out


def snap_walls(walls, axes, angle_tol_deg: float = 8.0):
    """Snap each wall's own centerline direction (p1 - p0) to whichever of
    the two dominant Manhattan axes (axes[:, 0], axes[:, 1]) it is closest
    to, but ONLY if that closest axis is within angle_tol_deg -- a wall
    that is not confidently Manhattan-aligned (e.g. 30-45 degrees off, a
    real diagonal feature) keeps its own PCA/plane-fit direction unchanged
    rather than being forced onto an axis it doesn't actually belong to.

    Snapping rotates the wall's p0/p1 about the wall's own midpoint so the
    new centerline is exactly parallel to the snapped axis while preserving
    the wall's length and (to first order) its position. "normal" (if
    present) is rotated by the same amount, preserving its original sign
    relative to the wall's direction, so it stays perpendicular to the
    (possibly snapped) direction. "offset_m" (if present) is recomputed
    from the (unchanged) midpoint against the new normal, for consistency.
    """
    candidates = _axis_candidates(axes)

    out = []
    for w in walls:
        w = dict(w)
        p0 = np.asarray(w["p0"], dtype=float)
        p1 = np.asarray(w["p1"], dtype=float)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9 or not candidates:
            out.append(w)
            continue
        d = seg / length

        best_axis, best_angle = None, None
        for axis in candidates:
            cos_a = np.clip(abs(float(np.dot(d, axis))), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_a)))
            if best_angle is None or angle < best_angle:
                best_axis, best_angle = axis, angle

        if best_axis is None or best_angle > angle_tol_deg:
            out.append(w)  # not confidently aligned -- leave as-is
            continue

        # Sign-correct so snapping can't flip the wall's own orientation
        # (which endpoint is p0 vs p1).
        sign = 1.0 if np.dot(d, best_axis) >= 0 else -1.0
        new_d = best_axis * sign

        mid = (p0 + p1) / 2.0
        new_p0 = mid - new_d * (length / 2.0)
        new_p1 = mid + new_d * (length / 2.0)
        w["p0"] = (float(new_p0[0]), float(new_p0[1]))
        w["p1"] = (float(new_p1[0]), float(new_p1[1]))

        new_normal_xy = None
        if w.get("normal") is not None:
            old_normal = np.asarray(w["normal"], dtype=float)
            perp = np.array([-new_d[1], new_d[0]])
            perp_sign = 1.0 if np.dot(old_normal[:2], perp) >= 0 else -1.0
            new_normal_xy = perp * perp_sign
            z = float(old_normal[2]) if old_normal.size > 2 else 0.0
            w["normal"] = (float(new_normal_xy[0]), float(new_normal_xy[1]), z)

        if "offset_m" in w and new_normal_xy is not None:
            w["offset_m"] = float(np.dot(mid, new_normal_xy))

        out.append(w)
    return out


# ---------------------------------------------------------------------------
# resolve_corners
# ---------------------------------------------------------------------------

# Two member walls at a candidate corner cluster are treated as a genuine
# (non-degenerate) intersection only if at least two of their directions
# differ by more than this many degrees; near-parallel members (e.g. two
# collinear pieces of the same wall run that should just concatenate) fall
# back to a plain centroid instead of an unstable/undefined line crossing.
_PARALLEL_ANGLE_DEG = 7.0

# If a resolved line-intersection point lands implausibly far from the raw
# cluster's own endpoints (can happen when directions are close to but not
# quite parallel, so the intersection extrapolates a long way out), that
# answer is numerically unstable -- fall back to centroid instead.
_MAX_RESOLVE_DIST_M = 2.0


def _wall_dir(w):
    p0 = np.asarray(w["p0"], dtype=float)
    p1 = np.asarray(w["p1"], dtype=float)
    v = p1 - p0
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _pairwise_max_angle_deg(dirs):
    """Largest angle (0-90 deg) between any two of the given unit direction
    vectors; a direction and its negation are treated as identical (a
    wall's centerline has no inherent sign).
    """
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos_a = np.clip(abs(float(np.dot(dirs[i], dirs[j]))), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_a)))
            best = max(best, angle)
    return best


def _line_intersection(pts, dirs):
    """Least-squares point minimizing total squared perpendicular distance
    to every (point, direction) infinite line: for a line through `pt`
    with unit direction `d`, the normal equation is
    (I - d @ d.T) @ (x - pt) = 0. Summing across all member lines gives a
    2x2 linear system `A @ x = b`. Returns None if A is (near-)singular
    (all member directions parallel -- no well-defined single intersection).
    """
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for pt, d in zip(pts, dirs):
        d = np.asarray(d, dtype=float)
        proj = np.outer(d, d)
        m = np.eye(2) - proj
        A += m
        b += m @ np.asarray(pt, dtype=float)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, b)


def resolve_corners(walls, tol_m: float = 0.25):
    """Snap nearby wall endpoints from DIFFERENT walls to a single shared
    corner point via least-squares line intersection (see
    _line_intersection), so corners are geometrically watertight --
    extended/trimmed to where each wall's own infinite centerline actually
    crosses its neighbor's, not just the average of two noisy raw
    endpoints.

    Clustering scaffold (endpoints within tol_m of a cluster's first
    member join it) and the CRITICAL self-wall guard mirror
    floorplan_geometry.snap_wall_endpoints: an endpoint is never added to a
    cluster that already contains an endpoint from the SAME wall (whether
    that other endpoint got there directly or via a bridging third wall)
    -- doing so would collapse that wall's own p0/p1 onto (near) the same
    point, destroying its length. This is the older pipeline's confirmed
    fix for a real bug; adopted verbatim here since the risk (transitive
    self-collapse) is identical regardless of how much more reliable the
    underlying wall geometry is.

    Per cluster of 2+ members from different walls:
      - if at least two member directions differ by more than
        _PARALLEL_ANGLE_DEG, resolve via _line_intersection;
      - if that intersection is None (singular) or lands more than
        _MAX_RESOLVE_DIST_M from the raw endpoints' centroid, or if all
        member directions are near-parallel to begin with (no real corner
        to intersect -- e.g. two collinear segments of one wall run), fall
        back to the plain centroid of the raw endpoints.

    Directions are computed once, up front, from each wall's ORIGINAL
    (pre-resolution) p0/p1 -- so resolution isn't order-dependent on which
    cluster happens to get processed first.
    """
    directions = [_wall_dir(w) for w in walls]

    endpoints = []
    for wi, w in enumerate(walls):
        endpoints.append((wi, "p0", np.asarray(w["p0"], dtype=float)))
        endpoints.append((wi, "p1", np.asarray(w["p1"], dtype=float)))

    clusters = []
    for (wi, key, pt) in endpoints:
        found = None
        for ci, members in enumerate(clusters):
            # Never join a cluster that already holds an endpoint from this
            # same wall -- see docstring's CRITICAL self-wall guard note.
            if any(m[0] == wi for m in members):
                continue
            rep_pt = members[0][2]
            if np.linalg.norm(pt - rep_pt) <= tol_m:
                found = ci
                break
        if found is None:
            clusters.append([(wi, key, pt)])
        else:
            clusters[found].append((wi, key, pt))

    out = [dict(w) for w in walls]
    for members in clusters:
        pts = [m[2] for m in members]
        dirs = [directions[m[0]] for m in members]

        target = None
        if len(members) >= 2 and _pairwise_max_angle_deg(dirs) > _PARALLEL_ANGLE_DEG:
            target = _line_intersection(pts, dirs)
            if target is not None:
                centroid = np.mean(pts, axis=0)
                if np.linalg.norm(target - centroid) > _MAX_RESOLVE_DIST_M:
                    target = None  # numerically unstable extrapolation
        if target is None:
            target = np.mean(pts, axis=0)

        for (wi, key, _pt) in members:
            out[wi][key] = (float(target[0]), float(target[1]))

    for w in out:
        p0 = np.asarray(w["p0"], dtype=float)
        p1 = np.asarray(w["p1"], dtype=float)
        w["length_m"] = float(np.linalg.norm(p1 - p0))

    return out


# ---------------------------------------------------------------------------
# pair_thickness
# ---------------------------------------------------------------------------

def _thickness_from_steps(run, min_thickness_m, max_thickness_m, min_overlap_frac):
    """Primary path: structure.group_wall_runs already chains a wall's
    inner/outer faces into the SAME run as separate WallSteps whenever
    their offset gap exceeds merge_offset_m but stays under max_relief_m
    (the common case for real wall thickness, see group_wall_runs's own
    docstring) -- so the back face is often already sitting right there in
    run["steps"], with no need to re-scan the raw point cloud.

    The main step (relative offset_m == 0.0) is the reference face. Among
    the OTHER steps, a candidate is accepted as the back face only if its
    along-wall (u) extent overlaps the main step's u-extent by at least
    min_overlap_frac OF THE MAIN STEP'S OWN LENGTH (not the shorter of the
    two) -- this is what tells a genuine back face (which runs the full
    length of the wall, like the main face) apart from a short relief step
    such as a pillar/pilaster front (which is fully CONTAINED within the
    main step's u-range and would score 100% overlap under a
    shorter-segment-normalized metric, but covers only a small fraction of
    the main step's own length).
    """
    steps = run.get("steps") or []
    if len(steps) < 2:
        return None
    main = min(steps, key=lambda st: abs(st.offset_m))
    main_len = main.u_max_m - main.u_min_m
    if main_len <= 0:
        return None

    best = None
    for st in steps:
        if st is main:
            continue
        thickness = abs(st.offset_m)
        if not (min_thickness_m <= thickness <= max_thickness_m):
            continue
        lo = max(main.u_min_m, st.u_min_m)
        hi = min(main.u_max_m, st.u_max_m)
        overlap = max(0.0, hi - lo)
        overlap_frac = overlap / main_len
        if overlap_frac < min_overlap_frac:
            continue
        if best is None or overlap_frac > best[0]:
            best = (overlap_frac, thickness)
    return best[1] if best else None


def _measure_thickness_from_points(run, points, min_thickness_m, max_thickness_m,
                                    min_overlap_frac, min_points=30, margin_m=0.1,
                                    gap_m=0.05):
    """Fallback path: directly search the raw point cloud near this wall's
    own footprint for a second, parallel, opposite-facing surface -- used
    when the run's own steps don't already carry a plausible back-face
    candidate (e.g. a caller-built wall dict with no/incomplete steps).

    Projects points near the wall's (u, z) footprint onto the wall's own
    normal (perpendicular coordinate relative to p0, a point on the main
    face's line), excludes points essentially on the main face itself, and
    single-linkage-chains what's left by perpendicular offset (gap_m) --
    mirroring structure._greedy_chain's 1-D clustering idea, but local to
    this function since it's a different representation (raw points, not
    WallStep stats). The largest resulting group that (a) has enough
    points, (b) sits within [min_thickness_m, max_thickness_m], and (c)
    spans enough of the wall's own length is reported as the back face.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return None

    p0 = np.asarray(run["p0"], dtype=float)
    p1 = np.asarray(run["p1"], dtype=float)
    u_vec = p1 - p0
    length = float(np.linalg.norm(u_vec))
    if length < 1e-9:
        return None
    u_vec = u_vec / length

    normal = run.get("normal")
    if normal is None:
        return None
    normal = np.asarray(normal[:2], dtype=float)
    n = np.linalg.norm(normal)
    if n < 1e-9:
        return None
    normal = normal / n

    steps = run.get("steps") or []
    if steps:
        z_min = min(st.z_min_m for st in steps)
        z_max = max(st.z_max_m for st in steps)
    else:
        z_min, z_max = -np.inf, np.inf

    xy = points[:, :2]
    rel = xy - p0
    u = rel @ u_vec
    off = rel @ normal

    z_ok = np.ones(len(points), dtype=bool)
    if points.shape[1] > 2 and np.isfinite(z_min) and np.isfinite(z_max):
        z_ok = (points[:, 2] >= z_min - margin_m) & (points[:, 2] <= z_max + margin_m)

    u_ok = (u >= -margin_m) & (u <= length + margin_m)
    band = z_ok & u_ok & (np.abs(off) <= max_thickness_m + margin_m)
    if not np.any(band):
        return None

    off_band = off[band]
    u_band = u[band]

    # Drop points essentially on the main face itself so only candidate
    # back-face points remain to be clustered.
    far = np.abs(off_band) >= (min_thickness_m * 0.5)
    if not np.any(far):
        return None
    cand_off = off_band[far]
    cand_u = u_band[far]

    order = np.argsort(cand_off)
    sorted_off = cand_off[order]
    sorted_u = cand_u[order]

    groups = []
    start = 0
    for i in range(1, len(sorted_off)):
        if sorted_off[i] - sorted_off[i - 1] > gap_m:
            groups.append((start, i))
            start = i
    groups.append((start, len(sorted_off)))

    best = None
    for (s, e) in groups:
        if (e - s) < min_points:
            continue
        seg_off = sorted_off[s:e]
        seg_u = sorted_u[s:e]
        thickness = float(np.median(np.abs(seg_off)))
        if not (min_thickness_m <= thickness <= max_thickness_m):
            continue
        u_span = float(seg_u.max() - seg_u.min())
        overlap_frac = min(u_span, length) / length
        if overlap_frac < min_overlap_frac:
            continue
        if best is None or (e - s) > best[0]:
            best = (e - s, thickness)
    return best[1] if best else None


def pair_thickness(walls, points, default_m: float = 0.10,
                    min_thickness_m: float = 0.03, max_thickness_m: float = 0.6,
                    min_overlap_frac: float = 0.5):
    """Determine each wall run's thickness by locating its "back face" --
    a second, parallel, opposite-facing surface near the wall's own
    location.

    Tries the run's own steps first (_thickness_from_steps -- often
    already implicit in how structure.group_wall_runs grouped the run's
    planes), then falls back to a direct raw-point-cloud search
    (_measure_thickness_from_points) when that comes up empty. If neither
    finds a plausible back face (single-sided wall -- common with a
    handheld scanner that only ever walks through the interior),
    thickness_m is set to `default_m` and thickness_source="assumed";
    otherwise thickness_source="measured".

    `points` is the full source point cloud (N,3) that steps/planes were
    derived from (same array passed to structure.group_wall_runs) -- only
    consulted by the fallback path.
    """
    points_arr = np.asarray(points, dtype=float) if points is not None else np.zeros((0, 3))

    out = []
    for run in walls:
        run = dict(run)
        thickness = _thickness_from_steps(run, min_thickness_m, max_thickness_m, min_overlap_frac)
        if thickness is None and points_arr.size:
            thickness = _measure_thickness_from_points(
                run, points_arr, min_thickness_m, max_thickness_m, min_overlap_frac
            )
        if thickness is not None:
            run["thickness_m"] = float(thickness)
            run["thickness_source"] = "measured"
        else:
            run["thickness_m"] = float(default_m)
            run["thickness_source"] = "assumed"
        out.append(run)
    return out


# ---------------------------------------------------------------------------
# recenter_walls
# ---------------------------------------------------------------------------

def recenter_walls(walls, points, z_floor: float, z_ceiling: float,
                    min_thickness_m: float = 0.04):
    """Shift each measured wall's centerline from its detected reference
    FACE to the midline between its two faces.

    Without this, the whole wall body (its full thickness) gets attributed
    to whichever adjacent room's plane happened to be the "main" step --
    p0/p1/offset_m sit on that face, not on the centerline, so the two
    rooms sharing the wall end up overlapping/absorbing the wall's
    internals instead of meeting cleanly at its midline. Must be called
    AFTER pair_thickness (needs thickness_m/thickness_source already set).

    Only walls with thickness_source == "measured" and thickness_m >=
    min_thickness_m are considered -- an "assumed" (single-sided,
    default-filled) thickness carries no evidence of where the back face
    actually is, so shifting would just be a guess in a random direction.

    Back-face side (which way, +normal or -normal, the second face lies)
    is determined two ways, in order:
      1. Preferred: one of the run's OWN steps (see pair_thickness/
         _thickness_from_steps) already sits at |offset_m| ~= thickness_m
         (+-0.06) -- its sign gives the side directly, no point search
         needed.
      2. Fallback: a perpendicular point-density peak at +-thickness_m
         (+-0.07 window), restricted to a mid-height band
         (z_floor+0.35 .. z_ceiling-0.35, to dodge floor/ceiling clutter)
         and to points along the wall's own u-span -- whichever side (+ or
         -) has >=50 points in that window wins.

    If neither method finds a side, the wall is returned unchanged (no
    guessing). Shifts p0, p1, and offset_m by thickness_m/2 toward the
    found back-face side, which places the centerline exactly on the
    face-pair midline. Non-mutating: returns a new list of shallow-copied
    dicts.
    """
    points = np.asarray(points, dtype=float) if points is not None else np.zeros((0, 3))
    xy = points[:, :2] if points.size else np.zeros((0, 2))
    z = points[:, 2] if points.size else np.zeros((0,))
    mid_band = (z > z_floor + 0.35) & (z < z_ceiling - 0.35)

    out = []
    for w in walls:
        w = dict(w)
        t = float(w.get("thickness_m", 0.0))
        if w.get("thickness_source") != "measured" or t < min_thickness_m:
            out.append(w)
            continue

        sign = None
        for st in w.get("steps", []):
            if abs(abs(st.offset_m) - t) < 0.06 and abs(st.offset_m) > 0.03:
                sign = 1.0 if st.offset_m > 0 else -1.0
                break

        axis_i = 0 if w["direction"] == "x" else 1
        u_i = 1 - axis_i

        if sign is None and xy.shape[0]:
            p0 = np.asarray(w["p0"], dtype=float)
            p1 = np.asarray(w["p1"], dtype=float)
            u_lo, u_hi = sorted([p0[u_i], p1[u_i]])
            band = mid_band & (xy[:, u_i] > u_lo) & (xy[:, u_i] < u_hi)
            dperp = xy[band, axis_i] - w["offset_m"]
            n_pos = int(np.count_nonzero(np.abs(dperp - t) < 0.07))
            n_neg = int(np.count_nonzero(np.abs(dperp + t) < 0.07))
            if max(n_pos, n_neg) >= 50:
                sign = 1.0 if n_pos >= n_neg else -1.0

        if sign is None:
            out.append(w)
            continue

        shift = sign * t / 2.0
        delta = np.zeros(2)
        delta[axis_i] = shift
        p0 = np.asarray(w["p0"], dtype=float)
        p1 = np.asarray(w["p1"], dtype=float)
        w["p0"] = (float(p0[0] + delta[0]), float(p0[1] + delta[1]))
        w["p1"] = (float(p1[0] + delta[0]), float(p1[1] + delta[1]))
        w["offset_m"] = float(w["offset_m"] + shift)
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# snap_endpoints_to_lines
# ---------------------------------------------------------------------------

def _point_seg_dist(pt, a, b):
    """Perpendicular (clamped) distance from `pt` to segment a-b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pt = np.asarray(pt, dtype=float)
    ab = b - a
    length_sq = float(ab @ ab)
    t = 0.0 if length_sq == 0.0 else float(np.clip((pt - a) @ ab / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * ab)))


def snap_endpoints_to_lines(walls, reach_m: float = 0.7, dangling_tol_m: float = 0.15):
    """Close T-junction corners that endpoint-to-endpoint clustering
    (resolve_corners) structurally cannot: a wall ending partway ALONG
    another wall's run, rather than at that other wall's own endpoint.

    Must be run AFTER resolve_corners. For each wall's own endpoint (p0,
    p1) that is farther than dangling_tol_m from every OTHER wall's own
    segment (i.e. not already touching anything -- resolve_corners already
    handled the endpoint-endpoint case), search for the nearest qualifying
    intersection of this wall's own infinite centerline with another
    wall's infinite centerline:
      - the two lines must not be (near-)parallel (2x2 cross-product
        system singular check);
      - the intersection point, projected onto THIS wall's own line, must
        be within reach_m of the dangling endpoint -- so the endpoint only
        extends/trims along its own centerline, it never jumps sideways;
      - the intersection point, projected onto the OTHER wall's line, must
        fall within 0.3 m of that other wall's own [p0, p1] extent -- a
        crossing far outside the other wall's actual footprint is not a
        real T-junction.

    Among all walls satisfying both conditions, the nearest intersection
    (by distance to the dangling endpoint) wins. If no candidate
    qualifies, the endpoint is left unchanged (real scan evidence: this is
    the pass that closed a 45 sq m living room corner that
    endpoint-endpoint clustering alone could not reach).

    Non-mutating: returns a new list of shallow-copied dicts; the second
    endpoint of a wall sees the (possibly already-moved) first endpoint's
    result, since a wall's own p0/p1 are re-read from `out` between the
    two endpoint passes.
    """
    out = [dict(w) for w in walls]

    for i, w in enumerate(out):
        p0 = np.asarray(w["p0"], dtype=float)
        p1 = np.asarray(w["p1"], dtype=float)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        d = seg / length

        for key, e in (("p0", p0), ("p1", p1)):
            nearest = min(
                _point_seg_dist(e, o["p0"], o["p1"])
                for j, o in enumerate(out) if j != i
            )
            if nearest <= dangling_tol_m:
                continue  # already touching something -- not dangling

            best = None
            for j, o in enumerate(out):
                if j == i:
                    continue
                a = np.asarray(o["p0"], dtype=float)
                d2 = np.asarray(o["p1"], dtype=float) - a
                length2 = float(np.linalg.norm(d2))
                if length2 < 1e-9:
                    continue
                d2 = d2 / length2

                denom = d[0] * d2[1] - d[1] * d2[0]
                if abs(denom) < 1e-6:
                    continue  # near-parallel -- no well-defined crossing

                r = a - p0
                t = (r[0] * d2[1] - r[1] * d2[0]) / denom   # along this wall
                s = (r[0] * d[1] - r[1] * d[0]) / denom     # along the other wall

                candidate = p0 + t * d
                dist = float(np.linalg.norm(candidate - e))
                if dist > reach_m:
                    continue
                if s < -0.3 or s > length2 + 0.3:
                    continue  # crossing point is far outside the other wall

                if best is None or dist < best[0]:
                    best = (dist, candidate)

            if best is not None:
                w[key] = (float(best[1][0]), float(best[1][1]))

        # Re-read p0/p1 so the second endpoint's dangling/geometry checks
        # see this wall's own already-moved first endpoint.
        p0 = np.asarray(w["p0"], dtype=float)
        p1 = np.asarray(w["p1"], dtype=float)

    return out
