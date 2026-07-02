"""Structure extraction: floor/ceiling selection, wall-run grouping with
relief (WallStep) preservation, and column/beam detection.

Operates on the flat list[Plane] produced by planes.detect_planes, on the
ALREADY axis-aligned cloud (see frame.dominant_axes / frame.axis_align).
Pure numpy -- no Open3D dependency, so every function here is unit-testable
without a real scan.
"""
from __future__ import annotations

import numpy as np

from .schema import WallStep, Column, Beam, new_column_id, new_beam_id


def extract_floor_ceiling(planes):
    """Pick the best floor and ceiling Plane (largest inlier count per label).

    Raises ValueError if a label is entirely missing among `planes` --
    callers need to know isolation/plane-detection failed upstream rather
    than silently getting None back and blowing up somewhere downstream.
    """
    floors = [p for p in planes if p.label == "floor"]
    ceilings = [p for p in planes if p.label == "ceiling"]
    if not floors:
        raise ValueError(
            "extract_floor_ceiling: no plane labeled 'floor' among the "
            f"{len(planes)} input planes. Check upstream plane detection "
            "(planes.detect_planes) and unit isolation."
        )
    if not ceilings:
        raise ValueError(
            "extract_floor_ceiling: no plane labeled 'ceiling' among the "
            f"{len(planes)} input planes. Check upstream plane detection "
            "(planes.detect_planes) and unit isolation."
        )
    floor = max(floors, key=lambda p: np.asarray(p.inlier_idx).size)
    ceiling = max(ceilings, key=lambda p: np.asarray(p.inlier_idx).size)
    return floor, ceiling


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _robust_range(vals, max_gap_m: float = 0.08):
    """1-D "largest contiguous group" range: sort vals, chain into groups
    split wherever a consecutive gap exceeds max_gap_m, and return the
    (min, max) of the LARGEST group by point count.

    detect_planes's RANSAC + DBSCAN occasionally lets a thin sliver of a
    spatially-nearby but geometrically unrelated surface leak into a plane's
    inliers -- e.g. a handful of points from a perpendicular face that
    happen to sit within dist_thresh of the fitted plane equation, and are
    then DBSCAN-bridged to the real cluster because they're within eps of
    it in full 3-D even though they're far from it along this one axis.
    That leak is always a small minority of the plane's points and clearly
    gap-separated along u/z (max_gap_m=0.08 m is well under detect_planes's
    default dbscan_eps=0.15 m but many times the fixtures'/real scans'
    typical point spacing), so taking the majority group discards it without
    eating into genuine extent -- unlike a plain min/max, which a single
    outlier point can drag by an arbitrary amount, or a percentile trim,
    which would also shave a bit off every clean, well-populated face.
    """
    vals = np.sort(np.asarray(vals, dtype=float))
    if vals.size <= 1:
        return float(vals.min()), float(vals.max())
    groups = []
    start = 0
    for i in range(1, vals.size):
        if vals[i] - vals[i - 1] > max_gap_m:
            groups.append((start, i))
            start = i
    groups.append((start, vals.size))
    lo, hi = max(groups, key=lambda g: g[1] - g[0])
    return float(vals[lo]), float(vals[hi - 1])


def _plane_axis_stats(plane, xyz, ex, ey):
    """Project a vertical plane's inlier points onto the (ex, ey) horizontal
    frame. Returns which of ex/ey the plane's normal is closer to (its
    "direction"), the plane's offset (its position along that axis), its
    along-wall extent (u, along the OTHER axis) and its vertical extent (z).
    """
    idx = np.asarray(plane.inlier_idx)
    pts = xyz[idx]
    normal = np.asarray(plane.normal, dtype=float)
    dot_x = abs(float(np.dot(normal, ex)))
    dot_y = abs(float(np.dot(normal, ey)))
    if dot_x >= dot_y:
        axis_name, axis_vec, u_vec, align = "x", ex, ey, dot_x
    else:
        axis_name, axis_vec, u_vec, align = "y", ey, ex, dot_y
    offset = float((pts @ axis_vec).mean())
    u_vals = pts @ u_vec
    z_vals = pts[:, 2]
    u_min, u_max = _robust_range(u_vals)
    z_min, z_max = _robust_range(z_vals)
    # Drop the same leaked outlier points from the raw XY set used for
    # column-footprint bounding boxes -- otherwise a stray point discarded
    # from u_min/u_max above would still blow out the footprint bbox.
    keep = (u_vals >= u_min) & (u_vals <= u_max) & (z_vals >= z_min) & (z_vals <= z_max)
    pts_kept = pts[keep]
    return dict(
        plane=plane,
        axis=axis_name,
        axis_vec=axis_vec,
        u_vec=u_vec,
        align=align,
        offset=offset,
        u_min=u_min,
        u_max=u_max,
        z_min=z_min,
        z_max=z_max,
        n_inliers=int(idx.size),
        centroid_xy=(float(pts_kept[:, 0].mean()), float(pts_kept[:, 1].mean())),
        pts_xy=pts_kept[:, :2],
    )


def _greedy_chain(items, key, max_gap):
    """Sort items by key(item) and split into maximal runs whose consecutive
    key gap is <= max_gap (single-linkage chaining along a 1-D axis).
    Returns list[list[item]].
    """
    items = sorted(items, key=key)
    if not items:
        return []
    clusters = [[items[0]]]
    for it in items[1:]:
        if key(it) - key(clusters[-1][-1]) <= max_gap:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return clusters


def group_wall_runs(
    vertical_planes,
    xyz,
    axes,
    merge_offset_m: float = 0.15,
    max_relief_m: float = 1.0,
    min_run_length_m: float = 1.0,
    angle_tol_deg: float = 10.0,
    full_height_frac: float = 0.5,
    *,
    return_used_indices: bool = False,
):
    """Group vertical Planes into wall "runs" (one per physical wall line).

    A run is a maximal set of same-direction, colinear-ish planes: their
    normal is parallel to a common horizontal axis (within angle_tol_deg,
    sign-agnostic -- a wall's two faces point opposite ways), and their
    perpendicular offset from that axis is within max_relief_m of the run's
    other members (chained transitively, see _greedy_chain).

    This is deliberately two-tiered:
      - merge_offset_m (default 0.15 m) controls whether two colinear planes
        collapse into the SAME WallStep, e.g. two DBSCAN pieces of one wall
        face split by a doorway (same physical surface, offset differs only
        by point-cloud noise).
      - max_relief_m (default 1.0 m) controls how far a face can protrude
        and still count as relief on the SAME wall rather than a different
        wall entirely -- e.g. a pillar front 0.3 m proud of its wall, or the
        ~0.2 m gap between a wall's inner/outer faces (offset > 0.15 m, so it
        does NOT collapse into the main step, but it is still far short of
        max_relief_m so it stays part of the same run as its own WallStep).
      Two distinct real walls in this pipeline's target buildings (room-scale
      Manhattan floorplans) are always many multiples of max_relief_m apart,
      so one generous constant safely tells "relief" from "a different wall".

    Only "full height" planes are eligible for run formation: a plane's own
    vertical extent must be >= full_height_frac of the tallest vertical
    plane's extent (among the input). This keeps beam side faces (elevated,
    short z-extent, but which can otherwise span the whole room and look
    "wall-length") out of wall-run formation, without needing floor_z /
    ceiling_z as parameters here. extract_columns_beams handles columns and
    beams separately, from the *unfiltered* vertical_planes list.

    Candidate runs whose main step spans less than min_run_length_m (default
    1.0 m) are dropped -- not long enough to be a real wall, most likely a
    column face caught in the same direction/offset bucket (see
    extract_columns_beams, which handles those explicitly).

    axes: 3x3 rotation whose first two columns give the two dominant
    horizontal (Manhattan) directions in the frame `xyz` is expressed in. By
    the time this function runs, axis_align() has already rotated the cloud
    so walls are parallel to world X/Y (see frame.py), so callers normally
    pass np.eye(3) here.

    Deviation from the brief's literal signature: `xyz` (the point cloud
    that every plane.inlier_idx indexes into) is added as the 2nd positional
    argument. Plane only stores (normal, d, label, inlier_idx) -- there is no
    way to derive a WallStep's u_min/u_max/z_min/z_max or a run's centerline
    endpoints from that alone; the point coordinates are required and Plane
    does not carry them.

    WallStep.offset_m is relative to the run's own main/reference face, per
    schema.py's WallStep docstring ("offset_m is signed along the wall
    normal relative to the wall's reference (main) face"): the main step
    always comes back with offset_m == 0.0, and every other step's offset_m
    is its signed distance from the main step (e.g. the pillar-front step
    in the west wall's run comes back at roughly +0.3, not the absolute
    world x-coordinate ~0.4). The run's own absolute world position is
    still available uncollapsed as the dict's top-level offset_m (see
    below) and in p0/p1, so nothing about the main face's true location is
    lost -- only the per-step values are re-based to satisfy the schema's
    contract.

    Returns a list[dict], one per wall run, each with keys:
      direction: "x" | "y" -- which dominant axis the run's normal is closest to
      normal: (float, float, float) unit vector along `direction`
      p0, p1: (x, y) centerline endpoints of the run's MAIN step, world frame
      offset_m: the main step's ABSOLUTE perpendicular offset (its world position)
      steps: list[WallStep] (including the main one, offset_m == 0.0),
             sorted by offset_m, relative to the main step as described above

    return_used_indices (keyword-only, default False): when True, return
    (runs, used_indices) instead of just runs, where used_indices is a
    set[int] of indices into the INPUT vertical_planes list that ended up
    consumed by some accepted run's steps. This exists so a caller can union
    it with extract_columns_beams's own used_indices and pass the result to
    extract_unclassified, which surfaces every vertical plane that neither
    function claimed -- so nothing detected by planes.detect_planes is ever
    silently dropped without at least being reported. Default False keeps
    the return type exactly a list[dict], unchanged from before this
    parameter existed, so every pre-existing caller/test keeps working
    without modification.
    """
    xyz = np.asarray(xyz, dtype=float)
    axes = np.asarray(axes, dtype=float)
    ex = _unit(axes[:, 0])
    ey = _unit(axes[:, 1])

    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    stats = []
    for i, p in enumerate(vertical_planes):
        s = _plane_axis_stats(p, xyz, ex, ey)
        if s["align"] < cos_tol:
            continue  # normal isn't close enough to either dominant axis
        s["src_index"] = i  # index into the ORIGINAL vertical_planes list --
        # tracked here (not reverse-engineered from output geometry later)
        # so it survives every filter/cluster step below intact.
        stats.append(s)
    if not stats:
        return ([], set()) if return_used_indices else []

    max_z_span = max(s["z_max"] - s["z_min"] for s in stats)
    height_thresh = full_height_frac * max_z_span
    stats = [s for s in stats if (s["z_max"] - s["z_min"]) >= height_thresh]

    runs = []
    used_indices = set()
    for axis_name in ("x", "y"):
        group = [s for s in stats if s["axis"] == axis_name]
        for cluster in _greedy_chain(group, key=lambda s: s["offset"], max_gap=max_relief_m):
            step_groups = _greedy_chain(cluster, key=lambda s: s["offset"], max_gap=merge_offset_m)
            steps = []
            for sg in step_groups:
                total_n = sum(x["n_inliers"] for x in sg)
                offset_avg = sum(x["offset"] * x["n_inliers"] for x in sg) / total_n
                steps.append(WallStep(
                    offset_m=offset_avg,
                    u_min_m=min(x["u_min"] for x in sg),
                    u_max_m=max(x["u_max"] for x in sg),
                    z_min_m=min(x["z_min"] for x in sg),
                    z_max_m=max(x["z_max"] for x in sg),
                ))
            main_step = max(steps, key=lambda st: st.u_max_m - st.u_min_m)
            if (main_step.u_max_m - main_step.u_min_m) < min_run_length_m:
                continue  # too short to be a wall -- likely a column face

            main_abs_offset = main_step.offset_m
            axis_vec = cluster[0]["axis_vec"]
            u_vec = cluster[0]["u_vec"]
            p0 = axis_vec * main_abs_offset + u_vec * main_step.u_min_m
            p1 = axis_vec * main_abs_offset + u_vec * main_step.u_max_m
            rebased_steps = sorted(
                (WallStep(
                    offset_m=st.offset_m - main_abs_offset,
                    u_min_m=st.u_min_m,
                    u_max_m=st.u_max_m,
                    z_min_m=st.z_min_m,
                    z_max_m=st.z_max_m,
                ) for st in steps),
                key=lambda st: st.offset_m,
            )
            runs.append(dict(
                direction=axis_name,
                normal=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
                offset_m=main_abs_offset,
                p0=(float(p0[0]), float(p0[1])),
                p1=(float(p1[0]), float(p1[1])),
                steps=rebased_steps,
            ))
            # The whole cluster (every step_group / step) is consumed by
            # this accepted run -- a plane that was chained here but whose
            # candidate run got dropped by the min_run_length_m check above
            # (the `continue` a few lines up) never reaches this point, so
            # it correctly stays out of used_indices.
            used_indices.update(s["src_index"] for s in cluster)
    if return_used_indices:
        return runs, used_indices
    return runs


def _cluster_by_centroid(items, gap_m):
    """Single-linkage clustering of items (each with a "centroid_xy" key):
    merge any two whose centroids are within gap_m. Returns list[list[item]].
    """
    n = len(items)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            ci = np.array(items[i]["centroid_xy"])
            cj = np.array(items[j]["centroid_xy"])
            if np.linalg.norm(ci - cj) <= gap_m:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    return list(groups.values())


def _extract_columns(stats, floor_z, ceiling_z, min_size_m, max_size_m, height_tol_m, cluster_gap_m):
    full_height = ceiling_z - floor_z
    candidates = []
    for s in stats:
        length = s["u_max"] - s["u_min"]
        if not (min_size_m <= length <= max_size_m):
            continue
        if abs(s["z_min"] - floor_z) > height_tol_m:
            continue
        if abs(s["z_max"] - ceiling_z) > height_tol_m:
            continue
        if full_height > 0 and (s["z_max"] - s["z_min"]) < 0.8 * full_height:
            continue
        candidates.append(s)

    columns = []
    used_indices = set()
    for i, group in enumerate(_cluster_by_centroid(candidates, cluster_gap_m)):
        pts_xy = np.vstack([g["pts_xy"] for g in group])
        x_min, y_min = pts_xy.min(axis=0)
        x_max, y_max = pts_xy.max(axis=0)
        z_min = min(g["z_min"] for g in group)
        z_max = max(g["z_max"] for g in group)
        footprint = [
            (float(x_min), float(y_min)),
            (float(x_max), float(y_min)),
            (float(x_max), float(y_max)),
            (float(x_min), float(y_max)),
        ]
        columns.append(Column(
            column_id=new_column_id(i),
            footprint=footprint,
            z_min_m=float(z_min),
            z_max_m=float(z_max),
        ))
        used_indices.update(g["src_index"] for g in group)
    return columns, used_indices


def _extract_beams(stats, floor_z, ceiling_z, elevation_m, ceiling_tol_m, gap_m, max_size_m):
    # A real beam side face spans most of the room (long, like a wall), so
    # excluding "compact" (<= max_size_m, column-scale) candidates keeps
    # stray column/pillar DBSCAN fragments -- which can otherwise sit at an
    # offset close enough to a real beam to get offset-chained into its
    # group (dragging its z_min/z_max off) -- out of beam candidacy
    # entirely, without needing to cross-reference the column detector.
    candidates = [
        s for s in stats
        if (s["z_min"] - floor_z) > elevation_m
        and (ceiling_z - s["z_max"]) <= ceiling_tol_m
        and (s["u_max"] - s["u_min"]) > max_size_m
    ]
    beams = []
    used_indices = set()
    idx = 0
    for axis_name in ("x", "y"):
        group = [s for s in candidates if s["axis"] == axis_name]
        for cluster in _greedy_chain(group, key=lambda s: s["offset"], max_gap=gap_m):
            offsets = [s["offset"] for s in cluster]
            u_min = min(s["u_min"] for s in cluster)
            u_max = max(s["u_max"] for s in cluster)
            z_min = min(s["z_min"] for s in cluster)
            z_max = max(s["z_max"] for s in cluster)
            mid_offset = 0.5 * (min(offsets) + max(offsets))
            depth = max(offsets) - min(offsets)
            axis_vec = cluster[0]["axis_vec"]
            u_vec = cluster[0]["u_vec"]
            p0 = axis_vec * mid_offset + u_vec * u_min
            p1 = axis_vec * mid_offset + u_vec * u_max
            beams.append(Beam(
                beam_id=new_beam_id(idx),
                p0=(float(p0[0]), float(p0[1])),
                p1=(float(p1[0]), float(p1[1])),
                width_m=float(u_max - u_min),
                depth_m=float(depth),
                z_min_m=float(z_min),
                z_max_m=float(z_max),
            ))
            used_indices.update(s["src_index"] for s in cluster)
            idx += 1
    return beams, used_indices


def extract_columns_beams(
    vertical_planes,
    xyz,
    floor_z: float,
    ceiling_z: float,
    min_size_m: float = 0.1,
    max_size_m: float = 0.6,
    planes=None,
    height_tol_m: float = 0.3,
    beam_elevation_m: float = 1.0,
    beam_ceiling_tol_m: float = 0.5,
    beam_offset_gap_m: float = 1.0,
    cluster_gap_m: float = None,
    *,
    return_used_indices: bool = False,
):
    """Split vertical_planes into Columns (compact floor-to-ceiling piers)
    and Beams (long faces hung near the ceiling that do not reach the floor).

    Column detection: a plane is a column-face candidate if its own
    along-face extent is between min_size_m and max_size_m (compact, unlike
    a wall, which is excluded by being longer than max_size_m) AND it spans
    close to the full floor_z..ceiling_z range (within height_tol_m at each
    end, and at least 80% of the total height). Candidates are grouped by
    XY proximity (centroid distance <= cluster_gap_m, default max_size_m --
    a pillar's front/side faces touch each other) into one Column per group;
    footprint is the axis-aligned bounding box of every member plane's
    inlier points. This pipeline targets Manhattan-aligned buildings (see
    frame.dominant_axes), so an AABB is the correct footprint representation
    for rectangular columns; it is not a general convex-hull footprint.

    Beam detection: a plane is a beam-side-face candidate if it does NOT
    reach near the floor (z_min_m - floor_z > beam_elevation_m, default
    1.0 m), DOES reach near the ceiling (ceiling_z - z_max_m <=
    beam_ceiling_tol_m, default 0.5 m), and is NOT compact (its own u-extent
    is > max_size_m). The elevation/ceiling checks distinguish a beam's
    vertical side faces from a real wall (which always reaches the floor) --
    a beam can otherwise span the full room width and look "wall-length".
    The compactness check exists because iterative RANSAC peeling can leave
    a small, elevated DBSCAN leftover fragment of a column/pillar face (not
    reaching the floor, purely as an artifact of point removal order) at an
    offset close enough to a real beam to get chained into its group by
    offset alone -- excluding compact (column-scale) candidates from beam
    candidacy up front avoids that without cross-referencing the column
    detector. Candidates are grouped by (direction, offset), chaining
    consecutive offsets <= beam_offset_gap_m apart (mirrors
    group_wall_runs's step clustering), into one Beam per group: width_m is
    the group's combined along-face extent, depth_m the spread between its
    member offsets (e.g. the two side faces of one beam), p0/p1 the
    centerline at the mid-offset.

    z_min_m/z_max_m on the returned Beam are read directly off its side
    faces' own vertical extent (in this pipeline's fixtures a beam's side
    faces already reach the ceiling, so this captures the true z-range).
    `planes` (the full, unfiltered plane list -- useful for locating the
    beam's horizontal bottom/soffit face, typically labeled "ceiling" by
    planes.label_plane since it is horizontal and sits in the upper half of
    the storey) is accepted for API symmetry with the brief and for future
    use (e.g. depth_m cross-check, or beams that don't reach the ceiling),
    but is currently unused -- see the task report for why it wasn't needed
    to pass this task's acceptance criteria.

    Deviation from the brief's literal signature: `xyz` (2nd positional arg)
    is required for the same reason as in group_wall_runs -- Plane alone has
    no coordinates to derive a footprint / span from.

    return_used_indices (keyword-only, default False): when True, return
    (columns, beams, used_indices) instead of just (columns, beams), where
    used_indices is a set[int] of indices into the INPUT vertical_planes
    list consumed by some accepted Column or Beam (their union -- a plane
    can in principle satisfy neither, either, or in edge cases both, since
    column and beam candidacy are checked independently). Paired with
    group_wall_runs's own return_used_indices, this lets a caller build the
    full set of "claimed" plane indices and pass whatever's left to
    extract_unclassified so nothing is silently dropped without at least
    being surfaced for review. Default False keeps the return type exactly
    (list[Column], list[Beam]), unchanged from before this parameter
    existed, so every pre-existing caller/test keeps working unmodified.

    Returns (list[Column], list[Beam]).
    """
    xyz = np.asarray(xyz, dtype=float)
    if cluster_gap_m is None:
        cluster_gap_m = max_size_m
    axes = np.eye(3)
    ex, ey = axes[:, 0], axes[:, 1]

    stats = []
    for i, p in enumerate(vertical_planes):
        s = _plane_axis_stats(p, xyz, ex, ey)
        s["src_index"] = i  # index into the ORIGINAL vertical_planes list
        stats.append(s)

    columns, col_used = _extract_columns(
        stats, floor_z, ceiling_z, min_size_m, max_size_m, height_tol_m, cluster_gap_m
    )
    beams, beam_used = _extract_beams(
        stats, floor_z, ceiling_z, beam_elevation_m, beam_ceiling_tol_m, beam_offset_gap_m,
        max_size_m
    )
    if return_used_indices:
        return columns, beams, col_used | beam_used
    return columns, beams


def extract_unclassified(vertical_planes, used_indices, xyz, axes=None, direction_tol_deg: float = 20.0):
    """Every vertical Plane whose index is NOT in `used_indices`, surfaced as
    a reviewable dict instead of being silently discarded.

    Motivation: on a real scan, planes.detect_planes can find far more
    vertical planes than group_wall_runs + extract_columns_beams together
    account for (a real house scan: 218 detected vertical planes vs. 25
    WallSteps / 11 runs / 8 beams / 0 columns -- ~88% unaccounted for). Some
    of that gap is genuine furniture/clutter noise, but some is plausibly a
    real short wall jog, a partial-height recess, or an occluded wall
    section that the deliberately conservative filters in group_wall_runs
    (min_run_length_m, full_height_frac) and extract_columns_beams
    (size/height/elevation/compactness thresholds) are excluding -- and a
    dropped real surface here is a real, costly downstream error (e.g. a
    wardrobe cutlist that doesn't notch around a real protruding pillar).
    Rather than loosen those already-tested, already-approved filters
    without more validation, this function makes the exclusion visible: it
    is the complement of used_indices within vertical_planes, so a human (or
    a later, more careful classification pass) can inspect exactly what got
    left out and decide if any of it is real.

    used_indices: a set[int] (or any container supporting `in`) of indices
    into THIS SAME vertical_planes list, as returned by
    group_wall_runs(..., return_used_indices=True) and/or
    extract_columns_beams(..., return_used_indices=True) -- typically their
    union, since a single plane may legitimately be claimed by either (or,
    same as a pillar face that is both a WallStep AND a Column, both).

    Deviation from the brief's literal `(vertical_planes, used_indices) ->
    list[dict]` signature: `xyz` is added as a 3rd positional argument, for
    the same reason group_wall_runs and extract_columns_beams both already
    require it -- Plane only stores (normal, d, label, inlier_idx), with no
    coordinates of its own, so the source cloud is needed to measure a
    plane's extent via this module's own _plane_axis_stats (reused here
    unchanged, so unclassified planes are measured with the exact same
    u_min/u_max/z_min/z_max/offset convention as every classified one,
    rather than a second, subtly-different convention). `axes` mirrors
    group_wall_runs's own optional 3x3 rotation (first two columns = the
    two dominant horizontal directions); defaults to np.eye(3), correct for
    callers on the already axis-aligned frame (see frame.axis_align), same
    as group_wall_runs's own docstring note that callers "normally pass
    np.eye(3) here".

    direction_tol_deg (default 20, deliberately more generous than
    group_wall_runs's own angle_tol_deg=10): a plane ends up unclassified
    for many reasons OTHER than poor axis alignment (too short, wrong
    elevation, wrong compactness, not full height, ...), so best-guess
    "direction" here uses a looser cutoff than the strict one group_wall_runs
    applies before it will actually build a run -- a wider net that still
    calls out a genuinely oblique/diagonal plane (e.g. furniture, a
    non-Manhattan real feature) by reporting direction=None rather than
    forcing a misleading guess onto it.

    Returns a list[dict], one per unclassified plane, ordered by
    plane_index ascending, each with keys:
      plane_index: int, this plane's index in the ORIGINAL vertical_planes
                   list (stable identifier for cross-referencing back to it)
      normal: (float, float, float), the plane's own stored unit normal
      direction: "x" | "y" | None -- best-guess dominant-axis alignment
                 within direction_tol_deg, or None if it isn't confidently
                 aligned to either (see above)
      n_points: int, len(plane.inlier_idx) -- how much real support this
                excluded surface has, the first thing a human reviewer
                needs to triage "probably real" vs. "probably noise"
      u_min_m, u_max_m, u_span_m: along-face extent (same "u" convention as
                 WallStep -- along whichever of x/y is NOT `direction`)
      z_min_m, z_max_m, z_span_m: vertical extent
      offset_m: perpendicular offset (world frame, absolute -- NOT rebased
                 to any reference face, since an unclassified plane by
                 definition doesn't belong to a run) along whichever of
                 ex/ey the plane's normal is closer to -- populated even
                 when direction is None, since _plane_axis_stats always
                 picks a closer axis to measure along; direction=None only
                 means that pick wasn't confident enough to report as a
                 real Manhattan-direction guess
    """
    xyz = np.asarray(xyz, dtype=float)
    if axes is None:
        axes = np.eye(3)
    axes = np.asarray(axes, dtype=float)
    ex = _unit(axes[:, 0])
    ey = _unit(axes[:, 1])
    cos_tol = np.cos(np.deg2rad(direction_tol_deg))

    out = []
    for i, p in enumerate(vertical_planes):
        if i in used_indices:
            continue
        s = _plane_axis_stats(p, xyz, ex, ey)
        direction = s["axis"] if s["align"] >= cos_tol else None
        out.append(dict(
            plane_index=i,
            normal=tuple(float(x) for x in np.asarray(p.normal, dtype=float)),
            direction=direction,
            n_points=int(np.asarray(p.inlier_idx).size),
            u_min_m=s["u_min"],
            u_max_m=s["u_max"],
            u_span_m=s["u_max"] - s["u_min"],
            z_min_m=s["z_min"],
            z_max_m=s["z_max"],
            z_span_m=s["z_max"] - s["z_min"],
            offset_m=s["offset"],
        ))
    return out
