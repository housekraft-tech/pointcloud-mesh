"""Openings: voids, trajectory visibility gate, walkthrough seeds, classification.

Operates on the wall-run dict shape produced by structure.group_wall_runs and
regularized by regularize.py (direction, p0, p1, offset_m, thickness_m, steps,
members -- see regularize.py's module docstring for the exact contract), plus
the furniture-free point cloud, the recovered scanner trajectory
(trajectory.approx_trajectory / load_trajectory), and trajectory.wall_crossings'
per-wall crossing-u lists.

Design note -- horizontal slicing: a wall's occupancy is never collapsed to a
single top-down projection. Instead wall_occupancy builds a 2-D (u, z) grid --
"a stack of horizontal slices" -- so a door reads as a floor-to-header void
column, a window as a mid-band void, and a furniture shadow as a void that
LOOKS open in the grid but fails the trajectory visibility gate (nobody ever
walked/saw through it because a piece of furniture, not open air, sits behind
it). Keeping height resolved is what makes those three cases distinguishable;
a 1-D u-only projection could not tell a window from a door from a shadow.

Module layout: occupancy/void geometry first (wall_occupancy, find_voids),
then the trajectory visibility gate (visibility_gate), then coarse-void edge
refinement against point density (refine_edges), then prior-aware
classification (classify_opening), then the top-level orchestrator
(detect_openings).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# occupancy / void geometry
# ---------------------------------------------------------------------------

def _wall_frame(wall):
    """Return (p0, u_vec, length, normal_xy) for a wall dict: u_vec is the
    unit vector along the wall's own centerline (direction="x" -> world X,
    "y" -> world Y -- matches structure.group_wall_runs/trajectory.wall_crossings'
    own u convention, u = the coordinate along whichever axis is NOT the
    wall's normal axis), normal_xy is the unit perpendicular (world XY).
    """
    p0 = np.asarray(wall["p0"], dtype=float)
    p1 = np.asarray(wall["p1"], dtype=float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        # degenerate: fall back to the wall's own declared direction so
        # callers still get a well-defined (if zero-length) frame.
        if wall.get("direction") == "x":
            u_vec = np.array([1.0, 0.0])
        else:
            u_vec = np.array([0.0, 1.0])
    else:
        u_vec = d / length
    normal_xy = np.array([-u_vec[1], u_vec[0]])
    return p0, u_vec, length, normal_xy


def wall_occupancy(wall, xyz, cell_m: float = 0.03, band_m: float = None, z_band=None):
    """Boolean (nu, nz) occupancy grid of `xyz` points within `band_m` of the
    wall's midline plane, projected onto the wall's own (u, z) frame.

    band_m defaults to thickness_m/2 + 0.08 (a real wall's own half-thickness
    plus slack for point-cloud noise / a slightly mis-measured thickness --
    wide enough to catch both wall faces, narrow enough to exclude furniture
    sitting a few cm off the wall).

    z_band (z_lo, z_hi), default full height of the points passed in: lets a
    caller (e.g. Task 10) request only a furniture-free mid-height slice
    instead of the whole wall -- see the module docstring's "horizontal
    slicing" note. wall_occupancy itself doesn't know z_floor/z_ceiling, so a
    caller must supply this explicitly if it wants anything other than "every
    point in the u/perp band, whatever z it's at".

    Returns (occ, u0, z0): occ[i, j] is True iff at least one point falls in
    the cell spanning u in [u0 + i*cell_m, u0 + (i+1)*cell_m) and z in
    [z0 + j*cell_m, z0 + (j+1)*cell_m). u0/z0 are the grid origin in world u /
    world z (u0 is snapped to the wall's own u=0 at p0, not re-based to the
    occupied points' own min -- so grid cells line up consistently across
    calls with the same wall).
    """
    xyz = np.asarray(xyz, dtype=float)
    p0, u_vec, length, normal_xy = _wall_frame(wall)
    thickness_m = float(wall.get("thickness_m", 0.1))
    if band_m is None:
        band_m = thickness_m / 2.0 + 0.08

    rel = xyz[:, :2] - p0
    u = rel @ u_vec
    perp = rel @ normal_xy
    z = xyz[:, 2]

    mask = (np.abs(perp) <= band_m) & (u >= -1e-6) & (u <= length + 1e-6)
    if z_band is not None:
        z_lo, z_hi = z_band
        mask &= (z >= z_lo) & (z <= z_hi)

    u_sel = u[mask]
    z_sel = z[mask]

    u0 = 0.0
    if z_sel.size:
        z0 = float(np.floor(z_sel.min() / cell_m) * cell_m)
        z_hi = float(np.ceil(z_sel.max() / cell_m) * cell_m)
    elif z_band is not None:
        z0 = float(np.floor(z_band[0] / cell_m) * cell_m)
        z_hi = float(np.ceil(z_band[1] / cell_m) * cell_m)
    else:
        z0 = 0.0
        z_hi = 0.0

    nu = max(1, int(np.ceil((length - u0) / cell_m)))
    nz = max(1, int(np.ceil((z_hi - z0) / cell_m)))
    occ = np.zeros((nu, nz), dtype=bool)

    if u_sel.size:
        iu = np.clip(((u_sel - u0) / cell_m).astype(int), 0, nu - 1)
        iz = np.clip(((z_sel - z0) / cell_m).astype(int), 0, nz - 1)
        occ[iu, iz] = True

    return occ, float(u0), float(z0)


def find_voids(occ, u0, z0, cell_m: float = 0.03, min_w_m: float = 0.55, min_h_m: float = 0.55):
    """Connected empty (~occ) regions of the occupancy grid, as rectangles
    {u0, u1, z0, z1} in world units (each tightened to its label's own
    bounding box), filtered to at least min_w_m x min_h_m.

    The grid is padded by one occupied ("wall") cell on every side first, so
    the true "outside the wall's own footprint" background merges with
    whatever real empty region touches the grid's edge only through that
    single padding ring -- not through the void itself. This lets a void that
    touches a real grid edge (e.g. a door reaching all the way down to the
    floor row, or a balcony opening reaching a side) still get its own label,
    distinct from the un-padded case where scipy.ndimage.label would treat
    "touches the border" as automatically connected to everything else
    touching any border (4-connectivity wraps a void around a corner into
    the outside-background region otherwise, silently swallowing real
    door/balcony voids that happen to touch the floor row or a side column).
    """
    occ = np.asarray(occ, dtype=bool)
    nu, nz = occ.shape
    padded = np.ones((nu + 2, nz + 2), dtype=bool)
    padded[1:-1, 1:-1] = occ

    empty = ~padded
    labeled, n_labels = ndimage.label(empty)

    voids = []
    for label in range(1, n_labels + 1):
        idx_u, idx_z = np.nonzero(labeled == label)
        # undo the padding offset
        idx_u = idx_u - 1
        idx_z = idx_z - 1
        # drop the outside-background pixels (the padding ring itself, and
        # any real cell that fell outside [0, nu) x [0, nz) -- shouldn't
        # happen since real cells only exist inside the original occ shape,
        # but guard anyway).
        keep = (idx_u >= 0) & (idx_u < nu) & (idx_z >= 0) & (idx_z < nz)
        idx_u = idx_u[keep]
        idx_z = idx_z[keep]
        if idx_u.size == 0:
            continue  # this label was purely the padding ring / background

        vu0 = u0 + idx_u.min() * cell_m
        vu1 = u0 + (idx_u.max() + 1) * cell_m
        vz0 = z0 + idx_z.min() * cell_m
        vz1 = z0 + (idx_z.max() + 1) * cell_m

        if (vu1 - vu0) < min_w_m or (vz1 - vz0) < min_h_m:
            continue

        voids.append({"u0": float(vu0), "u1": float(vu1), "z0": float(vz0), "z1": float(vz1)})

    return voids


# ---------------------------------------------------------------------------
# visibility gate
# ---------------------------------------------------------------------------

def _void_samples(void):
    """Center + 4 midpoints of half-edges of a void rect, as (u, z) pairs --
    5 sample points per the brief."""
    u0, u1, z0, z1 = void["u0"], void["u1"], void["z0"], void["z1"]
    uc, zc = (u0 + u1) / 2.0, (z0 + z1) / 2.0
    return [
        (uc, zc),
        ((u0 + uc) / 2.0, zc),
        ((uc + u1) / 2.0, zc),
        (uc, (z0 + zc) / 2.0),
        (uc, (zc + z1) / 2.0),
    ]


def visibility_gate(void, wall, trajectory, kdtree, ray_step_m: float = 0.10,
                     clear_r_m: float = 0.07, min_rays: int = 3,
                     max_sensor_dist_m: float = 7.0) -> bool:
    """True iff at least `min_rays` trajectory vertices had a clear line of
    sight through this void -- i.e. this is a real see-through opening, not
    an occupancy hole caused by furniture merely occluding the scanner's view
    of an otherwise-solid wall (a "shadow").

    For each trajectory vertex within max_sensor_dist_m of any of the void's
    5 sample points (center + 4 half-edge midpoints, see _void_samples), a
    ray is cast from the sensor position through that sample point and
    extended 0.4 m past the wall plane. The ray passes if NONE of its sample
    points along the way (starting 0.4 m from the sensor, spaced ray_step_m)
    have a cloud point within clear_r_m (queried against the prebuilt
    `kdtree`) -- i.e. nothing solid blocks the line of sight all the way
    through and past the wall. A furniture occlusion shadow fails this gate
    because rays from OTHER trajectory positions (that never got line of
    sight past the furniture) hit either the furniture itself or the wall
    surface; a real opening has at least min_rays clean rays through it.
    """
    p0, u_vec, length, normal_xy = _wall_frame(wall)
    samples = _void_samples(void)
    sample_xyz = []
    for (u, z) in samples:
        xy = p0 + u_vec * u
        sample_xyz.append(np.array([xy[0], xy[1], z]))

    traj = np.asarray(trajectory, dtype=float)
    if traj.size == 0:
        return False

    n_pass = 0
    for sensor in traj:
        for target in sample_xyz:
            dist = float(np.linalg.norm(target - sensor))
            if dist > max_sensor_dist_m or dist < 1e-6:
                continue
            direction = (target - sensor) / dist
            # extend 0.4 m past the wall plane (i.e. past the target point)
            total_len = dist + 0.4
            n_steps = int(np.floor((total_len - 0.4) / ray_step_m)) + 1
            if n_steps < 1:
                continue
            ts = 0.4 + np.arange(n_steps) * ray_step_m
            ts = ts[ts <= total_len]
            if ts.size == 0:
                continue
            ray_pts = sensor[None, :] + ts[:, None] * direction[None, :]
            counts = kdtree.query_ball_point(ray_pts, r=clear_r_m, return_length=True)
            blocked = bool(np.any(np.asarray(counts) > 0))
            if not blocked:
                n_pass += 1
                if n_pass >= min_rays:
                    return True
    return n_pass >= min_rays


# ---------------------------------------------------------------------------
# edge refinement
# ---------------------------------------------------------------------------

def _half_max_crossing(bin_centers, counts, coarse_edge, search_m, direction):
    """Find the density half-max crossing nearest to coarse_edge, searching
    within +-search_m, moving in `direction` (+1 or -1) from the coarse
    edge -- i.e. walking from the void's interior (low density) toward the
    solid wall (high density) and reporting where density crosses half of
    the local peak. Returns None if no crossing is found in range.
    """
    if counts.size == 0 or counts.max() <= 0:
        return None
    peak = float(counts.max())
    half = peak / 2.0

    in_range = np.abs(bin_centers - coarse_edge) <= search_m
    if not np.any(in_range):
        return None
    idxs = np.nonzero(in_range)[0]
    # order indices by distance moving outward from coarse_edge in `direction`
    order = idxs[np.argsort(direction * (bin_centers[idxs] - coarse_edge))]

    prev_below = None
    for i in order:
        above = counts[i] >= half
        if prev_below is not None and prev_below != above and above:
            return float(bin_centers[i])
        prev_below = above
    return None


def refine_edges(void, wall, xyz, search_m: float = 0.15, bin_m: float = 0.02):
    """Move each of the void's 4 edges to the density half-max crossing of
    the wall-band points' u (resp. z) marginal within +-search_m of the
    coarse edge; keep the coarse edge where no crossing is found.
    """
    xyz = np.asarray(xyz, dtype=float)
    p0, u_vec, length, normal_xy = _wall_frame(wall)
    thickness_m = float(wall.get("thickness_m", 0.1))
    band_m = thickness_m / 2.0 + 0.08

    rel = xyz[:, :2] - p0
    u = rel @ u_vec
    perp = rel @ normal_xy
    z = xyz[:, 2]
    band = np.abs(perp) <= band_m

    u0, u1, z0, z1 = void["u0"], void["u1"], void["z0"], void["z1"]

    # u marginal: points in the void's z-range (plus margin), histogram over u
    z_pad = search_m
    u_band_mask = band & (z >= z0 - z_pad) & (z <= z1 + z_pad)
    u_sel = u[u_band_mask]
    new_u0, new_u1 = u0, u1
    if u_sel.size:
        lo, hi = u0 - search_m, u1 + search_m
        n_bins = max(1, int(np.ceil((hi - lo) / bin_m)))
        counts, edges = np.histogram(u_sel, bins=n_bins, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2.0
        c0 = _half_max_crossing(centers, counts, u0, search_m, direction=-1)
        if c0 is not None:
            new_u0 = c0
        c1 = _half_max_crossing(centers, counts, u1, search_m, direction=+1)
        if c1 is not None:
            new_u1 = c1

    # z marginal: points in the void's u-range (plus margin), histogram over z
    u_pad = search_m
    z_band_mask = band & (u >= u0 - u_pad) & (u <= u1 + u_pad)
    z_sel = z[z_band_mask]
    new_z0, new_z1 = z0, z1
    if z_sel.size:
        lo, hi = z0 - search_m, z1 + search_m
        n_bins = max(1, int(np.ceil((hi - lo) / bin_m)))
        counts, edges = np.histogram(z_sel, bins=n_bins, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2.0
        c0 = _half_max_crossing(centers, counts, z0, search_m, direction=-1)
        if c0 is not None:
            new_z0 = c0
        c1 = _half_max_crossing(centers, counts, z1, search_m, direction=+1)
        if c1 is not None:
            new_z1 = c1

    out = dict(void)
    out["u0"], out["u1"], out["z0"], out["z1"] = new_u0, new_u1, new_z0, new_z1
    return out


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify_opening(void, crossing_us, z_floor: float, z_ceiling: float, priors,
                      exterior: bool = False) -> str:
    """Classify a gated void as "door" | "window" | "balcony_door" |
    "unknown_opening" per the brief's priority order:

      sill = void.z0 - z_floor; height = z1 - z0; width = u1 - u0;
      walked = any crossing u within the void's u-range.

      1. floor-touching (sill <= 0.15) AND (width >= balcony_min_w_m OR
         (walked AND exterior wall)) -> "balcony_door"
      2. floor-touching AND walked -> "door"
      3. floor-touching, NOT walked, height within door prior +- tol -> "door"
         (low confidence -- caller may want to inspect confidence separately)
      4. sill >= window_min_sill_m AND top below z_ceiling - 0.2 -> "window"
      5. else -> "unknown_opening"

    `exterior` (default False) mirrors the brief's "exterior wall" condition
    in the balcony_door rule; callers with interior/exterior wall
    classification available should pass it explicitly (detect_openings
    threads `exterior_flags` through here). Defaulting to False is
    deliberate and load-bearing: a walked, floor-touching, narrow (< the
    balcony width prior) opening -- an ordinary exterior door, like a front
    door -- must classify as "door", not "balcony_door", unless the caller
    positively asserts this wall is exterior AND wants the looser
    walked-implies-balcony heuristic applied. Without that guard every
    walked exterior door would misclassify as a balcony door.
    """
    sill = void["z0"] - z_floor
    height = void["z1"] - void["z0"]
    width = void["u1"] - void["u0"]
    walked = any(void["u0"] <= u <= void["u1"] for u in crossing_us)

    floor_touching = sill <= 0.15

    if floor_touching and (width >= priors["balcony_min_w_m"] or (walked and exterior)):
        return "balcony_door"
    if floor_touching and walked:
        return "door"
    if floor_touching and not walked and abs(height - priors["door_h_m"]) <= priors["door_h_tol_m"]:
        return "door"
    if sill >= priors["window_min_sill_m"] and void["z1"] < (z_ceiling - 0.2):
        return "window"
    return "unknown_opening"


# ---------------------------------------------------------------------------
# top-level orchestrator
# ---------------------------------------------------------------------------

def _seed_voids_from_crossings(wall, crossings, z_floor, priors):
    """Build seed void rects around every walkthrough crossing-u that landed
    inside a coverage hole of the wall's own `members` (i.e. a stretch of the
    wall's u-range where no member plane was ever detected -- a true opening,
    not merely a hole in the occupancy grid caused by a sparse/noisy patch of
    an otherwise-solid wall).
    """
    members = sorted(wall.get("members") or [])
    door_h = priors["door_h_m"]
    seeds = []
    for u in crossings:
        in_hole = True
        for (mu0, mu1) in members:
            if mu0 - 1e-6 <= u <= mu1 + 1e-6:
                in_hole = False
                break
        if not in_hole:
            continue
        seeds.append({
            "u0": float(u - 0.55), "u1": float(u + 0.55),
            "z0": float(z_floor), "z1": float(z_floor + door_h),
        })
    return seeds


def _union_voids(voids, seeds):
    """Union a list of occupancy-derived voids with seeded voids: a seed that
    overlaps an existing void is dropped (the real void already covers it);
    a seed with no overlap is appended as its own candidate.
    """
    out = list(voids)
    for seed in seeds:
        overlaps = False
        for v in voids:
            if seed["u0"] < v["u1"] and seed["u1"] > v["u0"] and seed["z0"] < v["z1"] and seed["z1"] > v["z0"]:
                overlaps = True
                break
        if not overlaps:
            out.append(seed)
    return out


def detect_openings(walls, xyz, trajectory, crossings, z_floor: float, z_ceiling: float,
                     priors, exterior_flags=None):
    """Per wall: occupancy -> voids, unioned with walkthrough-crossing seeds
    landing in a `members` coverage hole; gate every void against the
    trajectory (visibility_gate); refine survivors' edges against point
    density (refine_edges); classify (classify_opening); return
    {wall_idx: [{u0,u1,z0,z1,type,width_m,height_m,sill_m,walked,confidence}]}.

    Voids that fail the visibility gate are healing decisions: the wall
    stays solid there (they are simply omitted from the result), since a
    gate failure means no trajectory vertex ever had a clear line of sight
    through that void -- the occupancy hole is far more likely a furniture
    occlusion shadow than a real opening.

    `exterior_flags`: optional dict/list mapping wall_idx -> bool, used by
    classify_opening's balcony_door rule (exterior wall + walked). Defaults
    to False (not-asserted-exterior) for every wall if not supplied -- see
    classify_opening's own docstring for why False, not True, is the safe
    default: without positive evidence a wall is exterior, an ordinary
    walked door must not be reclassified as a balcony door just because it
    happens to sit on an outer wall.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=float)
    tree = cKDTree(xyz)

    out = {}
    for wi, wall in enumerate(walls):
        occ, u0, z0 = wall_occupancy(wall, xyz)
        cell_m = 0.03
        voids = find_voids(occ, u0, z0, cell_m=cell_m)

        wall_crossings_u = crossings.get(wi, []) if crossings else []
        seeds = _seed_voids_from_crossings(wall, wall_crossings_u, z_floor, priors)
        candidates = _union_voids(voids, seeds)

        exterior = False
        if exterior_flags is not None:
            if isinstance(exterior_flags, dict):
                exterior = exterior_flags.get(wi, False)
            else:
                exterior = exterior_flags[wi]

        results = []
        for void in candidates:
            if not visibility_gate(void, wall, trajectory, tree):
                continue
            refined = refine_edges(void, wall, xyz)
            opening_type = classify_opening(refined, wall_crossings_u, z_floor, z_ceiling, priors,
                                             exterior=exterior)
            sill = refined["z0"] - z_floor
            height = refined["z1"] - refined["z0"]
            width = refined["u1"] - refined["u0"]
            walked = any(refined["u0"] <= u <= refined["u1"] for u in wall_crossings_u)
            floor_touching = sill <= 0.15
            confidence = "high"
            if opening_type == "door" and floor_touching and not walked:
                confidence = "low"
            elif opening_type == "unknown_opening":
                confidence = "low"

            results.append({
                "u0": refined["u0"], "u1": refined["u1"],
                "z0": refined["z0"], "z1": refined["z1"],
                "type": opening_type,
                "width_m": width,
                "height_m": height,
                "sill_m": sill,
                "walked": walked,
                "confidence": confidence,
            })

        if results:
            out[wi] = results

    return out
