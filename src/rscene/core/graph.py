"""Patch relationships: coplanarity classes, adjacency, intersection lines.

Coplanarity is RECORDED, never applied. Two wall segments either side of a
door belong to one wall, and saying so is useful -- but their individually
fitted planes are left exactly as measured. Averaging them would be a global
snap by another name.

Intersection lines are how sharp edges get computed later: an edge is where
two fitted planes meet, never a polyline meshed from points.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .patches import Patch


def coplanarity_classes(patches: list[Patch], config: dict) -> list[list[int]]:
    """Group patches that lie on the same plane within tolerance.

    Returns lists of patch_id, each sorted ascending, the outer list sorted by
    first element. Patch planes are not modified.
    """
    dist_tol = float(config["coplanar_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["coplanar_angle_tol_deg"])))

    ordered = sorted(patches, key=lambda p: p.patch_id)
    parent = {p.patch_id: p.patch_id for p in ordered}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if abs(float(a.normal @ b.normal)) < cos_tol:
                continue
            if abs(a.d - b.d) > dist_tol:
                continue
            union(a.patch_id, b.patch_id)

    groups: dict[int, list[int]] = {}
    for p in ordered:
        groups.setdefault(find(p.patch_id), []).append(p.patch_id)
    return sorted([sorted(v) for v in groups.values()])


def perpendicular_offset(a: Patch, b: Patch) -> float:
    """Perpendicular distance between two parallel patches, centroid to centroid.

    `Patch.d` is the plane's distance from the WORLD ORIGIN. Differencing two
    `d` values to measure the gap between two surfaces is a trap: any error in
    a patch's fitted normal gets multiplied by that patch's distance from the
    origin (a lever arm), not by the small distance actually being measured.
    A 1 degree tilt on a patch 1.6 m from the origin costs ~20 mm of spurious
    offset even though the two surfaces are exactly where they should be --
    this is exactly what corrupted the golden room's step- and switch-box
    measurements before this helper existed (see task-10-report.md).

    This measures the gap the way it should be measured: along one patch's
    own normal, between the two patches' own centroids, which is invariant to
    where the world origin happens to sit. `abs()` is applied so canonical
    normal orientation (Task 6 review) never flips the sign.
    """
    return abs(float(a.normal @ (b.centroid - a.centroid)))


def intersection_line(a: Patch, b: Patch, config: dict | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Line where two planes meet, as (point, unit direction).

    Returns None when the planes are parallel (within angle tolerance) and
    therefore cannot provide a numerically stable intersection line.
    """
    if config is None:
        from ..config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    direction = np.cross(a.normal, b.normal)
    norm = float(np.linalg.norm(direction))

    # norm = sin(angle) between normals. Guard against numerical instability
    # by rejecting planes closer than min_intersection_angle_deg.
    min_sin_angle = float(np.sin(np.radians(config["min_intersection_angle_deg"])))
    if norm < min_sin_angle:
        return None
    direction = direction / norm

    # pick the point on the line closest to the origin
    matrix = np.vstack([a.normal, b.normal, direction])
    rhs = np.array([-a.d, -b.d, 0.0], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    return point, direction


def patch_adjacency(
    patches: list[Patch], xyz: np.ndarray, config: dict
) -> list[tuple[int, int]]:
    """Pairs of patches with supporting points within adjacency_radius_m.

    Returns sorted (lo, hi) patch_id pairs.
    """
    radius = float(config["adjacency_radius_m"])
    xyz = np.asarray(xyz, dtype=np.float64)

    ordered = sorted(patches, key=lambda q: q.patch_id)
    owner = {}
    clouds = []
    bounds = []
    for idx, p in enumerate(ordered):
        owner[idx] = p.patch_id
        cloud = xyz[p.point_idx]
        clouds.append(cloud)
        # Compute axis-aligned bounding box for this patch
        if len(cloud) > 0:
            lo = np.min(cloud, axis=0)
            hi = np.max(cloud, axis=0)
        else:
            lo = hi = p.centroid
        bounds.append((lo, hi))

    pairs: set[tuple[int, int]] = set()
    trees = [cKDTree(c) for c in clouds]
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            # Prefilter: check if bounding boxes (inflated by radius) overlap
            lo_i, hi_i = bounds[i]
            lo_j, hi_j = bounds[j]
            # Inflate by radius in each direction
            lo_i_inflated = lo_i - radius
            hi_i_inflated = hi_i + radius
            lo_j_inflated = lo_j - radius
            hi_j_inflated = hi_j + radius
            # Check for axis-aligned overlap
            if not (np.all(hi_i_inflated >= lo_j_inflated) and np.all(hi_j_inflated >= lo_i_inflated)):
                continue
            # Passed prefilter: do expensive tree-vs-tree query
            if trees[i].count_neighbors(trees[j], radius) > 0:
                pairs.add((min(owner[i], owner[j]), max(owner[i], owner[j])))

    return sorted(pairs)
