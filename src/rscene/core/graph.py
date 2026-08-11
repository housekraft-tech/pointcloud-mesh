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


def intersection_line(a: Patch, b: Patch) -> tuple[np.ndarray, np.ndarray] | None:
    """Line where two planes meet, as (point, unit direction).

    Returns None when the planes are parallel and therefore never meet.
    """
    direction = np.cross(a.normal, b.normal)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
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

    owner = {}
    clouds = []
    for p in sorted(patches, key=lambda q: q.patch_id):
        owner[len(clouds)] = p.patch_id
        clouds.append(xyz[p.point_idx])

    pairs: set[tuple[int, int]] = set()
    trees = [cKDTree(c) for c in clouds]
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            if trees[i].count_neighbors(trees[j], radius) > 0:
                pairs.add((min(owner[i], owner[j]), max(owner[i], owner[j])))

    return sorted(pairs)
