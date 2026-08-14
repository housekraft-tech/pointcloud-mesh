"""Faces: patches merged into the surfaces they actually belong to.

Plan 1 deliberately let one physical surface split into several patches --
region growing stops at anything it cannot cross, and a real scan of one room
produced 200 patches where perhaps 20 surfaces exist. Merging is how that is
repaired, and it is the one place in this pipeline where two surfaces are
allowed to become one, so the conditions are strict:

    coplanar within tolerance  AND  spatially adjacent

Coplanarity alone is not enough -- two walls on opposite sides of a building
can share a plane and are obviously not one surface. Adjacency alone is not
enough either, or a 75 mm step would merge into the wall behind it.

The merge tolerance is bounded above by the shallowest feature that must
survive. The golden room's groove is 12 mm deep, so the default sits at 5 mm.
Raising it past the groove depth erases the groove -- the exact failure this
rebuild exists to fix.

The merged plane is refitted from the union of the members' own points. It is
never an average of the input planes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_plane, plane_basis, plane_distance
from .graph import perpendicular_offset
from .patches import Patch


@dataclass
class Face:
    """One physical surface, backed by one or more patches."""

    face_id: int
    normal: np.ndarray                 # (3,) unit, canonically oriented
    d: float                           # normal @ x + d == 0 (ORIGIN-referenced)
    patch_ids: list[int]               # contributing patches, sorted
    point_idx: np.ndarray              # fitted members (drive the plane)
    loose_idx: np.ndarray              # recruited members (Task 3); never fitted
    n_points: int                      # len(point_idx)
    p95_residual_m: float              # over fitted members
    centroid: np.ndarray               # (3,)
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    role: Optional[str] = None         # set by classify (Task 6)
    interior_sign: Optional[int] = None  # set by occupancy (Task 5): +1 or -1

    def area_bound_m2(self) -> float:
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])

    def all_idx(self) -> np.ndarray:
        """Fitted plus recruited members, sorted. Every point this face owns."""
        return np.sort(np.concatenate([self.point_idx, self.loose_idx]))


def _bbox(xyz: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = xyz[idx]
    return pts.min(axis=0), pts.max(axis=0)


def _adjacent(xyz, a: Patch, b: Patch, gap: float) -> bool:
    """True when any point of a lies within `gap` of any point of b.

    A bounding-box test inflated by `gap` prunes the pair first; it can only
    reject pairs that are genuinely too far apart, never a true neighbour.
    """
    lo_a, hi_a = _bbox(xyz, a.point_idx)
    lo_b, hi_b = _bbox(xyz, b.point_idx)
    if np.any(lo_a - gap > hi_b) or np.any(lo_b - gap > hi_a):
        return False
    return cKDTree(xyz[a.point_idx]).count_neighbors(cKDTree(xyz[b.point_idx]), gap) > 0


def _finalise(face_id: int, xyz: np.ndarray, members: np.ndarray,
              patch_ids: list[int]) -> Face:
    pts = xyz[members]
    normal, d = fit_plane(pts)
    residual = np.abs(plane_distance(pts, normal, d))
    u, v = plane_basis(normal)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    us, vs = rel @ u, rel @ v
    return Face(
        face_id=face_id,
        normal=normal,
        d=d,
        patch_ids=sorted(patch_ids),
        point_idx=np.sort(members),
        loose_idx=np.zeros(0, dtype=np.int64),
        n_points=int(len(members)),
        p95_residual_m=float(np.percentile(residual, 95)),
        centroid=centroid,
        u_range=(float(us.min()), float(us.max())),
        v_range=(float(vs.min()), float(vs.max())),
    )


def merge_patches(patches: list[Patch], xyz: np.ndarray, config: dict) -> list[Face]:
    """Group patches into faces. Returns faces ordered by descending point count.

    Two patches join the same face when their normals agree, their PERPENDICULAR
    offset (never a difference of `d`) is within tolerance, and their points are
    spatially adjacent.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    dist_tol = float(config["face_merge_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))
    gap = float(config["face_merge_gap_m"])

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
            if perpendicular_offset(a, b) > dist_tol:
                continue
            if not _adjacent(xyz, a, b, gap):
                continue
            union(a.patch_id, b.patch_id)

    groups: dict[int, list[Patch]] = {}
    for p in ordered:
        groups.setdefault(find(p.patch_id), []).append(p)

    faces = []
    for root in sorted(groups):
        members = np.concatenate([p.point_idx for p in groups[root]])
        faces.append(_finalise(0, xyz, members, [p.patch_id for p in groups[root]]))

    faces.sort(key=lambda f: (-f.n_points, f.patch_ids[0]))
    for new_id, f in enumerate(faces):
        f.face_id = new_id
    return faces


def median_spacing(xyz: np.ndarray, sample_n: int = 50_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance -- the cloud's native point spacing.

    Sampled rather than exhaustive: the median is stable well below full size,
    and this is called once per run. Seeded for determinism.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    if len(xyz) > sample_n:
        rng = np.random.default_rng(seed)
        sample = xyz[np.sort(rng.choice(len(xyz), sample_n, replace=False))]
    else:
        sample = xyz
    dist, _ = cKDTree(sample).query(sample, k=2, workers=-1)
    return float(np.median(dist[:, 1]))


def apply_density_gate(
    faces: list[Face], xyz: np.ndarray, config: dict
) -> tuple[list[Face], list[Face]]:
    """Split faces into (kept, rejected) on in-plane fill ratio and area.

    Fill is the face's point count against the count a fully-sampled surface of
    the same extent would hold at the cloud's native spacing. A point count alone
    cannot distinguish a small dense feature from a large sparse chain -- and
    SLAM density falls off with range, so an absolute count is not portable
    across one scan, let alone between scans.

    The extent used here is a 2nd-98th percentile TRIM of the fitted members'
    in-plane coordinates, deliberately different from `area_bound_m2()` (a plain
    min/max over `u_range`/`v_range`). Region growing only requires 50 mm
    connectivity and 3 mm planarity, so a single stray member dragged a metre out
    inflates the plain bbox and deflates fill by the same factor -- easily enough
    to flip a genuinely dense wall from kept to rejected, which is a worse
    failure than admitting junk (the gate exists to stop concrete dust becoming
    objects, not to delete walls). `u_range`/`v_range`/`area_bound_m2()` on the
    Face itself are left untouched -- they are the face's true reported extent
    for downstream consumers -- so do not "helpfully" make the two consistent.

    Fill counts `f.n_points` (fitted members only, i.e. `f.point_idx`), never
    `f.all_idx()`. From Task 3 onward `loose_idx` holds recruited members
    attached under a looser tolerance than the fitted plane; deliberately, they
    cannot rescue a face whose own fitted points do not already support it.

    `median_spacing` is a single global figure over the whole cloud. SLAM
    density falls off with range, so a face far from the scanner is judged
    against a standard set mostly by near-scanner points, understating its
    fill -- an open calibration concern on real scans, not addressed here.

    Rejected faces are RETURNED, never dropped. The caller routes them to the
    scene's `unmodeled` set so a designer still sees that something is there.
    """
    spacing = median_spacing(xyz, seed=int(config["seed"]))
    min_fill = float(config["face_min_fill"])
    min_area = float(config["face_min_area_m2"])
    per_m2 = 1.0 / (spacing ** 2) if spacing > 0 else 0.0

    kept, rejected = [], []
    for f in sorted(faces, key=lambda g: g.face_id):
        area = f.area_bound_m2()
        if area < min_area:
            rejected.append(f)
            continue
        pts = xyz[f.point_idx]
        u, v = plane_basis(f.normal)
        rel = pts - f.centroid
        us, vs = rel @ u, rel @ v
        u_lo, u_hi = np.percentile(us, [2, 98])
        v_lo, v_hi = np.percentile(vs, [2, 98])
        trimmed_area = float((u_hi - u_lo) * (v_hi - v_lo))
        expected = trimmed_area * per_m2
        fill = (f.n_points / expected) if expected > 0 else 0.0
        (kept if fill >= min_fill else rejected).append(f)
    return kept, rejected
