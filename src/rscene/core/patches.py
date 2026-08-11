"""Region-growing planar patch extraction.

Whole-cloud RANSAC is what fuses a 75 mm step into its parent wall: one
dominant plane wins and swallows its neighbours. Growing regions from
low-curvature seeds, gated on BOTH normal agreement and spatial connectivity,
makes each formwork face its own patch by construction.

Each patch's plane is fitted only to its own points. Nothing here consults a
global frame, and no patch is ever merged into another.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_plane, plane_basis, plane_distance


@dataclass
class Patch:
    """A planar surface fitted to its own supporting points."""

    patch_id: int
    normal: np.ndarray                 # (3,) unit, canonically oriented
    d: float                           # normal @ x + d == 0
    point_idx: np.ndarray              # indices into the source cloud
    n_points: int
    p95_residual_m: float
    centroid: np.ndarray               # (3,)
    u_range: tuple[float, float]       # in-plane extent along plane_basis u
    v_range: tuple[float, float]       # in-plane extent along plane_basis v

    def area_bound_m2(self) -> float:
        """Area of the patch's in-plane bounding rectangle."""
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])


# Fewer than this many curvature-clean members and a fit on clean points
# alone would be numerically thin (fit_plane itself refuses below 3); fall
# back to fitting on every member rather than raising or discarding the
# patch. Below this floor there usually isn't enough of a "clean core" for
# the distinction to matter anyway.
_MIN_CLEAN_FOR_FIT = 3


def _fit_on_clean_members(
    xyz: np.ndarray, curvature: np.ndarray, members: np.ndarray, max_curvature: float
) -> tuple[np.ndarray, float]:
    """Fit a plane using only members whose curvature is within the seed gate.

    Contaminated (high-curvature) members are still patch members -- they are
    labelled, counted, and included in the reported residual -- they just
    don't get a vote on where the plane sits, so one edge-blended point can't
    drag the whole patch's fitted offset. Falls back to fitting on every
    member when too few clean ones are available.
    """
    clean = members[curvature[members] <= max_curvature]
    fit_pts = clean if len(clean) >= _MIN_CLEAN_FOR_FIT else members
    return fit_plane(xyz[fit_pts])


def _finalise(
    patch_id: int, xyz: np.ndarray, curvature: np.ndarray, members: np.ndarray, max_curvature: float
) -> Patch:
    """Fit the final plane (clean members only) and measure the patch's
    residual and extent over ALL members -- the residual is a truthfulness
    diagnostic and must reflect how well the plane explains every point the
    patch claims, edges included, not just the points that founded the fit.
    """
    normal, d = _fit_on_clean_members(xyz, curvature, members, max_curvature)
    pts = xyz[members]
    residual = np.abs(plane_distance(pts, normal, d))
    u, v = plane_basis(normal)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    us, vs = rel @ u, rel @ v
    return Patch(
        patch_id=patch_id,
        normal=normal,
        d=d,
        point_idx=np.sort(members),
        n_points=int(len(members)),
        p95_residual_m=float(np.percentile(residual, 95)),
        centroid=centroid,
        u_range=(float(us.min()), float(us.max())),
        v_range=(float(vs.min()), float(vs.max())),
    )


def extract_patches(
    xyz: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    config: dict,
) -> tuple[list[Patch], np.ndarray]:
    """Grow planar patches from low-curvature seeds.

    Returns (patches, labels). labels is (N,) int64 with the patch_id owning
    each point, or -1 where a point joined no patch. Unassigned points are the
    caller's to report -- they are never silently dropped.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64)

    tau = float(config["tau_fit_m"])
    cos_tol = float(np.cos(np.radians(config["patch_angle_tol_deg"])))
    radius = float(config["patch_connect_radius_m"])
    min_points = int(config["min_patch_points"])
    max_curvature = float(config["patch_max_curvature"])
    refit_interval = int(config["refit_interval"])

    tree = cKDTree(xyz)
    n = len(xyz)
    k_query = min(int(config["patch_neighbor_k"]), n)
    labels = np.full(n, -1, dtype=np.int64)
    patches: list[Patch] = []

    # Vectorised k-nearest precompute (Task 6 review finding 1): a single
    # batched call replaces one Python-level query_ball_point per point.
    # nbr_dist is ascending per row, so once its last entry already exceeds
    # `radius` we know every point within `radius` was captured; only when a
    # neighbourhood is denser than k_query do we fall back to an exact
    # radius query for that one point, which keeps results identical to an
    # unbounded radius search regardless of local density.
    nbr_dist, nbr_idx = tree.query(xyz, k=k_query, workers=-1)
    if k_query == 1:
        nbr_dist = nbr_dist[:, None]
        nbr_idx = nbr_idx[:, None]

    def neighbours_within_radius(i: int) -> np.ndarray:
        row_dist = nbr_dist[i]
        if k_query < n and row_dist[-1] < radius:
            cand = tree.query_ball_point(xyz[i], radius)
        else:
            cand = nbr_idx[i][row_dist <= radius]
        return np.sort(np.asarray(cand, dtype=np.int64))

    # flattest points first: seeds land mid-face, never on an edge
    seed_order = np.argsort(curvature, kind="stable")

    for seed in seed_order:
        if labels[seed] != -1:
            continue
        if curvature[seed] > max_curvature:
            # Edge/corner points blend normals from two faces (Task 6 review
            # finding 2); refusing to SEED from them stops a contaminated
            # point from ever founding its own spurious patch. This gate is
            # deliberately seed-only (Task 10): a high-curvature point may
            # still be RECRUITED below, once a patch's plane is already
            # established from a clean seed, subject to the existing
            # normal-agreement and tau_fit gates -- that's what lets a small,
            # finely-sampled feature (e.g. a recessed switch box) reach
            # min_patch_points at all. Recruited contaminated points don't
            # get a vote on the plane, though: _fit_on_clean_members excludes
            # them from every fit (periodic refit and the final one), so one
            # edge-blended recruit can't drag the whole patch's offset the
            # way it could before that split existed. Left at -1, counted
            # via unassigned_count, never dropped.
            continue

        pending_id = len(patches)
        members = [int(seed)]
        labels[seed] = pending_id

        plane_n = normals[seed].copy()
        plane_d = float(-plane_n @ xyz[seed])

        stack = [int(seed)]
        since_refit = 0
        while stack:
            current = stack.pop()
            for j in neighbours_within_radius(current):
                if labels[j] != -1:
                    continue
                if abs(float(normals[j] @ plane_n)) < cos_tol:
                    continue
                if abs(float(xyz[j] @ plane_n + plane_d)) > tau:
                    continue

                labels[j] = pending_id
                members.append(j)
                stack.append(j)

                since_refit += 1
                if since_refit >= refit_interval:
                    plane_n, plane_d = _fit_on_clean_members(
                        xyz, curvature, np.asarray(members, dtype=np.int64), max_curvature)
                    since_refit = 0

        member_arr = np.asarray(members, dtype=np.int64)
        if len(member_arr) < min_points:
            labels[member_arr] = -1        # release; may join a later patch
            continue

        patches.append(_finalise(pending_id, xyz, curvature, member_arr, max_curvature))

    return patches, labels


def unassigned_count(labels: np.ndarray) -> int:
    """Number of points that joined no patch. Reported, never dropped."""
    return int(np.count_nonzero(np.asarray(labels) == -1))
