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


def _finalise(patch_id: int, xyz: np.ndarray, members: np.ndarray) -> Patch:
    """Fit the final plane and measure the patch's residual and extent."""
    pts = xyz[members]
    normal, d = fit_plane(pts)
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
    refit_interval = int(config["refit_interval"])

    tree = cKDTree(xyz)
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int64)
    patches: list[Patch] = []

    # flattest points first: seeds land mid-face, never on an edge
    seed_order = np.argsort(curvature, kind="stable")

    for seed in seed_order:
        if labels[seed] != -1:
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
            # sorted() keeps neighbour visit order deterministic
            for j in sorted(tree.query_ball_point(xyz[current], radius)):
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
                    plane_n, plane_d = fit_plane(xyz[np.asarray(members)])
                    since_refit = 0

        member_arr = np.asarray(members, dtype=np.int64)
        if len(member_arr) < min_points:
            labels[member_arr] = -1        # release; may join a later patch
            continue

        patches.append(_finalise(pending_id, xyz, member_arr))

    return patches, labels


def unassigned_count(labels: np.ndarray) -> int:
    """Number of points that joined no patch. Reported, never dropped."""
    return int(np.count_nonzero(np.asarray(labels) == -1))
