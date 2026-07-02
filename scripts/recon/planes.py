"""Plane detection: iterative RANSAC plane segmentation + DBSCAN face splitting.

label_plane is pure (unit-testable). detect_planes wraps Open3D
(segment_plane + cluster_dbscan) behind a lazy import.
"""
from __future__ import annotations

import numpy as np

from .schema import Plane


def label_plane(normal, centroid_z, z_floor, z_ceiling, horizontal_min: float = 0.85) -> str:
    """Classify a plane as 'floor', 'ceiling', or 'vertical'.

    Near-horizontal planes (|n_z| >= horizontal_min) are floor or ceiling by
    whether their centroid sits in the lower or upper half of the storey.
    """
    nz = abs(float(normal[2]))
    if nz >= horizontal_min:
        mid = 0.5 * (z_floor + z_ceiling)
        return "floor" if centroid_z < mid else "ceiling"
    return "vertical"


def detect_planes(
    xyz,
    normals=None,
    dist_thresh: float = 0.02,
    min_inliers: int = 1500,
    max_planes: int = 60,
    dbscan_eps: float = 0.15,
    dbscan_min: int = 50,
    z_floor: float = None,
    z_ceiling: float = None,
    horizontal_min: float = 0.85,
    ransac_iters: int = 1000,
    seed: int | None = 0,
) -> list:
    """Detect planar patches by peeling RANSAC planes, splitting each into
    spatially-disjoint faces with DBSCAN. Returns a list[Plane] with normalized
    plane coefficients, labels, and original-cloud inlier indices.

    normals is accepted for API symmetry but unused (segment_plane needs none).
    """
    import open3d as o3d

    if seed is not None:
        o3d.utility.random.seed(seed)

    xyz = np.asarray(xyz, dtype=float)
    if z_floor is None:
        z_floor = float(xyz[:, 2].min())
    if z_ceiling is None:
        z_ceiling = float(xyz[:, 2].max())

    remaining = np.arange(xyz.shape[0])
    planes = []
    for _ in range(max_planes):
        if remaining.size < min_inliers:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[remaining])
        model, inl = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=ransac_iters
        )
        inl = np.asarray(inl, dtype=np.int64)
        if inl.size < min_inliers:
            break

        a, b, c, d = model
        normal = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0:
            break
        normal /= norm
        d = float(d) / norm

        inl_global = remaining[inl]
        sub = xyz[inl_global]
        subpcd = o3d.geometry.PointCloud()
        subpcd.points = o3d.utility.Vector3dVector(sub)
        labels = np.asarray(subpcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min))
        for lab in np.unique(labels):
            if lab < 0:
                continue
            mask = labels == lab
            if int(mask.sum()) < dbscan_min:
                continue
            idx = inl_global[mask]
            cz = float(xyz[idx, 2].mean())
            planes.append(
                Plane(
                    normal=tuple(normal),
                    d=d,
                    label=label_plane(normal, cz, z_floor, z_ceiling, horizontal_min),
                    inlier_idx=idx,
                )
            )

        keep = np.ones(remaining.size, dtype=bool)
        keep[inl] = False
        remaining = remaining[keep]

    return planes
