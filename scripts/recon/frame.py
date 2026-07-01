"""Coordinate framing: normal estimation, dominant-direction (Manhattan)
estimation, and axis alignment.

dominant_axes / axis_align are pure numpy (unit-testable). estimate_normals
wraps Open3D behind a lazy import.
"""
from __future__ import annotations

import numpy as np

from .schema import ScanData


def estimate_normals(xyz, radius: float = 0.06, max_nn: int = 30) -> np.ndarray:
    """Estimate per-point normals with Open3D (hybrid KD-tree search)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.asarray(pcd.normals)


def dominant_axes(normals, up=(0.0, 0.0, 1.0), horizontal_max: float = 0.5) -> np.ndarray:
    """Return a Z-rotation R aligning the dominant wall direction to the axes.

    Uses wall (near-vertical-surface) normals: their horizontal headings, taken
    mod 90 degrees, cluster at the building's orientation offset theta. R rotates
    the cloud by -theta so walls become axis-aligned (Manhattan). Returns a 3x3
    rotation about Z; identity if no wall normals are found.
    """
    n = np.asarray(normals, dtype=float)
    up = np.asarray(up, dtype=float)
    is_wall = np.abs(n @ up) < horizontal_max
    nh = n[is_wall]
    if nh.shape[0] == 0:
        return np.eye(3)

    # Wall headings modulo 90 deg cluster at the building's orientation offset.
    # Estimate it with a circular mean over the pi/2 period (period -> 2*pi via
    # x4), which is unbiased and wraps correctly (0 deg == 90 deg).
    ang = np.arctan2(nh[:, 1], nh[:, 0]) % (np.pi / 2.0)
    phase = 4.0 * ang
    mean_phase = np.arctan2(np.sin(phase).mean(), np.cos(phase).mean())
    theta = (mean_phase / 4.0) % (np.pi / 2.0)

    c, s = np.cos(-theta), np.sin(-theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_align(scan: ScanData, R) -> ScanData:
    """Rotate the cloud by R (points: xyz @ R.T). Returns a new ScanData."""
    R = np.asarray(R, dtype=float)
    rotated = scan.xyz @ R.T
    return ScanData(xyz=rotated, gps_time=scan.gps_time, rgb=scan.rgb, intensity=scan.intensity)
