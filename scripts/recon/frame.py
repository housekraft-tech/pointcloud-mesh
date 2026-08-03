"""Coordinate framing: normal estimation, dominant-direction (Manhattan)
estimation, and axis alignment.

dominant_axes / axis_align are pure numpy (unit-testable). estimate_normals
wraps Open3D behind a lazy import.
"""
from __future__ import annotations

from typing import NamedTuple

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


class FrameResidual(NamedTuple):
    """How well a single Manhattan grid actually describes the walls.

    All angles in degrees. `theta_deg` is the grid's orientation offset -- the
    same quantity dominant_axes rotates out -- and the rest describe how much
    the walls disagree with that grid once it is applied.
    """

    theta_deg: float
    theta_stderr_deg: float     # uncertainty of the grid orientation itself
    dispersion_deg: float       # circular spread of wall headings about the grid
    p50_dev_deg: float          # median |deviation| of a wall from its axis
    p90_dev_deg: float
    max_dev_deg: float
    frac_off_grid: float        # fraction deviating by more than off_grid_deg
    n: int


def deviation_mm(dev_deg, span_m: float = 10.0) -> float:
    """Lateral error a given angular deviation produces over `span_m`.

    This is the only form of the residual that is directly comparable with the
    rest of the error budget: an angle is not an error until it is multiplied
    by a lever arm.
    """
    return float(np.tan(np.deg2rad(float(dev_deg))) * span_m * 1000.0)


def axis_residuals(normals, up=(0.0, 0.0, 1.0), horizontal_max: float = 0.5,
                   off_grid_deg: float = 5.0) -> FrameResidual | None:
    """Measure how well the walls fit ONE Manhattan grid.

    dominant_axes returns a rotation but no indication of how good it is, so a
    building that is 0.5 degrees out of square -- 87mm over a 10m span -- looks
    exactly like one that is perfectly square. This reports the residual so the
    snap's cost is visible instead of assumed.

    A high `frac_off_grid` or `dispersion_deg` means the single-grid assumption
    is wrong (a wing at an odd angle, a curved or splayed wall). In that case
    the circular mean is being pulled by walls it does not describe, and the
    grid orientation itself is suspect -- not merely imprecise.

    Returns None when there are no wall normals to measure.
    """
    n = np.asarray(normals, dtype=float)
    up = np.asarray(up, dtype=float)
    nh = n[np.abs(n @ up) < horizontal_max]
    if nh.shape[0] == 0:
        return None

    # Same estimator as dominant_axes: headings mod 90 deg, circular mean taken
    # over the pi/2 period by mapping it to 2*pi (x4).
    ang = np.arctan2(nh[:, 1], nh[:, 0]) % (np.pi / 2.0)
    phase = 4.0 * ang
    C, S = np.cos(phase).mean(), np.sin(phase).mean()
    mean_phase = np.arctan2(S, C)
    theta = (mean_phase / 4.0) % (np.pi / 2.0)

    # Resultant length -> circular spread. R near 1 means the walls really do
    # share one grid; R near 0 means the mean is meaningless.
    R = float(np.hypot(C, S))
    R = min(max(R, 1e-12), 1.0 - 1e-12)
    disp = float(np.sqrt(-2.0 * np.log(R))) / 4.0            # radians, /4 undoes the x4
    se = float(np.sqrt((1.0 - R * R) / (nh.shape[0] * R * R))) / 4.0

    # Signed deviation of each wall from its nearest grid axis, in (-45, 45].
    dev = np.degrees((ang - theta + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
    adev = np.abs(dev)

    return FrameResidual(
        theta_deg=float(np.degrees(theta)),
        theta_stderr_deg=float(np.degrees(se)),
        dispersion_deg=float(np.degrees(disp)),
        p50_dev_deg=float(np.percentile(adev, 50)),
        p90_dev_deg=float(np.percentile(adev, 90)),
        max_dev_deg=float(adev.max()),
        frac_off_grid=float((adev > off_grid_deg).mean()),
        n=int(nh.shape[0]),
    )
