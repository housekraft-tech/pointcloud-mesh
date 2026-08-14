"""Frame estimation: gravity axis, reported XY rotation, storey levels.

Everything here is MEASURED AND REPORTED. Nothing is rotated, snapped or
regularised. Downstream stages may read the frame to interpret results; none
of them may use it to move a surface.
"""
from __future__ import annotations

import numpy as np

from .patches import Patch
from .scene import Frame


def estimate_frame(patches: list[Patch], config: dict) -> Frame:
    """Derive the gravity axis, dominant XY rotation and storey levels.

    The gravity axis comes from the largest horizontal patch (the floor slab).
    The XY rotation is the circular mean of vertical-patch azimuths modulo 90
    degrees -- reported so a designer knows how far off-square the building is,
    never applied.
    """
    tol_cos = float(np.cos(np.radians(config["floor_normal_tol_deg"])))

    # Patches between roughly 11.5 and 78.5 deg off vertical (i.e. neither
    # near-horizontal within floor_normal_tol_deg nor near-vertical within
    # vertical_normal_max_z) fall into neither set -- four such patches exist
    # on the real crop.
    vertical_max_z = float(config["vertical_normal_max_z"])
    horizontal = [p for p in patches if abs(float(p.normal[2])) >= tol_cos]
    vertical = [p for p in patches if abs(float(p.normal[2])) < vertical_max_z]

    if horizontal:
        floor_like = max(horizontal, key=lambda p: p.n_points)
        z_axis = floor_like.normal.copy()
        if z_axis[2] < 0:
            z_axis = -z_axis
    else:
        z_axis = np.array([0.0, 0.0, 1.0])

    floor_z: float | None = None
    ceiling_z: float | None = None
    if horizontal:
        heights = sorted(float(p.centroid[2]) for p in horizontal)
        floor_z, ceiling_z = heights[0], heights[-1]
        if ceiling_z == floor_z:
            ceiling_z = None

    # circular mean of azimuths folded into [0, 90): the building's squareness
    xy_rotation_deg = 0.0
    if vertical:
        azimuths = np.array([
            np.degrees(np.arctan2(float(p.normal[1]), float(p.normal[0])))
            for p in vertical
        ])
        folded = np.radians((azimuths % 90.0) * 4.0)     # 90 deg -> full circle
        weights = np.array([p.n_points for p in vertical], dtype=np.float64)
        mean_angle = np.arctan2(
            float(np.sum(weights * np.sin(folded))),
            float(np.sum(weights * np.cos(folded))),
        )
        xy_rotation_deg = float((np.degrees(mean_angle) / 4.0) % 90.0)

    return Frame(
        z_axis=[float(x) for x in z_axis],
        xy_rotation_deg=xy_rotation_deg,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
    )
