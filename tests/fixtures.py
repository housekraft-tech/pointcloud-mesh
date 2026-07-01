"""Synthetic multi-room house fixture for validating floorplan_geometry.py.

Confirmed via manual prototype run to exercise: a T-junction (partition
meets both exterior walls), a floor-to-ceiling door (sill=0, stresses the
opening flood-fill's floor-boundary handling), and a mid-wall window
(stresses the standard enclosed-void case), plus a sparse stray-point tail
matching the SLAM drift pattern seen in koushikexport.las/mujammelexport.las.
"""
import numpy as np


def _sample_face(u_range, z_range, offset_axis, offset_value, along_axis,
                  exclude_rects=None, spacing=0.016, noise_std=0.002, rng=None):
    rng = rng or np.random.default_rng(0)
    exclude_rects = exclude_rects or []
    u0, u1 = u_range
    z0, z1 = z_range
    us = np.arange(u0, u1, spacing)
    zs = np.arange(z0, z1, spacing)
    uu, zz = np.meshgrid(us, zs)
    uu = uu.ravel() + rng.normal(0, spacing * 0.2, uu.size)
    zz = zz.ravel() + rng.normal(0, spacing * 0.2, zz.size)

    keep = np.ones(uu.shape, dtype=bool)
    for (ru0, ru1, rz0, rz1) in exclude_rects:
        inside = (uu >= ru0) & (uu <= ru1) & (zz >= rz0) & (zz <= rz1)
        keep &= ~inside
    uu, zz = uu[keep], zz[keep]

    offset = offset_value + rng.normal(0, noise_std, uu.size)
    pts = np.zeros((len(uu), 3))
    if along_axis == "x":
        pts[:, 0] = uu
        pts[:, 1] = offset
    else:
        pts[:, 1] = uu
        pts[:, 0] = offset
    pts[:, 2] = zz
    return pts


def two_room_house(rng=None):
    """Returns (points (N,3), ground_truth dict).

    ground_truth["exterior_walls"]: list of {centerline: ((x0,y0),(x1,y1)), thickness_m}
    ground_truth["partition_walls"]: same shape, thickness 0.1
    ground_truth["openings"]: list of {wall, width_m, height_m, sill_m, type}
    """
    rng = rng or np.random.default_rng(42)
    z_full = (0.0, 2.7)
    faces = []

    window = [(4.0, 5.2, 0.9, 2.1)]
    faces.append(_sample_face((0, 6), z_full, "y", -0.1, "x", window, rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 0.1, "x", window, rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 5.9, "y", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 6.1, "y", rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 4.9, "x", rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 5.1, "x", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 0.1, "y", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", -0.1, "y", rng=rng))

    door = [(2.0, 2.9, 0.0, 2.1)]
    faces.append(_sample_face((0, 5), z_full, "x", 2.95, "y", door, rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 3.05, "y", door, rng=rng))

    real_points = np.vstack(faces)

    n_stray = 500
    stray = np.column_stack([
        rng.uniform(-10, 15, n_stray),
        rng.uniform(-20, 25, n_stray),
        rng.uniform(-5, 15, n_stray),
    ])

    all_points = np.vstack([real_points, stray])

    ground_truth = {
        "exterior_walls": [
            {"centerline": ((0, 0), (6, 0)), "thickness_m": 0.2},
            {"centerline": ((6, 0), (6, 5)), "thickness_m": 0.2},
            {"centerline": ((6, 5), (0, 5)), "thickness_m": 0.2},
            {"centerline": ((0, 5), (0, 0)), "thickness_m": 0.2},
        ],
        "partition_walls": [
            {"centerline": ((3, 0), (3, 5)), "thickness_m": 0.1},
        ],
        "openings": [
            {"wall": "south", "width_m": 1.2, "height_m": 1.2, "sill_m": 0.9, "type": "window"},
            {"wall": "partition", "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0, "type": "door"},
        ],
    }
    return all_points, ground_truth
