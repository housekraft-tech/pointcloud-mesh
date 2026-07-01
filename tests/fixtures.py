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


def _sample_horizontal(x_range, y_range, z_value, spacing=0.03, noise_std=0.002, rng=None):
    """Dense horizontal surface (floor/ceiling/beam-bottom) at constant z."""
    rng = rng or np.random.default_rng(0)
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel()
    yy = yy.ravel()
    zz = np.full(xx.shape, z_value) + rng.normal(0, noise_std, xx.size)
    return np.column_stack([xx, yy, zz])


def modular_house(rng=None):
    """Single-storey room with relief + a balcony + a neighbour blob.

    Returns (points (N,3), gps_time (N,), meta). Features (all rectangular):
      - 6 x 5 m room, floor z=0, ceiling z=2.7, exterior walls thickness 0.2
      - a 0.3 x 0.3 m pillar protruding from the west wall into the room
        (a WallStep in the west wall) full height
      - a beam under the ceiling spanning x, at y in [2.4,2.6], z in [2.4,2.7]
      - a door in the south wall (sill 0, height 2.1)
      - a window in the north wall (sill 0.9, top 2.1)
      - a wide balcony opening in the east wall (sill 0, height 2.1)
      - a neighbour facade blob ~10 m beyond the balcony (disconnected)
    """
    rng = rng or np.random.default_rng(7)
    z_full = (0.0, 2.7)
    faces = []

    door = [(2.0, 2.9, 0.0, 2.1)]          # south wall (along x), u in [2.0,2.9]
    window = [(4.0, 5.2, 0.9, 2.1)]        # north wall (along x)
    balcony = [(1.0, 4.0, 0.0, 2.1)]       # east wall (along y), u in [1.0,4.0]

    # South wall y=0 (double-sided) with door cut
    faces.append(_sample_face((0, 6), z_full, "y", -0.1, "x", door, rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 0.1, "x", door, rng=rng))
    # North wall y=5 with window cut
    faces.append(_sample_face((0, 6), z_full, "y", 4.9, "x", window, rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 5.1, "x", window, rng=rng))
    # West wall x=0 (double-sided)
    faces.append(_sample_face((0, 5), z_full, "x", -0.1, "y", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 0.1, "y", rng=rng))
    # East wall x=6 with balcony opening
    faces.append(_sample_face((0, 5), z_full, "x", 5.9, "y", balcony, rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 6.1, "y", balcony, rng=rng))

    # Pillar: footprint x in [0.1,0.4], y in [2.0,2.3], protrudes from west wall.
    # Exposed faces: front (x=0.4) and two sides (y=2.0, y=2.3).
    faces.append(_sample_face((2.0, 2.3), z_full, "x", 0.4, "y", rng=rng))   # front
    faces.append(_sample_face((0.1, 0.4), z_full, "y", 2.0, "x", rng=rng))   # side
    faces.append(_sample_face((0.1, 0.4), z_full, "y", 2.3, "x", rng=rng))   # side

    # Beam: spans x in [0,6], y in [2.4,2.6], z in [2.4,2.7].
    # Exposed: bottom (z=2.4) + two sides (y=2.4, y=2.6).
    faces.append(_sample_horizontal((0, 6), (2.4, 2.6), 2.4, rng=rng))       # bottom
    faces.append(_sample_face((0, 6), (2.4, 2.7), "y", 2.4, "x", rng=rng))   # side
    faces.append(_sample_face((0, 6), (2.4, 2.7), "y", 2.6, "x", rng=rng))   # side

    # Floor and ceiling
    faces.append(_sample_horizontal((0, 6), (0, 5), 0.0, rng=rng))
    faces.append(_sample_horizontal((0, 6), (0, 5), 2.7, rng=rng))

    real_points = np.vstack(faces)
    # walk-path sweep: time increases along y then x -> a serpentine inside
    gt = real_points
    gps_real = (gt[:, 1] / 5.0) * 100.0 + (gt[:, 0] / 6.0) * 10.0

    # Neighbour facade ~10 m beyond the balcony (east), multiple floor heights.
    n_neigh = 4000
    neigh = np.column_stack([
        rng.uniform(16.0, 17.0, n_neigh),
        rng.uniform(1.0, 4.0, n_neigh),
        rng.uniform(0.0, 8.0, n_neigh),
    ])
    gps_neigh = rng.uniform(40.0, 80.0, n_neigh)  # glimpsed while inside near balcony

    all_points = np.vstack([real_points, neigh])
    gps_time = np.concatenate([gps_real, gps_neigh])

    meta = {
        "footprint_m": (6.0, 5.0),
        "z_floor_m": 0.0,
        "z_ceiling_m": 2.7,
        "wall_thickness_m": 0.2,
        "exterior_walls": [
            {"centerline": ((0, 0), (6, 0))},
            {"centerline": ((6, 0), (6, 5))},
            {"centerline": ((6, 5), (0, 5))},
            {"centerline": ((0, 5), (0, 0))},
        ],
        "pillar": {"footprint": [(0.1, 2.0), (0.4, 2.0), (0.4, 2.3), (0.1, 2.3)],
                   "z": (0.0, 2.7), "protrusion_m": 0.3},
        "beam": {"p0": (0.0, 2.5), "p1": (6.0, 2.5), "width_m": 6.0,
                 "depth_m": 0.2, "z": (2.4, 2.7)},
        "openings": [
            {"wall": "south", "u_m": (2.0, 2.9), "sill_m": 0.0, "type": "door"},
            {"wall": "north", "u_m": (4.0, 5.2), "sill_m": 0.9, "type": "window"},
            {"wall": "east", "u_m": (1.0, 4.0), "sill_m": 0.0, "type": "balcony_door"},
        ],
        "neighbour_bbox": {"x": (16.0, 17.0), "y": (1.0, 4.0), "z": (0.0, 8.0)},
    }
    return all_points, gps_time, meta


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
