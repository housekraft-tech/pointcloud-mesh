import numpy as np

from rscene.config import merged_config
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box


def _room_points(spacing=0.02):
    floor = Box("floor", (0, 0, 0.0), (3, 2.5, 0.0)).sample_surface(spacing, faces=("z+",))
    ceil = Box("ceil", (0, 0, 2.75), (3, 2.5, 2.75)).sample_surface(spacing, faces=("z-",))
    wall = Box("wall", (0, 0, 0), (0, 2.5, 2.75)).sample_surface(spacing, faces=("x+",))
    return np.concatenate([floor, ceil, wall])


def _frame_from(xyz):
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())
    return estimate_frame(patches, merged_config())


def test_gravity_axis_is_recovered_from_the_floor():
    frame = _frame_from(_room_points())
    assert np.allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=1e-3)


def test_storey_levels_are_recovered():
    frame = _frame_from(_room_points())
    assert abs(frame.floor_z - 0.0) < 0.003
    assert abs(frame.ceiling_z - 2.75) < 0.003


def test_xy_rotation_is_reported_not_applied():
    """A yawed room reports its rotation; patch planes stay exactly as measured."""
    xyz = _room_points()
    yaw = np.radians(12.0)
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    rotated = xyz @ rot.T

    normals, curvature = estimate_normals(rotated, k=24)
    patches, _ = extract_patches(rotated, normals, curvature, merged_config())
    frame = estimate_frame(patches, merged_config())

    assert abs(frame.xy_rotation_deg - 12.0) < 1.0

    # the wall patch still sits where it was measured, un-rotated
    wall = max((p for p in patches if abs(p.normal[2]) < 0.2),
               key=lambda p: p.n_points)
    assert abs(abs(wall.normal[0]) - np.cos(yaw)) < 0.02


def test_frame_with_no_horizontal_patches_reports_none_levels():
    wall = Box("w", (0, 0, 0), (0, 2, 2.5)).sample_surface(0.02, faces=("x+",))
    frame = _frame_from(wall)
    assert frame.floor_z is None and frame.ceiling_z is None
