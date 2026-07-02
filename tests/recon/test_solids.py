import numpy as np
import pytest

from scripts.floorplan_schema import Wall, Opening
from scripts.recon.schema import WallStep, Column, Beam, Plane
from scripts.recon.solids import (
    wall_to_solid,
    slab_to_solid,
    column_to_solid,
    beam_to_solid,
    cut_openings,
)


def _make_wall(p0=(0.0, 0.0), p1=(4.0, 0.0), thickness_m=0.2, floor_z_m=0.0, ceiling_z_m=2.7):
    """Build a floorplan_schema.Wall with placeholder values for the fields
    solids.py doesn't consume (plane_front/back, region_*), matching this
    repo's shared Wall representation (p0/p1 are the wall CENTERLINE --
    see floorplan_schema.py's docstring and floorplan_reconstruct.py's
    _walls_to_obj_mesh, which builds the wall box centered on p0-p1).
    """
    d = np.array(p1) - np.array(p0)
    length = float(np.linalg.norm(d))
    return Wall(
        wall_id="wall_000",
        p0=p0,
        p1=p1,
        length_m=length,
        thickness_m=thickness_m,
        thickness_source="measured_3d",
        plane_front=[0.0, 1.0, 0.0, 0.0],
        plane_back=[0.0, 1.0, 0.0, -thickness_m],
        origin_xyz=(p0[0], p0[1], floor_z_m),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        floor_z_m=floor_z_m,
        ceiling_z_m=ceiling_z_m,
        region_band_m=0.3,
        region_corner_margin_m=0.5,
    )


def test_wall_to_solid_single_step_is_watertight_box():
    wall = _make_wall()
    steps = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7)]

    solid = wall_to_solid(wall, steps)

    assert solid.is_watertight
    expected_volume = 4.0 * 0.2 * 2.7
    assert solid.volume == pytest.approx(expected_volume, rel=0.02)

    extents = solid.bounds[1] - solid.bounds[0]
    assert extents == pytest.approx([4.0, 0.2, 2.7], abs=0.02)


def test_wall_to_solid_two_steps_adds_relief_volume():
    wall = _make_wall()
    base_step = WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7)
    # A pillar/relief step: offset well beyond the base wall's half-thickness
    # (0.1 m) so the union genuinely gains volume, not just overlapping mass.
    pillar_step = WallStep(offset_m=0.15, u_min_m=1.5, u_max_m=2.5, z_min_m=0.0, z_max_m=2.7)

    single = wall_to_solid(wall, [base_step])
    stepped = wall_to_solid(wall, [base_step, pillar_step])

    assert stepped.is_watertight
    assert stepped.volume > single.volume * 1.05


def test_slab_to_solid_extrudes_expected_volume():
    plane = Plane(normal=(0.0, 0.0, 1.0), d=0.0, label="floor", inlier_idx=np.array([]))
    polygon = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
    thickness = 0.15

    solid = slab_to_solid(plane, polygon, thickness)

    assert solid.is_watertight
    expected_volume = 3.0 * 2.0 * thickness
    assert solid.volume == pytest.approx(expected_volume, rel=0.02)


def test_column_to_solid_extrudes_footprint():
    column = Column(
        column_id="column_000",
        footprint=[(0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)],
        z_min_m=0.0,
        z_max_m=2.7,
    )

    solid = column_to_solid(column)

    assert solid.is_watertight
    expected_volume = 0.3 * 0.3 * 2.7
    assert solid.volume == pytest.approx(expected_volume, rel=0.02)


def test_beam_to_solid_extrudes_centerline_box():
    beam = Beam(
        beam_id="beam_000",
        p0=(0.0, 0.0),
        p1=(5.0, 0.0),
        width_m=0.2,
        depth_m=0.3,
        z_min_m=2.4,
        z_max_m=2.7,
    )

    solid = beam_to_solid(beam)

    assert solid.is_watertight
    expected_volume = 5.0 * 0.2 * (2.7 - 2.4)
    assert solid.volume == pytest.approx(expected_volume, rel=0.02)


def test_cut_openings_reduces_volume_by_opening_size_and_stays_watertight():
    wall = _make_wall()
    steps = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7)]
    wall_solid = wall_to_solid(wall, steps)

    opening = Opening(
        opening_id="opening_000",
        wall_id=wall.wall_id,
        type="door",
        u_min_m=1.0,
        u_max_m=1.9,
        sill_m=0.0,
        height_m=2.1,
        width_m=0.9,
        edge_method="density_half_max",
        both_faces_confirmed=True,
    )

    cut = cut_openings(wall_solid, [opening], wall)

    assert cut.is_watertight
    removed = wall_solid.volume - cut.volume
    expected_removed = (opening.u_max_m - opening.u_min_m) * opening.height_m * wall.thickness_m
    assert removed == pytest.approx(expected_removed, rel=0.1)


def test_cut_openings_no_openings_returns_equivalent_solid():
    wall = _make_wall()
    steps = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7)]
    wall_solid = wall_to_solid(wall, steps)

    result = cut_openings(wall_solid, [], wall)

    assert result.is_watertight
    assert result.volume == pytest.approx(wall_solid.volume, rel=1e-6)


def test_cut_openings_two_openings_both_removed():
    wall = _make_wall(p1=(6.0, 0.0))
    steps = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7)]
    wall_solid = wall_to_solid(wall, steps)

    door = Opening(
        opening_id="opening_000", wall_id=wall.wall_id, type="door",
        u_min_m=0.5, u_max_m=1.4, sill_m=0.0, height_m=2.1, width_m=0.9,
        edge_method="density_half_max", both_faces_confirmed=True,
    )
    window = Opening(
        opening_id="opening_001", wall_id=wall.wall_id, type="window",
        u_min_m=3.5, u_max_m=4.7, sill_m=0.9, height_m=1.2, width_m=1.2,
        edge_method="density_half_max", both_faces_confirmed=True,
    )

    cut = cut_openings(wall_solid, [door, window], wall)

    assert cut.is_watertight
    removed = wall_solid.volume - cut.volume
    expected_removed = (
        (door.u_max_m - door.u_min_m) * door.height_m * wall.thickness_m
        + (window.u_max_m - window.u_min_m) * window.height_m * wall.thickness_m
    )
    assert removed == pytest.approx(expected_removed, rel=0.1)


def test_cut_openings_sill_is_relative_to_nonzero_floor_z_m():
    """Regression test for the `z_min = wall.floor_z_m + opening.sill_m`
    conversion in `_opening_cutter_box` (scripts/recon/solids.py). Every
    other test in this file builds walls via `_make_wall()`'s default
    floor_z_m=0.0, where `floor_z_m + opening.sill_m == opening.sill_m` --
    so none of them can distinguish the correct absolute-Z conversion from
    a buggy version that used `opening.sill_m` directly and silently
    dropped the `+ floor_z_m` term.

    This test uses floor_z_m=3.0 and an opening spanning the wall's FULL
    u-range with sill_m=0.0, i.e. its cutter box should start exactly at
    the wall's own bottom face (z=floor_z_m=3.0) and punch up 1.0 m. If
    `+ floor_z_m` were dropped, the cutter's z-range would fall at
    [0.0, 1.0] -- entirely below the wall's real z-range of [3.0, 5.7] --
    so the boolean difference would remove *nothing* and the wall would
    come back completely unchanged (same volume, same z_min=3.0). With the
    correct conversion, the bottom 1.0 m slab across the wall's full width
    is genuinely sliced off, so the solid's z_min bound moves up to 4.0 and
    its volume shrinks. Chosen over comparing two walls' bounds directly
    because a partial-width notch doesn't move the overall bounding box at
    all (the uncut portions of the wall still span the full height) --
    spanning the full u-range makes the bug visible in the simple
    `solid.bounds` check the rest of this file already uses.
    """
    floor_z_m = 3.0
    ceiling_z_m = 5.7
    wall = _make_wall(floor_z_m=floor_z_m, ceiling_z_m=ceiling_z_m)
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=4.0, z_min_m=floor_z_m, z_max_m=ceiling_z_m)
    ]
    wall_solid = wall_to_solid(wall, steps)

    opening = Opening(
        opening_id="opening_000",
        wall_id=wall.wall_id,
        type="door",
        u_min_m=0.0,
        u_max_m=4.0,
        sill_m=0.0,
        height_m=1.0,
        width_m=4.0,
        edge_method="density_half_max",
        both_faces_confirmed=True,
    )

    cut = cut_openings(wall_solid, [opening], wall)

    assert cut.is_watertight
    # A buggy cutter (missing `+ floor_z_m`) wouldn't overlap the wall's
    # real z-range [3.0, 5.7] at all, leaving volume/bounds unchanged.
    assert cut.volume < wall_solid.volume * 0.99
    assert cut.bounds[0][2] == pytest.approx(floor_z_m + opening.height_m, abs=0.02)
    assert cut.bounds[1][2] == pytest.approx(ceiling_z_m, abs=0.02)
