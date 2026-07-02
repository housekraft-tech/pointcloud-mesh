from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Regression tests for the group_wall_runs <-> wall_to_solid offset-sign bug.
#
# structure.group_wall_runs measures WallStep.offset_m along a WORLD axis
# (whichever of ex=(1,0,0)/ey=(0,1,0) is closer to the run's own normal --
# see structure._plane_axis_stats and group_wall_runs's own docstring: "the
# main step always comes back with offset_m == 0.0, and every other step's
# offset_m is its signed distance from the main step" along that same
# world-positive axis). The old `_step_box` instead measured offset_m along
# a WALL-LOCAL `n_hat = (-u_hat.y, u_hat.x)` rotated off this particular
# wall's own, arbitrary p0->p1 ordering. For a wall whose own p0->p1
# happens to run in the world-negative direction along its snapped axis,
# those two conventions are exact sign opposites -- so a relief step (e.g.
# a pillar 0.3 m proud of its wall) got built on the WRONG side of the
# wall. The old `_step_box` also built every step as an independently
# thickness_m-wide slab centered on its own offset, so a relief step whose
# offset exceeded half the wall thickness (e.g. the pillar's 0.3 m, versus
# a typical ~0.1 m half-thickness) floated as a disconnected body instead
# of fusing into the main wall.
#
# NOTE on the wall run's own "direction" label: structure.py's own
# docstring for group_wall_runs's return value is explicit -- "direction:
# 'x'|'y' -- which dominant axis the run's normal is closest to" -- and
# this is verified empirically below (and against the real modular_house()
# west wall in test_wall_to_solid_on_real_west_wall_run_from_modular_house):
# a run with direction="x" has its `normal` along world X = (1,0,0), and
# (since u_vec is always the OTHER axis) its own p0->p1 travels along
# world Y. A wall running north-south (along Y) at x~=0 -- like
# modular_house()'s west wall -- is exactly this "direction='x'" case.
# ---------------------------------------------------------------------------

def _adapt_run_to_wall(run: dict, thickness_m: float = 0.2, floor_z_m: float = 0.0):
    """Minimal Wall-shaped adapter around a group_wall_runs run dict.

    wall_to_solid's eventual Task 17 wall type isn't finalized (see
    solids.py's module docstring: "these functions only touch p0, p1,
    thickness_m ..., and floor_z_m ... so any future wall object exposing
    at least those attributes will work unmodified"). group_wall_runs's
    run dict already carries p0/p1 (see its own docstring) but not
    thickness_m/floor_z_m (a run has no notion of assumed material
    thickness), so this fills those in from the caller -- exactly the kind
    of small adapter Task 17 will need to bridge group_wall_runs's output
    to wall_to_solid's input.
    """
    return SimpleNamespace(p0=run["p0"], p1=run["p1"], thickness_m=thickness_m, floor_z_m=floor_z_m)


def test_wall_to_solid_from_group_wall_runs_style_run_is_one_connected_solid_on_correct_side():
    """Synthetic run dict matching group_wall_runs's REAL output shape (see
    its docstring's "Returns a list[dict]..." section): direction="x" (so
    the run's own normal is world X, and its centerline runs along world Y
    -- see the module-level note above), a main step at offset_m=0.0, and a
    pillar-like relief step at offset_m=+0.3 (simulating a pillar 0.3 m
    proud of the wall, spanning a short sub-range of the wall's own u).
    """
    run = dict(
        direction="x",
        normal=(1.0, 0.0, 0.0),
        offset_m=0.0,
        p0=(0.0, 0.0),
        p1=(0.0, 5.0),
        steps=[
            WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
            WallStep(offset_m=0.3, u_min_m=2.0, u_max_m=2.3, z_min_m=0.0, z_max_m=2.7),
        ],
    )
    wall = _adapt_run_to_wall(run, thickness_m=0.2)

    solid = wall_to_solid(wall, run["steps"])

    # (a) one connected, watertight body -- not 3 disconnected pieces.
    assert solid.is_watertight
    bodies = solid.split(only_watertight=False)
    assert len(bodies) == 1, f"expected one connected wall+pillar solid, got {len(bodies)} bodies"

    # (b) the relief step is on the CORRECT side. This wall's centerline
    # runs along world Y (p0=(0,0), p1=(0,5)), so "u" is the Y coordinate
    # and the offset/perpendicular axis is world X. A +0.3 offset step
    # must therefore push the solid further in +X specifically over the
    # pillar's own u-range (y in [2.0, 2.3]), not elsewhere along the wall.
    verts = solid.vertices
    u_vals = verts[:, 1]
    in_pillar_u = (u_vals >= 1.95) & (u_vals <= 2.35)
    outside_pillar_u = (u_vals <= 1.5) | (u_vals >= 2.8)
    assert in_pillar_u.any() and outside_pillar_u.any()
    max_x_in_pillar_u = verts[in_pillar_u, 0].max()
    max_x_outside = verts[outside_pillar_u, 0].max()
    assert max_x_in_pillar_u > max_x_outside + 0.2, (
        f"pillar step did not protrude in +X where expected: "
        f"max_x_in_pillar_u={max_x_in_pillar_u}, max_x_outside={max_x_outside}"
    )
    # And it must not have built the relief on the wrong (-X) side either.
    assert verts[in_pillar_u, 0].max() > 0.0


try:
    import open3d  # noqa: F401
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_wall_to_solid_on_real_west_wall_run_from_modular_house():
    """End-to-end proof on REAL data: feed group_wall_runs's actual output
    for modular_house()'s west wall (the one with the real pillar, see
    tests/fixtures.py's modular_house docstring and its "pillar" meta
    entry: footprint x in [0.1, 0.4] -- i.e. the pillar protrudes in +X,
    INTO the room, away from the exterior at x~=0) through the fixed
    wall_to_solid, and confirm the pillar comes out as part of ONE
    connected watertight solid, protruding on the correct (+X, into the
    room) side -- not overlapping the exterior face, and not 3 separate
    bodies.
    """
    from scripts.recon.planes import detect_planes
    from scripts.recon.structure import group_wall_runs
    from tests.fixtures import modular_house

    pts, _, meta = modular_house()
    pts = pts[pts[:, 0] < 10.0]  # drop the far neighbour blob
    planes = detect_planes(
        pts, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )
    vertical_planes = [p for p in planes if p.label == "vertical"]
    axes = np.eye(3)  # cloud is already Manhattan-aligned by construction

    runs = group_wall_runs(vertical_planes, pts, axes)
    west_runs = [r for r in runs if r["direction"] == "x" and abs(r["offset_m"]) < 1.0]
    assert len(west_runs) == 1, "expected exactly one x-direction run near x=0 (west wall)"
    west = west_runs[0]
    assert len(west["steps"]) >= 3, "west run should carry both wall faces plus the pillar face"

    wall = _adapt_run_to_wall(west, thickness_m=meta["wall_thickness_m"], floor_z_m=meta["z_floor_m"])
    solid = wall_to_solid(wall, west["steps"])

    # (a) one connected, watertight solid -- the real-data equivalent of
    # the synthetic test above; this is the actual bug the prior code
    # review found (group_wall_runs's real west-wall output fed straight
    # into wall_to_solid produced 3 disconnected bodies).
    assert solid.is_watertight
    bodies = solid.split(only_watertight=False)
    assert len(bodies) == 1, f"expected one connected wall+pillar solid, got {len(bodies)} bodies"

    # (b) the pillar protrudes on the CORRECT side: further in +X over the
    # pillar's own u-range (y in meta["pillar"]["footprint"]'s y in
    # [2.0, 2.3]) than anywhere else along the wall. This is the real
    # end-to-end proof the bug is fixed: the pillar renders into the room
    # (+X), not through the exterior wall face (-X).
    pillar_u_min = min(p[1] for p in meta["pillar"]["footprint"])
    pillar_u_max = max(p[1] for p in meta["pillar"]["footprint"])
    verts = solid.vertices
    u_vals = verts[:, 1]
    in_pillar_u = (u_vals >= pillar_u_min - 0.05) & (u_vals <= pillar_u_max + 0.05)
    outside_pillar_u = (u_vals <= pillar_u_min - 0.5) | (u_vals >= pillar_u_max + 0.5)
    assert in_pillar_u.any() and outside_pillar_u.any()
    max_x_in_pillar_u = verts[in_pillar_u, 0].max()
    max_x_outside = verts[outside_pillar_u, 0].max()
    assert max_x_in_pillar_u > max_x_outside + 0.15, (
        f"pillar did not protrude into the room (+X) as expected: "
        f"max_x_in_pillar_u={max_x_in_pillar_u}, max_x_outside={max_x_outside}"
    )
    # The pillar's real front face is at x=0.4 (meta["pillar"]["footprint"]);
    # the solid's max-X bound over the pillar's u-range should land in that
    # neighbourhood (loose tolerance -- _step_box pads a step's own face by
    # another half wall-thickness beyond its measured offset, a documented
    # approximation, see solids.py's _step_box docstring), and must not
    # have grown on the wrong (-X, exterior) side.
    pillar_front_x = max(p[0] for p in meta["pillar"]["footprint"])
    assert max_x_in_pillar_u == pytest.approx(pillar_front_x, abs=0.15)
    assert verts[:, 0].min() > -0.3
