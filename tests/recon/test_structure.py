import numpy as np
import pytest

from scripts.recon.schema import Plane
from scripts.recon.structure import (
    extract_floor_ceiling,
    group_wall_runs,
    extract_columns_beams,
    extract_unclassified,
)


# ---------------------------------------------------------------------------
# extract_floor_ceiling
# ---------------------------------------------------------------------------

def _plane(label, n_inliers, normal=(0.0, 0.0, 1.0), d=0.0):
    return Plane(normal=normal, d=d, label=label, inlier_idx=np.arange(n_inliers))


def test_extract_floor_ceiling_picks_largest_by_inliers():
    planes = [
        _plane("floor", 100),
        _plane("floor", 500),      # largest floor -> should win
        _plane("ceiling", 50),
        _plane("ceiling", 300),    # largest ceiling -> should win
        _plane("vertical", 900),   # ignored
    ]
    floor, ceiling = extract_floor_ceiling(planes)
    assert floor.label == "floor"
    assert len(floor.inlier_idx) == 500
    assert ceiling.label == "ceiling"
    assert len(ceiling.inlier_idx) == 300


def test_extract_floor_ceiling_raises_on_missing_floor():
    planes = [_plane("ceiling", 300), _plane("vertical", 900)]
    with pytest.raises(ValueError, match="floor"):
        extract_floor_ceiling(planes)


def test_extract_floor_ceiling_raises_on_missing_ceiling():
    planes = [_plane("floor", 300), _plane("vertical", 900)]
    with pytest.raises(ValueError, match="ceiling"):
        extract_floor_ceiling(planes)


def test_extract_floor_ceiling_raises_on_empty_input():
    with pytest.raises(ValueError):
        extract_floor_ceiling([])


# ---------------------------------------------------------------------------
# group_wall_runs -- focused synthetic unit tests (hand-built planes, no fixture)
# ---------------------------------------------------------------------------

def _grid_points(x_val_or_range, y_val_or_range, z_range, n=200, axis="x", seed=0,
                  spacing=0.03, noise_std=0.001):
    """Build synthetic points on a vertical face as a dense regular grid (with
    small positional noise), mirroring tests/fixtures.py's _sample_face. A
    real meshgrid -- rather than a sparse random scatter -- is important
    here: group_wall_runs/extract_columns_beams use a robust "largest
    contiguous 1-D group" range statistic (structure._robust_range) to shed
    stray leaked points, and a sparse random scatter can itself contain
    gaps large enough to trip that same statistic, which a real point cloud
    (or this grid) does not. `n` is accepted for signature stability but
    unused -- grid density is controlled by `spacing`.
    """
    rng = np.random.default_rng(seed)
    u_range = y_val_or_range if axis == "x" else x_val_or_range
    offset = x_val_or_range if axis == "x" else y_val_or_range
    assert np.isscalar(offset), "offset must be a fixed scalar coordinate"

    us = np.arange(u_range[0], u_range[1], spacing)
    zs = np.arange(z_range[0], z_range[1], spacing)
    uu, zz = np.meshgrid(us, zs)
    uu = uu.ravel() + rng.normal(0, noise_std, uu.size)
    zz = zz.ravel() + rng.normal(0, noise_std, zz.size)
    off = np.full(uu.shape, offset) + rng.normal(0, noise_std, uu.size)

    pts = np.zeros((uu.size, 3))
    if axis == "x":
        pts[:, 0] = off
        pts[:, 1] = uu
    else:
        pts[:, 1] = off
        pts[:, 0] = uu
    pts[:, 2] = zz
    return pts


def test_group_wall_runs_merges_same_offset_and_splits_pillar_step():
    """Two point patches on the SAME wall plane (x=0, split by a doorway-like
    gap in u) must fold into one run with a single merged step, and a third
    patch 0.3 m further into the room (a pillar front face) must appear as a
    SEPARATE WallStep in that same run rather than a new run or an average.
    """
    z_full = (0.0, 2.7)
    main_a = _grid_points(0.0, (0.0, 2.0), z_full, n=300, axis="x", seed=1)
    main_b = _grid_points(0.0, (3.0, 5.0), z_full, n=300, axis="x", seed=2)
    pillar = _grid_points(0.3, (2.0, 2.3), z_full, n=200, axis="x", seed=3)

    xyz = np.vstack([main_a, main_b, pillar])
    n_a, n_b, n_p = len(main_a), len(main_b), len(pillar)
    idx_a = np.arange(0, n_a)
    idx_b = np.arange(n_a, n_a + n_b)
    idx_p = np.arange(n_a + n_b, n_a + n_b + n_p)

    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical", inlier_idx=idx_a),
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical", inlier_idx=idx_b),
        Plane(normal=(1.0, 0.0, 0.0), d=-0.3, label="vertical", inlier_idx=idx_p),
    ]
    axes = np.eye(3)
    runs = group_wall_runs(planes, xyz, axes)

    assert len(runs) == 1, "same-direction, colinear-with-relief planes must be ONE run"
    run = runs[0]
    assert run["direction"] == "x"

    # (a) main_a + main_b (same offset, split by a gap) collapse into one step
    offsets = sorted(st.offset_m for st in run["steps"])
    assert len(offsets) == 2, "same-offset pieces must merge into a single step, not be dropped/duplicated"

    # (b) the pillar's offset is preserved as its OWN step, not merged/averaged away
    assert abs(offsets[0] - 0.0) < 0.02
    assert abs(offsets[1] - 0.3) < 0.02

    # main step must be the long wall face (u spans the merged 0..5 extent),
    # not the short pillar patch (u spans 2.0..2.3)
    main_step = max(run["steps"], key=lambda st: st.u_max_m - st.u_min_m)
    assert main_step.u_max_m - main_step.u_min_m > 4.0
    assert abs(main_step.offset_m - 0.0) < 0.02


def test_group_wall_runs_step_offsets_are_relative_to_the_main_face():
    """Per schema.py's WallStep docstring, offset_m is relative to the run's
    reference (main) face, not an absolute world coordinate -- so the main
    step must always come back at exactly 0.0, regardless of where in world
    space the wall actually sits, and other steps carry the delta from it.
    """
    z_full = (0.0, 2.7)
    # A wall far from the origin (world x ~ 6), plus a pillar-like face
    # 0.3 m proud of it -- if offsets were absolute, these would read ~6.0
    # and ~6.3, not 0.0 and 0.3.
    main = _grid_points(6.0, (0.0, 5.0), z_full, axis="x", seed=50)
    pillar = _grid_points(6.3, (2.0, 2.3), z_full, axis="x", seed=51)

    xyz = np.vstack([main, pillar])
    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=-6.0, label="vertical", inlier_idx=np.arange(0, len(main))),
        Plane(normal=(1.0, 0.0, 0.0), d=-6.3, label="vertical",
              inlier_idx=np.arange(len(main), len(main) + len(pillar))),
    ]
    runs = group_wall_runs(planes, xyz, np.eye(3))
    assert len(runs) == 1
    run = runs[0]
    assert abs(run["offset_m"] - 6.0) < 0.02  # run-level offset stays absolute/world

    offsets = sorted(st.offset_m for st in run["steps"])
    assert abs(offsets[0] - 0.0) < 0.02   # the main face itself: relative offset 0
    assert abs(offsets[1] - 0.3) < 0.02   # the pillar face: +0.3 relative to main


def test_group_wall_runs_keeps_distant_offset_as_separate_run():
    """Two walls sharing a direction but far apart (e.g. opposite sides of a
    room) must NOT be merged into a single run."""
    z_full = (0.0, 2.7)
    west_a = _grid_points(0.0, (0.0, 5.0), z_full, n=400, axis="x", seed=10)
    west_b = _grid_points(0.2, (0.0, 5.0), z_full, n=400, axis="x", seed=11)
    east_a = _grid_points(6.0, (0.0, 5.0), z_full, n=400, axis="x", seed=12)
    east_b = _grid_points(6.2, (0.0, 5.0), z_full, n=400, axis="x", seed=13)

    xyz = np.vstack([west_a, west_b, east_a, east_b])
    sizes = [len(west_a), len(west_b), len(east_a), len(east_b)]
    offs = np.cumsum([0] + sizes)
    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical",
              inlier_idx=np.arange(offs[i], offs[i + 1]))
        for i in range(4)
    ]
    runs = group_wall_runs(planes, xyz, np.eye(3))
    assert len(runs) == 2
    run_offsets = sorted(r["offset_m"] for r in runs)
    # West run's main-step offset is one of its member faces (0.0 or 0.2);
    # east run's is one of (6.0, 6.2). Exact tie-break isn't the contract --
    # only that the two runs stay far apart and near their true wall.
    assert -0.1 <= run_offsets[0] <= 0.3
    assert 5.9 <= run_offsets[1] <= 6.3


def test_group_wall_runs_separates_perpendicular_directions():
    """A wall running along X (normal along Y) and a wall running along Y
    (normal along X) must never merge, regardless of offset overlap."""
    z_full = (0.0, 2.7)
    south = _grid_points((0.0, 6.0), 0.0, z_full, n=400, axis="y", seed=20)
    west = _grid_points(0.0, (0.0, 5.0), z_full, n=400, axis="x", seed=21)

    xyz = np.vstack([south, west])
    planes = [
        Plane(normal=(0.0, 1.0, 0.0), d=0.0, label="vertical", inlier_idx=np.arange(0, len(south))),
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical",
              inlier_idx=np.arange(len(south), len(south) + len(west))),
    ]
    runs = group_wall_runs(planes, xyz, np.eye(3))
    assert len(runs) == 2
    assert {r["direction"] for r in runs} == {"x", "y"}


def test_group_wall_runs_drops_short_non_wall_fragments():
    """A short, isolated patch (shorter than min_run_length_m) must not be
    reported as its own wall run (it is a column/feature, handled elsewhere)."""
    z_full = (0.0, 2.7)
    short = _grid_points((0.1, 0.4), 2.0, z_full, n=150, axis="y", seed=30)
    planes = [
        Plane(normal=(0.0, 1.0, 0.0), d=-2.0, label="vertical", inlier_idx=np.arange(len(short))),
    ]
    runs = group_wall_runs(planes, short, np.eye(3))
    assert runs == []


# ---------------------------------------------------------------------------
# extract_columns_beams -- light synthetic unit test (column-only, no Open3D)
# ---------------------------------------------------------------------------

def test_extract_columns_beams_finds_compact_floor_to_ceiling_column():
    z_full = (0.0, 2.7)
    front = _grid_points(0.4, (2.0, 2.3), z_full, n=200, axis="x", seed=40)
    side1 = _grid_points((0.1, 0.4), 2.0, z_full, n=200, axis="y", seed=41)
    side2 = _grid_points((0.1, 0.4), 2.3, z_full, n=200, axis="y", seed=42)
    # A long wall face that must NOT be mistaken for a column.
    wall = _grid_points(0.0, (0.0, 5.0), z_full, n=400, axis="x", seed=43)

    xyz = np.vstack([front, side1, side2, wall])
    sizes = [len(front), len(side1), len(side2), len(wall)]
    offs = np.cumsum([0] + sizes)
    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=-0.4, label="vertical", inlier_idx=np.arange(offs[0], offs[1])),
        Plane(normal=(0.0, 1.0, 0.0), d=-2.0, label="vertical", inlier_idx=np.arange(offs[1], offs[2])),
        Plane(normal=(0.0, 1.0, 0.0), d=-2.3, label="vertical", inlier_idx=np.arange(offs[2], offs[3])),
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical", inlier_idx=np.arange(offs[3], offs[4])),
    ]

    columns, beams = extract_columns_beams(planes, xyz, floor_z=0.0, ceiling_z=2.7)
    assert len(columns) == 1
    assert beams == []
    col = columns[0]
    xs = [p[0] for p in col.footprint]
    ys = [p[1] for p in col.footprint]
    assert abs(min(xs) - 0.1) < 0.03
    assert abs(max(xs) - 0.4) < 0.03
    assert abs(min(ys) - 2.0) < 0.03
    assert abs(max(ys) - 2.3) < 0.03
    assert abs(col.z_min_m - 0.0) < 0.05
    assert abs(col.z_max_m - 2.7) < 0.05


# ---------------------------------------------------------------------------
# Integration test on modular_house(): full structure pipeline
# ---------------------------------------------------------------------------

try:
    import open3d  # noqa: F401
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_structure_pipeline_on_modular_house():
    from scripts.recon.planes import detect_planes
    from tests.fixtures import modular_house

    pts, _, meta = modular_house()
    # drop the far neighbour blob so we test on the isolated unit
    pts = pts[pts[:, 0] < 10.0]
    planes = detect_planes(
        pts, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )

    floor, ceiling = extract_floor_ceiling(planes)
    floor_z = float(pts[floor.inlier_idx, 2].mean())
    ceiling_z = float(pts[ceiling.inlier_idx, 2].mean())
    assert abs(floor_z - meta["z_floor_m"]) < 0.02
    assert abs(ceiling_z - meta["z_ceiling_m"]) < 0.02

    vertical_planes = [p for p in planes if p.label == "vertical"]
    axes = np.eye(3)  # cloud is already Manhattan-aligned by construction

    runs = group_wall_runs(vertical_planes, pts, axes)
    assert len(runs) == 4, f"expected 4 exterior wall runs, got {len(runs)}"

    columns, beams = extract_columns_beams(vertical_planes, pts, floor_z, ceiling_z)

    # --- pillar -> Column ---
    assert len(columns) == 1
    col = columns[0]
    xs = [p[0] for p in col.footprint]
    ys = [p[1] for p in col.footprint]
    pf = meta["pillar"]["footprint"]
    pxs = [p[0] for p in pf]
    pys = [p[1] for p in pf]
    assert abs(min(xs) - min(pxs)) < 0.03
    assert abs(max(xs) - max(pxs)) < 0.03
    assert abs(min(ys) - min(pys)) < 0.03
    assert abs(max(ys) - max(pys)) < 0.03
    assert abs(col.z_min_m - meta["pillar"]["z"][0]) < 0.1
    assert abs(col.z_max_m - meta["pillar"]["z"][1]) < 0.1

    # --- beam -> Beam ---
    assert len(beams) == 1
    beam = beams[0]
    bx0, bx1 = sorted([beam.p0[0], beam.p1[0]])
    assert abs(bx0 - meta["beam"]["p0"][0]) < 0.03
    assert abs(bx1 - meta["beam"]["p1"][0]) < 0.03
    assert abs(beam.p0[1] - meta["beam"]["p0"][1]) < 0.03
    assert abs(beam.p1[1] - meta["beam"]["p1"][1]) < 0.03
    assert abs(beam.width_m - meta["beam"]["width_m"]) < 0.03
    assert abs(beam.z_min_m - meta["beam"]["z"][0]) < 0.05
    assert abs(beam.z_max_m - meta["beam"]["z"][1]) < 0.05

    # --- west wall run (bears the pillar) carries a step ~0.3 m proud of the wall ---
    west_runs = [r for r in runs if r["direction"] == "x" and abs(r["offset_m"]) < 1.0]
    assert len(west_runs) == 1, "expected exactly one x-direction run near x=0 (west wall)"
    west = west_runs[0]
    offsets = sorted(st.offset_m for st in west["steps"])
    spread = offsets[-1] - offsets[0]
    assert len(offsets) >= 2, "west run must carry more than one step (wall face + pillar face)"
    assert 0.15 < spread < 0.6, (
        f"expected a step offset spread consistent with the pillar's 0.3m protrusion, got {spread}"
    )


# ---------------------------------------------------------------------------
# return_used_indices -- default-shape stability + extract_unclassified
# ---------------------------------------------------------------------------

def test_group_wall_runs_default_return_shape_unchanged():
    """Default call (return_used_indices left at its default False) must
    keep returning a plain list[dict] -- the exact pre-existing shape --
    so every caller that doesn't opt in is completely unaffected."""
    z_full = (0.0, 2.7)
    pts = _grid_points(0.0, (0.0, 5.0), z_full, axis="x", seed=60)
    planes = [Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical", inlier_idx=np.arange(len(pts)))]
    runs = group_wall_runs(planes, pts, np.eye(3))
    assert isinstance(runs, list)
    assert len(runs) == 1
    assert isinstance(runs[0], dict)


def test_extract_columns_beams_default_return_shape_unchanged():
    """Default call must keep returning the exact pre-existing (columns,
    beams) 2-tuple, not a 3-tuple."""
    z_full = (0.0, 2.7)
    front = _grid_points(0.4, (2.0, 2.3), z_full, axis="x", seed=61)
    side1 = _grid_points((0.1, 0.4), 2.0, z_full, axis="y", seed=62)
    side2 = _grid_points((0.1, 0.4), 2.3, z_full, axis="y", seed=63)
    xyz = np.vstack([front, side1, side2])
    sizes = [len(front), len(side1), len(side2)]
    offs = np.cumsum([0] + sizes)
    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=-0.4, label="vertical", inlier_idx=np.arange(offs[0], offs[1])),
        Plane(normal=(0.0, 1.0, 0.0), d=-2.0, label="vertical", inlier_idx=np.arange(offs[1], offs[2])),
        Plane(normal=(0.0, 1.0, 0.0), d=-2.3, label="vertical", inlier_idx=np.arange(offs[2], offs[3])),
    ]
    result = extract_columns_beams(planes, xyz, floor_z=0.0, ceiling_z=2.7)
    assert isinstance(result, tuple)
    assert len(result) == 2
    columns, beams = result
    assert len(columns) == 1


def _build_unclassified_fixture():
    """Five hand-built vertical Planes with a KNOWN classification outcome:

      0: a real wall (long, full height, direction x, offset 0.0)
         -> must be consumed by group_wall_runs.
      1: a compact floor-to-ceiling pier (direction x, offset 3.0 -- far
         enough from the wall's offset that it does NOT chain into its run)
         -> too short (0.3m) for group_wall_runs's min_run_length_m, so
         NOT used there, but IS a valid column -> consumed by
         extract_columns_beams.
      2: an elevated, non-compact, near-ceiling face (direction y) -- a
         beam side face -> excluded from group_wall_runs by the
         full_height_frac filter (its z-span is well under half the
         wall's), but IS a valid beam -> consumed by extract_columns_beams.
      3: an isolated axis-aligned fragment that is small AND doesn't reach
         near the floor (fails the column's floor check) AND doesn't clear
         beam_elevation_m (fails the beam's elevation check) AND is too
         short + not full height for group_wall_runs -- genuinely
         unclaimed by both. direction should read "x" (its stored normal
         is exactly axis-aligned).
      4: the same isolated geometry as #3, but with a 45-degree diagonal
         stored normal -- also genuinely unclaimed by both, and used to
         confirm direction reads back None (not a forced x/y guess).

    Returns (planes, xyz).
    """
    z_full = (0.0, 2.7)
    isolated_z = (0.5, 1.2)  # neither near-floor nor near-elevated-beam-height

    wall_pts = _grid_points(0.0, (0.0, 5.0), z_full, axis="x", seed=70)
    column_pts = _grid_points(3.0, (2.0, 2.3), z_full, axis="x", seed=71)
    beam_pts = _grid_points((0.0, 5.0), 2.5, (2.0, 2.7), axis="y", seed=72)
    isolated_axis_pts = _grid_points(10.0, (2.0, 2.3), isolated_z, axis="x", seed=73)
    isolated_diag_pts = _grid_points(10.5, (2.0, 2.3), isolated_z, axis="x", seed=74)

    chunks = [wall_pts, column_pts, beam_pts, isolated_axis_pts, isolated_diag_pts]
    sizes = [len(c) for c in chunks]
    offs = np.cumsum([0] + sizes)
    xyz = np.vstack(chunks)

    half_sqrt2 = float(np.sqrt(0.5))
    planes = [
        Plane(normal=(1.0, 0.0, 0.0), d=0.0, label="vertical",
              inlier_idx=np.arange(offs[0], offs[1])),                    # 0: wall
        Plane(normal=(1.0, 0.0, 0.0), d=-3.0, label="vertical",
              inlier_idx=np.arange(offs[1], offs[2])),                    # 1: column
        Plane(normal=(0.0, 1.0, 0.0), d=-2.5, label="vertical",
              inlier_idx=np.arange(offs[2], offs[3])),                    # 2: beam
        Plane(normal=(1.0, 0.0, 0.0), d=-10.0, label="vertical",
              inlier_idx=np.arange(offs[3], offs[4])),                    # 3: isolated, axis-aligned
        Plane(normal=(half_sqrt2, half_sqrt2, 0.0), d=-10.5, label="vertical",
              inlier_idx=np.arange(offs[4], offs[5])),                    # 4: isolated, diagonal
    ]
    return planes, xyz


def test_extract_unclassified_returns_exactly_the_leftover_indices():
    planes, xyz = _build_unclassified_fixture()
    axes = np.eye(3)

    runs, run_used = group_wall_runs(planes, xyz, axes, return_used_indices=True)
    columns, beams, cb_used = extract_columns_beams(
        planes, xyz, floor_z=0.0, ceiling_z=2.7, return_used_indices=True
    )

    # Sanity-check the ground truth this test relies on before asserting on
    # extract_unclassified itself, so a failure here points at a fixture
    # mistake rather than a bug in the new function.
    assert len(runs) == 1, "plane 0 (the wall) should form exactly one run"
    assert run_used == {0}
    assert len(columns) == 1, "plane 1 (the compact floor-to-ceiling pier) should be a Column"
    assert len(beams) == 1, "plane 2 (the elevated non-compact face) should be a Beam"
    assert cb_used == {1, 2}

    used_indices = run_used | cb_used
    assert used_indices == {0, 1, 2}

    unclassified = extract_unclassified(planes, used_indices, xyz, axes)
    got_indices = {u["plane_index"] for u in unclassified}

    # No false positives (a used plane wrongly appearing) or false
    # negatives (an unused plane missing).
    assert got_indices == {3, 4}, f"expected exactly {{3, 4}} leftover, got {got_indices}"

    by_index = {u["plane_index"]: u for u in unclassified}

    axis_aligned = by_index[3]
    assert axis_aligned["direction"] == "x"
    assert axis_aligned["n_points"] == len(planes[3].inlier_idx)
    assert axis_aligned["normal"] == (1.0, 0.0, 0.0)
    assert abs(axis_aligned["z_span_m"] - 0.7) < 0.05
    assert abs(axis_aligned["u_span_m"] - 0.3) < 0.05

    diagonal = by_index[4]
    assert diagonal["direction"] is None, "a genuinely 45-degree normal must not be forced onto x or y"
    assert diagonal["n_points"] == len(planes[4].inlier_idx)
    assert abs(diagonal["z_span_m"] - 0.7) < 0.05


# ---------------------------------------------------------------------------
# extract_unclassified -- integration test on modular_house()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_extract_unclassified_accounts_for_every_plane_on_modular_house():
    """This fixture is clean/synthetic (no real-scan clutter), so it's
    entirely plausible every vertical plane ends up classified and
    unclassified comes back empty -- that's fine. What must hold exactly,
    regardless: every plane is accounted for exactly once (no plane
    missing, no double counting between "used" and "unclassified")."""
    from scripts.recon.planes import detect_planes
    from tests.fixtures import modular_house

    pts, _, meta = modular_house()
    pts = pts[pts[:, 0] < 10.0]
    planes = detect_planes(
        pts, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )

    floor, ceiling = extract_floor_ceiling(planes)
    floor_z = float(pts[floor.inlier_idx, 2].mean())
    ceiling_z = float(pts[ceiling.inlier_idx, 2].mean())

    vertical_planes = [p for p in planes if p.label == "vertical"]
    axes = np.eye(3)

    runs, run_used = group_wall_runs(vertical_planes, pts, axes, return_used_indices=True)
    columns, beams, cb_used = extract_columns_beams(
        vertical_planes, pts, floor_z, ceiling_z, return_used_indices=True
    )
    used_indices = run_used | cb_used

    unclassified = extract_unclassified(vertical_planes, used_indices, pts, axes)
    unclassified_indices = {u["plane_index"] for u in unclassified}

    assert len(unclassified) + len(used_indices) == len(vertical_planes), (
        "every input plane must be accounted for exactly once between "
        "used_indices and extract_unclassified's output"
    )
    assert unclassified_indices.isdisjoint(used_indices), "no plane may be both used AND unclassified"
    assert unclassified_indices | used_indices == set(range(len(vertical_planes))), (
        "no plane index may be missing from used_indices + unclassified"
    )
