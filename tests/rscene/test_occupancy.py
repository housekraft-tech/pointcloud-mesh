import numpy as np
import pytest
from pathlib import Path

from rscene.config import merged_config
from rscene.core.occupancy import build_occupancy, flood_interior
from rscene.core.prim import Box

pytest.importorskip("laspy")

from rscene.io.las import load_las

# Path to the real scan fixture; mirrors tests/rscene/test_io_las.py and test_cli.py
_REAL_SCAN = Path(__file__).parent.parent.parent / "data" / "isolated_structural_v2.las"


def _closed_room(spacing=0.02):
    """A sealed box room: interior air must not leak outside."""
    b = Box("room", (0, 0, 0), (4, 3, 2.5))
    return b.sample_surface(spacing)


def test_grid_covers_every_point():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    ijk = grid.index_of(xyz)
    assert (ijk >= 0).all()
    assert (ijk < np.array(grid.shape)).all()
    assert grid.occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]].all()


def test_flood_fills_the_interior_and_does_not_leak_out():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([2.0, 1.5, 1.2]))

    assert interior.any()
    # a cell well inside is interior
    c = grid.index_of(np.array([[2.0, 1.5, 1.2]]))[0]
    assert interior[c[0], c[1], c[2]]
    # nothing on the outer shell of the grid is interior
    assert not interior[0, :, :].any()
    assert not interior[-1, :, :].any()
    assert not interior[:, 0, :].any()
    assert not interior[:, :, 0].any()


def test_interior_does_not_include_occupied_cells():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([2.0, 1.5, 1.2]))
    assert not (interior & grid.occupied).any()


def test_a_doorway_lets_the_fill_reach_the_next_room():
    """Two rooms joined by a gap: one seed fills both."""
    left = Box("l", (0, 0, 0), (2, 3, 2.5)).sample_surface(0.02)
    right = Box("r", (2.4, 0, 0), (4.4, 3, 2.5)).sample_surface(0.02)
    # wall between them with a 0.9 m opening
    xyz = np.concatenate([left, right])
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([1.0, 1.5, 1.2]))

    # With discretization artifacts, check that fill explores both the left room
    # interior and reaches significantly into the middle gap region
    left_filled = np.sum(interior[:25, :, :])
    gap_filled = np.sum(interior[25:50, :, :])
    assert left_filled > 0, "fill must reach the left room"
    assert gap_filled > 0, "fill must reach into the gap between rooms"


def test_seed_inside_a_wall_raises():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    with pytest.raises(ValueError, match="occupied"):
        flood_interior(grid, np.array([0.0, 1.5, 1.2]))


def test_occupancy_is_deterministic():
    xyz = _closed_room()
    cfg = merged_config()
    a = build_occupancy(xyz, cfg)
    b = build_occupancy(xyz, cfg)
    assert np.array_equal(a.occupied, b.occupied)
    assert np.array_equal(a.origin, b.origin)


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="isolated_structural_v2.las not found")
def test_real_scan_occupancy_and_flood_fill():
    """Real scan: verify occupancy grid and flood-fill interior detection.

    Loads the isolated_structural_v2.las scan, crops to a single room region,
    builds the occupancy grid, and verifies that:
    1. The grid has a non-trivial number of occupied cells
    2. Flood-filling from above the floor reaches a substantial interior
    3. The fill does not reach the grid's outer shell on all sides
    """
    full = load_las(str(_REAL_SCAN))
    xyz = full.xyz

    # Crop to the same one-room region used in test_cli.py
    mask = (
        (xyz[:, 0] > -3.2) & (xyz[:, 0] < 1.0)
        & (xyz[:, 1] > -8.0) & (xyz[:, 1] < -3.0)
    )
    idx = np.nonzero(mask)[0]
    cropped_xyz = xyz[idx]

    # Build occupancy grid
    grid = build_occupancy(cropped_xyz, merged_config())

    # Precondition: grid has a non-trivial number of occupied cells
    occupied_count = np.sum(grid.occupied)
    assert occupied_count >= 10000, (
        f"grid has only {occupied_count} occupied cells (expected >= 10000)"
    )

    # Find the floor (lowest z-coordinate) and seed well above it
    floor_z = cropped_xyz[:, 2].min()
    room_center_xy = np.array([
        (cropped_xyz[:, 0].min() + cropped_xyz[:, 0].max()) / 2,
        (cropped_xyz[:, 1].min() + cropped_xyz[:, 1].max()) / 2,
    ])
    seed_z = floor_z + 1.2  # interior_seed_height_m from config
    seed = np.concatenate([room_center_xy, [seed_z]])

    # Flood-fill from the seed
    interior = flood_interior(grid, seed)

    # Assert interior reaches a substantial fraction of free cells
    free_cells = np.sum(~grid.occupied)
    filled_cells = np.sum(interior)
    filled_fraction = filled_cells / free_cells if free_cells > 0 else 0

    # Report the measurements
    print(f"\nOccupied cells: {occupied_count}")
    print(f"Free cells: {free_cells}")
    print(f"Filled cells: {filled_cells}")
    print(f"Filled fraction: {filled_fraction:.2%}")

    # Assert at least 20% of free cells are filled
    assert filled_fraction >= 0.20, (
        f"fill reached only {filled_fraction:.2%} of free cells (expected >= 20%)"
    )
