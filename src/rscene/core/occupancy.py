"""Voxel occupancy and the interior flood-fill.

This exists to answer two questions and no others: which side of a surface is
inside, and does a void pass all the way through. It never defines geometry --
its cells are 50 mm and its edges are stair-stepped, both unacceptable for a
model whose deliverable is sub-3 mm.

The mechanism is a flood-fill rather than the walk trajectory the spec
originally named. See spec section 4.4.1: a 360 degree scanner's time-slice
centroid is the centre of what was seen, not where the scanner stood, and every
sensor-geometry field in the reference LAS exports is zero. The fill needs no
path -- seed it in air above the floor and interior is whatever it reaches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyGrid:
    """A binary voxel grid. `origin` is the world position of cell (0,0,0)."""

    origin: np.ndarray          # (3,)
    cell_m: float
    occupied: np.ndarray        # (nx, ny, nz) bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occupied.shape

    def index_of(self, points: np.ndarray) -> np.ndarray:
        """World points -> integer cell indices, clipped into the grid."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ijk = np.floor((pts - self.origin) / self.cell_m).astype(np.int64)
        return np.clip(ijk, 0, np.array(self.shape) - 1)


def build_occupancy(xyz: np.ndarray, config: dict) -> OccupancyGrid:
    """Mark every cell containing at least one point as occupied.

    One cell of padding is added on all sides so the flood-fill always has an
    exterior shell to be bounded by, and so `index_of` never lands on an edge
    cell for a real point.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    cell = float(config["occupancy_cell_m"])
    origin = xyz.min(axis=0) - cell
    extent = xyz.max(axis=0) + cell - origin
    shape = tuple(int(np.ceil(e / cell)) + 1 for e in extent)

    occupied = np.zeros(shape, dtype=bool)
    ijk = np.floor((xyz - origin) / cell).astype(np.int64)
    ijk = np.clip(ijk, 0, np.array(shape) - 1)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return OccupancyGrid(origin=origin, cell_m=cell, occupied=occupied)


_NEIGHBOURS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


def flood_interior(grid: OccupancyGrid, seed_xyz: np.ndarray) -> np.ndarray:
    """6-connected flood-fill of free cells from a seed. Returns a bool array.

    Interior is whatever the fill reaches without passing through an occupied
    cell. A sealed room bounds it; a doorway lets it through to the next room;
    a hole in the envelope lets it escape to the grid boundary, which is itself
    diagnostic.

    Raises if the seed lands in an occupied cell -- silently returning an empty
    fill would look identical to a sealed void and hide the real problem.

    WARNING: this requires a watertight occupancy envelope. Real scans do not
    have one -- a single missing voxel from a scan hole, an occlusion, or
    glazing that returns nothing lets the fill escape into the grid's padding
    ring, which wraps the entire model. Measured on
    data/isolated_structural_v2.las at 50 mm cells, flood-fill from a
    real-scan crop filled 100.00% of free cells (the full bounding box, not
    an interior); the full isolated scan leaked identically, so it is not a
    cropping artefact. Use `interior_by_enclosure` for real data -- it is
    local (per-Z-slice horizontal enclosure), needs no watertight envelope,
    and is unaffected by leaks. Keep using `flood_interior` only for a
    genuinely closed envelope: a synthetic sealed room, or a repaired scan.
    """
    seed = grid.index_of(np.asarray(seed_xyz, dtype=np.float64).reshape(1, 3))[0]
    si, sj, sk = int(seed[0]), int(seed[1]), int(seed[2])
    if grid.occupied[si, sj, sk]:
        raise ValueError(
            f"interior seed {np.asarray(seed_xyz).tolist()} lands in an occupied "
            f"cell {(si, sj, sk)}; choose a seed in open air"
        )

    nx, ny, nz = grid.shape
    interior = np.zeros(grid.shape, dtype=bool)
    interior[si, sj, sk] = True
    queue = deque([(si, sj, sk)])

    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in _NEIGHBOURS:
            a, b, c = i + di, j + dj, k + dk
            if not (0 <= a < nx and 0 <= b < ny and 0 <= c < nz):
                continue
            if interior[a, b, c] or grid.occupied[a, b, c]:
                continue
            interior[a, b, c] = True
            queue.append((a, b, c))
    return interior


def interior_by_enclosure(grid: OccupancyGrid) -> np.ndarray:
    """Mark free cells as interior when enclosed horizontally, per Z slice.

    Default interior definition for real scan data. Flood-fill (`flood_interior`)
    needs a watertight occupancy envelope, which real scans do not have -- see
    its docstring for measured leak figures. This mechanism instead asks, for
    every free cell independently: does a ray along -x, +x, -y and +y each hit
    occupancy before leaving the grid, all within this cell's own Z slice? A
    cell inside a room is walled on all four sides and passes; a cell outside
    the building escapes in at least one direction and fails; a cell on a
    balcony correctly reads as outside. It needs no connectivity to a seed and
    no closed envelope, so a single missing voxel elsewhere in the grid cannot
    leak it -- the check is local per cell. O(n) via cumulative sums along x
    and y, and deterministic.
    """
    occ = grid.occupied

    c = np.cumsum(occ, axis=0)
    # c[i] = count of occupied cells at indices 0..i (inclusive) along x.
    # For a free cell (occ[i] == 0), c[i] > 0 means an occupied cell exists
    # at some index < i, i.e. strictly before it -- exactly "hit going -x".
    any_neg_x = c > 0
    # c[-1] is the total occupied count along the whole x column; c[-1] - c[i]
    # is the count at indices strictly greater than i (since c[i] itself adds
    # nothing extra for a free cell), i.e. "hit going +x".
    any_pos_x = (c[-1][None, :, :] - c) > 0

    c = np.cumsum(occ, axis=1)
    any_neg_y = c > 0
    any_pos_y = (c[:, -1][:, None, :] - c) > 0

    return (~occ) & any_neg_x & any_pos_x & any_neg_y & any_pos_y
