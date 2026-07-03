"""Point-cloud cleanup: percentile bbox crop, statistical outlier removal, and
furniture removal.

percentile_crop and remove_furniture are pure numpy (unit-testable anywhere).
remove_outliers wraps Open3D and imports it lazily so this module loads
without Open3D.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .schema import Beam, Column, ScanData


def percentile_crop(scan: ScanData, lo: float = 1.0, hi: float = 99.0, margin_m: float = 0.5) -> ScanData:
    """Keep points inside the per-axis [lo, hi] percentile bbox, expanded by margin.

    Culls gross SLAM-drift / stray outliers without assuming a fixed extent.
    """
    xyz = scan.xyz
    lo_b = np.percentile(xyz, lo, axis=0) - margin_m
    hi_b = np.percentile(xyz, hi, axis=0) + margin_m
    mask = np.all((xyz >= lo_b) & (xyz <= hi_b), axis=1)
    return scan.subset(mask)


def remove_outliers(scan: ScanData, nb_neighbors: int = 20, std_ratio: float = 2.0) -> ScanData:
    """Statistical outlier removal (Open3D). Returns the kept-points subset."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan.xyz)
    _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return scan.subset(np.asarray(keep_idx, dtype=np.int64))


def remove_furniture(
    scan: ScanData,
    walls: Sequence[dict],
    z_floor: float,
    z_ceiling: float,
    columns: Iterable[Column] = (),
    beams: Iterable[Beam] = (),
    dist_m: float = 0.15,
    floor_band_m: float = 0.12,
    ceiling_band_m: float = 0.20,
) -> tuple:
    """Drop points not explained by a structural element (furniture/clutter removal).

    This is a GEOMETRIC-CORRIDOR filter, not a plane-membership filter: it keeps
    a full-height/full-band corridor around every wall step face (± dist_m and
    the wall's own thickness), a thin floor/ceiling band everywhere, and column
    footprints / beam boxes. Everything else in room interiors -- sofas, tables,
    clutter -- is not architecturally "explained" by any of those and drops.

    A prior experiment (scripts/experiments/strip_furniture.py) tried keeping
    only points that belong to a single detected structural plane and it tore
    holes in walls (door/window frames, skirting not cleanly claimed by one
    plane vanished with the furniture). This corridor rule -- distance to a
    wall step's face plane, bounded by that step's own u/z extent -- was
    validated as the fix (scripts/experiments/strip_furniture_v2.py) and is
    what this function implements, pure numpy.

    Returns (kept scan, dropped count).
    """
    xyz = scan.xyz
    z = xyz[:, 2]

    mask = (np.abs(z - z_floor) <= floor_band_m) | (np.abs(z - z_ceiling) <= ceiling_band_m)

    for wall in walls:
        direction = wall["direction"]
        base_offset = wall["offset_m"]
        thickness_m = wall.get("thickness_m", 0.0)
        # direction is the axis the wall's normal is closest to (its
        # perpendicular axis); the other horizontal axis is "u" along the run.
        if direction == "x":
            perp = xyz[:, 0]
            u = xyz[:, 1]
        else:
            perp = xyz[:, 1]
            u = xyz[:, 0]

        for step in wall["steps"]:
            offset = base_offset + step.offset_m
            step_mask = (
                (np.abs(perp - offset) <= dist_m + thickness_m / 2.0)
                & (u >= step.u_min_m - dist_m)
                & (u <= step.u_max_m + dist_m)
                & (z >= step.z_min_m - dist_m)
                & (z <= step.z_max_m + dist_m)
            )
            mask |= step_mask

    for column in columns:
        footprint = np.asarray(column.footprint, dtype=float)
        lo_xy = footprint.min(axis=0) - dist_m
        hi_xy = footprint.max(axis=0) + dist_m
        col_mask = (
            (xyz[:, 0] >= lo_xy[0]) & (xyz[:, 0] <= hi_xy[0])
            & (xyz[:, 1] >= lo_xy[1]) & (xyz[:, 1] <= hi_xy[1])
            & (z >= column.z_min_m - dist_m) & (z <= column.z_max_m + dist_m)
        )
        mask |= col_mask

    for beam in beams:
        p0 = np.asarray(beam.p0, dtype=float)
        p1 = np.asarray(beam.p1, dtype=float)
        d = p1 - p0
        length = np.linalg.norm(d)
        d_unit = d / length if length > 0 else np.array([1.0, 0.0])
        n_unit = np.array([-d_unit[1], d_unit[0]])

        rel = xyz[:, :2] - p0
        along = rel @ d_unit
        across = rel @ n_unit

        half_len_margin = beam.depth_m / 2.0 + dist_m
        half_width = beam.width_m / 2.0 + dist_m
        beam_mask = (
            (along >= -half_len_margin) & (along <= length + half_len_margin)
            & (np.abs(across) <= half_width)
            & (z >= beam.z_min_m - dist_m) & (z <= beam.z_max_m + dist_m)
        )
        mask |= beam_mask

    return scan.subset(mask), int((~mask).sum())
