"""Point-cloud cleanup: percentile bbox crop and statistical outlier removal.

percentile_crop is pure numpy (unit-testable anywhere). remove_outliers wraps
Open3D and imports it lazily so this module loads without Open3D.
"""
from __future__ import annotations

import numpy as np

from .schema import ScanData


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
