"""Chunked LAS/LAZ loader -> ScanData.

Reads XYZ (scaled float64) plus gps_time / RGB / intensity when present.
Chunked to bound memory on 20M+ point SLAM scans; optional uniform subsample.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import laspy

from .schema import ScanData

_CHUNK = 3_000_000


def load_scan(path: str, max_points: Optional[int] = None, rng_seed: int = 0) -> ScanData:
    """Load a LAS/LAZ file into a ScanData.

    gps_time / RGB / intensity are populated only when the point format
    carries them and (for RGB) at least one channel is non-zero. RGB is
    downscaled to uint8. If max_points is given and exceeded, a uniform
    random subsample of that size is returned (deterministic via rng_seed).
    """
    with laspy.open(path) as reader:
        dims = set(reader.header.point_format.dimension_names)
        has_time = "gps_time" in dims
        has_rgb = {"red", "green", "blue"}.issubset(dims)
        has_intensity = "intensity" in dims

        xyz_chunks, t_chunks, rgb_chunks, i_chunks = [], [], [], []
        for pts in reader.chunk_iterator(_CHUNK):
            xyz_chunks.append(
                np.column_stack(
                    [np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)]
                ).astype(np.float64)
            )
            if has_time:
                t_chunks.append(np.asarray(pts.gps_time, dtype=np.float64))
            if has_rgb:
                rgb_chunks.append(
                    np.column_stack(
                        [np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)]
                    )
                )
            if has_intensity:
                i_chunks.append(np.asarray(pts.intensity))

    xyz = np.concatenate(xyz_chunks) if xyz_chunks else np.zeros((0, 3))
    gps_time = np.concatenate(t_chunks) if has_time else None
    intensity = np.concatenate(i_chunks) if has_intensity else None

    rgb = None
    if has_rgb:
        rgb16 = np.concatenate(rgb_chunks)
        if np.any(rgb16):
            rgb = (rgb16 >> 8).astype(np.uint8) if rgb16.max() > 255 else rgb16.astype(np.uint8)

    scan = ScanData(xyz=xyz, gps_time=gps_time, rgb=rgb, intensity=intensity)

    if max_points is not None and scan.n > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(scan.n, size=max_points, replace=False)
        scan = scan.subset(idx)

    return scan
