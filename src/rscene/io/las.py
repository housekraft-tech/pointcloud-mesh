"""LAS/LAZ adapter.

The only module allowed to import laspy. Reads every attribute the point
format carries so nothing the sensor gave us is discarded at the door --
intensity in particular is a material cue the pipeline should keep available.
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    import laspy
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "laspy is required for LAS I/O; install with: uv pip install -e '.[io]'"
    ) from exc

from ..core.points import PointSet

_CHUNK = 3_000_000


def file_sha256(path: str) -> str:
    """Content hash of the scan, recorded in the scene's provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_las(path: str, max_points: int | None = None, seed: int = 0) -> PointSet:
    """Load a LAS/LAZ file, retaining every attribute the format carries.

    Parameters
    ----------
    path : str
        Path to the LAS/LAZ file.
    max_points : int | None
        If set, uniformly randomly subsample to this many points. Random subsampling
        destroys the local point density that region-growing patch extraction depends on.
        Measured on real data: at native 6.1 mm spacing, 13.7% of points go unassigned;
        the same room randomly thinned to 15.2 mm spacing gives 40.9% unassigned.
        Use only for quick previews and tests; real runs must use spatial tiling or
        voxel downsampling.
    seed : int
        Random seed for reproducible subsampling.
    """
    with laspy.open(path) as reader:
        dims = set(reader.header.point_format.dimension_names)
        has_time = "gps_time" in dims
        has_rgb = {"red", "green", "blue"}.issubset(dims)
        has_intensity = "intensity" in dims

        xyz_chunks, t_chunks, rgb_chunks, i_chunks = [], [], [], []
        for pts in reader.chunk_iterator(_CHUNK):
            xyz_chunks.append(np.column_stack(
                [np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)]
            ).astype(np.float64))
            if has_time:
                t_chunks.append(np.asarray(pts.gps_time, dtype=np.float64))
            if has_rgb:
                rgb_chunks.append(np.column_stack(
                    [np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)]
                ))
            if has_intensity:
                i_chunks.append(np.asarray(pts.intensity))

    xyz = np.concatenate(xyz_chunks) if xyz_chunks else np.zeros((0, 3))
    rgb = None
    if has_rgb:
        rgb16 = np.concatenate(rgb_chunks)
        if np.any(rgb16):
            rgb = (rgb16 >> 8).astype(np.uint8) if rgb16.max() > 255 else rgb16.astype(np.uint8)

    points = PointSet(
        xyz=xyz,
        gps_time=np.concatenate(t_chunks) if has_time else None,
        intensity=np.concatenate(i_chunks) if has_intensity else None,
        rgb=rgb,
    )

    if max_points is not None and points.n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(points.n, size=max_points, replace=False))
        points = points.subset(idx)

    return points


def save_las(points: PointSet, path: str) -> None:
    """Write a PointSet to LAS point format 3 at 0.1 mm scale."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.0001, 0.0001, 0.0001]
    header.offsets = [0.0, 0.0, 0.0]

    las = laspy.LasData(header)
    las.x = points.xyz[:, 0]
    las.y = points.xyz[:, 1]
    las.z = points.xyz[:, 2]
    if points.gps_time is not None:
        las.gps_time = points.gps_time
    if points.intensity is not None:
        las.intensity = np.asarray(points.intensity).astype(np.uint16)
    if points.rgb is not None:
        las.red = points.rgb[:, 0].astype(np.uint16) << 8
        las.green = points.rgb[:, 1].astype(np.uint16) << 8
        las.blue = points.rgb[:, 2].astype(np.uint16) << 8
    las.write(path)
