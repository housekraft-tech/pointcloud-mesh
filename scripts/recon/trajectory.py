"""Scanner walk-path recovery.

No trajectory sidecar exists for the current scans, but gps_time is present, so
approx_trajectory reconstructs a coarse path by binning points in time and
taking each time-slice centroid. Good enough to anchor unit isolation and to
seed visibility rays. load_trajectory parses an explicit trajectory file when
the user exports one from SLAM GO POST.
"""
from __future__ import annotations

import numpy as np


def approx_trajectory(gps_time: np.ndarray, xyz: np.ndarray, dt_s: float = 0.25) -> np.ndarray:
    """Approximate the sensor path as time-slice centroids, ordered by time.

    Bins points into dt_s-second slices; each non-empty slice contributes its
    centroid as one path vertex. Returns (M, 3). Coarse but monotonic in time.
    """
    gps_time = np.asarray(gps_time, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if gps_time.size == 0:
        return np.zeros((0, 3))

    t0, t1 = gps_time.min(), gps_time.max()
    n_bins = max(1, int(np.ceil((t1 - t0) / dt_s)))
    edges = np.linspace(t0, t1, n_bins + 1)
    bin_idx = np.clip(np.digitize(gps_time, edges[1:-1]), 0, n_bins - 1)

    path = []
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_xyz = xyz[order]
    # split into contiguous groups by bin id
    boundaries = np.flatnonzero(np.diff(sorted_bins)) + 1
    for group in np.split(sorted_xyz, boundaries):
        if group.size:
            path.append(group.mean(axis=0))
    return np.asarray(path)


def load_trajectory(path: str) -> np.ndarray:
    """Load an explicit trajectory file (whitespace or comma separated).

    Auto-detects columns: >=4 numeric columns are treated as [time, x, y, z]
    (sorted by time); exactly 3 columns are treated as [x, y, z] in file order.
    Returns (M, 3).
    """
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#;":
                continue
            parts = line.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue  # header / non-numeric line
            if len(vals) >= 3:
                rows.append(vals)
    if not rows:
        return np.zeros((0, 3))

    ncols = min(len(r) for r in rows)
    arr = np.array([r[:ncols] for r in rows], dtype=float)
    if ncols >= 4:
        arr = arr[np.argsort(arr[:, 0])]
        return arr[:, 1:4]
    return arr[:, 0:3]
