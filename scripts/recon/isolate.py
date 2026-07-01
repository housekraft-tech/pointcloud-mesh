"""Isolate the scanned unit from drift and neighbour geometry.

Two steps, both pure numpy/scipy:
  select_z_band  -- find the primary storey's floor/ceiling Z from the
                    dominant floor+ceiling density spikes.
  isolate_unit   -- keep the Z-band, then the XY-connected region reachable
                    from the scanner path, then a distance-from-path cap.
                    Removes the neighbouring building seen through the balcony
                    (wrong height AND across an air gap) while keeping the
                    walked balcony (connected).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .schema import ScanData


def select_z_band(z, bin_m: float = 0.05, min_height_m: float = 1.8, max_height_m: float = 4.5):
    """Return (z_floor, z_ceiling) from the two dominant horizontal density spikes.

    Anchors on the tallest histogram bin (the densest horizontal surface) and
    pairs it with the tallest bin a plausible storey-height away.
    """
    z = np.asarray(z, dtype=float)
    lo, hi = float(z.min()), float(z.max())
    n_bins = max(1, int(np.ceil((hi - lo) / bin_m)))
    hist, edges = np.histogram(z, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if hist.max() == 0:
        return lo, hi

    anchor = int(np.argmax(hist))
    z_a = centers[anchor]
    thresh = 0.2 * hist.max()
    cand = np.flatnonzero(hist >= thresh)

    best, best_score = None, -1
    for c in cand:
        d = abs(centers[c] - z_a)
        if min_height_m <= d <= max_height_m and hist[c] > best_score:
            best, best_score = c, hist[c]

    if best is None:
        return float(z_a - min_height_m / 2.0), float(z_a + min_height_m / 2.0)
    z_floor, z_ceiling = sorted([float(z_a), float(centers[best])])
    return z_floor, z_ceiling


def isolate_unit(
    scan: ScanData,
    trajectory: np.ndarray,
    z_band,
    z_margin_m: float = 0.15,
    cell_m: float = 0.25,
    max_gap_cells: int = 1,
    max_dist_m: float = 8.0,
):
    """Extract the unit: Z-band + trajectory-connected XY region + distance cap.

    Returns (ScanData, stats). stats = {kept, dropped, z_floor, z_ceiling}.
    If trajectory is empty, anchors connectivity on the densest XY cell and
    skips the distance cap.
    """
    z_floor, z_ceiling = z_band
    z = scan.xyz[:, 2]
    z_mask = (z >= z_floor - z_margin_m) & (z <= z_ceiling + z_margin_m)
    band = scan.subset(z_mask)
    n_total = scan.n
    if band.n == 0:
        return band, {"kept": 0, "dropped": n_total, "z_floor": z_floor, "z_ceiling": z_ceiling}

    xy = band.xyz[:, :2]
    mins = xy.min(axis=0)
    ij = np.floor((xy - mins) / cell_m).astype(np.int64)
    nx = int(ij[:, 0].max()) + 1
    ny = int(ij[:, 1].max()) + 1
    occ = np.zeros((nx, ny), dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True

    # Bridge small gaps, then connected-component label (8-connectivity).
    struct = ndimage.generate_binary_structure(2, 2)
    occ_bridged = ndimage.binary_dilation(occ, structure=struct, iterations=max_gap_cells)
    labels, _ = ndimage.label(occ_bridged, structure=struct)

    traj = np.asarray(trajectory, dtype=float)
    if traj.shape[0] > 0:
        tij = np.floor((traj[:, :2] - mins) / cell_m).astype(np.int64)
        tij[:, 0] = np.clip(tij[:, 0], 0, nx - 1)
        tij[:, 1] = np.clip(tij[:, 1], 0, ny - 1)
        anchor_labels = np.unique(labels[tij[:, 0], tij[:, 1]])
    else:
        flat = int(np.argmax(np.bincount(labels[occ].ravel())[1:]) + 1) if occ.any() else 0
        anchor_labels = np.array([flat])
    anchor_labels = anchor_labels[anchor_labels > 0]

    keep_cell = np.isin(labels, anchor_labels) & occ
    point_keep = keep_cell[ij[:, 0], ij[:, 1]]

    # Distance-from-path cap (only when we have a path).
    if traj.shape[0] > 0 and max_dist_m is not None:
        tree = cKDTree(traj[:, :2])
        dist, _ = tree.query(xy, k=1)
        point_keep &= dist <= max_dist_m

    unit = band.subset(point_keep)
    stats = {
        "kept": unit.n,
        "dropped": n_total - unit.n,
        "z_floor": z_floor,
        "z_ceiling": z_ceiling,
    }
    return unit, stats
