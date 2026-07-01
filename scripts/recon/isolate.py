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
    max_dist_m: float = None,
):
    """Extract the unit: Z-band, then the largest connected XY region.

    The scanned unit is the dominant connected occupied blob in the primary
    storey; the neighbouring building glimpsed through a balcony is a smaller,
    physically disconnected blob (an air gap separates them), so anchoring on
    the largest connected component removes it robustly -- independent of the
    (approximate) trajectory, which for a rough gps_time path can otherwise
    wander onto far geometry. The trajectory is used only for an optional
    distance cap, which can never re-admit dropped points.

    Returns (ScanData, stats). stats = {kept, dropped, z_floor, z_ceiling}.
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

    # Bridge small gaps (doorway thresholds), then label 8-connected components.
    struct = ndimage.generate_binary_structure(2, 2)
    occ_bridged = ndimage.binary_dilation(occ, structure=struct, iterations=max_gap_cells)
    labels, n_labels = ndimage.label(occ_bridged, structure=struct)

    point_label = labels[ij[:, 0], ij[:, 1]]  # each point's component
    if n_labels == 0:
        return band, {"kept": band.n, "dropped": n_total - band.n, "z_floor": z_floor, "z_ceiling": z_ceiling}

    # Largest component by point count (label 0 is background; ignore it).
    counts = np.bincount(point_label, minlength=n_labels + 1)
    counts[0] = 0
    unit_label = int(np.argmax(counts))
    point_keep = point_label == unit_label

    # Optional distance-from-path cap (only removes; never adds).
    traj = np.asarray(trajectory, dtype=float)
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
