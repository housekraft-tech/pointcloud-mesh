"""Inspect a LAS scan's Z-density histogram to determine its primary story's
Z-band, for use with floorplan_reconstruct.py's --z-band flag.

Automatic Z-band detection (find_dense_z_band) has NOT proven reliable across
different real scans and sampling densities during development -- real scans
can contain a second dense structure (another floor, a stairwell) or enough
background scatter to defeat a fixed threshold. This script prints the
Z-histogram directly so you can pick the band by eye: look for two dense
spikes (floor and ceiling slabs -- far denser than the room-volume bins
between them) bracketing a region of moderate-but-real density, separated
from any OTHER dense region by a genuine near-empty gap.

Usage:
    python scripts/find_z_band.py [input.las] [bin_m]
"""
import sys
from pathlib import Path

import numpy as np

try:
    from mesh_common import load_las_as_o3d, recenter_pcd, log, log_stage
except ImportError:
    from scripts.mesh_common import load_las_as_o3d, recenter_pcd, log, log_stage

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_BIN_M = 0.1


def main(input_path, bin_m=DEFAULT_BIN_M):
    with log_stage(f"Loading {input_path}"):
        pcd = load_las_as_o3d(Path(input_path))
        recenter_pcd(pcd)
        z = np.asarray(pcd.points)[:, 2]

    with log_stage(f"Building Z-histogram (bin={bin_m * 1000:.0f}mm)"):
        n_bins = max(int(np.ceil((z.max() - z.min()) / bin_m)), 4)
        counts, edges = np.histogram(z, bins=n_bins)

    log(f"Z range: {z.min():.3f} to {z.max():.3f} ({z.size:,} points)")
    log(f"Peak bin density: {counts.max():,} points")
    log("Bins with non-trivial density (>= 0.5% of peak) -- look for spikes bracketing a real story:")
    threshold = max(1, int(counts.max() * 0.005))
    for count, edge in zip(counts, edges[:-1]):
        if count >= threshold:
            marker = " <== dense spike (likely floor/ceiling slab)" if count > counts.max() * 0.3 else ""
            log(f"  z={edge:8.3f}  {count:>10,} pts{marker}")

    log(
        "Pick z_min/z_max bracketing one story's spikes (inclusive of the floor/ceiling "
        "spikes themselves), then run: python scripts/floorplan_reconstruct.py "
        f"{input_path} <output_dir> --z-band z_min,z_max"
    )


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    bin_arg = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BIN_M
    main(inp, bin_arg)
