"""Fast test run on a small spatial patch with finer settings.

Crops 3m radius around scan center. 8mm voxels, Poisson depth 11.
Run this first to preview quality (~5-15 min on a 32-core VM).

Usage:
    python scripts/reconstruct_mesh_test.py [input.laz] [output.obj]
"""

import sys
from pathlib import Path

from mesh_common import ROOT, MeshConfig, run_pipeline

DEFAULT_INPUT = ROOT / "data" / "koushikexport.laz"
DEFAULT_OUTPUT = ROOT / "output" / "mesh_test.obj"

TEST_CONFIG = MeshConfig(
    name="TEST PATCH (3m radius, 8mm voxel, depth 11)",
    voxel_size=0.008,
    poisson_depth=11,
    density_trim_percentile=5,
    outlier_std_ratio=2.5,
    crop_radius_m=3.0,
)

if __name__ == "__main__":
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run_pipeline(inp, outp, TEST_CONFIG)
