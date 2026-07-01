"""Smoke test — small point count, coarse mesh, fast end-to-end check.

Does NOT produce a high-quality mesh. Use this to verify the pipeline works,
then run reconstruct_mesh_detail.py for real output.

Usage:
    python scripts/reconstruct_mesh_test.py [input.laz] [output.obj]
"""

import sys
from pathlib import Path

from mesh_common import ROOT, MeshConfig, run_pipeline

DEFAULT_INPUT = ROOT / "data" / "koushikexport.laz"
DEFAULT_OUTPUT = ROOT / "output" / "mesh_test.obj"

# ~1-3 min on a 32-core VM: tiny crop, 100k point cap, coarse voxel, low Poisson depth.
TEST_CONFIG = MeshConfig(
    name="SMOKE TEST (flow check only — not for quality)",
    voxel_size=0.05,
    poisson_depth=8,
    density_trim_percentile=5,
    crop_radius_m=1.5,
    max_points=100_000,
    skip_outlier_removal=True,
)

if __name__ == "__main__":
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run_pipeline(inp, outp, TEST_CONFIG)
