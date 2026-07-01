"""Full-scan high-detail mesh — finer than the original 15mm pipeline.

8mm voxels, Poisson depth 11 on the entire cloud (~30-90 min on 32-core VM).
Run reconstruct_mesh_test.py first to validate quality on a small patch.

Usage:
    python scripts/reconstruct_mesh_detail.py [input.laz] [output.obj]
"""

import sys
from pathlib import Path

from mesh_common import ROOT, MeshConfig, run_pipeline

DEFAULT_INPUT = ROOT / "data" / "koushikexport.laz"
DEFAULT_OUTPUT = ROOT / "output" / "mesh_detail.obj"

DETAIL_CONFIG = MeshConfig(
    name="FULL DETAIL (8mm voxel, depth 11)",
    voxel_size=0.008,
    poisson_depth=11,
    density_trim_percentile=5,
    outlier_std_ratio=2.5,
    crop_radius_m=None,
)

if __name__ == "__main__":
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run_pipeline(inp, outp, DETAIL_CONFIG)
