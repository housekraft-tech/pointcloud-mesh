"""Full-scan production mesh at native point spacing (~15mm).

15mm voxels, Poisson depth 10 — matched to ~4.4M points after downsample.
Run reconstruct_mesh_test.py first to verify the pipeline (~20-30 min on 32-core VM).

Usage:
    python scripts/reconstruct_mesh_detail.py [input.laz] [output.obj]
"""

import sys
from pathlib import Path

from mesh_common import ROOT, MeshConfig, run_pipeline

DEFAULT_INPUT = ROOT / "data" / "koushikexport.laz"
DEFAULT_OUTPUT = ROOT / "output" / "mesh_detail.obj"

DETAIL_CONFIG = MeshConfig(
    name="FULL DETAIL (15mm voxel, depth 10)",
    voxel_size=0.015,
    poisson_depth=10,
    density_trim_percentile=5,
    outlier_std_ratio=2.0,
    crop_radius_m=None,
)

if __name__ == "__main__":
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run_pipeline(inp, outp, DETAIL_CONFIG)
