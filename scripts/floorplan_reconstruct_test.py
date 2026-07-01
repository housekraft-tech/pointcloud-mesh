"""Fast smoke test for floorplan_reconstruct.py: small spatial crop + point cap,
mirrors reconstruct_mesh_test.py's test-patch pattern.

Usage:
    python scripts/floorplan_reconstruct_test.py [input.las] [output_dir] [crop_radius_m] [max_points]
"""
import sys
from pathlib import Path

try:
    from mesh_common import load_las_as_o3d, recenter_pcd, log
    from floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG, render_floorplan_image
except ImportError:
    from scripts.mesh_common import load_las_as_o3d, recenter_pcd, log
    from scripts.floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG, render_floorplan_image
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "floorplan_test"


def main(input_path, output_dir, crop_radius_m=8.0, max_points=1_500_000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = load_las_as_o3d(Path(input_path), crop_radius_m=crop_radius_m, max_points=max_points)
    recenter_pcd(pcd)
    xyz = np.asarray(pcd.points)

    manifest, walls = build_floorplan_outputs(xyz, DEFAULT_CONFIG)
    log(f"Test-patch: detected {manifest['wall_count']} walls")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    wall_dicts = [{"p0": np.array(w.p0), "p1": np.array(w.p1),
                   "thickness_m": w.thickness_m, "length_m": w.length_m} for w in walls]
    render_floorplan_image(wall_dicts, {}, str(output_dir / "floorplan.png"))
    log(f"Wrote {output_dir}/manifest.json and floorplan.png -- inspect these against the real building before trusting a full run")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    crop_r = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    max_pts = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    main(inp, outp, crop_r, max_pts)
