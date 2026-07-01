import pytest
o3d = pytest.importorskip("open3d")
import json
import numpy as np
import laspy
from scripts.floorplan_reconstruct import main as floorplan_main
from tests.fixtures import two_room_house


def _write_synthetic_las(path, points):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(str(path))


def test_floorplan_reconstruct_cli_writes_all_outputs(tmp_path):
    pts, _gt = two_room_house()
    las_path = tmp_path / "synthetic.las"
    _write_synthetic_las(las_path, pts)
    out_dir = tmp_path / "output"

    floorplan_main(str(las_path), str(out_dir))

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["wall_count"] == 5

    assert (out_dir / "floorplan.png").exists()
    assert (out_dir / "reconstructed.obj").exists()
