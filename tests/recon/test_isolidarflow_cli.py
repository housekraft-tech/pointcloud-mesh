"""End-to-end CLI test for scripts/isolidarflow.py.

Writes the synthetic modular_house() fixture to a temp LAS, runs the whole
pipeline via run(), and asserts the five deliverables exist and the manifest
carries the structural elements the fixture was built to exercise.
"""
import json

import numpy as np
import laspy

from tests.fixtures import modular_house


def _write_las(path, xyz, t):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.001, 0.001, 0.001]
    h.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.gps_time = t
    las.write(str(path))


def test_isolidarflow_end_to_end(tmp_path):
    from scripts.isolidarflow import run, DEFAULT_CONFIG

    pts, gps, meta = modular_house()
    in_las = tmp_path / "modular_house.las"
    _write_las(in_las, pts, gps)

    out_dir = tmp_path / "out"

    config = dict(DEFAULT_CONFIG)
    config["remove_outliers"] = False  # keep the test fast + deterministic

    result = run(str(in_las), str(out_dir), config)

    # --- five deliverables exist ---
    for name in ("model.glb", "floorplan.dxf", "floorplan.svg", "manifest.json", "report.txt"):
        assert (out_dir / name).exists(), f"missing deliverable: {name}"

    manifest = json.loads((out_dir / "manifest.json").read_text())

    # --- structural element counts ---
    assert len(manifest["walls"]) >= 4, f"expected >=4 walls, got {len(manifest['walls'])}"
    assert len(manifest["columns"]) == 1, f"expected exactly 1 column, got {len(manifest['columns'])}"
    assert len(manifest["beams"]) >= 1, f"expected >=1 beam, got {len(manifest['beams'])}"

    opening_types = {o["type"] for o in manifest["openings"]}
    assert "door" in opening_types, f"no door found; opening types = {opening_types}"
    assert "window" in opening_types, f"no window found; opening types = {opening_types}"
    assert "balcony_door" in opening_types, f"no balcony_door found; opening types = {opening_types}"

    # --- isolation removed the neighbour blob: no element inside neighbour_bbox ---
    nb = meta["neighbour_bbox"]
    (nx0, nx1), (ny0, ny1) = nb["x"], nb["y"]

    def _inside(x, y):
        return nx0 <= x <= nx1 and ny0 <= y <= ny1

    for w in manifest["walls"]:
        for (x, y) in (w["p0"], w["p1"]):
            assert not _inside(x, y), f"wall endpoint {(x, y)} inside neighbour bbox"
    for c in manifest["columns"]:
        for (x, y) in c["footprint"]:
            assert not _inside(x, y), f"column corner {(x, y)} inside neighbour bbox"
    for b in manifest["beams"]:
        for (x, y) in (b["p0"], b["p1"]):
            assert not _inside(x, y), f"beam endpoint {(x, y)} inside neighbour bbox"

    # run() returns a summary dict pointing at the artifacts + parsed manifest
    assert result["manifest"] is manifest or result["manifest"]["walls"]
