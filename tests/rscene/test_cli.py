import json
from pathlib import Path

import numpy as np
import pytest

from rscene.core.points import PointSet
from rscene.core.prim import Box

pytest.importorskip("laspy")

from rscene.cli import main
from rscene.io.las import load_las, save_las

# Path to the real scan fixture; mirrors tests/rscene/test_io_las.py so both
# suites point at a single location for easy file swaps.
_REAL_SCAN = Path(__file__).parent.parent.parent / "data" / "isolated_structural_v2.las"


def _write_scan(tmp_path):
    pts = np.concatenate([
        Box("floor", (0, 0, 0), (2, 2, 0)).sample_surface(0.01, faces=("z+",)),
        Box("wall", (0, 0, 0), (0, 2, 2.5)).sample_surface(0.01, faces=("x+",)),
    ])
    path = tmp_path / "scan.las"
    save_las(PointSet(xyz=pts), str(path))
    return str(path)


def test_cli_writes_scene_and_report(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"

    assert main(["patches", scan, str(out)]) == 0
    assert (out / "scene.json").exists()
    assert (out / "report.md").exists()


def test_scene_json_records_patches_and_provenance(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"
    main(["patches", scan, str(out)])

    payload = json.loads((out / "scene.json").read_text())
    assert len(payload["patches"]) >= 2
    assert payload["provenance"]["scan_sha256"]
    assert payload["provenance"]["config"]["tau_fit_m"] == 0.003


def test_report_states_the_unassigned_count(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"
    main(["patches", scan, str(out)])

    report = (out / "report.md").read_text()
    assert "Unassigned points" in report


def test_two_runs_produce_identical_scene_json(tmp_path):
    scan = _write_scan(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    main(["patches", scan, str(a)])
    main(["patches", scan, str(b)])

    assert (a / "scene.json").read_text() == (b / "scene.json").read_text()


def test_missing_input_exits_nonzero(tmp_path):
    assert main(["patches", str(tmp_path / "nope.las"), str(tmp_path / "out")]) == 2


def test_max_points_warns_on_stderr(tmp_path, capsys):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"

    main(["patches", scan, str(out), "--max-points", "50"])

    captured = capsys.readouterr()
    assert "subsampling" in captured.err.lower() or "subsample" in captured.err.lower()
    assert "spatial subregion" in captured.err.lower() or "spatial" in captured.err.lower()


def test_max_points_omitted_prints_no_warning(tmp_path, capsys):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"

    main(["patches", scan, str(out)])

    captured = capsys.readouterr()
    assert "subsampl" not in captured.err.lower()


def test_unassigned_buckets_sum_to_unassigned_total(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"
    main(["patches", scan, str(out)])

    payload = json.loads((out / "scene.json").read_text())
    buckets = payload["diagnostics"]["unassigned_buckets"]
    assert sum(buckets.values()) == payload["unassigned_points"]

    report = (out / "report.md").read_text()
    for name in ("high_curvature", "no_plane_within_tolerance", "isolated", "other"):
        assert name in report


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="real scan fixture not present")
def test_real_scan_cli_end_to_end(tmp_path):
    from rscene.core.scene import scene_from_json

    full = load_las(str(_REAL_SCAN))
    xyz = full.xyz
    mask = (
        (xyz[:, 0] > -3.2) & (xyz[:, 0] < 1.0)
        & (xyz[:, 1] > -8.0) & (xyz[:, 1] < -3.0)
    )
    idx = np.nonzero(mask)[0]
    cropped = full.subset(idx)

    cropped_path = tmp_path / "cropped.las"
    save_las(cropped, str(cropped_path))

    out = tmp_path / "out"
    rc = main([
        "patches", str(cropped_path), str(out),
        "--set", "patch_neighbor_k=64",
    ])
    assert rc == 0
    assert (out / "scene.json").exists()
    assert (out / "report.md").exists()

    payload = json.loads((out / "scene.json").read_text())
    scene = scene_from_json((out / "scene.json").read_text())
    assert scene.patches

    assert 4.0 <= scene.frame.xy_rotation_deg <= 6.5
    assert 2.6 <= (scene.frame.ceiling_z - scene.frame.floor_z) <= 2.9

    residuals = sorted(p.p95_residual_m for p in scene.patches)
    median_residual = residuals[len(residuals) // 2]
    assert median_residual < 0.005

    buckets = payload["diagnostics"]["unassigned_buckets"]
    assert sum(buckets.values()) == payload["unassigned_points"]
