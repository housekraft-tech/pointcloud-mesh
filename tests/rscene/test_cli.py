import json
from pathlib import Path

import numpy as np
import pytest

from rscene.config import merged_config
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
    for name in ("high_curvature", "plane_exists_but_unassigned", "isolated", "no_plane_within_tolerance"):
        assert name in report


def test_unassigned_buckets_classify_each_kind_of_point_correctly():
    # Synthetic case with one hand-placed point per bucket, so each is known
    # in advance rather than only checking the sum (which the inverted
    # bucket bug survived undetected under for a whole review cycle).
    from rscene.cli import _classify_unassigned
    from rscene.core.patches import Patch

    config = merged_config()
    max_curvature = float(config["patch_max_curvature"])
    tau = float(config["tau_fit_m"])

    # One patch: the plane x = 0.
    patch = Patch(
        patch_id=0, normal=np.array([1.0, 0.0, 0.0]), d=0.0,
        point_idx=np.array([0]), n_points=1, p95_residual_m=0.0,
        centroid=np.zeros(3), u_range=(0.0, 1.0), v_range=(0.0, 1.0),
    )

    # idx 0: high_curvature -- curvature far above the gate, regardless of
    # position or normal.
    p_high_curv = np.array([50.0, 50.0, 50.0])

    # idx 1: plane_exists_but_unassigned -- low curvature, normal agrees
    # with the patch's normal, and it sits well within tau_fit_m of the
    # plane x = 0.
    p_matched = np.array([0.0005, 5.0, 5.0])

    # idx 2: isolated -- low curvature, normal disagrees with the patch (no
    # plane match), and it is far from every other point (fewer than
    # _MIN_NEIGHBOURS neighbours within patch_connect_radius_m).
    p_isolated = np.array([100.0, 100.0, 100.0])

    # idx 3-6: no_plane_within_tolerance -- a small dense cluster whose
    # normal disagrees with the patch (no plane match) but which has
    # plenty of close neighbours, so it is not isolated either.
    p_cluster = np.array([
        [10.0, 10.0, 10.0],
        [10.01, 10.0, 10.0],
        [10.0, 10.01, 10.0],
        [10.0, 10.0, 10.01],
    ])

    xyz = np.vstack([p_high_curv, p_matched, p_isolated, p_cluster])
    n = len(xyz)

    normals = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    normals[1] = [1.0, 0.0, 0.0]        # matches the patch's normal

    curvature = np.full(n, 0.0001)
    curvature[0] = max_curvature * 10   # far above the gate

    labels = np.full(n, -1, dtype=np.int64)

    buckets = _classify_unassigned(xyz, normals, curvature, labels, [patch], config)

    assert sum(buckets.values()) == n

    # Re-derive each point's bucket the same way _classify_unassigned does,
    # to assert against by index rather than only against totals.
    idx_high_curv, idx_matched, idx_isolated = 0, 1, 2
    idx_cluster = 3

    # high_curvature point is counted once, matched/isolated/cluster points
    # are not high curvature.
    assert buckets["high_curvature"] == 1

    # The matched point is the only one whose normal agrees with the patch
    # and whose distance to the plane is within tau.
    dist_matched = abs(float(xyz[idx_matched] @ patch.normal + patch.d))
    assert dist_matched <= tau
    assert buckets["plane_exists_but_unassigned"] == 1

    # The isolated point and the 4-point cluster together make up the
    # remaining points that don't match the plane; isolated has none of its
    # own kind nearby, the cluster has plenty of its own kind nearby.
    assert buckets["isolated"] == 1
    assert buckets["no_plane_within_tolerance"] == 4
    assert idx_isolated == 2 and idx_cluster == 3   # documents point layout


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
