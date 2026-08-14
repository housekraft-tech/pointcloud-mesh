import numpy as np
import pytest

from rscene.core.points import PointSet

laspy = pytest.importorskip("laspy")

from rscene.io.las import file_sha256, load_las, save_las


def _sample(tmp_path):
    n = 500
    rng = np.random.default_rng(0)
    ps = PointSet(
        xyz=rng.uniform(0, 5, (n, 3)),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=rng.integers(0, 65535, n).astype(np.uint16),
        rgb=rng.integers(0, 255, (n, 3)).astype(np.uint8),
    )
    path = tmp_path / "scan.las"
    save_las(ps, str(path))
    return ps, str(path)


def test_round_trip_preserves_geometry_to_the_las_scale(tmp_path):
    original, path = _sample(tmp_path)
    restored = load_las(path)

    assert restored.n == original.n
    assert np.allclose(restored.xyz, original.xyz, atol=0.0005)   # 0.1 mm scale


def test_round_trip_preserves_every_attribute(tmp_path):
    original, path = _sample(tmp_path)
    restored = load_las(path)

    assert np.allclose(restored.gps_time, original.gps_time)
    assert np.array_equal(restored.intensity, original.intensity)
    assert np.array_equal(restored.rgb, original.rgb)


def test_max_points_subsamples_deterministically(tmp_path):
    _, path = _sample(tmp_path)
    a = load_las(path, max_points=100, seed=7)
    b = load_las(path, max_points=100, seed=7)

    assert a.n == 100
    assert np.array_equal(a.xyz, b.xyz)


def test_max_points_above_the_count_is_a_no_op(tmp_path):
    original, path = _sample(tmp_path)
    assert load_las(path, max_points=10_000).n == original.n


def test_sha256_is_stable_and_content_dependent(tmp_path):
    _, path = _sample(tmp_path)
    assert file_sha256(path) == file_sha256(path)

    other = tmp_path / "other.las"
    save_las(PointSet(xyz=np.zeros((10, 3))), str(other))
    assert file_sha256(path) != file_sha256(str(other))
