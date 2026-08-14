import numpy as np
import pytest
from pathlib import Path

from rscene.core.points import PointSet

laspy = pytest.importorskip("laspy")

from rscene.io.las import file_sha256, load_las, save_las

# Path to the real scan fixture; extracted to a single location for easy file swaps
_REAL_SCAN = Path(__file__).parent.parent.parent / "data" / "isolated_structural_v2.las"


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
    assert restored.intensity.dtype == np.uint16
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


def test_all_zero_rgb_returns_none(tmp_path):
    """Synthetic test: all-zero RGB should return None, not a zero array."""
    n = 100
    ps = PointSet(
        xyz=np.random.rand(n, 3),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=np.ones(n, dtype=np.uint16),
        rgb=np.zeros((n, 3), dtype=np.uint8),
    )
    path = tmp_path / "zero_rgb.las"
    save_las(ps, str(path))
    restored = load_las(str(path))
    assert restored.rgb is None


def test_nonzero_rgb_round_trips_as_uint8(tmp_path):
    """Synthetic test: non-zero RGB survives round trip as uint8."""
    n = 100
    rng = np.random.default_rng(42)
    original_rgb = rng.integers(0, 255, (n, 3)).astype(np.uint8)
    ps = PointSet(
        xyz=rng.uniform(0, 5, (n, 3)),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=rng.integers(0, 65535, n).astype(np.uint16),
        rgb=original_rgb,
    )
    path = tmp_path / "color_rgb.las"
    save_las(ps, str(path))
    restored = load_las(str(path))
    assert restored.rgb is not None
    assert restored.rgb.shape == (n, 3)
    assert restored.rgb.dtype == np.uint8
    assert np.array_equal(restored.rgb, original_rgb)


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="isolated_structural_v2.las not found")
def test_real_scan_rgb_contract(tmp_path):
    """Real scan: verify RGB satisfies the contract (None or (N, 3) uint8).

    Reports which case was observed. If the file carries all-zero RGB,
    this assertion must pass with rgb=None. If a future variant carries
    colour, this assertion must still pass with an (N, 3) uint8 array.
    """
    restored = load_las(str(_REAL_SCAN))
    if restored.rgb is None:
        print("Observed: rgb is None (all-zero case)")
    else:
        assert restored.rgb.shape == (restored.n, 3)
        assert restored.rgb.dtype == np.uint8
        print(f"Observed: rgb is ({restored.n}, 3) uint8 with values in [{restored.rgb.min()}, {restored.rgb.max()}]")


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="isolated_structural_v2.las not found")
def test_real_scan_chunk_boundary_and_row_alignment(tmp_path):
    """Real scan: verify chunk boundary crossing and row alignment.

    The 3.6 M point scan exceeds _CHUNK=3_000_000, exercising the
    concatenation logic across chunk boundaries.
    """
    points = load_las(str(_REAL_SCAN))

    # Verify point count
    assert points.n == 3_608_374

    # Verify attributes are present and row-aligned
    assert points.gps_time is not None
    assert points.intensity is not None
    assert len(points.gps_time) == points.n
    assert len(points.intensity) == points.n

    # Check gps_time for monotonicity
    is_monotonic = np.all(np.diff(points.gps_time) >= 0)
    if not is_monotonic:
        # Report actual behavior
        diffs = np.diff(points.gps_time)
        negative_diffs = np.where(diffs < 0)[0]
        print(f"gps_time is NOT monotonically non-decreasing: {len(negative_diffs)} decreasing transitions")
        if len(negative_diffs) > 0:
            print(f"  First few: indices {negative_diffs[:5]} with diffs {diffs[negative_diffs[:5]]}")

    # Verify xyz has no NaN or inf
    assert not np.any(np.isnan(points.xyz))
    assert not np.any(np.isinf(points.xyz))
