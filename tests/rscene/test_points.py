import numpy as np
import pytest

from rscene.core.points import PointSet, add_gaussian_noise


def _sample_set(n=10):
    return PointSet(
        xyz=np.arange(3 * n, dtype=np.float64).reshape(n, 3),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=np.arange(n, dtype=np.uint16),
        rgb=np.zeros((n, 3), dtype=np.uint8),
    )


def test_n_reports_point_count():
    assert _sample_set(7).n == 7


def test_subset_keeps_every_attribute_row_aligned():
    ps = _sample_set(10)
    keep = np.array([0, 3, 9])
    sub = ps.subset(keep)

    assert sub.n == 3
    assert np.array_equal(sub.xyz, ps.xyz[keep])
    assert np.array_equal(sub.gps_time, ps.gps_time[keep])
    assert np.array_equal(sub.intensity, ps.intensity[keep])
    assert np.array_equal(sub.rgb, ps.rgb[keep])


def test_subset_tolerates_absent_optional_attributes():
    ps = PointSet(xyz=np.zeros((5, 3)))
    sub = ps.subset(np.array([True, False, True, False, True]))
    assert sub.n == 3
    assert sub.gps_time is None and sub.intensity is None and sub.rgb is None


def test_mismatched_attribute_length_is_rejected():
    with pytest.raises(ValueError, match="row count"):
        PointSet(xyz=np.zeros((5, 3)), gps_time=np.zeros(4))


def test_noise_has_the_requested_sigma_and_is_reproducible():
    xyz = np.zeros((200_000, 3))
    a = add_gaussian_noise(xyz, sigma_m=0.002, rng=np.random.default_rng(0))
    b = add_gaussian_noise(xyz, sigma_m=0.002, rng=np.random.default_rng(0))

    assert np.array_equal(a, b)                      # same seed, same result
    assert abs(a.std() - 0.002) < 0.0001
    assert a is not xyz                              # input not mutated
    assert np.array_equal(xyz, np.zeros((200_000, 3)))
