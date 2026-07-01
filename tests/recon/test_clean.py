import numpy as np

from scripts.recon.schema import ScanData
from scripts.recon.clean import percentile_crop


def test_percentile_crop_drops_far_outliers():
    rng = np.random.default_rng(0)
    core = rng.random((100, 3))  # in [0,1]^3
    outliers = np.array([[100.0, 100.0, 100.0], [-50.0, 0.0, 0.0]])
    xyz = np.vstack([core, outliers])
    scan = ScanData(xyz=xyz)
    out = percentile_crop(scan, lo=1.0, hi=99.0, margin_m=0.1)
    assert out.n >= 95
    assert out.n <= 100  # the two far outliers are gone
    assert out.xyz.max() < 10.0


def test_percentile_crop_keeps_aligned_attrs():
    xyz = np.vstack([np.zeros((10, 3)), np.array([[1000.0, 1000.0, 1000.0]])])
    t = np.arange(11, dtype=float)
    scan = ScanData(xyz=xyz, gps_time=t)
    out = percentile_crop(scan, lo=0.0, hi=90.0, margin_m=0.0)
    assert out.gps_time is not None
    assert out.n == out.gps_time.shape[0]
    assert 1000.0 not in out.xyz
