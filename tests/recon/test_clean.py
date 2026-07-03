import numpy as np

from scripts.recon.schema import ScanData, WallStep
from scripts.recon.clean import percentile_crop, remove_furniture


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


def test_furniture_blob_dropped_structure_kept():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 6, 3000), rng.uniform(0, 4, 3000), np.zeros(3000)])
    wall = np.column_stack([rng.uniform(0, 6, 3000), np.full(3000, 4.0), rng.uniform(0, 2.7, 3000)])
    sofa = np.column_stack([rng.uniform(2, 3, 2000), rng.uniform(1.5, 2.5, 2000), rng.uniform(0.3, 1.0, 2000)])
    scan = ScanData(xyz=np.vstack([floor, wall, sofa]), gps_time=None, rgb=None, intensity=None)
    walls = [dict(direction="y", offset_m=4.0, p0=(0.0, 4.0), p1=(6.0, 4.0),
                  thickness_m=0.1, steps=[WallStep(0.0, 0.0, 6.0, 0.0, 2.7)])]
    kept, dropped = remove_furniture(scan, walls, z_floor=0.0, z_ceiling=2.7)
    assert dropped >= 1900                      # sofa gone
    assert kept.n >= 5900                       # floor + wall kept
    assert kept.xyz[:, 1].max() > 3.9           # wall points survived
