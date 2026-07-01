import pytest
o3d = pytest.importorskip("open3d")  # skips locally on the 32-bit dev machine; runs on the Jarvis Labs VM
import numpy as np
from scripts.mesh_common import crop_pcd_to_percentile_bounds


def test_crop_pcd_to_percentile_bounds_drops_stray_points():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=[3.0, 2.5, 1.3], scale=0.5, size=(2000, 3))
    stray = rng.uniform(-20, 20, size=(20, 3))
    xyz = np.vstack([real, stray])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    cropped_pcd, stats = crop_pcd_to_percentile_bounds(pcd, low_pct=1.0, high_pct=99.0, margin_m=0.5)
    assert len(cropped_pcd.points) < len(pcd.points)
    assert stats["dropped_fraction"] < 0.05
