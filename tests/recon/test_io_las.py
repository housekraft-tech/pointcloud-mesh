import numpy as np
import laspy

from scripts.recon.io_las import load_scan, save_scan_las
from scripts.recon.schema import ScanData


def _write_las(path, xyz, t=None, rgb=None):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.001, 0.001, 0.001]
    h.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(h)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    if t is not None:
        las.gps_time = t
    if rgb is not None:
        las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(str(path))


def test_load_scan_roundtrip(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=float)
    p = tmp_path / "a.las"
    _write_las(p, xyz, t=np.array([10.0, 11.0, 12.0]))
    s = load_scan(str(p))
    assert s.n == 3
    assert np.allclose(np.sort(s.xyz[:, 0]), [0, 1, 4], atol=1e-3)
    assert s.gps_time is not None
    assert np.allclose(np.sort(s.gps_time), [10, 11, 12])


def test_load_scan_rgb_all_zero_is_none(tmp_path):
    xyz = np.random.default_rng(0).random((20, 3))
    p = tmp_path / "b.las"
    _write_las(p, xyz)  # no rgb set -> all zero
    s = load_scan(str(p))
    assert s.rgb is None


def test_load_scan_rgb_downscaled(tmp_path):
    xyz = np.zeros((2, 3))
    rgb = np.array([[65535, 0, 65535], [0, 65535, 0]], dtype=np.uint16)
    p = tmp_path / "c.las"
    _write_las(p, xyz, rgb=rgb)
    s = load_scan(str(p))
    assert s.rgb is not None
    assert s.rgb.dtype == np.uint8
    assert s.rgb.max() == 255


def test_load_scan_subsample(tmp_path):
    xyz = np.random.default_rng(1).random((1000, 3))
    p = tmp_path / "d.las"
    _write_las(p, xyz)
    s = load_scan(str(p), max_points=100)
    assert s.n == 100


def test_save_scan_roundtrip(tmp_path):
    xyz = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    t = np.array([100.0, 101.0])
    p = tmp_path / "out.las"
    save_scan_las(ScanData(xyz=xyz, gps_time=t), str(p))
    s = load_scan(str(p))
    assert s.n == 2
    assert np.allclose(np.sort(s.xyz[:, 2]), [0.3, 3.0], atol=1e-3)
    assert s.gps_time is not None and np.allclose(np.sort(s.gps_time), [100.0, 101.0])
