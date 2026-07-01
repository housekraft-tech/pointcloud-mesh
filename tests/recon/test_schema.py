import numpy as np

from scripts.recon.schema import ScanData, Plane, WallStep, Column, Beam, new_column_id, new_beam_id


def test_scandata_subset_keeps_aligned_fields():
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    t = np.array([1.0, 2.0, 3.0, 4.0])
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.uint8)
    s = ScanData(xyz=xyz, gps_time=t, rgb=rgb, intensity=None)
    assert s.n == 4
    sub = s.subset(np.array([True, False, True, False]))
    assert sub.n == 2
    assert np.allclose(sub.gps_time, [1.0, 3.0])
    assert np.array_equal(sub.rgb, [[1, 1, 1], [3, 3, 3]])
    assert sub.intensity is None


def test_scandata_subset_with_index_array():
    xyz = np.arange(9, dtype=float).reshape(3, 3)
    s = ScanData(xyz=xyz)
    sub = s.subset(np.array([2, 0]))
    assert np.allclose(sub.xyz, [[6, 7, 8], [0, 1, 2]])


def test_plane_signed_distance():
    p = Plane(normal=(0.0, 0.0, 1.0), d=-2.0, label="floor", inlier_idx=np.array([]))
    dist = p.signed_distance(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [5.0, 5.0, 1.5]]))
    assert np.allclose(dist, [0.0, 1.0, -0.5])


def test_ids_zero_padded():
    assert new_column_id(3) == "column_003"
    assert new_beam_id(11) == "beam_011"


def test_wallstep_and_element_fields():
    ws = WallStep(offset_m=0.1, u_min_m=1.0, u_max_m=1.3, z_min_m=0.0, z_max_m=2.4)
    assert ws.offset_m == 0.1
    c = Column(column_id="column_000", footprint=[(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)], z_min_m=0.0, z_max_m=2.4)
    assert len(c.footprint) == 4
    b = Beam(beam_id="beam_000", p0=(0, 0), p1=(3, 0), width_m=0.2, depth_m=0.3, z_min_m=2.1, z_max_m=2.4)
    assert b.width_m == 0.2
