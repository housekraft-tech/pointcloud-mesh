import numpy as np

from scripts.recon.schema import ScanData
from scripts.recon.frame import dominant_axes, axis_align


def _rot_z(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_dominant_axes_recovers_rotation():
    # Manhattan wall normals along +-X and +-Y, then rotate the world by 30 deg.
    base = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    base = np.repeat(base, 200, axis=0)
    rot = base @ _rot_z(30).T
    R = dominant_axes(rot)
    # Applying R should bring headings back to multiples of 90 deg.
    fixed = rot @ R.T
    ang = np.arctan2(fixed[:, 1], fixed[:, 0]) % (np.pi / 2.0)
    # every heading is ~0 mod 90 (within ~1 deg)
    err = np.minimum(ang, np.pi / 2.0 - ang)
    assert np.rad2deg(err).max() < 1.5


def test_dominant_axes_ignores_floor_normals():
    # mostly-vertical (floor/ceiling) normals must not drive the estimate
    normals = np.array([[0, 0, 1]] * 500 + [[1, 0, 0]] * 100, dtype=float)
    R = dominant_axes(normals)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_axis_align_rotates_points():
    xyz = np.array([[1.0, 0.0, 0.0]])
    scan = ScanData(xyz=xyz)
    out = axis_align(scan, _rot_z(90))
    assert np.allclose(out.xyz[0], [0.0, 1.0, 0.0], atol=1e-9)
