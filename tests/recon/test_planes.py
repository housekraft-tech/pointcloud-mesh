import numpy as np
import pytest

from scripts.recon.planes import label_plane


def test_label_plane_floor():
    assert label_plane((0, 0, 1), centroid_z=0.02, z_floor=0.0, z_ceiling=2.7) == "floor"


def test_label_plane_ceiling():
    assert label_plane((0, 0, -1), centroid_z=2.68, z_floor=0.0, z_ceiling=2.7) == "ceiling"


def test_label_plane_vertical():
    assert label_plane((1, 0, 0), centroid_z=1.3, z_floor=0.0, z_ceiling=2.7) == "vertical"
    assert label_plane((0, 1, 0.05), centroid_z=1.3, z_floor=0.0, z_ceiling=2.7) == "vertical"


try:
    import open3d  # noqa: F401
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_detect_planes_on_modular_house():
    from scripts.recon.planes import detect_planes
    from tests.fixtures import modular_house

    pts, _, meta = modular_house()
    # drop the far neighbour blob so we test on the isolated unit
    pts = pts[pts[:, 0] < 10.0]
    planes = detect_planes(
        pts, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )
    labels = [p.label for p in planes]
    assert "floor" in labels
    assert "ceiling" in labels
    assert labels.count("vertical") >= 4  # at least the 4 exterior walls


def test_detect_planes_accepts_seed_param():
    import inspect
    from scripts.recon.planes import detect_planes

    sig = inspect.signature(detect_planes)
    assert "seed" in sig.parameters
    assert sig.parameters["seed"].default == 0
