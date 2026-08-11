import numpy as np

from rscene.core.normals import estimate_normals
from rscene.core.prim import Box


def test_normals_on_a_flat_face_point_along_the_face_normal():
    pts = Box("f", (0, 0, 0), (2, 2, 0)).sample_surface(0.02, faces=("z+",))
    normals, curvature = estimate_normals(pts, k=16)

    assert np.abs(normals[:, 2]).min() > 0.99
    assert curvature.max() < 0.01


def test_curvature_is_higher_at_an_edge_than_on_a_face():
    box = Box("b", (0, 0, 0), (1, 1, 1))
    pts = box.sample_surface(0.02, faces=("z+", "x+"))
    normals, curvature = estimate_normals(pts, k=16)

    near_edge = np.abs(pts[:, 0] - 1.0) < 0.01
    on_face = pts[:, 0] < 0.5
    assert curvature[near_edge].mean() > curvature[on_face].mean() * 5


def test_normals_are_canonically_oriented_and_unit_length():
    pts = Box("f", (0, 0, 0), (1, 1, 0)).sample_surface(0.05, faces=("z+",))
    normals, _ = estimate_normals(pts, k=12)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # canonical: dominant component positive, so all z-normals point +z
    assert (normals[:, 2] > 0).all()


def test_estimation_is_deterministic():
    pts = Box("f", (0, 0, 0), (1, 1, 0)).sample_surface(0.05, faces=("z+",))
    a, ca = estimate_normals(pts, k=12)
    b, cb = estimate_normals(pts, k=12)
    assert np.array_equal(a, b) and np.array_equal(ca, cb)
