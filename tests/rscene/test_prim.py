import numpy as np

from rscene.core.prim import FACE_KEYS, Box


def test_sampled_points_lie_on_the_box_surface():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 2.0, 3.0))
    pts = box.sample_surface(spacing_m=0.05)

    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 2.0, 3.0])
    # every point sits inside the box, and touches at least one face
    assert np.all(pts >= lo - 1e-9)
    assert np.all(pts <= hi + 1e-9)
    on_a_face = np.isclose(pts, lo).any(axis=1) | np.isclose(pts, hi).any(axis=1)
    assert on_a_face.all()


def test_single_face_sampling_is_planar_and_correctly_placed():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 2.0, 3.0))
    pts = box.sample_surface(spacing_m=0.1, faces=("x+",))

    assert np.allclose(pts[:, 0], 1.0)
    assert pts[:, 1].min() == 0.0 and np.isclose(pts[:, 1].max(), 2.0)
    assert pts[:, 2].min() == 0.0 and np.isclose(pts[:, 2].max(), 3.0)


def test_sampling_is_deterministic():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    assert np.array_equal(box.sample_surface(0.05), box.sample_surface(0.05))


def test_all_six_faces_are_produced():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    per_face = {k: box.sample_surface(0.25, faces=(k,)) for k in FACE_KEYS}
    assert len(per_face) == 6
    assert all(len(v) > 0 for v in per_face.values())
