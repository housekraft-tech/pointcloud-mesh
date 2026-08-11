import numpy as np
import pytest

from rscene.core.fitting import canonical_normal, fit_plane, plane_basis, plane_distance


def test_canonical_normal_makes_dominant_component_positive():
    assert np.allclose(canonical_normal(np.array([0.0, 0.0, -1.0])), [0.0, 0.0, 1.0])
    assert np.allclose(canonical_normal(np.array([-0.9, 0.1, 0.0])), [0.9, -0.1, 0.0])


def test_opposite_normals_canonicalise_to_the_same_direction():
    n = np.array([0.6, -0.8, 0.0])
    assert np.allclose(canonical_normal(n), canonical_normal(-n))


def test_fit_plane_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 4, 5000), rng.uniform(0, 3, 5000),
                           np.full(5000, 2.75)])
    normal, d = fit_plane(pts)

    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)
    # plane is z = 2.75  ->  1*z - 2.75 = 0
    assert abs(d + 2.75) < 1e-9


def test_fit_plane_recovers_offset_between_two_parallel_planes():
    """The core property the whole rebuild depends on: a 75 mm step is 75 mm."""
    rng = np.random.default_rng(1)
    face = np.column_stack([np.zeros(4000), rng.uniform(0, 4, 4000),
                            rng.uniform(0, 2.75, 4000)])
    step = np.column_stack([np.full(4000, 0.075), rng.uniform(2.8, 3.15, 4000),
                            rng.uniform(0, 2.75, 4000)])

    n1, d1 = fit_plane(face)
    n2, d2 = fit_plane(step)

    assert np.allclose(n1, n2, atol=1e-9)          # canonical -> same direction
    assert abs(abs(d1 - d2) - 0.075) < 1e-6


def test_plane_distance_is_signed_and_scaled_in_metres():
    normal, d = np.array([0.0, 0.0, 1.0]), -2.0
    pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.5], [0.0, 0.0, 1.5]])
    assert np.allclose(plane_distance(pts, normal, d), [0.0, 0.5, -0.5])


def test_plane_basis_is_orthonormal_and_deterministic():
    n = canonical_normal(np.array([0.0, 0.0, 1.0]))
    u, v = plane_basis(n)

    assert abs(u @ v) < 1e-12
    assert abs(u @ n) < 1e-12 and abs(v @ n) < 1e-12
    assert abs(np.linalg.norm(u) - 1) < 1e-12 and abs(np.linalg.norm(v) - 1) < 1e-12
    assert np.array_equal(u, plane_basis(n)[0])


def test_fit_plane_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 3"):
        fit_plane(np.zeros((2, 3)))
