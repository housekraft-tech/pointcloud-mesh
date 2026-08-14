import numpy as np

from itertools import combinations

from rscene.config import merged_config
from rscene.core.graph import perpendicular_offset
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.points import add_gaussian_noise
from rscene.core.prim import Box


def _max_pairwise_offset(patches):
    """Largest perpendicular_offset among all pairs -- the two farthest-apart
    same-normal patches, without relying on origin-referenced Patch.d."""
    return max(perpendicular_offset(a, b) for a, b in combinations(patches, 2))


def _wall_with_step(spacing=0.008, noise_m=0.001):
    """A wall face at x=0 with a 75 mm rectangular extrusion standing on it.

    The wall's own face is sampled only where the extrusion does not cover it,
    exactly as a scanner would see it.
    """
    below = Box("wall_lo", (0.0, 0.0, 0.0), (0.0, 2.80, 2.75))
    above = Box("wall_hi", (0.0, 3.15, 0.0), (0.0, 4.00, 2.75))
    step_face = Box("step_face", (0.075, 2.80, 0.0), (0.075, 3.15, 2.75))
    step_side_a = Box("step_a", (0.0, 2.80, 0.0), (0.075, 2.80, 2.75))
    step_side_b = Box("step_b", (0.0, 3.15, 0.0), (0.075, 3.15, 2.75))

    pts = np.concatenate([
        below.sample_surface(spacing, faces=("x+",)),
        above.sample_surface(spacing, faces=("x+",)),
        step_face.sample_surface(spacing, faces=("x+",)),
        step_side_a.sample_surface(spacing, faces=("y-",)),
        step_side_b.sample_surface(spacing, faces=("y+",)),
    ])
    return add_gaussian_noise(pts, noise_m, np.random.default_rng(0))


def test_a_75mm_step_survives_as_its_own_patch():
    """The defining test of the rebuild: the step must NOT be absorbed."""
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, labels = extract_patches(xyz, normals, curvature, merged_config())

    x_facing = [p for p in patches if abs(p.normal[0]) > 0.99]
    assert len(x_facing) >= 2, "the step was absorbed into the wall face"

    assert abs(_max_pairwise_offset(x_facing) - 0.075) < 0.003


def test_the_step_side_faces_are_found_as_separate_patches():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    y_facing = [p for p in patches if abs(p.normal[1]) > 0.99]
    assert len(y_facing) >= 2

    assert abs(_max_pairwise_offset(y_facing) - 0.35) < 0.003   # step is 350 mm wide


def test_no_surface_is_snapped_to_another():
    """Two nearly-parallel faces stay distinct rather than collapsing to one.

    The gap between the faces (30 mm in y) is INSIDE patch_connect_radius_m
    (50 mm), so region growth genuinely has the opportunity to jump between
    them -- unlike a 10x-radius gap, which is trivially disjoint and cannot
    exercise the invariant this test exists to guard. What must keep the two
    surfaces apart is the 20 mm offset between their planes (in x), not
    spatial separation.
    """
    a = Box("a", (0.0, 0.0, 0.0), (0.0, 2.0, 2.0)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0.02, 2.03, 0.0), (0.02, 4.03, 2.0)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    x_facing = [p for p in patches if abs(p.normal[0]) > 0.99]
    assert len(x_facing) == 2
    assert abs(perpendicular_offset(x_facing[0], x_facing[1]) - 0.02) < 0.002


def test_labels_cover_every_point_or_mark_it_unassigned():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, labels = extract_patches(xyz, normals, curvature, merged_config())

    assert labels.shape == (len(xyz),)
    for p in patches:
        assert np.array_equal(np.sort(p.point_idx), np.sort(np.flatnonzero(labels == p.patch_id)))
    assert set(np.unique(labels)) <= {-1} | {p.patch_id for p in patches}


def test_patch_records_its_own_residual_and_extent():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    big = max(patches, key=lambda p: p.n_points)
    assert big.p95_residual_m < 0.004          # ~1 mm noise, 3 mm tolerance
    assert big.u_range[1] > big.u_range[0]
    assert big.v_range[1] > big.v_range[0]


def test_extraction_is_deterministic():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    a, la = extract_patches(xyz, normals, curvature, merged_config())
    b, lb = extract_patches(xyz, normals, curvature, merged_config())

    assert np.array_equal(la, lb)
    assert [p.d for p in a] == [p.d for p in b]
