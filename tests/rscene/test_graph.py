import numpy as np

from rscene.config import merged_config
from rscene.core.graph import coplanarity_classes, intersection_line, patch_adjacency, perpendicular_offset
from rscene.core.normals import estimate_normals
from rscene.core.patches import Patch, extract_patches
from rscene.core.prim import Box


def _patch(pid, normal, d):
    # Default centroid sits ON the plane the patch claims (normal @ x + d ==
    # 0), matching how a real fitted Patch's centroid always lies on its own
    # plane. Coplanarity now measures the gap via perpendicular_offset
    # (centroid to centroid), so a centroid inconsistent with (normal, d)
    # would silently misrepresent the patch being tested.
    normal = np.array(normal, dtype=float)
    centroid = -d * normal
    return Patch(
        patch_id=pid, normal=normal, d=d,
        point_idx=np.array([0]), n_points=1, p95_residual_m=0.0,
        centroid=centroid, u_range=(0.0, 1.0), v_range=(0.0, 1.0),
    )


def test_coplanar_patches_share_a_class_but_keep_their_own_planes():
    a = _patch(0, (1, 0, 0), -2.000)
    b = _patch(1, (1, 0, 0), -2.002)     # 2 mm apart: same plane within tolerance
    c = _patch(2, (1, 0, 0), -2.500)     # clearly a different plane

    classes = coplanarity_classes([a, b, c], merged_config())

    assert [0, 1] in classes and [2] in classes
    assert a.d == -2.000 and b.d == -2.002      # untouched, never averaged


def test_parallel_but_offset_patches_are_not_coplanar():
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, (1, 0, 0), -0.075)             # the 75 mm step
    assert coplanarity_classes([a, b], merged_config()) == [[0], [1]]


def test_patches_with_different_normals_are_not_coplanar():
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, (0, 1, 0), 0.0)
    assert coplanarity_classes([a, b], merged_config()) == [[0], [1]]


def test_intersection_line_of_two_perpendicular_planes():
    a = _patch(0, (1, 0, 0), 0.0)        # x = 0
    b = _patch(1, (0, 1, 0), 0.0)        # y = 0
    point, direction = intersection_line(a, b, merged_config())

    assert np.allclose(np.abs(direction), [0, 0, 1])
    assert abs(point[0]) < 1e-9 and abs(point[1]) < 1e-9


def test_intersection_line_of_parallel_planes_is_none():
    assert intersection_line(_patch(0, (1, 0, 0), 0.0),
                             _patch(1, (1, 0, 0), -0.2), merged_config()) is None


def test_intersection_line_of_near_parallel_planes_rejects_numerically_unstable_case():
    # Planes at 0.001° apart: sin(0.001°) ≈ 1.7e-5, which is much smaller than
    # the default min_intersection_angle_deg threshold (~0.5°, sin ≈ 0.0087).
    # This should return None to avoid numerically unstable computation.
    angle_rad = np.radians(0.001)
    # Create a second normal by rotating (1,0,0) by tiny angle around z
    normal2 = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, normal2, 0.0)
    config = merged_config()
    assert intersection_line(a, b, config) is None


def test_intersection_line_uses_default_config_when_not_provided():
    # Verifies the no-config fallback path works correctly.
    # When called without config, intersection_line should use DEFAULT_CONFIG.
    a = _patch(0, (1, 0, 0), 0.0)        # x = 0
    b = _patch(1, (0, 1, 0), 0.0)        # y = 0
    point, direction = intersection_line(a, b)  # no config argument

    assert np.allclose(np.abs(direction), [0, 0, 1])
    assert abs(point[0]) < 1e-9 and abs(point[1]) < 1e-9


def test_adjacent_patches_are_detected_and_distant_ones_are_not():
    corner = np.concatenate([
        Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",)),
        Box("b", (0, 0, 0), (2, 0, 2)).sample_surface(0.01, faces=("y+",)),
    ])
    far = Box("c", (5, 5, 0), (5, 7, 2)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([corner, far])

    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())
    pairs = patch_adjacency(patches, xyz, merged_config())

    ids_near = {p.patch_id for p in patches if p.centroid[0] < 3 and p.centroid[1] < 3}
    ids_far = {p.patch_id for p in patches if p.centroid[0] > 3}

    assert any(a in ids_near and b in ids_near for a, b in pairs)
    assert not any((a in ids_far) != (b in ids_far) for a, b in pairs)


def test_perpendicular_offset_is_symmetric_under_slightly_different_normals():
    # Two parallel-ish patches whose fitted normals differ by a tiny tilt --
    # exactly the situation that made the old a.normal-only projection
    # asymmetric. With the bisector, argument order must not matter.
    angle_rad = np.radians(0.02)
    normal_a = np.array([1.0, 0.0, 0.0])
    normal_b = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

    a = _patch(0, normal_a, 0.0)
    b = _patch(1, normal_b, -0.045)
    a.centroid = np.array([0.0, 0.0, 0.0])
    b.centroid = np.array([0.045, 0.01, 0.02])

    assert perpendicular_offset(a, b) == perpendicular_offset(b, a)


def test_perpendicular_offset_raises_on_opposed_normals():
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, (-1, 0, 0), -0.045)
    try:
        perpendicular_offset(a, b)
        assert False, "expected ValueError for opposed normals"
    except ValueError:
        pass


def test_coplanar_patches_far_from_origin_share_a_class_despite_d_lever_arm():
    # Two genuinely coplanar patches, 8-15 m from the world origin, whose
    # fitted normals differ by ~0.2 deg (ordinary fitting noise). Differencing
    # a.d and b.d directly (the old, forbidden predicate) lever-arms that 0.2
    # deg disagreement by the patches' distance-from-origin components and
    # produces a spurious ~30 mm gap -- comfortably over coplanar_dist_tol_m
    # (5 mm) -- even though the true perpendicular offset between the patches
    # (measured centroid to centroid, via perpendicular_offset) is under 1 mm.
    # This is the false-negative failure mode from the review: 241 of 331
    # genuinely coplanar pairs on the real crop were missed this way.
    normal_a = np.array([1.0, 0.0, 0.0])
    centroid_a = np.array([10.0, 8.0, -3.0])
    a = _patch(0, normal_a, -float(normal_a @ centroid_a))
    a.centroid = centroid_a

    theta = np.radians(0.2)
    normal_b = np.array([np.cos(theta), np.sin(theta), 0.0])
    centroid_b = np.array([10.0, 8.5, -2.5])   # shifted WITHIN the x=10 plane
    b = _patch(1, normal_b, -float(normal_b @ centroid_b))
    b.centroid = centroid_b

    config = merged_config()
    dist_tol = float(config["coplanar_dist_tol_m"])

    # Sanity: the forbidden d-difference predicate would have rejected this
    # pair (proving this is a real regression test, not a vacuous one).
    assert abs(a.d - b.d) > dist_tol
    # The real perpendicular offset is small -- these ARE the same surface.
    assert perpendicular_offset(a, b) < dist_tol

    classes = coplanarity_classes([a, b], config)
    assert classes == [[0, 1]]


def test_coplanar_by_d_but_truly_offset_patches_land_in_different_classes():
    # Converse of the case above: two patches at similar distance from the
    # origin whose d values coincidentally agree to sub-micron precision
    # (a small normal tilt exactly cancels a real spatial offset via the
    # origin lever arm), yet whose true perpendicular offset is ~20 mm. The
    # old d-difference predicate calls this coplanar; perpendicular_offset
    # must not. This is the false-positive failure mode from the review: 65
    # of 155 pairs the old predicate called coplanar were really 5-29 mm
    # apart.
    normal_a = np.array([1.0, 0.0, 0.0])
    centroid_a = np.array([10.0, 0.0, 0.0])
    a = _patch(0, normal_a, -float(normal_a @ centroid_a))
    a.centroid = centroid_a

    theta = np.radians(-0.2287253629674964)
    normal_b = np.array([np.cos(theta), np.sin(theta), 0.0])
    centroid_b = np.array([10.04, 10.0, 0.0])
    b = _patch(1, normal_b, -float(normal_b @ centroid_b))
    b.centroid = centroid_b

    config = merged_config()
    dist_tol = float(config["coplanar_dist_tol_m"])

    # Sanity: the forbidden d-difference predicate would have accepted this
    # pair as coplanar.
    assert abs(a.d - b.d) < dist_tol
    # The real perpendicular offset is ~20 mm -- these are NOT the same
    # surface.
    assert perpendicular_offset(a, b) > 0.015

    classes = coplanarity_classes([a, b], config)
    assert classes == [[0], [1]]


def test_classes_and_pairs_are_deterministically_ordered():
    ps = [_patch(2, (1, 0, 0), 0.0), _patch(0, (1, 0, 0), 0.0), _patch(1, (0, 1, 0), 0.0)]
    classes = coplanarity_classes(ps, merged_config())
    assert classes == sorted(classes)
    assert all(c == sorted(c) for c in classes)
