import numpy as np

from rscene.config import merged_config
from rscene.core.graph import coplanarity_classes, intersection_line, patch_adjacency
from rscene.core.normals import estimate_normals
from rscene.core.patches import Patch, extract_patches
from rscene.core.prim import Box


def _patch(pid, normal, d):
    return Patch(
        patch_id=pid, normal=np.array(normal, dtype=float), d=d,
        point_idx=np.array([0]), n_points=1, p95_residual_m=0.0,
        centroid=np.zeros(3), u_range=(0.0, 1.0), v_range=(0.0, 1.0),
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
    point, direction = intersection_line(a, b)

    assert np.allclose(np.abs(direction), [0, 0, 1])
    assert abs(point[0]) < 1e-9 and abs(point[1]) < 1e-9


def test_intersection_line_of_parallel_planes_is_none():
    assert intersection_line(_patch(0, (1, 0, 0), 0.0),
                             _patch(1, (1, 0, 0), -0.2)) is None


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


def test_classes_and_pairs_are_deterministically_ordered():
    ps = [_patch(2, (1, 0, 0), 0.0), _patch(0, (1, 0, 0), 0.0), _patch(1, (0, 1, 0), 0.0)]
    classes = coplanarity_classes(ps, merged_config())
    assert classes == sorted(classes)
    assert all(c == sorted(c) for c in classes)
