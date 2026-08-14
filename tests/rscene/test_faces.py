import numpy as np

from rscene.config import merged_config
from rscene.core.faces import (
    _finalise,
    apply_density_gate,
    median_spacing,
    merge_patches,
    recruit_points,
)
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.points import add_gaussian_noise
from rscene.core.prim import Box


def _extract(xyz, overrides=None):
    cfg = merged_config(overrides or {})
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, labels = extract_patches(xyz, normals, curv, cfg)
    return patches, labels, cfg


def test_two_fragments_of_one_face_merge():
    """A face split by a gap narrower than face_merge_gap_m becomes one face.

    The gap (80 mm) must sit strictly between patch_connect_radius_m (50 mm)
    -- so extraction cannot bridge it and the fixture actually exercises
    merge_patches -- and face_merge_gap_m (150 mm) -- so merging can.
    """
    a = Box("a", (0, 0.00, 0), (0, 1.96, 2.5)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0, 2.04, 0), (0, 4.00, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    x_patches = [p for p in patches if abs(p.normal[0]) > 0.99]
    assert len(x_patches) >= 2, (
        "fixture is wrong: extraction already merged the fragments, so this "
        "test would pass without merge_patches doing anything"
    )

    faces = merge_patches(patches, xyz, cfg)
    x_facing = [f for f in faces if abs(f.normal[0]) > 0.99]
    assert len(x_facing) == 1, f"expected one merged face, got {len(x_facing)}"
    assert x_facing[0].n_points == sum(p.n_points for p in patches
                                       if abs(p.normal[0]) > 0.99)


def test_a_12mm_groove_is_not_merged_into_its_wall():
    """The merge tolerance must stay below the shallowest feature."""
    wall_a = Box("wa", (0.000, 0.0, 0), (0.000, 1.00, 2.5)).sample_surface(0.008, faces=("x+",))
    groove = Box("g", (-0.012, 1.00, 0), (-0.012, 1.06, 2.5)).sample_surface(0.008, faces=("x+",))
    wall_b = Box("wb", (0.000, 1.06, 0), (0.000, 2.50, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([wall_a, groove, wall_b])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)

    offsets = sorted({round(abs(f.d), 4) for f in faces if abs(f.normal[0]) > 0.99})
    assert len(offsets) >= 2, "the groove was merged into the wall"
    assert abs((offsets[-1] - offsets[0]) - 0.012) < 0.003, offsets


def test_distant_coplanar_patches_do_not_merge():
    """Coplanar but far apart is two faces, not one -- merging needs adjacency."""
    a = Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    b = Box("b", (0, 8, 0), (0, 10, 2)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    assert len([f for f in faces if abs(f.normal[0]) > 0.99]) == 2


def test_merged_plane_is_refitted_from_the_union():
    a = Box("a", (0, 0.00, 0), (0, 1.96, 2.5)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0, 2.04, 0), (0, 4.00, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    face = [f for f in merge_patches(patches, xyz, cfg) if abs(f.normal[0]) > 0.99][0]

    assert face.p95_residual_m < cfg["tau_fit_m"] * 1.5
    assert face.v_range[1] - face.v_range[0] > 3.5      # spans both fragments
    assert len(face.patch_ids) >= 2


def test_a_lone_patch_becomes_a_single_patch_face():
    xyz = Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    assert all(len(f.patch_ids) >= 1 for f in faces)
    assert sum(f.n_points for f in faces) == sum(p.n_points for p in patches)


def test_merging_is_deterministic():
    a = Box("a", (0, 0.00, 0), (0, 1.96, 2.5)).sample_surface(0.01, faces=("x+",))
    b = Box("b", (0, 2.04, 0), (0, 4.00, 2.5)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([a, b])
    patches, _, cfg = _extract(xyz)

    f1 = merge_patches(patches, xyz, cfg)
    f2 = merge_patches(patches, xyz, cfg)
    assert [f.face_id for f in f1] == [f.face_id for f in f2]
    assert [f.d for f in f1] == [f.d for f in f2]
    assert [f.patch_ids for f in f1] == [f.patch_ids for f in f2]


def test_median_spacing_recovers_the_sample_grid():
    xyz = Box("f", (0, 0, 0), (2, 2, 0)).sample_surface(0.01, faces=("z+",))
    assert abs(median_spacing(xyz) - 0.01) < 0.001


def test_a_dense_surface_passes_the_gate():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    kept, rejected = apply_density_gate(faces, xyz, cfg)
    assert len(kept) >= 1 and len(rejected) == 0


def test_a_sparse_chain_is_rejected():
    """A scattered chain spanning a large box at low fill is not a surface."""
    dense = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    rng = np.random.default_rng(0)
    chain = np.column_stack([
        np.zeros(300),
        rng.uniform(5.0, 5.5, 300),
        rng.uniform(0.0, 0.5, 300),
    ])
    xyz = np.concatenate([dense, chain])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    kept, rejected = apply_density_gate(faces, xyz, cfg)

    for f in kept:
        assert not (f.centroid[1] > 4.5), "the sparse chain survived the gate"
    assert sum(f.n_points for f in kept) + sum(f.n_points for f in rejected) == \
        sum(f.n_points for f in faces)


def test_rejected_faces_are_returned_not_discarded():
    """No silent drops: the gate hands rejects back for the unmodeled bucket."""
    dense = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(dense)
    faces = merge_patches(patches, dense, cfg)
    kept, rejected = apply_density_gate(faces, dense, cfg)
    assert len(kept) + len(rejected) == len(faces)


def test_a_bbox_outlier_does_not_flip_a_dense_wall_to_rejected():
    """A single far member inflates area_bound_m2() but must not tank fill.

    Region growing only needs 50 mm connectivity and 3 mm planarity, so a
    stray member dragged far from the rest of a real wall is not hypothetical.
    Fill must be computed against a trimmed extent, not the raw bbox, or one
    outlier point can flip a genuinely dense wall from kept to rejected.
    """
    dense = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    outlier = np.array([[0.0, 100.0, 100.0]])
    xyz = np.concatenate([dense, outlier])
    members = np.arange(len(xyz))

    face = _finalise(0, xyz, members, [0])
    # the raw bbox is dominated by the outlier -- confirms the fixture is real
    assert face.area_bound_m2() > 1000.0

    cfg = merged_config()
    kept, rejected = apply_density_gate([face], xyz, cfg)
    assert len(kept) == 1 and len(rejected) == 0, "outlier member sank a real wall's fill"


def test_recruitment_claims_points_lying_on_a_face():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    before = int(np.count_nonzero(labels < 0))
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    after = int(np.count_nonzero(new_labels < 0))

    assert after <= before
    assert sum(len(f.loose_idx) for f in faces) == before - after


def test_recruits_do_not_move_the_plane():
    """A recruit is a member for accounting, never for fitting."""
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    planes_before = [(f.normal.copy(), f.d) for f in faces]
    recruit_points(faces, xyz, normals, labels, cfg)
    for (n0, d0), f in zip(planes_before, faces):
        assert np.array_equal(n0, f.normal)
        assert d0 == f.d


def test_a_point_far_from_every_face_is_not_recruited():
    xyz = np.concatenate([
        Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",)),
        np.array([[5.0, 5.0, 5.0]]),
    ])
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    assert new_labels[-1] == -1


def test_no_point_is_recruited_twice():
    xyz = np.concatenate([
        Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",)),
        Box("b", (0.5, 0, 0), (0.5, 2, 2)).sample_surface(0.008, faces=("x+",)),
    ])
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    recruit_points(faces, xyz, normals, labels, cfg)

    claimed = np.concatenate([f.loose_idx for f in faces]) if faces else np.zeros(0)
    assert len(claimed) == len(np.unique(claimed))


def test_recruitment_is_deterministic():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    f1 = merge_patches(patches, xyz, cfg)
    l1 = recruit_points(f1, xyz, normals, labels, cfg)
    f2 = merge_patches(patches, xyz, cfg)
    l2 = recruit_points(f2, xyz, normals, labels, cfg)

    assert np.array_equal(l1, l2)
    assert [f.loose_idx.tolist() for f in f1] == [f.loose_idx.tolist() for f in f2]


def test_recruitment_claims_a_noisy_tail_growth_left_behind():
    """Sensor noise (real p95 residual 2.36 mm) puts a genuine tail past
    tau_fit_m (3 mm) but inside recruit_dist_tol_m (8 mm) -- exactly the
    points growth cannot admit but recruitment should.
    """
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    xyz = add_gaussian_noise(xyz, 0.0025, np.random.default_rng(1))
    patches, labels, cfg = _extract(xyz)

    before = int(np.count_nonzero(labels < 0))
    assert before >= 100, "fixture leaves nothing for recruitment to do"

    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    after = int(np.count_nonzero(new_labels < 0))

    assert after <= before // 2, f"recruited too few: {before - after}/{before}"
    assert sum(len(f.loose_idx) for f in faces) == before - after


def test_recruitment_reaches_across_a_gap_growth_cannot_cross():
    """recruit_max_reach_m (100 mm) exceeds patch_connect_radius_m (50 mm),
    so a cluster 70 mm past the wall's edge is out of growth's reach but
    inside recruitment's.
    """
    face = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    zs = np.arange(0.2, 1.8, 0.02)
    near = np.column_stack([np.zeros_like(zs), np.full_like(zs, 2.07), zs])
    xyz = np.concatenate([face, near])
    cluster_idx = np.arange(len(face), len(xyz))

    patches, labels, cfg = _extract(xyz)
    assert np.all(labels[cluster_idx] < 0), "fixture is wrong: growth already reached the cluster"

    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    assert np.all(new_labels[cluster_idx] >= 0), "reachable cluster was not recruited"


def test_recruitment_does_not_cross_a_gap_beyond_reach():
    """The converse of the reach test: a cluster past recruit_max_reach_m
    (150 mm here) must stay unassigned, proving the reach gate is real."""
    face = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    zs = np.arange(0.2, 1.8, 0.02)
    far = np.column_stack([np.zeros_like(zs), np.full_like(zs, 2.15), zs])
    xyz = np.concatenate([face, far])
    cluster_idx = np.arange(len(face), len(xyz))

    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    assert np.all(new_labels[cluster_idx] < 0), "cluster beyond reach was recruited anyway"
