import numpy as np

from rscene.config import merged_config
from rscene.core.faces import apply_density_gate, median_spacing, merge_patches
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
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
