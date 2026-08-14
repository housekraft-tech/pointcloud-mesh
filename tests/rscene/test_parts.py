from pathlib import Path

import numpy as np
import pytest

from rscene.config import merged_config
from rscene.core.classify import classify_faces
from rscene.core.faces import apply_density_gate, merge_patches
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.occupancy import assign_interior_sides, build_occupancy, interior_by_enclosure
from rscene.core.parts import _is_free_standing_leaf, assemble_walls
from rscene.core.patches import extract_patches
from rscene.core.prim import Box

THICKNESS = 0.200


def _wall_with_two_faces(spacing=0.008):
    """A free-standing wall slab, both faces visible."""
    a = Box("fa", (0.0, 0, 0), (0.0, 3.0, 2.5)).sample_surface(spacing, faces=("x+",))
    b = Box("fb", (THICKNESS, 0, 0), (THICKNESS, 3.0, 2.5)).sample_surface(spacing, faces=("x-",))
    floor = Box("fl", (-1, 0, 0), (1.5, 3.0, 0)).sample_surface(0.02, faces=("z+",))
    xyz = np.concatenate([a, b, floor])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)
    return xyz, faces, cfg


def test_two_opposing_faces_become_one_wall_with_measured_thickness():
    xyz, faces, cfg = _wall_with_two_faces()
    walls, unpaired = assemble_walls(faces, xyz, cfg)

    paired = [w for w in walls if w.face_b is not None]
    assert len(paired) == 1, f"expected one paired wall, got {len(paired)}"
    t = paired[0].thickness
    assert abs(t.value - THICKNESS) < 0.003, f"thickness {t.value}"
    assert t.method
    assert t.n_points > 0


def test_thickness_is_not_measured_by_differencing_d():
    """Guard the origin lever-arm trap: the same wall far from the origin
    must measure the same thickness."""
    xyz, faces, cfg = _wall_with_two_faces()
    walls, _ = assemble_walls(faces, xyz, cfg)
    near = [w for w in walls if w.face_b is not None][0].thickness.value

    shifted = xyz + np.array([40.0, 25.0, 0.0])
    normals, curv = estimate_normals(shifted, k=cfg["normal_k"])
    patches, _ = extract_patches(shifted, normals, curv, cfg)
    faces2 = merge_patches(patches, shifted, cfg)
    classify_faces(faces2, estimate_frame(patches, cfg), cfg)
    walls2, _ = assemble_walls(faces2, shifted, cfg)
    far = [w for w in walls2 if w.face_b is not None][0].thickness.value

    assert abs(near - far) < 0.001, f"thickness moved with the origin: {near} vs {far}"


def test_a_lone_face_is_reported_unpaired_not_invented():
    a = Box("fa", (0, 0, 0), (0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    cfg = merged_config()
    normals, curv = estimate_normals(a, k=cfg["normal_k"])
    patches, _ = extract_patches(a, normals, curv, cfg)
    faces = merge_patches(patches, a, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    walls, unpaired = assemble_walls(faces, a, cfg)
    assert len(unpaired) >= 1
    for w in walls:
        if w.face_b is None:
            assert w.thickness is None


def test_faces_too_far_apart_are_not_paired():
    a = Box("fa", (0.0, 0, 0), (0.0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    b = Box("fb", (1.5, 0, 0), (1.5, 3.0, 2.5)).sample_surface(0.01, faces=("x-",))
    xyz = np.concatenate([a, b])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    walls, _ = assemble_walls(faces, xyz, cfg)
    assert all(w.face_b is None for w in walls), "1.5 m apart is not one wall"


def test_wall_length_and_height_are_measured():
    xyz, faces, cfg = _wall_with_two_faces()
    walls, _ = assemble_walls(faces, xyz, cfg)
    w = [x for x in walls if x.face_b is not None][0]
    assert abs(w.length.value - 3.0) < 0.01
    assert abs(w.height.value - 2.5) < 0.01


def test_assembly_is_deterministic():
    xyz, faces, cfg = _wall_with_two_faces()
    w1, u1 = assemble_walls(faces, xyz, cfg)
    w2, u2 = assemble_walls(faces, xyz, cfg)
    assert [w.wall_id for w in w1] == [w.wall_id for w in w2]
    assert [None if w.thickness is None else w.thickness.value for w in w1] == \
           [None if w.thickness is None else w.thickness.value for w in w2]
    assert u1 == u2


# --- local overlap thickness measurement -----------------------------------


def test_curved_wall_thickness_is_measured_locally_not_by_global_offset():
    """Two faces that are not flat -- each has a mild bow across its extent --
    must still measure a thickness close to the LOCAL gap, not a number
    dragged around by the far corners' curvature."""
    rng = np.random.default_rng(0)
    y = np.linspace(0, 3.0, 200)
    z = np.linspace(0, 2.5, 60)
    yy, zz = np.meshgrid(y, z)
    yy, zz = yy.ravel(), zz.ravel()

    # a mild bow in x as a function of y, several mm over the wall's extent,
    # applied identically to both faces so the true local gap is constant.
    bow = 0.004 * np.sin(yy / 3.0 * np.pi)
    xa = 0.0 + bow
    xb = THICKNESS + bow

    face_a_pts = np.stack([xa, yy, zz], axis=1)
    face_b_pts = np.stack([xb, yy, zz], axis=1)
    floor = Box("fl", (-1, 0, 0), (1.5, 3.0, 0)).sample_surface(0.02, faces=("z+",))
    xyz = np.concatenate([face_a_pts, face_b_pts, floor])

    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    walls, _ = assemble_walls(faces, xyz, cfg)
    paired = [w for w in walls if w.face_b is not None]
    assert len(paired) == 1
    assert abs(paired[0].thickness.value - THICKNESS) < 0.003


# --- door-leaf exclusion -----------------------------------------------


def _two_separate_wall_pairs():
    """Two independent, spatially separate wall pairs (200 mm thick each)."""
    a1 = Box("a1", (0.0, 0, 0), (0.0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    b1 = Box("b1", (THICKNESS, 0, 0), (THICKNESS, 3.0, 2.5)).sample_surface(0.01, faces=("x-",))
    a2 = Box("a2", (10.0, 0, 0), (10.0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    b2 = Box("b2", (10.0 + THICKNESS, 0, 0), (10.0 + THICKNESS, 3.0, 2.5)).sample_surface(0.01, faces=("x-",))
    floor = Box("fl", (-1, 0, 0), (12.0, 3.0, 0)).sample_surface(0.02, faces=("z+",))
    xyz = np.concatenate([a1, b1, a2, b2, floor])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)
    return xyz, faces, cfg


def test_face_with_interior_air_on_both_sides_is_excluded_from_pairing():
    """A face pair standing in for a door leaf -- interior air on BOTH sides,
    the positive test `_is_free_standing_leaf` looks for -- must not pair,
    even though geometrically it looks exactly like the wall pair next to
    it. `interior` is constructed directly here so the test controls exactly
    which sides read as interior, independent of whatever `interior_by_enclosure`
    would compute for this (non-enclosed) synthetic scene."""
    xyz, faces, cfg = _two_separate_wall_pairs()
    wall_faces = sorted([f for f in faces if f.role == "wall"], key=lambda f: f.face_id)
    assert len(wall_faces) == 4
    by_id = {f.face_id: f for f in wall_faces}

    grid = build_occupancy(xyz, cfg)
    interior = np.zeros(grid.shape, dtype=bool)
    step = grid.cell_m * float(cfg["interior_probe_cells"])

    # Pair partner = the other wall face closest in x. First pair (x < 5) is
    # a real wall: interior only on the side facing AWAY from its partner.
    # Second pair (x > 5) is the "leaf": interior on BOTH sides.
    # The sample taken here mirrors `_probe_interior_fraction`'s own sampling
    # (up to 256 points, evenly spaced across `point_idx`) exactly, so the
    # cells marked interior are the same cells the probe actually reads --
    # a different, sparser sample can miss those cells even when the
    # intended geometry is identical.
    for f in wall_faces:
        partner = min(
            (g for g in wall_faces if g.face_id != f.face_id),
            key=lambda g: abs(g.centroid[0] - f.centroid[0]),
        )
        away = f.centroid - partner.centroid
        idx = f.point_idx
        take = idx if len(idx) <= 256 else idx[
            np.linspace(0, len(idx) - 1, 256).astype(np.int64)
        ]
        pts = xyz[take]
        if f.centroid[0] < 5.0:
            sign = 1.0 if float(f.normal @ away) > 0 else -1.0
            probe = pts + sign * step * f.normal
            ijk = grid.index_of(probe)
            interior[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        else:
            for sign in (1.0, -1.0):
                probe = pts + sign * step * f.normal
                ijk = grid.index_of(probe)
                interior[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    walls, unpaired = assemble_walls(faces, xyz, cfg, grid=grid, interior=interior)
    paired = [w for w in walls if w.face_b is not None]
    assert len(paired) == 1
    a, b = paired[0].face_a, paired[0].face_b
    assert all(by_id[fid].centroid[0] < 5.0 for fid in (a, b)), \
        "the both-sides-interior pair must not have paired"

    unpaired_faces = [by_id[fid] for fid in unpaired]
    assert len(unpaired_faces) == 2
    assert all(f.centroid[0] > 5.0 for f in unpaired_faces)
    for w in walls:
        if w.face_b is None:
            assert w.thickness is None


def test_leaf_test_is_a_noop_when_no_interior_data_exists():
    """A synthetic fixture with no enclosed room around it (just a wall slab
    and a floor) has essentially no interior cells anywhere, so the
    both-sides-interior probe reads ~0 on both sides of every face and the
    leaf test never fires -- no separate synthetic-vs-real branch is needed
    for this to behave. Covered structurally by
    test_two_opposing_faces_become_one_wall_with_measured_thickness already;
    this test makes the no-op behaviour explicit and exercises the
    self-building occupancy path (grid/interior omitted)."""
    xyz, faces, cfg = _wall_with_two_faces()
    walls, _ = assemble_walls(faces, xyz, cfg)
    assert len([w for w in walls if w.face_b is not None]) == 1


# --- real scan ---------------------------------------------------------

_REAL_SCAN = Path(__file__).parent.parent.parent / "data" / "isolated_structural_v2.las"


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="isolated_structural_v2.las not found")
def test_real_scan_wall_assembly():
    """Real scan: pairing finds real opposing-face pairs where most walls are
    seen from one side only, and does not invent pairs or thicknesses.

    Measured this session (patch_neighbor_k=64, crop x in (-3.2, 1.0),
    y in (-8.0, -3.0)): 56 wall faces, 5 mutual pairs, thicknesses in the
    60-207 mm range.
    """
    pytest.importorskip("laspy")
    from rscene.io.las import load_las

    full = load_las(str(_REAL_SCAN))
    xyz = full.xyz

    mask = (
        (xyz[:, 0] > -3.2) & (xyz[:, 0] < 1.0)
        & (xyz[:, 1] > -8.0) & (xyz[:, 1] < -3.0)
    )
    idx = np.nonzero(mask)[0]
    cropped_xyz = xyz[idx]

    cfg = merged_config({"patch_neighbor_k": 64})
    normals, curv = estimate_normals(cropped_xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(cropped_xyz, normals, curv, cfg)
    all_faces = merge_patches(patches, cropped_xyz, cfg)
    faces, _rejected = apply_density_gate(all_faces, cropped_xyz, cfg)
    assert len(faces) >= 60

    frame = estimate_frame(patches, cfg)
    classify_faces(faces, frame, cfg)

    grid = build_occupancy(cropped_xyz, cfg)
    interior = interior_by_enclosure(grid)
    assign_interior_sides(faces, grid, interior, cfg, cropped_xyz)

    wall_faces = [f for f in faces if f.role == "wall"]
    assert len(wall_faces) >= 40  # measured 56

    # reuse the grid/interior already built above rather than recomputing
    walls, unpaired = assemble_walls(faces, cropped_xyz, cfg, grid=grid, interior=interior)

    # accounting: every wall face is in exactly one Wall, paired or not
    assert len(walls) + sum(1 for w in walls if w.face_b is not None) == len(wall_faces)

    # pairing is neither dead nor promiscuous (measured 5 paired, 51 single)
    paired = [w for w in walls if w.face_b is not None]
    assert 2 <= len(paired) <= 15
    assert len(walls) - len(paired) >= 25, "almost everything paired: overlap/mutual gate broken"

    # every measured thickness is in the physical band and carries provenance
    for w in paired:
        assert 0.05 <= w.thickness.value <= 0.45
        assert w.thickness.n_points > 0 and w.thickness.method

    # unpaired walls are honest: no invented thickness
    for w in walls:
        if w.face_b is None:
            assert w.thickness is None

    print(f"\nwall faces: {len(wall_faces)}")
    print(f"paired: {len(paired)}, unpaired: {len(walls) - len(paired)}")
    for w in sorted(paired, key=lambda w: w.thickness.value):
        print(
            f"  {w.wall_id}: thickness={w.thickness.value * 1000:.1f} mm "
            f"n_points={w.thickness.n_points} spread(p95)={w.thickness.p95_residual * 1000:.2f} mm "
            f"method={w.thickness.method}"
        )
    leaf_ids = {
        f.face_id for f in wall_faces
        if _is_free_standing_leaf(f, grid, interior, cfg, cropped_xyz)
    }
    print(f"faces excluded from pairing (positive both-sides-interior test): {len(leaf_ids)}")
    if leaf_ids:
        for fid in sorted(leaf_ids):
            f = next(x for x in wall_faces if x.face_id == fid)
            print(f"  excluded face {fid}: n_points={f.n_points} centroid_z={f.centroid[2]:.2f}")
