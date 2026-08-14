import numpy as np
import pytest
from pathlib import Path

from rscene.config import merged_config
from rscene.core.classify import classify_faces
from rscene.core.faces import apply_density_gate, merge_patches
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box

pytest.importorskip("laspy")

from rscene.io.las import load_las

# Path to the real scan fixture; mirrors tests/rscene/test_occupancy.py and test_cli.py
_REAL_SCAN = Path(__file__).parent.parent.parent / "data" / "isolated_structural_v2.las"


def _room():
    xyz = Box("room", (0, 0, 0), (3, 2.5, 2.5)).sample_surface(0.015)
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    frame = estimate_frame(patches, cfg)
    return faces, frame, cfg


def test_a_closed_room_yields_a_floor_a_ceiling_and_walls():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    roles = [f.role for f in faces]
    assert roles.count("floor") == 1
    assert roles.count("ceiling") == 1
    assert roles.count("wall") >= 4


def test_every_face_gets_a_role():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    assert all(f.role is not None for f in faces)
    assert set(f.role for f in faces) <= {"floor", "ceiling", "wall", "oblique", "unknown"}


def test_a_horizontal_face_away_from_both_slabs_is_not_a_slab():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    floor = [f for f in faces if f.role == "floor"][0]
    ceiling = [f for f in faces if f.role == "ceiling"][0]
    assert floor.centroid[2] < ceiling.centroid[2]


def test_classification_is_deterministic():
    faces_a, frame, cfg = _room()
    classify_faces(faces_a, frame, cfg)
    faces_b, _, _ = _room()
    classify_faces(faces_b, frame, cfg)
    assert [f.role for f in faces_a] == [f.role for f in faces_b]


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="isolated_structural_v2.las not found")
def test_real_scan_classify_faces():
    """Real scan: face classification assigns roles and satisfies structural properties.

    Measured this session (fresh run, current working tree):
    - patches / merged faces: 196 / 90
    - frame floor_z / ceiling_z: −0.276 / 2.519
    - classify (band 0.30): floor 5, ceiling 11, wall 56, unknown 10, oblique 7
    - points by role: floor 70,964 · ceiling 139,317 · wall 170,871 · unknown 10,334 · oblique 862
    - largest floor / ceiling face: 63,602 pts at z=−0.24 / 76,684 pts at z=2.50
    """
    full = load_las(str(_REAL_SCAN))
    xyz = full.xyz

    # Crop to the same one-room region used in test_cli.py
    mask = (
        (xyz[:, 0] > -3.2) & (xyz[:, 0] < 1.0)
        & (xyz[:, 1] > -8.0) & (xyz[:, 1] < -3.0)
    )
    idx = np.nonzero(mask)[0]
    cropped_xyz = xyz[idx]

    # Build the pipeline up to classification
    cfg = merged_config({"patch_neighbor_k": 64})
    normals, curv = estimate_normals(cropped_xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(cropped_xyz, normals, curv, cfg)
    all_faces = merge_patches(patches, cropped_xyz, cfg)
    faces, _rejected = apply_density_gate(all_faces, cropped_xyz, cfg)
    frame = estimate_frame(patches, cfg)

    # Preconditions
    assert len(faces) >= 60
    assert frame.floor_z is not None and frame.ceiling_z is not None
    assert frame.ceiling_z - frame.floor_z > 2.0

    # Classify
    classify_faces(faces, frame, cfg)

    # Assertions
    assert all(f.role is not None for f in faces)
    assert set(f.role for f in faces) <= {"floor", "ceiling", "wall", "oblique", "unknown"}

    pts = lambda role: sum(f.n_points for f in faces if f.role == role)

    # a dominant slab face on each level (measured 63,602 and 76,684)
    assert max((f.n_points for f in faces if f.role == "floor"), default=0) > 30_000
    assert max((f.n_points for f in faces if f.role == "ceiling"), default=0) > 30_000

    # floors sit at floor level, ceilings at ceiling level — role vs frame consistency
    for f in faces:
        if f.role == "floor":
            assert abs(f.centroid[2] - frame.floor_z) <= cfg["classify_slab_band_m"]
        if f.role == "ceiling":
            assert abs(f.centroid[2] - frame.ceiling_z) <= cfg["classify_slab_band_m"]

    # walls are the largest vertical population (measured 56 faces, 170,871 pts)
    assert sum(1 for f in faces if f.role == "wall") >= 40
    assert pts("wall") > 100_000

    # the residual buckets stay residual (measured unknown 10,334 pts, oblique 862 pts)
    assert pts("unknown") + pts("oblique") < 0.10 * sum(f.n_points for f in faces)

    # Report role distribution
    for role in ["floor", "ceiling", "wall", "oblique", "unknown"]:
        count = sum(1 for f in faces if f.role == role)
        points = pts(role)
        total_points = sum(f.n_points for f in faces)
        fraction = points / total_points if total_points > 0 else 0
        print(f"{role:10} {count:3} faces {points:7} points ({fraction:5.2%})")
