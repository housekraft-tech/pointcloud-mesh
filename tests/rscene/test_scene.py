import json

import numpy as np

from rscene.core.patches import Patch
from rscene.core.scene import (
    Frame, Measurement, Provenance, Scene, scene_from_json, scene_to_json,
)


def _scene():
    return Scene(
        provenance=Provenance(
            scan_path="koushik.las", scan_sha256="abc123",
            pipeline_version="0.1.0", timestamp="2026-08-11T00:00:00Z",
            config={"tau_fit_m": 0.003},
        ),
        frame=Frame(z_axis=[0.0, 0.0, 1.0], xy_rotation_deg=1.4,
                    floor_z=0.0, ceiling_z=2.75),
        patches=[Patch(
            patch_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-2.0,
            point_idx=np.array([0, 1, 2]), n_points=3, p95_residual_m=0.0021,
            centroid=np.array([2.0, 1.0, 1.4]), u_range=(-1.0, 1.0), v_range=(-1.4, 1.4),
        )],
        coplanarity_classes=[[0]],
        adjacency=[],
        unassigned_points=42,
        diagnostics={"p95_residual_m": 0.0021},
    )


def test_measurement_carries_provenance_and_uncertainty():
    m = Measurement(value=0.2031, method="face-to-face raw points",
                    n_points=18422, p95_residual=0.0021)
    assert m.to_dict() == {
        "value": 0.2031, "method": "face-to-face raw points",
        "n_points": 18422, "p95_residual": 0.0021, "n_bins": None,
    }


def test_scene_round_trips_through_json():
    original = _scene()
    restored = scene_from_json(scene_to_json(original))

    assert restored.provenance.scan_sha256 == "abc123"
    assert restored.frame.ceiling_z == 2.75
    assert restored.unassigned_points == 42
    assert len(restored.patches) == 1
    assert np.allclose(restored.patches[0].normal, [1.0, 0.0, 0.0])
    assert restored.patches[0].d == -2.0
    assert np.array_equal(restored.patches[0].point_idx, [0, 1, 2])


def test_serialisation_is_deterministic():
    assert scene_to_json(_scene()) == scene_to_json(_scene())


def test_json_keys_are_sorted_so_diffs_stay_readable():
    payload = json.loads(scene_to_json(_scene()))
    assert list(payload.keys()) == sorted(payload.keys())


def test_unassigned_points_are_recorded_not_dropped():
    payload = json.loads(scene_to_json(_scene()))
    assert payload["unassigned_points"] == 42


def test_measurement_round_trips_through_its_dict_form():
    m = Measurement(value=0.2031, method="face-to-face raw points",
                    n_points=18422, p95_residual=0.0021)
    assert Measurement.from_dict(m.to_dict()) == m


def test_scene_round_trip_preserves_point_idx_dtype():
    original = _scene()
    restored = scene_from_json(scene_to_json(original))
    assert restored.patches[0].point_idx.dtype == np.int64


def test_measurement_n_bins_round_trips():
    m = Measurement(value=0.2031, method="local-bin-median",
                    n_points=18422, p95_residual=0.0021, n_bins=7)
    assert m.to_dict()["n_bins"] == 7
    restored = Measurement.from_dict(m.to_dict())
    assert restored.n_bins == 7
    assert restored == m


def test_scene_round_trips_walls_and_features():
    from rscene.core.faces import Face
    from rscene.core.features import Feature
    from rscene.core.parts import ThicknessField, ThicknessSample, ThicknessSegment, Wall

    face = Face(
        face_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-2.0, patch_ids=[0, 1],
        point_idx=np.array([0, 1, 2]), loose_idx=np.array([3]), n_points=3,
        p95_residual_m=0.0021, centroid=np.array([2.0, 1.0, 1.4]),
        u_range=(-1.0, 1.0), v_range=(-1.4, 1.4), role="wall", interior_sign=1,
    )
    thickness_field = ThicknessField(
        samples=[
            ThicknessSample(u=0.1, v=0.2, value=0.203, n_points=12),
            ThicknessSample(u=0.9, v=0.2, value=0.451, n_points=9),
        ],
        segments=[
            ThicknessSegment(start=0.0, end=1.2, thickness=0.203, n_bins=4,
                              n_points=48, is_candidate_column=False),
            ThicknessSegment(start=1.2, end=1.6, thickness=0.451, n_bins=2,
                              n_points=18422, is_candidate_column=True),
        ],
    )
    wall = Wall(
        wall_id="W01", face_a=0, face_b=1,
        thickness=Measurement(0.2031, "face-to-face perpendicular offset", 18422, 0.0021, n_bins=4),
        thickness_field=thickness_field,
        length=Measurement(4.182, "face in-plane extent", 18422, 0.0021),
        height=Measurement(2.748, "face in-plane extent", 18422, 0.0021),
        centroid=np.array([2.0, 1.0, 1.4]), normal=np.array([1.0, 0.0, 0.0]),
    )
    feature = Feature(
        feature_id="F01", kind="extrusion", parent_face=0, source_face=2,
        u_range=(2.80, 3.15), v_range=(0.0, 2.75),
        depth=Measurement(0.075, "perpendicular offset to parent face", 900, 0.0022),
        rect_fit=0.94,
    )
    scene = _scene()
    scene.faces = [face]
    scene.walls = [wall]
    scene.features = [feature]
    scene.unmodeled = [7, 9]

    restored = scene_from_json(scene_to_json(scene))

    rf = restored.faces[0]
    assert rf.face_id == face.face_id
    assert np.array_equal(rf.normal, face.normal)
    assert rf.normal.dtype == np.float64
    assert rf.d == face.d
    assert rf.patch_ids == face.patch_ids
    assert np.array_equal(rf.point_idx, face.point_idx)
    assert rf.point_idx.dtype == np.int64
    assert np.array_equal(rf.loose_idx, [3])
    assert rf.loose_idx.dtype == np.int64
    assert rf.n_points == face.n_points
    assert rf.p95_residual_m == face.p95_residual_m
    assert np.array_equal(rf.centroid, face.centroid)
    assert rf.u_range == face.u_range and isinstance(rf.u_range, tuple)
    assert rf.v_range == face.v_range and isinstance(rf.v_range, tuple)
    assert rf.role == "wall"
    assert rf.interior_sign == 1

    rw = restored.walls[0]
    assert rw.wall_id == "W01"
    assert rw.face_a == 0
    assert rw.face_b == 1
    assert rw.thickness.value == 0.2031
    assert rw.thickness.method == "face-to-face perpendicular offset"
    assert rw.thickness.n_points == 18422
    assert rw.thickness.p95_residual == 0.0021
    assert rw.thickness.n_bins == 4
    assert rw.length.value == 4.182
    assert rw.height.value == 2.748
    assert np.array_equal(rw.centroid, wall.centroid)
    assert np.array_equal(rw.normal, wall.normal)

    rtf = rw.thickness_field
    assert rtf is not None
    assert len(rtf.samples) == 2
    assert rtf.samples[0] == ThicknessSample(u=0.1, v=0.2, value=0.203, n_points=12)
    assert rtf.samples[1] == ThicknessSample(u=0.9, v=0.2, value=0.451, n_points=9)
    assert len(rtf.segments) == 2
    assert rtf.segments[0] == ThicknessSegment(
        start=0.0, end=1.2, thickness=0.203, n_bins=4, n_points=48,
        is_candidate_column=False,
    )
    assert rtf.segments[1] == ThicknessSegment(
        start=1.2, end=1.6, thickness=0.451, n_bins=2, n_points=18422,
        is_candidate_column=True,
    )

    rft = restored.features[0]
    assert rft.feature_id == "F01"
    assert rft.kind == "extrusion"
    assert rft.parent_face == 0
    assert rft.source_face == 2
    assert rft.u_range == (2.80, 3.15)
    assert rft.v_range == (0.0, 2.75)
    assert rft.depth.value == 0.075
    assert rft.rect_fit == 0.94

    assert restored.unmodeled == [7, 9]


def test_an_unpaired_wall_serialises_a_null_thickness():
    from rscene.core.parts import Wall
    scene = _scene()
    scene.walls = [Wall(
        wall_id="W02", face_a=3, face_b=None, thickness=None,
        thickness_field=None,
        length=Measurement(2.0, "face in-plane extent", 500, 0.002),
        height=Measurement(2.5, "face in-plane extent", 500, 0.002),
        centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]),
    )]
    restored = scene_from_json(scene_to_json(scene))
    assert restored.walls[0].thickness is None
    assert restored.walls[0].thickness_field is None
    assert restored.walls[0].face_b is None


def test_empty_faces_walls_features_unmodeled_round_trip():
    scene = _scene()
    restored = scene_from_json(scene_to_json(scene))
    assert restored.faces == []
    assert restored.walls == []
    assert restored.features == []
    assert restored.unmodeled == []


def test_frame_with_none_floor_and_ceiling_round_trips():
    frame_with_nones = Frame(z_axis=[0.0, 0.0, 1.0], xy_rotation_deg=1.4,
                             floor_z=None, ceiling_z=None)
    scene = Scene(
        provenance=Provenance(
            scan_path="test.las", scan_sha256="def456",
            pipeline_version="0.1.0", timestamp="2026-08-11T00:00:00Z",
            config={},
        ),
        frame=frame_with_nones,
        patches=[],
        coplanarity_classes=[],
        adjacency=[],
        unassigned_points=0,
        diagnostics={},
    )
    restored = scene_from_json(scene_to_json(scene))
    assert restored.frame.floor_z is None
    assert restored.frame.ceiling_z is None
