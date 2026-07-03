import json

import numpy as np

from scripts.recon.schema import (
    ScanData, Plane, WallStep, Column, Beam, new_column_id, new_beam_id,
    Feature, WallSegment, feature_to_dict, wall_segment_to_dict,
    ElementDims, check_priors, build_manifest,
)

PRIORS = {"door_h_m": 2.13, "door_h_tol_m": 0.25, "ceiling_m": 2.75, "ceiling_tol_m": 0.35}


def test_scandata_subset_keeps_aligned_fields():
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    t = np.array([1.0, 2.0, 3.0, 4.0])
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.uint8)
    s = ScanData(xyz=xyz, gps_time=t, rgb=rgb, intensity=None)
    assert s.n == 4
    sub = s.subset(np.array([True, False, True, False]))
    assert sub.n == 2
    assert np.allclose(sub.gps_time, [1.0, 3.0])
    assert np.array_equal(sub.rgb, [[1, 1, 1], [3, 3, 3]])
    assert sub.intensity is None


def test_scandata_subset_with_index_array():
    xyz = np.arange(9, dtype=float).reshape(3, 3)
    s = ScanData(xyz=xyz)
    sub = s.subset(np.array([2, 0]))
    assert np.allclose(sub.xyz, [[6, 7, 8], [0, 1, 2]])


def test_plane_signed_distance():
    p = Plane(normal=(0.0, 0.0, 1.0), d=-2.0, label="floor", inlier_idx=np.array([]))
    dist = p.signed_distance(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [5.0, 5.0, 1.5]]))
    assert np.allclose(dist, [0.0, 1.0, -0.5])


def test_ids_zero_padded():
    assert new_column_id(3) == "column_003"
    assert new_beam_id(11) == "beam_011"


def test_wallstep_and_element_fields():
    ws = WallStep(offset_m=0.1, u_min_m=1.0, u_max_m=1.3, z_min_m=0.0, z_max_m=2.4)
    assert ws.offset_m == 0.1
    c = Column(column_id="column_000", footprint=[(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)], z_min_m=0.0, z_max_m=2.4)
    assert len(c.footprint) == 4
    b = Beam(beam_id="beam_000", p0=(0, 0), p1=(3, 0), width_m=0.2, depth_m=0.3, z_min_m=2.1, z_max_m=2.4)
    assert b.width_m == 0.2


# ---------------------------------------------------------------------------
# Architecture Revision R1: Feature / WallSegment
# ---------------------------------------------------------------------------

def test_feature_fields_and_default_meta():
    f = Feature(kind="opening", u_min_m=1.0, u_max_m=1.9, z_min_m=0.0, z_max_m=2.1, offset_m=0.0)
    assert f.kind == "opening"
    assert f.meta == {}
    f2 = Feature(kind="opening", u_min_m=1.0, u_max_m=1.9, z_min_m=0.0, z_max_m=2.1,
                 offset_m=0.0, meta={"type": "door", "walked": True})
    assert f2.meta == {"type": "door", "walked": True}


def test_feature_to_dict_is_json_serializable():
    f = Feature(kind="groove", u_min_m=0.2, u_max_m=0.5, z_min_m=0.0, z_max_m=2.4, offset_m=-0.05)
    d = feature_to_dict(f)
    assert json.loads(json.dumps(d))["kind"] == "groove"


def test_wallsegment_fields_and_unassigned_room():
    ws = WallSegment(wall_id="wall_000", room_id=None, u_min_m=0.0, u_max_m=3.0, features=[])
    assert ws.room_id is None
    assert ws.features == []


def test_wallsegment_to_dict_is_json_serializable_with_nested_features():
    feat = Feature(kind="face", u_min_m=0.0, u_max_m=3.0, z_min_m=0.0, z_max_m=2.4, offset_m=0.0)
    ws = WallSegment(wall_id="wall_000", room_id="room_001", u_min_m=0.0, u_max_m=3.0, features=[feat])
    d = wall_segment_to_dict(ws)
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["room_id"] == "room_001"
    assert round_tripped["features"][0]["kind"] == "face"


# ---------------------------------------------------------------------------
# ElementDims / check_priors (Task 8)
# ---------------------------------------------------------------------------

def test_door_height_prior_flags():
    ok = ElementDims(0.9, 2.10, 0.1, "measured", [])
    odd = ElementDims(0.9, 2.60, 0.1, "measured", [])
    assert check_priors("door", ok, PRIORS) == []
    assert any("door_height" in f for f in check_priors("door", odd, PRIORS))


def test_ceiling_height_prior_flags():
    ok = ElementDims(0.0, 2.75, 0.0, "measured", [])
    odd = ElementDims(0.0, 3.50, 0.0, "measured", [])
    assert check_priors("ceiling", ok, PRIORS) == []
    assert any("ceiling_height_unusual" in f for f in check_priors("ceiling", odd, PRIORS))


# ---------------------------------------------------------------------------
# build_manifest (Task 8)
# ---------------------------------------------------------------------------

def test_manifest_roundtrips_to_json():
    m = build_manifest([], {}, [], [], [], z_floor=-0.25, z_ceiling=2.50, config={"seed": 0})
    assert json.loads(json.dumps(m))["storey"]["height_m"] == 2.75


def test_manifest_walls_openings_columns_beams_rooms():
    walls = [
        {"p0": (0.0, 0.0), "p1": (5.0, 0.0), "thickness_m": 0.12,
         "thickness_source": "measured", "steps": [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0,
                                                              z_min_m=0.0, z_max_m=2.4)]},
    ]
    openings = {0: [{"u0": 1.0, "u1": 1.9, "z0": 0.0, "z1": 2.55, "type": "door",
                     "walked": True, "confidence": 0.9}]}
    columns = [Column(column_id="ignored", footprint=[(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)],
                       z_min_m=0.0, z_max_m=2.4)]
    beams = [Beam(beam_id="ignored", p0=(0, 0), p1=(3, 0), width_m=0.2, depth_m=0.3,
                   z_min_m=2.1, z_max_m=2.4)]
    rooms = [[(0, 0), (5, 0), (5, 4), (0, 4)]]

    m = build_manifest(walls, openings, columns, beams, rooms,
                        z_floor=0.0, z_ceiling=2.4, config={})

    assert m["walls"][0]["wall_id"] == "wall_000"
    assert abs(m["walls"][0]["length_m"] - 5.0) < 1e-9
    assert m["walls"][0]["thickness_source"] == "measured"
    assert len(m["walls"][0]["steps"]) == 1

    op = m["openings"][0]
    assert op["opening_id"] == "opening_000"
    assert op["wall_id"] == "wall_000"
    assert op["type"] == "door"
    assert abs(op["width_m"] - 0.9) < 1e-9
    assert abs(op["height_m"] - 2.55) < 1e-9
    assert op["walked"] is True
    assert any("door_height_unusual" in f for f in op["flags"])

    assert m["columns"][0]["column_id"] == "column_000"
    assert abs(m["columns"][0]["height_m"] - 2.4) < 1e-9

    assert m["beams"][0]["beam_id"] == "beam_000"
    assert abs(m["beams"][0]["span_m"] - 3.0) < 1e-9

    assert m["rooms"][0]["room_id"] == "room_000"
    assert abs(m["rooms"][0]["area_m2"] - 20.0) < 1e-9

    json.loads(json.dumps(m))  # whole manifest is JSON-ready
