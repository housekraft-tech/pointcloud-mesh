"""Task 9 (Architecture Revision R1): wall -> Feature classification, room
ownership split into WallSegments, per-segment watertight meshing, and the
room-collection assembly.

Conventions asserted here (see scripts/recon/model.py's docstring):
  - Feature/WallSegment u-coordinates are WALL-LOCAL (metres from wall.p0
    along p0->p1), the same frame openings.py and floorplan2d.py use.
  - z_min_m/z_max_m are absolute world Z.
"""
import numpy as np
import pytest
from shapely.geometry import Polygon

from scripts.recon.schema import WallStep, Feature, WallSegment
from scripts.recon.model import (
    classify_wall_features,
    split_wall_into_segments,
    build_room_model,
)
from scripts.recon.solids import segment_to_mesh


def _wall(p0=(0.0, 0.0), p1=(6.0, 0.0), thickness_m=0.2, floor_z_m=0.0,
          ceiling_z_m=2.7, steps=None):
    """A group_wall_runs-style run dict (the shape build_manifest/build_room
    consume). Steps carry ABSOLUTE world-u (as group_wall_runs produces)."""
    if steps is None:
        steps = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7)]
    return dict(p0=p0, p1=p1, thickness_m=thickness_m, floor_z_m=floor_z_m,
                ceiling_z_m=ceiling_z_m, steps=steps)


# ---------------------------------------------------------------------------
# classifier (the gap this task fills)
# ---------------------------------------------------------------------------

def test_classify_main_step_is_a_face_spanning_floor_to_ceiling():
    wall = _wall()
    feats = classify_wall_features(wall, z_floor=0.0, z_ceiling=2.7)
    faces = [f for f in feats if f.kind == "face"]
    assert len(faces) == 1
    f = faces[0]
    assert f.u_min_m == pytest.approx(0.0)
    assert f.u_max_m == pytest.approx(6.0)
    assert f.z_min_m == pytest.approx(0.0)
    assert f.z_max_m == pytest.approx(2.7)


def test_classify_recessed_step_is_a_groove():
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=-0.1, u_min_m=2.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7),
    ]
    feats = classify_wall_features(_wall(steps=steps), 0.0, 2.7)
    grooves = [f for f in feats if f.kind == "groove"]
    assert len(grooves) == 1
    assert grooves[0].offset_m == pytest.approx(-0.1)
    assert grooves[0].u_min_m == pytest.approx(2.0)
    assert grooves[0].u_max_m == pytest.approx(4.0)


def test_classify_proud_wide_step_is_l_extrusion_narrow_is_column_attach():
    wide = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=0.2, u_min_m=1.0, u_max_m=3.0, z_min_m=0.0, z_max_m=2.7),
    ]
    narrow = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=0.3, u_min_m=2.0, u_max_m=2.3, z_min_m=0.0, z_max_m=2.7),
    ]
    fw = classify_wall_features(_wall(steps=wide), 0.0, 2.7)
    fn = classify_wall_features(_wall(steps=narrow), 0.0, 2.7)
    assert any(f.kind == "l_extrusion" for f in fw)
    assert not any(f.kind == "column_attach" for f in fw)
    assert any(f.kind == "column_attach" for f in fn)
    assert not any(f.kind == "l_extrusion" for f in fn)


def test_classify_elevated_ceiling_reaching_step_is_beam_soffit():
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=2.4, z_max_m=2.7),
    ]
    feats = classify_wall_features(_wall(steps=steps), z_floor=0.0, z_ceiling=2.7)
    soffits = [f for f in feats if f.kind == "beam_soffit"]
    assert len(soffits) == 1
    # soffit keeps its elevated bottom (a hollow gap below), not flattened to floor
    assert soffits[0].z_min_m == pytest.approx(2.4)
    assert soffits[0].z_max_m == pytest.approx(2.7)


def test_classify_openings_become_opening_features():
    wall = _wall()
    openings = [{"u0": 2.0, "u1": 2.9, "z0": 0.0, "z1": 2.1, "type": "door", "walked": True}]
    feats = classify_wall_features(wall, 0.0, 2.7, openings=openings)
    ops = [f for f in feats if f.kind == "opening"]
    assert len(ops) == 1
    assert ops[0].u_min_m == pytest.approx(2.0)
    assert ops[0].u_max_m == pytest.approx(2.9)
    assert ops[0].z_max_m == pytest.approx(2.1)
    assert ops[0].meta.get("type") == "door"


def test_classify_uses_wall_local_u_when_p0_not_at_origin():
    """group_wall_runs emits WallStep.u_min_m in ABSOLUTE world-u; a wall
    whose p0 sits away from the world origin must still get features in
    wall-local u [0, length]."""
    steps = [WallStep(offset_m=0.0, u_min_m=3.0, u_max_m=9.0, z_min_m=0.0, z_max_m=2.7)]
    wall = _wall(p0=(2.0, 3.0), p1=(2.0, 9.0), steps=steps)  # runs along +Y from y=3
    feats = classify_wall_features(wall, 0.0, 2.7)
    face = [f for f in feats if f.kind == "face"][0]
    assert face.u_min_m == pytest.approx(0.0)
    assert face.u_max_m == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# ownership split
# ---------------------------------------------------------------------------

def test_wall_spanning_two_rooms_splits_into_two_segments():
    wall = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    feats = classify_wall_features(wall, 0.0, 2.7)
    room_a = ("Room_A", Polygon([(0, 0), (3, 0), (3, 4), (0, 4)]))
    room_b = ("Room_B", Polygon([(3, 0), (6, 0), (6, 4), (3, 4)]))
    segs = split_wall_into_segments(wall, feats, [room_a, room_b], wall_id="wall_000")

    assert len(segs) == 2
    by_room = {s.room_id: s for s in segs}
    assert set(by_room) == {"Room_A", "Room_B"}
    assert by_room["Room_A"].u_min_m == pytest.approx(0.0)
    assert by_room["Room_A"].u_max_m == pytest.approx(3.0)
    assert by_room["Room_B"].u_min_m == pytest.approx(3.0)
    assert by_room["Room_B"].u_max_m == pytest.approx(6.0)

    # each is its own watertight mesh with the expected half-volume
    for s in segs:
        mesh = segment_to_mesh(s, wall)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(3.0 * 0.2 * 2.7, rel=0.05)


def test_wall_no_room_claims_becomes_one_unassigned_segment():
    wall = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    feats = classify_wall_features(wall, 0.0, 2.7)
    # a room far from this wall (no overlapping edge)
    far = ("Room_X", Polygon([(0, 20), (3, 20), (3, 24), (0, 24)]))
    segs = split_wall_into_segments(wall, feats, [far], wall_id="wall_000")

    assert len(segs) == 1
    assert segs[0].room_id is None
    assert segs[0].u_min_m == pytest.approx(0.0)
    assert segs[0].u_max_m == pytest.approx(6.0)
    mesh = segment_to_mesh(segs[0], wall)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(6.0 * 0.2 * 2.7, rel=0.05)


def test_partial_room_ownership_keeps_unassigned_stretches():
    """A room claiming only the middle of a wall leaves the two end stretches
    as room_id=None segments -- never silently dropped (the v4/v5 bug R1
    fixes)."""
    wall = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    feats = classify_wall_features(wall, 0.0, 2.7)
    mid = ("Room_M", Polygon([(2, 0), (4, 0), (4, 4), (2, 4)]))
    segs = sorted(split_wall_into_segments(wall, feats, [mid], wall_id="w"),
                  key=lambda s: s.u_min_m)
    assert len(segs) == 3
    assert [s.room_id for s in segs] == [None, "Room_M", None]
    assert segs[1].u_min_m == pytest.approx(2.0)
    assert segs[1].u_max_m == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# per-segment opening cut only where the opening's u-range overlaps
# ---------------------------------------------------------------------------

def test_opening_cuts_only_the_segment_it_overlaps():
    wall = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    # door at u in [4.0, 4.9], which lies in Room_B's ownership half [3, 6]
    openings = [{"u0": 4.0, "u1": 4.9, "z0": 0.0, "z1": 2.1, "type": "door"}]
    feats = classify_wall_features(wall, 0.0, 2.7, openings=openings)
    room_a = ("Room_A", Polygon([(0, 0), (3, 0), (3, 4), (0, 4)]))
    room_b = ("Room_B", Polygon([(3, 0), (6, 0), (6, 4), (3, 4)]))
    segs = {s.room_id: s for s in
            split_wall_into_segments(wall, feats, [room_a, room_b], wall_id="w")}

    mesh_a = segment_to_mesh(segs["Room_A"], wall)
    mesh_b = segment_to_mesh(segs["Room_B"], wall)
    assert mesh_a.is_watertight and mesh_b.is_watertight

    # A: uncut half -> full half volume
    assert mesh_a.volume == pytest.approx(3.0 * 0.2 * 2.7, rel=0.05)
    # B: half minus the door
    door_vol = 0.9 * 0.2 * 2.1
    assert mesh_b.volume == pytest.approx(3.0 * 0.2 * 2.7 - door_vol, rel=0.1)


# ---------------------------------------------------------------------------
# relief preserved (not flattened) in the segment mesh
# ---------------------------------------------------------------------------

def _segment_from(wall, feats, u0, u1, room_id=None):
    clipped = []
    for f in feats:
        lo = max(f.u_min_m, u0)
        hi = min(f.u_max_m, u1)
        if hi - lo <= 1e-6:
            continue
        clipped.append(Feature(f.kind, lo, hi, f.z_min_m, f.z_max_m, f.offset_m, dict(f.meta)))
    return WallSegment(wall_id="w", room_id=room_id, u_min_m=u0, u_max_m=u1, features=clipped)


def test_groove_recesses_and_reduces_volume():
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=-0.1, u_min_m=2.0, u_max_m=4.0, z_min_m=0.0, z_max_m=2.7),
    ]
    wall = _wall(steps=steps)
    feats = classify_wall_features(wall, 0.0, 2.7)
    seg = _segment_from(wall, feats, 0.0, 6.0)
    plain_seg = _segment_from(_wall(), classify_wall_features(_wall(), 0.0, 2.7), 0.0, 6.0)

    grooved = segment_to_mesh(seg, wall)
    plain = segment_to_mesh(plain_seg, _wall())
    assert grooved.is_watertight
    # the recess removed material over the groove's 2 m x 2.7 m x 0.1 m box
    assert grooved.volume < plain.volume
    removed = plain.volume - grooved.volume
    assert removed == pytest.approx(2.0 * 2.7 * 0.1, rel=0.25)


def test_l_extrusion_protrudes_beyond_wall_thickness():
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=0.2, u_min_m=1.0, u_max_m=3.0, z_min_m=0.0, z_max_m=2.7),
    ]
    wall = _wall(steps=steps)  # thickness 0.2 -> half 0.1
    feats = classify_wall_features(wall, 0.0, 2.7)
    seg = _segment_from(wall, feats, 0.0, 6.0)
    mesh = segment_to_mesh(seg, wall)
    assert mesh.is_watertight
    perp_extent = (mesh.bounds[1] - mesh.bounds[0])[1]  # wall runs along X -> perp is Y
    assert perp_extent > 0.2 + 0.05  # protrudes past the plain 0.2 m thickness
    assert mesh.volume > 6.0 * 0.2 * 2.7  # gained relief volume


def test_beam_soffit_segment_leaves_hollow_gap_below():
    steps = [
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=0.0, z_max_m=2.7),
        WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=6.0, z_min_m=2.4, z_max_m=2.7),
    ]
    wall = _wall(steps=steps)
    feats = classify_wall_features(wall, 0.0, 2.7)
    soffit = [f for f in feats if f.kind == "beam_soffit"]
    assert len(soffit) == 1
    # a segment carrying ONLY the beam soffit -> its mesh starts at the soffit,
    # leaving open air below (not flattened to the floor).
    seg = WallSegment(wall_id="w", room_id=None, u_min_m=0.0, u_max_m=6.0, features=soffit)
    mesh = segment_to_mesh(seg, wall)
    assert mesh.is_watertight
    assert mesh.bounds[0][2] == pytest.approx(2.4, abs=0.03)


# ---------------------------------------------------------------------------
# final assembly
# ---------------------------------------------------------------------------

def test_build_room_model_produces_room_collections_and_unassigned():
    wall_top = _wall(p0=(0.0, 4.0), p1=(6.0, 4.0))       # shared between rooms? no - top of both
    wall_bottom = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    # a wall no room edge claims
    lonely = _wall(p0=(0.0, 20.0), p1=(2.0, 20.0),
                   steps=[WallStep(0.0, 0.0, 2.0, 0.0, 2.7)])
    rooms = [Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])]
    model = build_room_model([wall_bottom, wall_top, lonely], {}, [], [], rooms,
                             z_floor=0.0, z_ceiling=2.7)

    room_keys = [k for k in model if k.startswith("Room_")]
    assert len(room_keys) == 1
    assert "Walls_unassigned" in model
    assert len(model["Walls_unassigned"]) >= 1
    # every mesh in every collection is watertight
    for meshes in model.values():
        for m in meshes:
            assert m.is_watertight


def test_build_room_model_counts_dropped_segments_not_silent(monkeypatch):
    """A segment whose meshing fails (e.g. _ensure_watertight raising on
    degenerate geometry) must be COUNTED on model.drops, never silently
    absorbed -- and the rest of the model must still assemble."""
    import scripts.recon.model as model_mod

    wall_bottom = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    wall_top = _wall(p0=(0.0, 4.0), p1=(6.0, 4.0))
    rooms = [Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])]

    real_segment_to_mesh = segment_to_mesh
    calls = {"n": 0}

    def _flaky_segment_to_mesh(seg, wall):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("forced degenerate-geometry failure for test")
        return real_segment_to_mesh(seg, wall)

    monkeypatch.setattr(model_mod, "segment_to_mesh", _flaky_segment_to_mesh, raising=False)
    import scripts.recon.solids as solids_mod
    monkeypatch.setattr(solids_mod, "segment_to_mesh", _flaky_segment_to_mesh)

    model = build_room_model([wall_bottom, wall_top], {}, [], [], rooms,
                             z_floor=0.0, z_ceiling=2.7)

    assert hasattr(model, "drops"), "build_room_model must expose a drop report"
    assert len(model.drops["segments"]) == 1
    dropped = model.drops["segments"][0]
    assert "wall_id" in dropped and "error" in dropped
    assert "forced degenerate-geometry failure" in dropped["error"]
    # the model must still assemble successfully for everything else
    room_keys = [k for k in model if k.startswith("Room_")]
    assert len(room_keys) == 1
    for meshes in model.values():
        for m in meshes:
            assert m.is_watertight


def test_build_room_model_columns_and_floor_panel():
    from scripts.recon.schema import Column
    wall = _wall(p0=(0.0, 0.0), p1=(6.0, 0.0))
    rooms = [Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])]
    col = Column(column_id="column_000",
                 footprint=[(1.0, 1.0), (1.3, 1.0), (1.3, 1.3), (1.0, 1.3)],
                 z_min_m=0.0, z_max_m=2.7)
    model = build_room_model([wall], {}, [col], [], rooms, 0.0, 2.7)
    assert "Columns" in model and len(model["Columns"]) == 1
    # the room collection carries a floor panel in addition to wall segments
    room_key = [k for k in model if k.startswith("Room_")][0]
    assert len(model[room_key]) >= 2
