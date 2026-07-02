import trimesh
import pytest

from scripts.recon.assemble import build_scene, write_glb, COLLECTION_NAMES

try:
    import bpy  # noqa: F401
    HAVE_BPY = True
except Exception:
    HAVE_BPY = False


def _box(extents=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translate)
    return mesh


def test_build_scene_names_nodes_by_type():
    elements = {
        "wall": [_box(translate=(0, 0, 0))],
        "door": [_box(translate=(5, 0, 0))],
    }
    scene = build_scene(elements)
    assert isinstance(scene, trimesh.Scene)
    names = set(scene.graph.nodes_geometry)
    assert names == {"wall_00", "door_00"}


def test_build_scene_zero_pads_sequential_same_type_names():
    elements = {
        "wall": [_box(translate=(0, 0, 0)), _box(translate=(2, 0, 0))],
    }
    scene = build_scene(elements)
    names = set(scene.graph.nodes_geometry)
    assert names == {"wall_00", "wall_01"}


def test_build_scene_many_same_type_names_are_sorted_and_padded():
    elements = {"column": [_box(translate=(i, 0, 0)) for i in range(3)]}
    scene = build_scene(elements)
    names = set(scene.graph.nodes_geometry)
    assert names == {"column_00", "column_01", "column_02"}


def test_write_glb_roundtrip_preserves_node_names(tmp_path):
    elements = {
        "wall": [_box(translate=(0, 0, 0)), _box(translate=(2, 0, 0))],
        "door": [_box(translate=(5, 0, 0))],
    }
    scene = build_scene(elements)
    out_path = tmp_path / "scene.glb"
    write_glb(scene, str(out_path))

    assert out_path.exists()

    reloaded = trimesh.load(str(out_path))
    assert isinstance(reloaded, trimesh.Scene)
    reloaded_names = set(reloaded.graph.nodes_geometry)
    assert reloaded_names == {"wall_00", "wall_01", "door_00"}
    assert len(reloaded.geometry) == 3


@pytest.mark.skipif(not HAVE_BPY, reason="bpy not available")
def test_write_blend_creates_expected_collections(tmp_path):
    from scripts.recon.assemble import write_blend

    elements = {
        "wall": [_box(translate=(0, 0, 0))],
        "door": [_box(translate=(5, 0, 0))],
    }
    out_path = tmp_path / "scene.blend"
    write_blend(elements, str(out_path))

    assert out_path.exists()

    import bpy as _bpy

    collection_names = {c.name for c in _bpy.data.collections}
    assert set(COLLECTION_NAMES.values()).issubset(collection_names)

    wall_coll = _bpy.data.collections["Walls"]
    door_coll = _bpy.data.collections["Doors"]
    assert len(wall_coll.objects) == 1
    assert len(door_coll.objects) == 1


def test_write_blend_raises_clear_error_without_bpy(monkeypatch):
    if HAVE_BPY:
        pytest.skip("bpy is available in this environment; cannot test the fallback path")

    from scripts.recon.assemble import write_blend

    with pytest.raises(RuntimeError, match="bpy not available"):
        write_blend({"wall": [_box()]}, "unused.blend")
