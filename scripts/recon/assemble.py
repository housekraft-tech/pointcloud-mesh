"""Final scene assembly: named GLB export + optional Blender collections.

Takes the per-element-type mesh lists produced by the rest of the plane-first
pipeline (walls, floor, ceiling, doors, windows, columns, beams) and:

- `build_scene` packs them into a single `trimesh.Scene` with deterministic,
  zero-padded node names (`wall_00`, `wall_01`, `door_00`, ...) so downstream
  tools (and humans) can address individual solids by name.
- `write_glb` writes that scene to a glTF-binary (.glb) file.
- `write_blend` optionally builds a native Blender scene with one named
  collection per element type, guarded behind an import of `bpy` (Blender's
  Python module), which is only present when running inside Blender itself.

Node/index naming: within each element-type key, meshes are named
`{type}_{i:02d}` using their position in the input list (0-based, 2-digit
zero padding -- matches the sizes we expect per element type in a single
housing unit; if a type ever exceeds 100 elements the index simply widens,
`build_scene` uses plain str formatting so it still produces valid, sortable
names).
"""
from __future__ import annotations

import trimesh

# Maps the singular element-type keys used throughout the pipeline (see
# scripts/recon/schema.py, floorplan_schema.py) to the plural, capitalized
# Blender collection names required by the spec. Deliberately spelled out
# (not derived via `key.capitalize() + "s"`) because "floor" and "ceiling"
# don't pluralize, and being explicit here is clearer than a naive string
# transform that would need special-casing anyway.
COLLECTION_NAMES = {
    "wall": "Walls",
    "floor": "Floor",
    "ceiling": "Ceiling",
    "door": "Doors",
    "window": "Windows",
    "column": "Columns",
    "beam": "Beams",
}


def _node_name(element_type: str, index: int) -> str:
    return f"{element_type}_{index:02d}"


def build_scene(elements: dict) -> trimesh.Scene:
    """Pack per-type mesh lists into one named trimesh.Scene.

    `elements` maps element-type strings (e.g. "wall", "door") to lists of
    `trimesh.Trimesh` objects. Every mesh becomes a scene node named
    `{type}_{i:02d}`, where `i` is that mesh's 0-based position within its
    type's list. Element-type keys are processed in sorted order so the
    resulting scene graph is built deterministically.
    """
    scene = trimesh.Scene()
    for element_type in sorted(elements):
        meshes = elements[element_type]
        for i, mesh in enumerate(meshes):
            name = _node_name(element_type, i)
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    return scene


def write_glb(scene: trimesh.Scene, path: str) -> str:
    """Export a trimesh.Scene to a GLB file at `path`. Returns `path`."""
    scene.export(file_obj=path, file_type="glb")
    return path


def write_blend(elements: dict, path: str) -> str:
    """Build a Blender scene with one named collection per element type and
    save it to `path` as a .blend file.

    Requires Blender's `bpy` module, which is only importable when running
    inside Blender (or a matching bundled Python). If `bpy` isn't available,
    raises RuntimeError("bpy not available") so callers can catch it and
    fall back to GLB-only output rather than crashing unpredictably.
    """
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError("bpy not available") from exc

    # Start from an empty scene so repeated calls / batch runs don't
    # accumulate leftover default-scene objects (Cube, Camera, Light).
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene_collection = bpy.context.scene.collection

    collections = {}
    for collection_name in COLLECTION_NAMES.values():
        coll = bpy.data.collections.new(collection_name)
        scene_collection.children.link(coll)
        collections[collection_name] = coll

    for element_type in sorted(elements):
        collection_name = COLLECTION_NAMES.get(element_type)
        if collection_name is None:
            # Unknown element type: skip rather than silently misfile it
            # into an arbitrary collection.
            continue
        target_collection = collections[collection_name]
        meshes = elements[element_type]
        for i, mesh in enumerate(meshes):
            name = _node_name(element_type, i)
            bl_mesh = bpy.data.meshes.new(name=f"{name}_mesh")
            bl_mesh.from_pydata(
                [tuple(v) for v in mesh.vertices],
                [],
                [tuple(f) for f in mesh.faces],
            )
            bl_mesh.update()
            bl_obj = bpy.data.objects.new(name, bl_mesh)
            target_collection.objects.link(bl_obj)

    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path
