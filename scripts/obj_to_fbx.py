"""Blender headless script: convert an OBJ mesh to FBX.

Usage:
    blender --background --python scripts/obj_to_fbx.py -- input.obj output.fbx
"""

import sys

import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_path = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)

try:
    bpy.ops.wm.obj_import(filepath=input_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=input_path)

bpy.ops.export_scene.fbx(filepath=output_path, use_selection=False, embed_textures=True)

print(f"Exported FBX to {output_path}")
