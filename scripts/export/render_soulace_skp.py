"""Render the stacked native Soulace SketchUp handover through SketchUp."""
import json
from pathlib import Path

from skp_client import rb


root = Path("output_final/soulace_asbuilt_v2").resolve()
model = (root / "Soulace_all_levels_asbuilt.skp").as_posix()
image = (root / "Soulace_all_levels_native_iso.png").as_posix()
code = f'''Sketchup.open_file({json.dumps(model)});
m = Sketchup.active_model;
["PCM_CEILINGS", "PCM_CEILINGS_DROPPED"].each {{ |name|
  layer = m.layers[name]; layer.visible = false if layer }};
v = m.active_view; b = m.bounds; c = b.center;
span = [b.width, b.height, b.depth].max;
eye = Geom::Point3d.new(c.x + 1.30*span, c.y - 1.45*span, c.z + 0.92*span);
v.camera = Sketchup::Camera.new(eye, c, Geom::Vector3d.new(0, 0, 1), true);
v.zoom_extents;
ok = v.write_image({{filename: {json.dumps(image)}, width: 1900, height: 1450,
                    antialias: true, transparent: false, compression: 0.92}});
{{ok: ok, model: m.path, image: {json.dumps(image)}}}.to_json'''
print(rb(code, timeout=600))
