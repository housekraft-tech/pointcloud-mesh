"""Reopen the native Soulace handover files and audit SketchUp groups."""
import json
from pathlib import Path

from skp_client import rb


root = Path("output_final/soulace_asbuilt_v2").resolve()
names = [
    "Soulace_L0_ground_asbuilt.skp",
    "Soulace_L1_first_asbuilt.skp",
    "Soulace_L2_second_asbuilt.skp",
    "Soulace_all_levels_asbuilt.skp",
]
for name in names:
    path = (root / name).as_posix()
    code = f'''Sketchup.open_file({json.dumps(path)});
    m = Sketchup.active_model;
    gs = m.entities.grep(Sketchup::Group);
    cs = m.entities.grep(Sketchup::ComponentInstance);
    items = gs + cs;
    walls = items.select {{ |g| g.name.include?("wall_") || g.name.include?("parapet_") }};
    floors = items.select {{ |g| g.name.include?("floor_") }};
    beams = items.select {{ |g| g.name.include?("beam_") }};
    {{file: m.path, groups: gs.length, components: cs.length, items: items.length, wall_groups: walls.length,
      wall_manifold: walls.count {{ |g| g.manifold? }},
      floor_groups: floors.length, floor_manifold: floors.count {{ |g| g.manifold? }},
      beam_groups: beams.length, beam_manifold: beams.count {{ |g| g.manifold? }},
      nonmanifold_walls: walls.reject {{ |g| g.manifold? }}.map {{ |g|
        edges = g.entities.grep(Sketchup::Edge);
        [g.name, edges.count {{ |e| e.faces.length == 0 }},
         edges.count {{ |e| e.faces.length == 1 }},
         edges.count {{ |e| e.faces.length > 2 }}] }},
      all_manifold: items.count {{ |g| g.manifold? }},
      faces: items.sum {{ |g| g.definition.entities.grep(Sketchup::Face).length }},
      tags: m.layers.map {{ |layer| layer.name }}}}.to_json'''
    response = rb(code, timeout=600)
    print(name, response.get("result", response.get("error")))
