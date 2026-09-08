"""Offline and reopened-native checks for the Soulace v8 whole-house model."""
import json
from pathlib import Path

import trimesh

from export_rectangular_skp import invoke


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output_final" / "soulace_full_house_planes_v8"
NATIVE = OUT / "Soulace_full_house_wall_and_floor_planes.skp"
RUBY = ROOT / "scripts/export/ruby/pcm_coverage_baseline.rb"


def offline():
    data = json.loads((OUT / "additions.build.json").read_text())
    walls = [part for part in data["parts"] if part["kind"] == "wall_paired_planes"]
    floors = [part for part in data["parts"] if part["kind"] == "floor_wall_joined"]
    assert len(walls) == 85
    assert len(floors) == 9
    assert sum(part["level"] == 0 for part in walls) == 46
    assert sum(part["level"] == 1 for part in walls) == 39
    assert sum(bool(part["thickness_verified"]) for part in walls) == 26
    assert len({part["name"] for part in data["parts"]}) == len(data["parts"])
    for part in data["parts"]:
        mesh = trimesh.Trimesh(part["v"], part["f"], process=False)
        assert mesh.is_watertight, part["name"]
        assert abs(float(mesh.volume)) > 1e-7, part["name"]
        if part["kind"] == "wall_paired_planes":
            assert [plane["side"] for plane in part["wall_plane_pair"]] == ["A", "B"]
            assert part["modeled_thickness_m"] > 0.05
    audit = json.loads((OUT / "full_house_plane_audit.json").read_text())
    assert audit["new_wall_runs_discovered"] == 0
    assert audit["levels"]["0"]["common_finish_floor_z_m"] == 0.0
    assert abs(audit["levels"]["1"]["common_finish_floor_z_m"] - 3.2041) < 1e-9
    return {
        "walls": len(walls),
        "floors": len(floors),
        "measured_thickness_walls": 26,
        "modeled_thickness_walls": 59,
        "all_new_meshes_closed": True,
    }


def native():
    assert NATIVE.exists() and NATIVE.stat().st_size > 1_000_000
    target = OUT / "native_full_house_checks.json"
    code = r'''
      load RUBY;
      raise 'Could not reopen whole-house model' unless Sketchup.open_file(NATIVE);
      m=Sketchup.active_model;groups=m.entities.grep(Sketchup::Group);
      created=groups.select do |g|
        ['wall_paired_planes','floor_wall_joined'].include?(g.get_attribute('CoverageBaseline','kind','')) &&
          [0,1].include?(g.get_attribute('CoverageBaseline','level',nil))
      end;
      rows=created.map do |g|
        faces=g.entities.grep(Sketchup::Face);
        side_rows=['A','B'].map do |side|
          selected=faces.select{|f|f.get_attribute('WallPlane','side','')==side};
          {side:side,faces:selected.length,area_m2:selected.sum{|f|f.area}/(PCMCoverageBaseline::SCALE**2)}
        end;
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),level:g.get_attribute('CoverageBaseline','level',nil),
         hidden:g.hidden?,reference_only:g.get_attribute('CoverageBaseline','reference_only',false),manifold:g.manifold?,
         volume_m3:g.volume/(PCMCoverageBaseline::SCALE**3),faces:faces.length,sides:side_rows,
         thickness_verified:g.get_attribute('ObservedEvidence','thickness_verified',nil),
         bounds_z_m:[g.bounds.min.z/PCMCoverageBaseline::SCALE,g.bounds.max.z/PCMCoverageBaseline::SCALE]}
      end;
      old=groups.select do |g|
        g.name.match?(/^L[01]_/) && ['wall_measured','wall_inferred','floor'].include?(g.get_attribute('CoverageBaseline','kind',''))
      end;
      ground=groups.find{|g|g.name=='Ground - ONE CONTINUOUS PLANE (simplified)'};
      raise 'Single cyan ground is missing' unless ground;
      stairs=groups.select{|g|g.get_attribute('CoverageBaseline','kind','').include?('stair')};
      result={path:m.path,groups:groups.length,created:rows,
        old_lower_wall_floor_references:old.map{|g|{name:g.name,hidden:g.hidden?,reference_only:g.get_attribute('CoverageBaseline','reference_only',false)}},
        ground:{faces:ground.entities.grep(Sketchup::Face).length,edges:ground.entities.grep(Sketchup::Edge).length,
          loops:ground.entities.grep(Sketchup::Face).map{|f|f.loops.length},z_m:[ground.bounds.min.z/PCMCoverageBaseline::SCALE,ground.bounds.max.z/PCMCoverageBaseline::SCALE]},
        stairs:stairs.map{|g|{name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),level:g.get_attribute('CoverageBaseline','level',nil),reference_only:g.get_attribute('CoverageBaseline','reference_only',false)}},
        scenes:m.pages.map{|page|page.name}};
      File.write(TARGET,JSON.pretty_generate(result));
      {groups:groups.length,created:rows.length,manifold:rows.count{|row|row[:manifold]},wall_side_faces:rows.sum{|row|row[:sides].sum{|side|side[:faces]}}}.to_json
    '''
    replacements = {
        "RUBY": json.dumps(RUBY.as_posix()),
        "NATIVE": json.dumps(NATIVE.as_posix()),
        "TARGET": json.dumps(target.as_posix()),
    }
    for key, value in replacements.items():
        code = code.replace(key, value)
    result = invoke(code)
    data = json.loads(target.read_text())
    assert result["groups"] == 509
    assert result["created"] == 94
    assert result["manifold"] == 94
    walls = [row for row in data["created"] if row["kind"] == "wall_paired_planes"]
    floors = [row for row in data["created"] if row["kind"] == "floor_wall_joined"]
    assert len(walls) == 85 and len(floors) == 9
    for wall in walls:
        assert not wall["hidden"] and not wall["reference_only"]
        assert all(side["faces"] > 0 for side in wall["sides"]), wall["name"]
    for floor in floors:
        assert not floor["hidden"] and floor["manifold"]
        expected = 0.0 if floor["level"] == 0 else 3.2041
        assert abs(floor["bounds_z_m"][1] - expected) < 1e-6, floor["name"]
    assert data["old_lower_wall_floor_references"]
    assert all(
        row["hidden"] and row["reference_only"]
        for row in data["old_lower_wall_floor_references"]
    )
    assert data["ground"]["faces"] == 1
    assert data["ground"]["edges"] == 4
    assert data["ground"]["loops"] == [1]
    assert max(abs(z + 0.5334) for z in data["ground"]["z_m"]) < 1e-6
    assert any(row["kind"] == "roof_stair_clean" for row in data["stairs"])
    required_scenes = {
        "3D", "Top", "Front", "Side", "L0 3D", "L0 Top",
        "L1 3D", "L1 Top", "L2 3D", "L2 Top",
    }
    assert required_scenes.issubset(data["scenes"])
    audit_path = OUT / "native_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["existing_geometry_preserved"]
    assert audit["existing_transforms_preserved"]
    assert audit["addition_area_pass"]
    audit.update(
        reopened_native_verified=True,
        reopened_group_count=509,
        closed_new_solids_verified=94,
        paired_lower_wall_groups_verified=85,
        paired_lower_wall_side_faces_verified=result["wall_side_faces"],
        lower_floor_planes_verified=9,
        old_lower_wall_floor_fragments_hidden_as_references=True,
        cyan_ground_single_plane_unchanged=True,
        cyan_ground_z_m=-0.5334,
        l0_interior_finish_floor_z_m=0.0,
        l1_interior_finish_floor_z_m=3.2041,
        parking_and_interior_elevations_remain_distinct=True,
        no_new_wall_runs_discovered=True,
        site_accuracy_certified=False,
    )
    audit_path.write_text(json.dumps(audit, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps({"offline": offline(), "native": native()}, indent=2), flush=True)
