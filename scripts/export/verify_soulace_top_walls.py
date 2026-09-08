"""Native solid, ground-plane and inter-storey scene verification."""
import json
from model_soulace_top_walls import ROOT,OUT
from export_rectangular_skp import invoke
from verify_observed_recovery import main as verify_native

NATIVE='Soulace_top_floor_wall_revision.skp'


def main():
    verify_native('soulace',folder_override=str(OUT),native_name=NATIVE)
    target=OUT/'native_wall_ground_checks.json'
    ground=json.loads((ROOT/'output_final/soulace_single_ground_plane_v3/single_ground.config.json').read_text())
    code=f'''
      m=Sketchup.active_model;groups=m.entities.grep(Sketchup::Group);
      ground=groups.find {{ |g| g.name=={json.dumps(ground['group_name'])} }};
      raise 'Ground missing' unless ground;
      fs=ground.entities.grep(Sketchup::Face);es=ground.entities.grep(Sketchup::Edge);
      wall_rows=groups.select {{ |g| g.get_attribute('CoverageBaseline','kind','')=='wall_solid_refit' }}.map do |g|
        {{name:g.name,manifold:g.manifold?,volume_m3:g.volume/(PCMCoverageBaseline::SCALE**3),
          faces:g.entities.grep(Sketchup::Face).length,hidden:g.hidden?}}
      end;
      stairs=groups.select {{ |g| g.get_attribute('CoverageBaseline','kind','')=='stair_clean' }};
      roof=groups.find {{ |g| g.get_attribute('CoverageBaseline','kind','')=='roof_stair_scan' }};
      payload={{wall_solids:wall_rows,ground:{{faces:fs.length,edges:es.length,loops:fs.map {{ |f| f.loops.length }},
        world_vertices_m:fs.flat_map {{ |f| f.vertices }}.uniq.map {{ |v| p=v.position.transform(ground.transformation);[p.x,p.y,p.z].map {{ |c| c/PCMCoverageBaseline::SCALE }} }},hidden:ground.hidden?}},
        inter_floor_stairs_visible_on_l2:stairs.all? {{ |g| PCMCoverageBaseline.visible_on_level?(g,2) }},
        roof_stair_visible_on_l2:roof && PCMCoverageBaseline.visible_on_level?(roof,2),
        roof_bounds_m:roof ? [roof.bounds.min.to_a,roof.bounds.max.to_a].map {{ |a| a.map {{ |c| c/PCMCoverageBaseline::SCALE }} }} : nil}};
      File.write({json.dumps(target.as_posix())},JSON.pretty_generate(payload));
      {{walls:wall_rows.length,solid_walls:wall_rows.count {{ |w| w[:manifold] }},ground_faces:fs.length}}.to_json
    '''
    print(invoke(code),flush=True)
    a=json.loads(target.read_text());assert len(a['wall_solids'])==11
    assert all(w['manifold'] and w['volume_m3']>0 for w in a['wall_solids'])
    fit=json.loads((OUT/'wall_fit_audit.json').read_text())
    expected={w['new_name']:w['volume_m3'] for w in fit['walls'] if 'new_name' in w}
    for w in a['wall_solids']:assert abs(w['volume_m3']-expected[w['name']])<1e-5,w['name']
    g=a['ground'];assert g['faces']==1 and g['edges']==4 and g['loops']==[1] and not g['hidden']
    assert all(abs(v[2]-ground['datum_m'])<1e-9 for v in g['world_vertices_m'])
    assert a['inter_floor_stairs_visible_on_l2'] and a['roof_stair_visible_on_l2']
    audit=json.loads((OUT/'native_audit.json').read_text());assert audit['addition_area_pass']
    audit.update(native_wall_solids_verified=True,single_ground_unchanged_verified=True,stair_scene_membership_verified=True)
    (OUT/'native_audit.json').write_text(json.dumps(audit,indent=2))
    print('11 native solid wall groups verified; single cyan ground unchanged; stair scene membership repaired.',flush=True)


if __name__=='__main__':main()
