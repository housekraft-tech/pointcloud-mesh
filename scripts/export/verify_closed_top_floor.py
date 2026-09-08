"""Reopen the whole-house SKP and verify solids, volumes and preserved ground."""
import json
from close_soulace_top_floor import ROOT,OUT
from export_rectangular_skp import invoke
from verify_observed_recovery import main as verify_native

NATIVE='Soulace_closed_top_floor_and_clean_stairs.skp'


def main():
    verify_native('soulace',folder_override=str(OUT),native_name=NATIVE)
    target=OUT/'native_solid_checks.json'
    ground=json.loads((ROOT/'output_final/soulace_single_ground_plane_v3/single_ground.config.json').read_text())
    code=f'''
      m=Sketchup.active_model;groups=m.entities.grep(Sketchup::Group);
      g=groups.find {{ |x| x.name=={json.dumps(ground['group_name'])} }};raise 'Ground missing' unless g;
      fs=g.entities.grep(Sketchup::Face);es=g.entities.grep(Sketchup::Edge);
      solids=groups.select {{ |x| ['wall_solid_continuous','roof_stair_clean'].include?(x.get_attribute('CoverageBaseline','kind','')) }};
      rows=solids.map {{ |x| {{name:x.name,kind:x.get_attribute('CoverageBaseline','kind',''),manifold:x.manifold?,volume_m3:x.volume/(PCMCoverageBaseline::SCALE**3),hidden:x.hidden?,visible_on_l2:PCMCoverageBaseline.visible_on_level?(x,2),thickness_verified:x.get_attribute('ObservedEvidence','thickness_verified',nil)}} }};
      payload={{solids:rows,ground:{{faces:fs.length,edges:es.length,loops:fs.map {{ |f| f.loops.length }},hidden:g.hidden?,
        vertices_m:fs.flat_map {{ |f| f.vertices }}.uniq.map {{ |v| p=v.position.transform(g.transformation);[p.x,p.y,p.z].map {{ |c| c/PCMCoverageBaseline::SCALE }} }} }} }};
      File.write({json.dumps(target.as_posix())},JSON.pretty_generate(payload));
      {{solids:rows.length,closed:rows.count {{ |r| r[:manifold] }},ground_faces:fs.length}}.to_json
    '''
    print(invoke(code),flush=True);a=json.loads(target.read_text())
    expected={r['new_name']:r['volume_m3'] for r in json.loads((OUT/'closure_audit.json').read_text())['walls'] if 'new_name' in r}
    expected['Top-floor roof stair - clean measured run']=json.loads((OUT/'roof_stair_fit.json').read_text())['volume_m3']
    assert len(a['solids'])==22
    for row in a['solids']:
        assert row['manifold'] and row['visible_on_l2'] and not row['hidden'],row
        assert abs(row['volume_m3']-expected[row['name']])<1e-4,row
    g=a['ground'];assert g['faces']==1 and g['edges']==4 and g['loops']==[1] and not g['hidden']
    assert all(abs(v[2]-ground['datum_m'])<1e-9 for v in g['vertices_m'])
    audit=json.loads((OUT/'native_audit.json').read_text());assert audit['addition_area_pass']
    audit.update(new_closed_solids_verified=22,ground_single_face_unchanged=True,staircase_closed_solid_verified=True)
    (OUT/'native_audit.json').write_text(json.dumps(audit,indent=2));print('All 21 wall assemblies + roof stair verified closed after reopening.',flush=True)


if __name__=='__main__':main()
