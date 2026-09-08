"""Reopen the native file and verify a literal single, hole-free ground face."""
import json
from make_soulace_single_ground import ROOT,OUT,NATIVE
from export_rectangular_skp import invoke
from verify_observed_recovery import main as verify_native


def main():
    verify_native('soulace',folder_override=str(OUT),native_name=NATIVE)
    config=json.loads((OUT/'single_ground.config.json').read_text())
    target=OUT/'single_face_reopened.json'
    code=f'''
      m=Sketchup.active_model;
      raise 'Wrong active model' unless File.expand_path(m.path)==File.expand_path({json.dumps((OUT/NATIVE).as_posix())});
      g=m.entities.grep(Sketchup::Group).find {{ |x| x.name=={json.dumps(config['group_name'])} }};
      raise 'Missing single ground group' unless g;
      fs=g.entities.grep(Sketchup::Face);es=g.entities.grep(Sketchup::Edge);
      payload={{faces:fs.length,edges:es.length,loops:fs.map {{ |f| f.loops.length }},
        world_vertices_m:fs.flat_map {{ |f| f.vertices }}.uniq.map {{ |v| p=v.position.transform(g.transformation);[p.x,p.y,p.z].map {{ |c| c/PCMCoverageBaseline::SCALE }} }},
        normal:fs.first.normal.to_a,hidden:g.hidden?,reference_only:g.get_attribute('CoverageBaseline','reference_only',false)}};
      File.write({json.dumps(target.as_posix())},JSON.pretty_generate(payload));
      {{faces:fs.length,edges:es.length}}.to_json
    '''
    print(invoke(code),flush=True)
    check=json.loads(target.read_text())
    assert check['faces']==1 and check['edges']==4 and check['loops']==[1]
    assert len(check['world_vertices_m'])==4
    assert all(abs(v[2]-config['datum_m'])<1e-9 for v in check['world_vertices_m'])
    assert abs(check['normal'][2]-1)<1e-9
    assert not check['hidden'] and not check['reference_only']
    audit=json.loads((OUT/'native_audit.json').read_text());audit['single_face_reopened_verified']=True
    (OUT/'native_audit.json').write_text(json.dumps(audit,indent=2))
    print('Verified: 1 horizontal face, 4 edges, 1 outer loop, 0 holes. Original geometry preserved.',flush=True)


if __name__=='__main__':main()
