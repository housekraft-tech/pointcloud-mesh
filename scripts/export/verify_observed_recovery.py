"""Reopen the delivered SKP and check all group counts and surface areas."""
import argparse
import json
from pathlib import Path
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]


def main(project,refresh_floor=False,folder_override=None,native_name=None):
    folder=Path(folder_override).resolve() if folder_override else ROOT/f'output_final/scan_recovered_v1/{project}'
    path=folder/(native_name or ('Soulace_stairs_and_lower_floor.skp' if project=='soulace' else 'Engrance_scan_recovered.skp'))
    audit=json.loads((folder/'native_audit.json').read_text());target=folder/'reopened_groups.json'
    ruby=(ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb').as_posix()
    code=f'load {json.dumps(ruby)}; PCMCoverageBaseline.preserve_active({json.dumps(path.as_posix())}); raise "Reopen failed" unless Sketchup.open_file({json.dumps(path.as_posix())}); m=Sketchup.active_model;'
    if refresh_floor:
        code+='''m.pages.to_a.select { |p| p.name.start_with?('Ground and floors') || p.name.start_with?('Observed stairs') }.each { |p| m.pages.erase(p) };'''
        code+=f'''PCMCoverageBaseline.detail_scenes(m,{json.dumps(path.as_posix())},lambda {{ |g| g.get_attribute('CoverageBaseline','kind','').include?('floor') && g.get_attribute('CoverageBaseline','level',0)==0 }},'Ground and floors');
          PCMCoverageBaseline.detail_scenes(m,{json.dumps(path.as_posix())},lambda {{ |g| g.get_attribute('CoverageBaseline','kind','')=='stair_observed' }},'Observed stairs');
          m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;
          m.entities.grep(Sketchup::Group).each {{ |g| g.hidden=PCMCoverageBaseline.default_hidden?(g) }};
          raise 'Save failed' unless m.save({json.dumps(path.as_posix())});
          raise 'Reopen failed' unless Sketchup.open_file({json.dumps(path.as_posix())});m=Sketchup.active_model;
        '''
    # Commit display settings into each saved style, so changing scenes does not
    # restore the old sky/ground or heavy profile edges. Geometry is untouched.
    code+=f'''m.styles.each do |style|
      m.styles.selected_style=style;
      m.rendering_options['DrawGround']=false;m.rendering_options['DrawHorizon']=false;
      m.rendering_options['EdgeDisplayMode']=0;m.rendering_options['DrawSilhouettes']=false;
      m.rendering_options['BackgroundColor']=Sketchup::Color.new(247,248,250);
      m.styles.update_selected_style;
    end;
    m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;
    m.entities.grep(Sketchup::Group).each {{ |g| g.hidden=PCMCoverageBaseline.default_hidden?(g) }};
    m.active_view.refresh;
    raise 'Image failed' unless m.active_view.write_image(filename:{json.dumps((folder/'native_3d.png').as_posix())},width:1800,height:1300,antialias:true,transparent:false);
    raise 'Save failed' unless m.save({json.dumps(path.as_posix())});
    raise 'Reopen failed' unless Sketchup.open_file({json.dumps(path.as_posix())});m=Sketchup.active_model;
    '''
    code+=f'''File.write({json.dumps(target.as_posix())},JSON.pretty_generate({{groups:PCMCoverageBaseline.snapshot(m),scenes:m.pages.map {{ |p| p.name }},path:m.path,
      reference_visibility:m.entities.grep(Sketchup::Group).select {{ |g| g.get_attribute('CoverageBaseline','reference_only',false) }}.map {{ |g| {{name:g.name,hidden:g.hidden?}} }}
    }})); {{verified_file:m.path}}.to_json'''
    print(invoke(code),flush=True)
    observed=json.loads(target.read_text());expected={p['name']:p for p in audit['groups']}
    assert set(expected)=={p['name'] for p in observed['groups']}
    for group in observed['groups']:
        orig=expected[group['name']]
        assert group['faces']==orig['faces'],group['name']
        assert abs(group['area_m2']-orig['area_m2'])<1e-8,group['name']
    reference=observed.get('reference_visibility',[])
    assert {g['name'] for g in reference}==set(audit.get('reference_only_groups',[]))
    assert all(g['hidden'] for g in reference),'Superseded rough reference visible in default scene'
    audit['reference_only_hidden_on_reopen']=True
    audit['native_reopen_verified']=True;audit['scenes']=observed['scenes']
    (folder/'native_audit.json').write_text(json.dumps(audit,indent=2));print('All saved/reopened group counts and areas match',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('project',choices=['soulace','engrance']);ap.add_argument('--refresh-ground-views',action='store_true');ap.add_argument('--folder');ap.add_argument('--native-name');args=ap.parse_args();main(args.project,args.refresh_ground_views,args.folder,args.native_name)
