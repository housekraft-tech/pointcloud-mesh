"""Refine only our newly generated v7 floor groups; retain the draft SKP."""
import json
import trimesh
from check_soulace_floor_junctions import ROOT,OUT
from export_rectangular_skp import invoke
from verify_soulace_wall_floor_planes import NATIVE


def main():
    target=OUT/NATIVE
    if target.exists():raise FileExistsError(target)
    config=OUT/'additions.build.json';audit_path=OUT/'native_audit.json'
    data=json.loads(config.read_text())
    code='''
      load RUBY;
      m=Sketchup.active_model;
      raise 'Expected our unmodified draft' unless ['Soulace_inner_outer_wall_planes_joined_floors.skp','Soulace_paired_wall_planes_floor_junctions_checked.skp'].include?(File.basename(m.path)) && !m.modified?;
      data=JSON.parse(File.read(CONFIG));
      before=PCMCoverageBaseline.snapshot(m);
      transforms=m.entities.grep(Sketchup::Group).to_h{|g|[g.name,g.transformation.to_a]};
      floors=m.entities.grep(Sketchup::Group).select{|g|g.get_attribute('CoverageBaseline','kind','')=='floor_wall_joined'};
      recorded=JSON.parse(File.read(AUDIT));
      expected=recorded['groups'].map{|g|g['name']}.select{|n|n.match?(/^L2 floor plane [0-9]{2} - joined to wall faces$/)};
      raise 'No generated floor groups' if expected.empty?;
      raise 'Unexpected floor groups' unless floors.map(&:name).sort==expected.sort;
      m.start_operation('Close remaining new floor junctions',true);
      begin
        m.entities.erase_entities(floors);
        data['parts'].select{|p|p['kind']=='floor_wall_joined'}.each do |part|
          mesh=Geom::PolygonMesh.new(part['v'].length,part['f'].length);
          ids=part['v'].map{|v|mesh.add_point(Geom::Point3d.new(*v.map{|c|c*PCMCoverageBaseline::SCALE}))};
          part['f'].each{|f|mesh.add_polygon(*f.map{|i|ids[i]})};
          g=m.entities.add_group;raise 'Floor import failed' unless g.entities.fill_from_mesh(mesh,true,Geom::PolygonMesh::AUTO_SOFTEN);
          PCMCoverageBaseline.consolidate_planes(g);
          g.name=part['name'];g.set_attribute('CoverageBaseline','kind','floor_wall_joined');g.set_attribute('CoverageBaseline','level',2);
          g.set_attribute('ObservedEvidence','construction_solid',true);g.set_attribute('ObservedEvidence','source',data['source_label']);
          g.layer=m.layers['OBSERVED_L2_FLOOR_WALL_JOINED'];
          material=m.materials.add('Joined floor '+g.name);material.color=Sketchup::Color.new(*part['colour']);g.material=material;
          g.entities.grep(Sketchup::Face).each{|f|f.material=material;f.back_material=material};
          raise 'New floor is not a native solid: '+g.name unless g.manifold?;
        end;
        m.commit_operation;
      rescue => e
        m.abort_operation;raise e;
      end;
      after=PCMCoverageBaseline.snapshot(m);by_name=after.to_h{|g|[g[:name],g]};
      before.reject{|g|expected.include?(g[:name])}.each do |old|
        g=by_name[old[:name]];raise 'Untouched geometry changed' unless g && g[:faces]==old[:faces] && (g[:area_m2]-old[:area_m2]).abs<1e-9;
      end;
      m.entities.grep(Sketchup::Group).each do |g|
        next if g.get_attribute('CoverageBaseline','kind','')=='floor_wall_joined';
        raise 'Untouched geometry moved' unless transforms[g.name]==g.transformation.to_a;
      end;
      PCMCoverageBaseline.finish(m,TARGET,data['label'],['L2 3D','L2 Top']);
      PCMCoverageBaseline.detail_scenes(m,TARGET,lambda{|g|g.get_attribute('CoverageBaseline','kind','')=='wall_paired_planes'},'Inner and outer wall planes',[1,0,0],[]);
      PCMCoverageBaseline.detail_scenes(m,TARGET,lambda{|g|['wall_paired_planes','floor_wall_joined'].include?(g.get_attribute('CoverageBaseline','kind',''))},'Joined walls and floors',[1,0,0],['3D','Top']);
      PCMCoverageBaseline.detail_scenes(m,TARGET,lambda{|g|g.get_attribute('CoverageBaseline','kind','')=='roof_stair_clean'},'Clean roof staircase',[0,-1,0],[]);
      m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;
      m.entities.grep(Sketchup::Group).each{|g|g.hidden=PCMCoverageBaseline.default_hidden?(g)};
      raise 'Save failed' unless m.save(TARGET);
      audit=JSON.parse(File.read(AUDIT));audit['path']=TARGET;audit['groups']=PCMCoverageBaseline.snapshot(m);audit['scenes']=m.pages.map(&:name);
      audit['local_floor_contact_refinement']=true;audit['draft_native_retained']=true;
      File.write(AUDIT,JSON.pretty_generate(audit));{path:m.path,groups:audit['groups'].length}.to_json
    '''
    replacements={'RUBY':(ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb').as_posix(),'CONFIG':config.as_posix(),'TARGET':target.as_posix(),'AUDIT':audit_path.as_posix()}
    for key,val in replacements.items():code=code.replace(key,json.dumps(val))
    print(invoke(code),flush=True)
    audit=json.loads(audit_path.read_text());observed={g['name']:g for g in audit['groups']}
    checks=[]
    for p in data['parts']:
        area=float(trimesh.Trimesh(p['v'],p['f'],process=False).area);native=observed[p['name']]['area_m2']
        checks.append({'name':p['name'],'source_area_m2':area,'native_area_m2':native,'delta_m2':native-area,'area_pass':bool(abs(native-area)<max(.000025,area*1e-5))})
    assert all(c['area_pass'] for c in checks),checks
    audit['addition_area_checks']=checks;audit['addition_area_pass']=True;audit_path.write_text(json.dumps(audit,indent=2))
    print('Native area checks passed; draft and original models retained.',flush=True)


if __name__=='__main__':main()
