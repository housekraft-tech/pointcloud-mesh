"""Amend only our final generated floor groups, keeping a native backup."""
import json
import trimesh
from check_soulace_floor_junctions import ROOT,OUT
from export_rectangular_skp import invoke
from verify_soulace_wall_floor_planes import NATIVE


def main():
    path=OUT/NATIVE;backup=OUT/'Soulace_floor_junctions_before_pore_cleanup.skp'
    if not backup.exists():raise FileNotFoundError('Create the draft backup before amending')
    config=OUT/'additions.build.json';data=json.loads(config.read_text());audit_path=OUT/'native_audit.json'
    code='''
      load RUBY;m=Sketchup.active_model;
      raise 'Unexpected or modified active model' unless m.path.gsub('\\\\','/').downcase==TARGET.downcase && !m.modified?;
      data=JSON.parse(File.read(CONFIG));parts=data['parts'].select{|p|p['kind']=='floor_wall_joined'};
      groups=m.entities.grep(Sketchup::Group);selected=groups.select{|g|g.get_attribute('CoverageBaseline','kind','')=='floor_wall_joined'};
      raise 'Floor identity changed' unless selected.map(&:name).sort==parts.map{|p|p['name']}.sort;
      before=PCMCoverageBaseline.snapshot(m);m.start_operation('Finish generated floor sampling holes',true);
      begin
        parts.each do |part|
          g=selected.find{|x|x.name==part['name']};g.entities.clear!;
          mesh=Geom::PolygonMesh.new(part['v'].length,part['f'].length);
          ids=part['v'].map{|v|mesh.add_point(Geom::Point3d.new(*v.map{|c|c*PCMCoverageBaseline::SCALE}))};
          part['f'].each{|f|mesh.add_polygon(*f.map{|i|ids[i]})};
          raise 'Fill failed' unless g.entities.fill_from_mesh(mesh,true,Geom::PolygonMesh::AUTO_SOFTEN);
          PCMCoverageBaseline.consolidate_planes(g);g.entities.grep(Sketchup::Face).each{|f|f.material=g.material;f.back_material=g.material};
          raise 'Floor is not closed' unless g.manifold?;
        end;m.commit_operation;
      rescue => e;m.abort_operation;raise e;end;
      after=PCMCoverageBaseline.snapshot(m);by_name=after.to_h{|g|[g[:name],g]};
      before.reject{|g|parts.any?{|p|p['name']==g[:name]}}.each do |g|
        row=by_name[g[:name]];raise 'Unrelated geometry changed' unless row && row[:faces]==g[:faces] && (row[:area_m2]-g[:area_m2]).abs<1e-9;
      end;
      [['L2 3D','native_l2_3d.png'],['L2 Top','native_l2_top.png'],['Joined walls and floors 3D','detail_joined_walls_and_floors_3d.png'],['Joined walls and floors Top','detail_joined_walls_and_floors_top.png']].each do |name,file|
        page=m.pages.find{|p|p.name==name};raise 'Scene missing' unless page;m.pages.selected_page=page;m.active_view.camera=page.camera;
        groups.each do |g|
          g.hidden=if name.start_with?('Joined')
            !['floor_wall_joined','wall_paired_planes'].include?(g.get_attribute('CoverageBaseline','kind',''))
          else
            PCMCoverageBaseline.default_hidden?(g) || !PCMCoverageBaseline.visible_on_level?(g,2)
          end;
        end;
        m.active_view.refresh;raise 'Image failed' unless m.active_view.write_image(filename:File.join(FOLDER,file),width:1800,height:1300,antialias:true,transparent:false);
      end;
      m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;groups.each{|g|g.hidden=PCMCoverageBaseline.default_hidden?(g)};
      raise 'Save failed' unless m.save(TARGET);
      audit=JSON.parse(File.read(AUDIT));audit['groups']=PCMCoverageBaseline.snapshot(m);audit['small_floor_holes_finished']=true;
      File.write(AUDIT,JSON.pretty_generate(audit));{path:m.path,groups:audit['groups'].length}.to_json
    '''
    for key,val in {'RUBY':(ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb').as_posix(),'TARGET':path.as_posix(),'CONFIG':config.as_posix(),'FOLDER':OUT.as_posix(),'AUDIT':audit_path.as_posix()}.items():code=code.replace(key,json.dumps(val))
    print(invoke(code),flush=True)
    audit=json.loads(audit_path.read_text());observed={g['name']:g for g in audit['groups']};checks=[]
    for p in data['parts']:
        expected=float(trimesh.Trimesh(p['v'],p['f'],process=False).area);native=observed[p['name']]['area_m2']
        checks.append({'name':p['name'],'source_area_m2':expected,'native_area_m2':native,'delta_m2':native-expected,'area_pass':abs(native-expected)<max(.000025,expected*1e-5)})
    assert all(c['area_pass'] for c in checks);audit['addition_area_checks']=checks;audit['addition_area_pass']=True
    audit_path.write_text(json.dumps(audit,indent=2));print('Final native areas verified.',flush=True)


if __name__=='__main__':main()
