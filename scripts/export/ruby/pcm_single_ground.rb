# One native face, not a joined collection of triangles. User-requested simplified
# ground only. The previous house and stair geometry is preserved without edits.
require 'sketchup.rb'
require 'json'
load File.join(__dir__,'pcm_coverage_baseline.rb')

module PCMSingleGround
  def self.build(config_path)
    data=JSON.parse(File.read(config_path));destination=data['destination']
    raise 'Refusing to overwrite handover' if File.exist?(destination)
    backup=PCMCoverageBaseline.preserve_active(destination)
    raise 'Cannot open source' unless Sketchup.open_file(data['source_native'])
    model=Sketchup.active_model;before=PCMCoverageBaseline.snapshot(model)
    originals=model.entities.grep(Sketchup::Group)
    transforms=originals.to_h { |g| [g.name,g.transformation.to_a] }
    raise 'Unexpected source group count' unless before.length==data['expected_base_groups']
    references=data['reference_source_names']
    raise 'Missing reference targets' unless (references-originals.map(&:name)).empty?
    model.start_operation('Single continuous ground plane',true)
    begin
      originals.each { |g| g.set_attribute('CoverageBaseline','reference_only',true) if references.include?(g.name) }
      group=model.entities.add_group;group.name=data['group_name']
      points=data['outline_m'].map { |v| Geom::Point3d.new(*v.map { |c| c*PCMCoverageBaseline::SCALE }) }
      face=group.entities.add_face(points)
      raise 'Could not create the single ground face' unless face
      face.reverse! if face.normal.z<0
      tag='L0_SINGLE_GROUND_PLANE_SIMPLIFIED';group.layer=model.layers[tag] || model.layers.add(tag)
      group.set_attribute('CoverageBaseline','level',0)
      group.set_attribute('CoverageBaseline','kind',data['kind'])
      group.set_attribute('GroundSimplification','scan_only_reconstruction',false)
      group.set_attribute('GroundSimplification','note',data['note'])
      group.set_attribute('GroundSimplification','datum_m',data['datum_m'])
      material=model.materials.add('Single ground plane cyan');material.color=Sketchup::Color.new(*data['colour'])
      group.material=material;face.material=material;face.back_material=material
      model.commit_operation
    rescue => e
      model.abort_operation;raise e
    end
    after=PCMCoverageBaseline.snapshot(model);by_name=after.to_h { |g| [g[:name],g] }
    originals.each do |g|
      old=before.find { |r| r[:name]==g.name };now=by_name[g.name]
      raise "Changed existing geometry: #{g.name}" unless old[:faces]==now[:faces] && (old[:area_m2]-now[:area_m2]).abs<1e-9
      raise "Moved existing geometry: #{g.name}" unless transforms[g.name]==g.transformation.to_a
    end
    faces=group.entities.grep(Sketchup::Face)
    raise 'Ground is not exactly one face without holes' unless faces.length==1 && faces[0].loops.length==1
    raise 'Ground does not have four boundary edges' unless group.entities.grep(Sketchup::Edge).length==4
    actual=faces[0].area/(PCMCoverageBaseline::SCALE**2)
    raise 'Single ground area mismatch' if (actual-data['area_m2']).abs>1e-7
    PCMCoverageBaseline.finish(model,destination,data['label'])
    PCMCoverageBaseline.detail_scenes(model,destination,lambda { |g| g.name==data['group_name'] },'Single ground plane')
    PCMCoverageBaseline.detail_scenes(model,destination,lambda { |g|
      g.get_attribute('CoverageBaseline','kind','').include?('floor') &&
      g.get_attribute('CoverageBaseline','level',0)==0 &&
      !g.get_attribute('CoverageBaseline','reference_only',false)
    },'Ground and raised interior')
    PCMCoverageBaseline.detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='stair_clean' },'Clean upper stairs')
    # Retain an explicit comparison scene of the fitted cyan patches, not the raw
    # underside/scaffolding. Original source files remain untouched as well.
    PCMCoverageBaseline.detail_scenes(model,destination,lambda { |g|
      g.get_attribute('CoverageBaseline','kind','')=='floor_clean' ||
      (g.get_attribute('CoverageBaseline','kind','').include?('floor') &&
       g.get_attribute('CoverageBaseline','level',0)==0 &&
       !g.get_attribute('CoverageBaseline','reference_only',false) && g.name!=data['group_name'])
    },'Previous ground reference')
    model.pages.selected_page=model.pages[0];model.active_view.camera=model.pages[0].camera
    model.entities.grep(Sketchup::Group).each { |g| g.hidden=PCMCoverageBaseline.default_hidden?(g) }
    model.set_attribute('CoverageBaseline','note',data['note'])
    model.set_attribute('CoverageBaseline','added_hole_filling',true)
    model.set_attribute('GroundSimplification','scan_only_reconstruction',false)
    raise 'Save failed' unless model.save(destination)
    audit={path:destination,source_native:data['source_native'],previous_session_backup:backup,
      existing_geometry_preserved:true,existing_transforms_preserved:true,base_groups:before.length,
      groups:after,reference_only_groups:references,scenes:model.pages.map(&:name),
      single_ground:{name:group.name,faces:faces.length,loops:faces[0].loops.length,
        edges:group.entities.grep(Sketchup::Edge).length,area_m2:actual,datum_m:data['datum_m']},
      addition_area_pass:true,design_simplification_not_asbuilt:true}
    File.write(File.join(File.dirname(destination),'native_audit.json'),JSON.pretty_generate(audit))
    {path:destination,groups:after.length,ground_faces:faces.length,ground_holes:faces[0].loops.length-1,preserved_base:true}.to_json
  end
end
