# Restore or import the coverage-first LiDAR baseline without rectangle gates.
require 'sketchup.rb'
require 'json'

module PCMCoverageBaseline
  SCALE = 39.37007874015748

  def self.default_hidden?(group)
    group.get_attribute('CoverageBaseline','kind','').include?('ceiling') || group.get_attribute('CoverageBaseline','reference_only',false)
  end

  def self.visible_on_level?(group,level)
    return true if level.nil?
    levels=group.get_attribute('CoverageBaseline','scene_levels',nil)
    levels ? levels.include?(level) : group.get_attribute('CoverageBaseline','level',0)==level
  end

  def self.preserve_active(destination)
    model=Sketchup.active_model
    return nil unless model.modified?
    backup=File.join(File.dirname(destination),"previous_session_#{Time.now.strftime('%Y%m%d_%H%M%S')}.skp")
    raise 'Backup already exists' if File.exist?(backup)
    raise 'Could not preserve unsaved session' unless model.save_copy(backup)
    backup
  end

  def self.snapshot(model)
    model.entities.grep(Sketchup::Group).map do |g|
      faces=g.entities.grep(Sketchup::Face)
      {name:g.name,faces:faces.length,area_m2:faces.sum { |f| f.area }/(SCALE*SCALE)}
    end
  end

  def self.consolidate_planes(group)
    # Remove only internal triangulation, never perimeter or opening edges.
    edges=group.entities.grep(Sketchup::Edge).select do |edge|
      faces=edge.faces
      next false unless faces.length==2 && faces[0].normal.dot(faces[1].normal)>1.0-1e-10
      plane=faces[0].plane
      faces[1].vertices.all? { |v| v.position.distance_to_plane(plane).abs<1e-5 }
    end
    group.entities.erase_entities(edges) unless edges.empty?
  end

  def self.label_wall_planes(group,planes)
    planes.each do |plane|
      n=Geom::Vector3d.new(*plane['normal']);offset=plane['offset_m']
      faces=group.entities.grep(Sketchup::Face).select do |face|
        face.vertices.all? do |v|
          p=v.position
          ((p.x*n.x+p.y*n.y+p.z*n.z)/SCALE-offset).abs<1e-6
        end
      end
      raise 'Missing wall side plane' if faces.empty?
      faces.each do |face|
        face.set_attribute('WallPlane','side',plane['side'])
        face.set_attribute('WallPlane','source',plane['source'])
        face.set_attribute('WallPlane','offset_m',offset)
      end
    end
    group.set_attribute('WallPlane','paired_sides',true)
  end

  def self.finish(model,destination,label,image_names=nil)
    ['GeoReference','Reconstruction','temp'].each do |name|
      dictionary=model.attribute_dictionary(name,false)
      model.attribute_dictionaries.delete(dictionary) if dictionary
    end
    model.set_attribute('CoverageBaseline','label',label)
    model.set_attribute('CoverageBaseline','rectangular_selection',false)
    model.set_attribute('CoverageBaseline','added_hole_filling',false)
    model.set_attribute('CoverageBaseline','semantic_wall_identity_verified',false)
    model.set_attribute('CoverageBaseline','note','Coverage-first candidate/LiDAR overlap; not all raw scan geometry or certified dimensions')
    model.options['UnitsOptions']['LengthUnit']=2
    model.options['UnitsOptions']['LengthPrecision']=0
    model.pages.to_a.each { |page| model.pages.erase(page) }
    model.rendering_options['DisplaySketchAxes']=false
    model.rendering_options['DrawGround']=false
    model.rendering_options['DrawHorizon']=false
    model.rendering_options['EdgeDisplayMode']=0
    model.rendering_options['DrawSilhouettes']=false
    model.rendering_options['DrawHidden']=false
    model.rendering_options['DisplaySectionCuts']=false
    model.rendering_options['BackgroundColor']=Sketchup::Color.new(247,248,250)
    model.styles.update_selected_style
    groups=model.entities.grep(Sketchup::Group)
    levels=groups.map { |g| g.get_attribute('CoverageBaseline','level',0) }.uniq.sort
    sets=[['',nil]]
    sets+=levels.map { |l| ["L#{l} ",l] } if levels.length>1
    view=model.active_view
    sets.each do |prefix,level|
      groups.each do |g|
        kind=g.get_attribute('CoverageBaseline','kind','')
        g.hidden=default_hidden?(g) || !visible_on_level?(g,level)
        g.layer.visible=true
      end
      bounds=Geom::BoundingBox.new
      groups.each { |g| bounds.add(g.bounds) unless g.hidden? }
      c=bounds.center;s=[bounds.width,bounds.height,bounds.depth].max
      views=[['3D',Geom::Point3d.new(c.x+1.3*s,c.y-1.5*s,c.z+1.25*s),Geom::Vector3d.new(0,0,1)],
             ['Top',Geom::Point3d.new(c.x,c.y,c.z+s),Geom::Vector3d.new(0,1,0)],
             ['Front',Geom::Point3d.new(c.x,c.y-s,c.z),Geom::Vector3d.new(0,0,1)],
             ['Side',Geom::Point3d.new(c.x+s,c.y,c.z),Geom::Vector3d.new(0,0,1)]]
      views.each do |name,eye,up|
        cam=Sketchup::Camera.new(eye,c,up,false)
        aspect=1800.0/1300.0;cam.aspect_ratio=aspect
        right=cam.direction.cross(cam.up).normalize;vertical=cam.up.normalize
        corners=(0..7).map { |i| bounds.corner(i)-c }
        xs=corners.map { |p| p.dot(right) };ys=corners.map { |p| p.dot(vertical) }
        cam.height=[ys.max-ys.min,(xs.max-xs.min)/aspect].max*1.10
        view.camera=cam;view.refresh
        model.pages.add(prefix+name)
        image=File.join(File.dirname(destination),"native_#{(prefix+name).strip.downcase.gsub(' ','_')}.png")
        if image_names.nil? || image_names.include?(prefix+name)
          raise 'Image failed' unless view.write_image(filename:image,width:1800,height:1300,antialias:true,transparent:false)
        end
      end
    end
    groups.each { |g| g.hidden=default_hidden?(g) }
    model.pages.selected_page=model.pages[0];view.camera=model.pages[0].camera
    model.rendering_options['DisplaySketchAxes']=false
    model.pages[0].update
    raise 'Native save failed' unless model.save(destination)
  end

  def self.restore(config_path,destination)
    raise 'Refusing to overwrite handover' if File.exist?(destination)
    config=JSON.parse(File.read(config_path))
    backup=preserve_active(destination)
    raise 'Could not open baseline' unless Sketchup.open_file(config['native_source'])
    model=Sketchup.active_model
    before=snapshot(model)
    groups=model.entities.grep(Sketchup::Group)
    expected=config['groups'].keys.sort
    raise 'Unexpected baseline groups' unless groups.map { |g| g.name }.sort==expected
    groups.each do |g|
      item=config['groups'][g.name]
      g.set_attribute('CoverageBaseline','level',item['level'])
      g.set_attribute('CoverageBaseline','kind',item['kind'])
      matrix=config['source_to_common'][item['level'].to_s].map(&:dup)
      3.times { |i| matrix[i][3]*=SCALE }
      g.transform!(Geom::Transformation.new(matrix.transpose.flatten))
    end
    after=snapshot(model)
    before_by_name=before.to_h { |g| [g[:name],g] }
    after.each do |g|
      old=before_by_name[g[:name]]
      raise 'Restore changed source faces' unless g[:faces]==old[:faces]
      raise 'Restore changed source area' if (g[:area_m2]-old[:area_m2]).abs>1e-9
    end
    finish(model,destination,config['label'])
    audit={path:destination,source:config['native_source'],previous_session_backup:backup,
           source_faces:before.sum { |g| g[:faces] },output_faces:after.sum { |g| g[:faces] },
           exact_group_face_area_preservation:true,groups:after,scenes:model.pages.map { |p| p.name }}
    File.write(File.join(File.dirname(destination),'native_audit.json'),JSON.pretty_generate(audit))
    {path:destination,groups:after.length,faces:audit[:output_faces],preserved:true}.to_json
  end

  def self.build(json_path,destination)
    raise 'Refusing to overwrite handover' if File.exist?(destination)
    data=JSON.parse(File.read(json_path));backup=preserve_active(destination)
    model=Sketchup.active_model
    model.start_operation('Build coverage-first LiDAR baseline',true)
    begin
      model.entities.clear!
      model.definitions.purge_unused;model.materials.purge_unused
      materials={}
      data['parts'].each do |part|
        mesh=Geom::PolygonMesh.new(part['v'].length,part['f'].length)
        ids=part['v'].map { |v| mesh.add_point(Geom::Point3d.new(*v.map { |c| c*SCALE })) }
        part['f'].each { |f| mesh.add_polygon(*f.map { |i| ids[i] }) }
        g=model.entities.add_group
        raise 'Mesh fill failed' unless g.entities.fill_from_mesh(mesh,true,Geom::PolygonMesh::AUTO_SOFTEN)
        consolidate_planes(g) if part['merge_coplanar_faces']
        label_wall_planes(g,part['wall_plane_pair']) if part['wall_plane_pair']
        g.name=part['name'];kind=part['kind'];level=part['level'] || 0
        tag="LIDAR_L#{level}_#{kind.upcase}";g.layer=model.layers[tag] || model.layers.add(tag)
        g.set_attribute('CoverageBaseline','level',level);g.set_attribute('CoverageBaseline','kind',kind)
        g.set_attribute('CoverageBaseline','reference_only',true) if part['reference_only']
        g.set_attribute('CoverageBaseline','scene_levels',part['scene_levels']) if part['scene_levels']
        g.set_attribute('ObservedEvidence','semantic_classification_verified',false)
        g.set_attribute('ObservedEvidence','thickness_verified',false)
        g.set_attribute('ObservedEvidence','source',part['evidence_status'] || 'Unverified architectural hypothesis')
        g.set_attribute('ObservedEvidence','support_cutoff_mm',part['support_cutoff_mm']) if part['support_cutoff_mm']
        colour=part['colour'] || [194,204,207];key=colour.join('_')
        material=materials[key] ||= begin
          m=model.materials.add("LiDAR #{key}");m.color=Sketchup::Color.new(*colour);m
        end
        g.material=material
        g.entities.grep(Sketchup::Face).each { |f| f.material=material;f.back_material=material }
      end
      model.commit_operation
    rescue => e
      model.abort_operation;raise e
    end
    finish(model,destination,data['label'])
    measured=snapshot(model)
    File.write(File.join(File.dirname(destination),'native_audit.json'),JSON.pretty_generate({path:destination,previous_session_backup:backup,groups:measured,scenes:model.pages.map { |p| p.name }}))
    {path:destination,groups:measured.length,faces:measured.sum { |g| g[:faces] }}.to_json
  end

  def self.detail_scenes(model,destination,selector,prefix,side_offset=[1,0,0],image_names=nil)
    groups=model.entities.grep(Sketchup::Group)
    selected=groups.select { |g| selector.call(g) }
    return if selected.empty?
    groups.each { |g| g.hidden=!selected.include?(g) }
    bounds=Geom::BoundingBox.new;selected.each { |g| bounds.add(g.bounds) }
    c=bounds.center;s=[bounds.width,bounds.height,bounds.depth].max;view=model.active_view
    [['3D',[1.3,-1.5,1.05],[0,0,1]],['Side',side_offset,[0,0,1]],['Top',[0,0,1],[0,1,0]]].each do |name,offset,up|
      eye=Geom::Point3d.new(c.x+s*offset[0],c.y+s*offset[1],c.z+s*offset[2])
      camera=Sketchup::Camera.new(eye,c,Geom::Vector3d.new(*up),false)
      camera.aspect_ratio=1800.0/1300.0
      right=camera.direction.cross(camera.up).normalize;vertical=camera.up.normalize
      corners=(0..7).map { |i| bounds.corner(i)-c }
      xs=corners.map { |p| p.dot(right) };ys=corners.map { |p| p.dot(vertical) }
      camera.height=[ys.max-ys.min,(xs.max-xs.min)/camera.aspect_ratio].max*1.12
      model.rendering_options['DrawGround']=false
      model.rendering_options['DrawHorizon']=false
      model.rendering_options['EdgeDisplayMode']=0
      model.rendering_options['DrawSilhouettes']=false
      model.rendering_options['BackgroundColor']=Sketchup::Color.new(247,248,250)
      model.styles.update_selected_style
      view.camera=camera;view.refresh;model.pages.add(prefix+' '+name)
      file=File.join(File.dirname(destination),"detail_#{prefix.downcase.gsub(' ','_')}_#{name.downcase}.png")
      if image_names.nil? || image_names.include?(name)
        raise 'Detail image failed' unless view.write_image(filename:file,width:1800,height:1300,antialias:true,transparent:false)
      end
    end
  end

  def self.append(json_path,destination)
    raise 'Refusing to overwrite handover' if File.exist?(destination)
    data=JSON.parse(File.read(json_path));backup=preserve_active(destination)
    raise 'Could not open coverage baseline' unless Sketchup.open_file(data['source_native'])
    model=Sketchup.active_model;before=snapshot(model)
    original_transforms=model.entities.grep(Sketchup::Group).to_h { |g| [g.name,g.transformation.to_a] }
    raise 'Wrong base model' unless before.length==data['expected_base_groups']
    reference_names=data['reference_source_names'] || []
    raise 'Missing reference groups' unless (reference_names-before.map { |g| g[:name] }).empty?
    model.start_operation('Add observed scan surfaces',true)
    begin
      model.entities.grep(Sketchup::Group).each do |g|
        g.set_attribute('CoverageBaseline','reference_only',true) if reference_names.include?(g.name)
        levels=(data['source_scene_levels'] || {})[g.name]
        g.set_attribute('CoverageBaseline','scene_levels',levels) if levels
      end
      data['parts'].each do |part|
        File.write(File.join(File.dirname(destination),'export_progress.txt'),"#{Time.now}: preparing #{part['name']} (#{part['f'].length} triangles)")
        mesh=Geom::PolygonMesh.new(part['v'].length,part['f'].length)
        ids=part['v'].map { |v| mesh.add_point(Geom::Point3d.new(*v.map { |c| c*SCALE })) }
        part['f'].each { |f| mesh.add_polygon(*f.map { |i| ids[i] }) }
        g=model.entities.add_group
        File.write(File.join(File.dirname(destination),'export_progress.txt'),"#{Time.now}: filling #{part['name']}")
        raise 'Mesh fill failed' unless g.entities.fill_from_mesh(mesh,true,Geom::PolygonMesh::AUTO_SOFTEN)
        consolidate_planes(g) if part['merge_coplanar_faces']
        label_wall_planes(g,part['wall_plane_pair']) if part['wall_plane_pair']
        g.name=part['name'];kind=part['kind'];level=part['level'] || 0
        tag="OBSERVED_L#{level}_#{kind.upcase}";g.layer=model.layers[tag] || model.layers.add(tag)
        g.set_attribute('CoverageBaseline','level',level);g.set_attribute('CoverageBaseline','kind',kind)
        g.set_attribute('CoverageBaseline','scene_levels',part['scene_levels']) if part['scene_levels']
        g.set_attribute('CoverageBaseline','reference_only',true) if part['reference_only']
        g.set_attribute('ObservedEvidence','construction_solid',part['construction_solid'] || false)
        unless part['thickness_verified'].nil?
          g.set_attribute('ObservedEvidence','thickness_verified',part['thickness_verified'])
          g.set_attribute('ObservedEvidence','modeled_thickness_m',part['modeled_thickness_m'])
          g.set_attribute('ObservedEvidence','measured_face_positions_local_m',part['measured_face_positions_m'])
        end
        g.set_attribute('ObservedEvidence','source',data['source_label'] || 'Raw-LAS bounded surface recovery')
        g.set_attribute('ObservedEvidence','semantic_classification_verified',false)
        material=model.materials.add('Observed '+g.name);material.color=Sketchup::Color.new(*(part['colour'] || [94,159,171]))
        g.material=material
        g.entities.grep(Sketchup::Face).each { |f| f.material=material;f.back_material=material }
      end
      model.commit_operation
    rescue => e
      model.abort_operation;raise e
    end
    after=snapshot(model);by_name=after.to_h { |g| [g[:name],g] }
    before.each do |g|
      same=by_name[g[:name]]
      raise 'Existing geometry changed' unless same && same[:faces]==g[:faces] && (same[:area_m2]-g[:area_m2]).abs<1e-9
    end
    model.entities.grep(Sketchup::Group).each do |g|
      if original_transforms.key?(g.name)
        raise 'Existing object moved' unless original_transforms[g.name]==g.transformation.to_a
      end
    end
    finish(model,destination,data['label'],data['render_image_names'])
    if data['parts'].any? { |p| p['kind']=='wall_paired_planes' }
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='wall_paired_planes' },'Inner and outer wall planes')
      detail_scenes(model,destination,lambda { |g| ['wall_paired_planes','floor_wall_joined'].include?(g.get_attribute('CoverageBaseline','kind','')) },'Joined walls and floors')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='roof_stair_clean' },'Clean roof staircase',[0,-1,0])
    elsif data['parts'].any? { |p| p['kind']=='wall_solid_continuous' }
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='wall_solid_continuous' },'Continuous top floor walls')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='roof_stair_clean' },'Clean roof staircase',[0,-1,0])
    elsif data['parts'].any? { |p| p['kind']=='roof_stair_scan' }
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='roof_stair_scan' },'Top floor roof stair',[0,-1,0])
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='stair_clean' },'Inter-floor stairs L1-L2')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='wall_solid_refit' },'Modeled top floor walls')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','').include?('floor') && g.get_attribute('CoverageBaseline','level',0)==0 && !default_hidden?(g) },'Ground and raised interior')
    elsif data['parts'].any? { |p| p['kind']=='stair_clean' }
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='stair_clean' },'Clean upper stairs')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='stair_observed' && g.get_attribute('CoverageBaseline','reference_only',false) },'Original upper scan reference')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','').include?('floor') && g.get_attribute('CoverageBaseline','level',0)==0 && !g.get_attribute('CoverageBaseline','reference_only',false) },'Clean ground and floors')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='lower_floor_observed' },'Original cyan floor reference')
    elsif data['parts'].any? { |p| p['kind'].include?('stair') }
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='stair_observed' },'Observed stairs')
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','').include?('floor') && g.get_attribute('CoverageBaseline','level',0)==0 },'Ground and floors')
    else
      detail_scenes(model,destination,lambda { |g| g.get_attribute('CoverageBaseline','kind','')=='scan_detail' },'Recovered evidence')
      names=before.map { |g| g[:name] }
      detail_scenes(model,destination,lambda { |g| names.include?(g.name) && !g.get_attribute('CoverageBaseline','kind','').include?('ceiling') },'Baseline only')
    end
    model.pages.selected_page=model.pages[0];model.active_view.camera=model.pages[0].camera
    model.entities.grep(Sketchup::Group).each { |g| g.hidden=default_hidden?(g) }
    model.set_attribute('CoverageBaseline','note',data['note'] || 'Previous coverage preserved; cyan/orange additions are scan-supported surface evidence, not verified construction solids or certified dimensions')
    model.set_attribute('CoverageBaseline','added_hole_filling',data['small_hole_interpolation'] || false)
    raise 'Save failed' unless model.save(destination)
    all_references=model.entities.grep(Sketchup::Group).select { |g| g.get_attribute('CoverageBaseline','reference_only',false) }.map(&:name)
    audit={path:destination,source_native:data['source_native'],previous_session_backup:backup,existing_geometry_preserved:true,existing_transforms_preserved:true,
           base_groups:before.length,base_faces:before.sum { |g| g[:faces] },groups:after,scenes:model.pages.map { |p| p.name },reference_only_groups:all_references,
           scene_level_overrides:data['source_scene_levels'] || {}}
    File.write(File.join(File.dirname(destination),'native_audit.json'),JSON.pretty_generate(audit))
    {path:destination,groups:after.length,faces:after.sum { |g| g[:faces] },preserved_base:true}.to_json
  end
end
