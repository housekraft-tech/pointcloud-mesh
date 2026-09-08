# Editable rectangular surface candidates. Never infer hidden solid thickness.
require 'sketchup.rb'
require 'json'

module PCMRectangular
  SCALE = 39.37007874015748

  def self.stamp(model, data)
    # The reused SketchUp document can carry assertions from a previous export.
    # These are new local-coordinate models, not the template's geolocation.
    ['ScanSupport','GeoReference','Reconstruction','temp'].each do |name|
      dictionary = model.attribute_dictionary(name, false)
      model.attribute_dictionaries.delete(dictionary) if dictionary
    end
    strict = data['parts'].all? { |p| p['evidence_status'] == 'bounded_scan_support' }
    model.set_attribute('Reconstruction','label',data['label'])
    model.set_attribute('Reconstruction','mode',strict ? 'strict_rectangles' : 'continuity_assisted_rectangles')
    model.set_attribute('Reconstruction','no_inferred_continuity',strict)
    model.set_attribute('Reconstruction','target_mm',10.0)
    model.set_attribute('Reconstruction','support_threshold_mm',50.0)
    model.set_attribute('Reconstruction','whole_surface_distance_bound_mm',50.0) if strict
    model.set_attribute('Reconstruction','semantic_wall_identity_verified',false)
    model.set_attribute('Reconstruction','accuracy','Scan agreement only; not independent dimensional certification')
    model.set_attribute('Reconstruction','continuity',strict ? 'No continuity fill; omitted regions remain unknown' : 'Amber faces may contain unobserved area beyond 50 mm; see audit.json')
    strict
  end

  def self.build(input, destination)
    raise 'Refusing to overwrite native model' if File.exist?(destination)
    data = JSON.parse(File.read(input))
    model = Sketchup.active_model
    backup = nil
    if model.modified?
      backup = File.join(File.dirname(destination), "previous_unsaved_session_#{Time.now.strftime('%Y%m%d_%H%M%S')}.skp")
      raise 'Could not preserve unsaved session' unless model.save_copy(backup)
    end
    model.start_operation('LiDAR rectangular reconstruction', true)
    begin
      model.entities.clear!
      model.pages.to_a.each { |p| model.pages.erase(p) }
      model.definitions.purge_unused
      model.materials.purge_unused
      model.options['UnitsOptions']['LengthUnit'] = 2
      model.options['UnitsOptions']['LengthPrecision'] = 0
      stamp(model, data)
      groups = {}; materials = {}; failures = []
      data['parts'].each do |part|
        name = part['source_object']
        group = groups[name] ||= begin
          g = model.entities.add_group
          g.name = name
          kind = part['kind'].start_with?('wall') ? 'WALL' : part['kind'].upcase
          tag = "RECT_L#{part['level'] || 0}_#{kind}"
          g.layer = model.layers[tag] || model.layers.add(tag)
          g.set_attribute('Reconstruction', 'source_object', name)
          g.set_attribute('Reconstruction', 'level', part['level'] || 0)
          g
        end
        colour = part['colour']; key = colour.join('_')
        material = materials[key] ||= begin
          m = model.materials.add("Rect #{key}"); m.color = Sketchup::Color.new(*colour); m
        end
        vertices = part['v'].map { |v| Geom::Point3d.new(*v.map { |c| c*SCALE }) }
        part['quads'].each do |indices|
          face = group.entities.add_face(indices.map { |i| vertices[i] })
          unless face
            failures << part['name']; next
          end
          face.material = material; face.back_material = material
          face.set_attribute('Reconstruction', 'rectangle_id', part['name'])
          face.set_attribute('Reconstruction', 'evidence_status', part['evidence_status'])
          face.set_attribute('Reconstruction', 'supported_sample_pct', part['supported_sample_pct'])
        end
      end
      raise "Native face creation failures: #{failures.take(10)}" unless failures.empty?
      groups.values.each do |g|
        g.entities.grep(Sketchup::Edge).each do |edge|
          faces = edge.faces
          if faces.length == 2 && faces[0].normal.parallel?(faces[1].normal)
            edge.soft = true; edge.smooth = true; edge.hidden = true
          end
        end
      end
      model.commit_operation
    rescue => e
      model.abort_operation
      raise e
    end
    model.layers.purge_unused if model.layers.respond_to?(:purge_unused)
    model.rendering_options['DisplaySketchAxes'] = false
    model.rendering_options['DrawGround'] = false
    model.rendering_options['DrawHorizon'] = false
    model.rendering_options['EdgeDisplayMode'] = 1
    model.rendering_options['DrawSilhouettes'] = false
    model.rendering_options['DrawHidden'] = false
    model.rendering_options['DisplaySectionCuts'] = false
    model.rendering_options['BackgroundColor'] = Sketchup::Color.new(247,248,250)
    model.layers.each { |t| t.visible = !t.name.include?('CEILING') }
    view = model.active_view
    levels = data['parts'].map { |p| p['level'] || 0 }.uniq.sort
    scene_sets = [['',nil]]
    scene_sets += levels.map { |l| ["L#{l} ",l] } if levels.length > 1
    scene_sets.each do |prefix, level|
      model.layers.each do |t|
        next unless t.name.start_with?('RECT_')
        t.visible = !t.name.include?('CEILING') && (level.nil? || t.name.start_with?("RECT_L#{level}_"))
      end
      bounds = Geom::BoundingBox.new
      model.entities.grep(Sketchup::Group).each { |g| bounds.add(g.bounds) if g.layer.visible? }
      centre = bounds.center; span = [bounds.width,bounds.height,bounds.depth].max
      views = [
      ['3D', Geom::Point3d.new(centre.x+1.3*span,centre.y-1.5*span,centre.z+1.25*span), Geom::Vector3d.new(0,0,1)],
      ['Top',Geom::Point3d.new(centre.x,centre.y,centre.z+span),Geom::Vector3d.new(0,1,0)],
      ['Front',Geom::Point3d.new(centre.x,centre.y-span,centre.z),Geom::Vector3d.new(0,0,1)],
      ['Side',Geom::Point3d.new(centre.x+span,centre.y,centre.z),Geom::Vector3d.new(0,0,1)]
    ]
      views.each do |name,eye,up|
      cam = Sketchup::Camera.new(eye,centre,up,false)
      aspect = 1800.0/1300.0; cam.aspect_ratio = aspect
      right = cam.direction.cross(cam.up).normalize; vertical = cam.up.normalize
      corners = (0..7).map { |i| bounds.corner(i)-centre }
      xs = corners.map { |p| p.dot(right) }; ys = corners.map { |p| p.dot(vertical) }
      cam.height = [ys.max-ys.min,(xs.max-xs.min)/aspect].max*1.10
      view.camera = cam; view.refresh
      model.pages.add(prefix+name)
      image = File.join(File.dirname(destination), "native_#{(prefix+name).strip.downcase.gsub(' ','_')}.png")
      raise 'Native image export failed' unless view.write_image(filename:image,width:1800,height:1300,antialias:true,transparent:false)
      end
    end
    model.layers.each { |t| t.visible = !t.name.include?('CEILING') }
    model.pages.selected_page = model.pages[0]
    view.camera = model.pages[0].camera
    model.styles.update_selected_style if model.styles.respond_to?(:update_selected_style)
    raise 'Native save failed' unless model.save(destination)
    {path:destination, previous_session_backup:backup, groups:groups.length, requested_rectangles:data['parts'].length}.to_json
  end
end
