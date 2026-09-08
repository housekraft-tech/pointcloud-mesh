# Native export of an intentionally OPEN, scan-supported CAD subset.
# Never call repair_shell, find_faces, pushpull or any gap-filling operation.
require 'sketchup.rb'
require 'json'

module PCMOverlapSurfaces
  M_TO_IN = 39.37007874015748

  def self.build(json_path, skp_path)
    raise "Refusing to overwrite #{skp_path}" if File.exist?(skp_path)
    data = JSON.parse(File.read(json_path))
    m = Sketchup.active_model
    if m.modified?
      backup = File.join(File.dirname(skp_path), "session_before_overlap_#{Time.now.strftime('%Y%m%d_%H%M%S')}.skp")
      raise 'Could not preserve the active model' unless m.save_copy(backup)
    end
    m.start_operation('Build overlap-only surfaces', true)
    m.entities.clear!
    m.pages.to_a.each { |page| m.pages.erase(page) }
    m.definitions.purge_unused
    m.materials.purge_unused
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.options['UnitsOptions']['LengthPrecision'] = 0
    m.set_attribute('ScanSupport', 'target_mm', 10.0)
    m.set_attribute('ScanSupport', 'maximum_retained_distance_mm', 50.0)
    m.set_attribute('ScanSupport', 'release', 'Incomplete open surfaces; not a certified as-built')
    mats = {}
    created = []
    data['parts'].each do |part|
      mesh = Geom::PolygonMesh.new(part['v'].length, part['f'].length)
      ids = part['v'].map { |v| mesh.add_point(Geom::Point3d.new(*v.map { |x| x*M_TO_IN })) }
      part['f'].each do |face|
        indices = face.map { |i| ids[i] }
        mesh.add_polygon(*indices) if indices.uniq.length == 3
      end
      group = m.entities.add_group
      group.entities.add_faces_from_mesh(mesh, Geom::PolygonMesh::AUTO_SOFTEN)
      group.name = part['name']
      level = part['level'] || part['name'][/L(\d)_/, 1].to_i
      kind = part['kind'].start_with?('wall') ? 'WALL_SURFACES' : part['kind'].upcase
      tag = "OVERLAP_L#{level}_#{kind}"
      group.layer = m.layers[tag] || m.layers.add(tag)
      material = mats[kind] ||= begin
        material = m.materials.add("Scan supported #{kind}")
        material.color = Sketchup::Color.new(*(part['colour'] || [191,183,170]))
        material
      end
      group.material = material
      group.entities.grep(Sketchup::Face).each do |face|
        face.material = material
        face.back_material = material
      end
      group.set_attribute('ScanSupport', 'cutoff_mm', part['support_cutoff_mm'])
      group.set_attribute('ScanSupport', 'open_surface', true)
      created << group
    end
    m.commit_operation
    m.definitions.purge_unused
    m.layers.purge_unused if m.layers.respond_to?(:purge_unused)
    options = m.rendering_options
    options['DisplaySketchAxes'] = false
    options['DrawGround'] = false
    options['DrawHorizon'] = false
    options['EdgeDisplayMode'] = 0
    options['DrawSilhouettes'] = false
    options['BackgroundColor'] = Sketchup::Color.new(245,246,248)
    m.layers.each do |tag|
      tag.visible = !tag.name.include?('CEILING')
    end
    view = m.active_view
    centre = m.bounds.center
    span = [m.bounds.width,m.bounds.height,m.bounds.depth].max
    view.camera = Sketchup::Camera.new(
      Geom::Point3d.new(centre.x+1.3*span,centre.y-1.5*span,centre.z+1.2*span),
      centre, Geom::Vector3d.new(0,0,1), false)
    view.zoom_extents
    m.pages.add('All levels — overlap only')
    (0..2).each do |level|
      m.layers.each do |tag|
        next unless tag.name.start_with?('OVERLAP_')
        tag.visible = tag.name.start_with?("OVERLAP_L#{level}_") && !tag.name.include?('CEILING')
      end
      view.zoom_extents
      m.pages.add("L#{level} — overlap only")
    end
    m.layers.each { |tag| tag.visible = !tag.name.include?('CEILING') }
    view.zoom_extents
    m.pages.selected_page = m.pages[0]
    raise 'SketchUp save failed' unless m.save(skp_path)
    {path:m.path, groups:created.length,
     faces:created.sum { |g| g.entities.grep(Sketchup::Face).length },
     scenes:m.pages.map { |p| p.name }, open_surfaces:true}.to_json
  end
end
