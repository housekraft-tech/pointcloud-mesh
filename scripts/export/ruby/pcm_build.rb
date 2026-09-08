# Build the model in SketchUp from our own arrays, not from an importer.
#
# The Collada file is correct -- 74 named nodes, one geometry each -- and
# SketchUp's importer flattens it anyway into a single component holding 6,012
# loose edges and 2,340 faces. The parts, which are the whole point of the
# handover, do not survive.
#
# So the geometry is handed over as plain numbers and assembled here: one
# Sketchup::Group per part, named, built from a PolygonMesh. Nothing is
# interpreted by an importer, the coordinates are exactly the ones measured,
# and every part arrives as a thing a designer can click.
require 'sketchup.rb'
require 'json'

module PCMBuild
  M_TO_IN = 39.3700787401575

  # A wall side is one rectangle, not two triangles and a diagonal.
  #
  # Every face here arrives as triangles, so a plain box comes into SketchUp as
  # twelve faces with eighteen edges through the middle of its sides -- which
  # is what "broken into fragments" looks like, and it is unusable for
  # push/pull. An edge between two faces that lie in the SAME plane is not a
  # real edge, so it is erased, and the two faces become one.
  def self.merge_coplanar(ents, tol = 1e-6)
    ents.grep(Sketchup::Edge).each do |e|
      next unless e.valid? && e.faces.length == 2
      a, b = e.faces
      next unless a.normal.parallel?(b.normal)
      # parallel is not enough: two parallel faces can sit on different planes
      next if (a.plane[3] - b.plane[3]).abs > tol &&
              (a.plane[3] + b.plane[3]).abs > tol
      e.erase!
    end
  end

  # Boolean libraries can leave harmless zero-face edges, and SketchUp can
  # import a triangulated bottom cap with overlapping coplanar triangles.  The
  # source mesh is watertight, but those import artefacts make SketchUp refuse
  # to identify the group as a Solid.  Rebuild only the affected bottom cap
  # from its boundary edges and remove orphan edges; measured outer vertices do
  # not move.
  def self.repair_shell(group, tol = 1e-5)
    ents = group.entities
    ents.grep(Sketchup::Edge).select { |e| e.valid? && e.faces.empty? }.each do |e|
      e.erase! if e.valid?
    end
    return if group.manifold?
    zmin = group.bounds.min.z
    bottoms = ents.grep(Sketchup::Face).select do |f|
      f.vertices.all? { |v| (v.position.z - zmin).abs < tol }
    end
    bottoms.each { |f| f.erase! if f.valid? }
    ents.grep(Sketchup::Edge).select { |e| e.valid? && e.faces.empty? }.each do |e|
      e.erase! if e.valid?
    end
    boundary = ents.grep(Sketchup::Edge).select do |e|
      e.valid? && e.faces.length == 1 &&
        (e.start.position.z - zmin).abs < tol &&
        (e.end.position.z - zmin).abs < tol
    end
    boundary.each { |e| e.find_faces if e.valid? }
    merge_coplanar(ents)
    ents.grep(Sketchup::Edge).select { |e| e.valid? && e.faces.empty? }.each do |e|
      e.erase! if e.valid?
    end
  end

  def self.build(json_path, skp_path, colours = true)
    t0 = Time.now
    data = JSON.parse(File.read(json_path))
    # NOT Sketchup.file_new: on a modified model that raises a "save changes?"
    # modal, SketchUp stops responding, and the socket waits forever. Clearing
    # the model we are already in has the same effect and asks nothing.
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2      # metres
    m.entities.clear!
    m.definitions.purge_unused
    m.materials.purge_unused
    m.start_operation('build from scan', true)
    mats = {}
    tags = {}
    made = 0
    data['parts'].each do |p|
      if p['dae']
        # PolygonMesh produces non-manifold cap edges on a handful of valid,
        # return-dense opening shells. SketchUp's own Collada path reads those
        # same watertight meshes correctly. Import creates a definition but no
        # instance when driven by Ruby, so place the new definition explicitly.
        before = m.definitions.to_a
        ok = m.import(p['dae'], {merge_coplanar_faces: true, validate: true})
        raise "Collada import failed: #{p['dae']}" unless ok
        definition = (m.definitions.to_a - before).last
        raise "Collada import made no definition: #{p['dae']}" unless definition
        definition.name = p['name']
        g = m.entities.add_instance(definition, Geom::Transformation.new)
      else
        pts = p['v'].map { |v| Geom::Point3d.new(v[0]*M_TO_IN, v[1]*M_TO_IN, v[2]*M_TO_IN) }
        mesh = Geom::PolygonMesh.new(pts.length, p['f'].length)
        # add_point may weld two coordinates inside SketchUp's geometric
        # tolerance and returns the index that actually exists.  Referencing the
        # original sequential indices after such a weld shifts every later face
        # onto the wrong vertex (the failure only appeared on dense arch/niche
        # returns).  Always map source vertex ids to PolygonMesh's real ids.
        point_ids = pts.map { |q| mesh.add_point(q) }
        p['f'].each { |f| mesh.add_polygon(*f.map { |i| point_ids[i] }) }
        g = m.entities.add_group
        g.entities.add_faces_from_mesh(mesh, Geom::PolygonMesh::AUTO_SOFTEN)
        merge_coplanar(g.entities)
        repair_shell(g)
      end
      g.name = p['name']
      tag_name = case p['kind']
                 when 'floor' then 'PCM_FLOORS_CLEAN'
                 when 'ceiling' then 'PCM_CEILINGS'
                 when 'dropped_ceiling' then 'PCM_CEILINGS_DROPPED'
                 when 'beam' then 'PCM_BEAMS'
                 when 'column' then 'PCM_COLUMNS'
                 when 'wall_measured' then 'PCM_WALLS_MEASURED'
                 when 'wall_inferred' then 'PCM_WALLS_INFERRED_REVIEW'
                 when 'wall_profile_review' then 'PCM_WALLS_PROFILE_REVIEW'
                 else 'PCM_FEATURES_REVIEW'
                 end
      tags[tag_name] ||= m.layers[tag_name] || m.layers.add(tag_name)
      g.layer = tags[tag_name]
      if colours && p['colour']
        key = p['kind']
        mats[key] ||= begin
          mm = m.materials.add(key)
          mm.color = Sketchup::Color.new(*p['colour'])
          mm
        end
        g.material = mats[key]
      end
      made += 1
    end
    # Keep the designer view clean on first open.  Nothing is deleted: ceiling
    # and unvalidated feature groups remain available on explicit review tags.
    tags['PCM_CEILINGS'].visible = false if tags['PCM_CEILINGS']
    tags['PCM_CEILINGS_DROPPED'].visible = false if tags['PCM_CEILINGS_DROPPED']
    tags['PCM_FEATURES_REVIEW'].visible = false if tags['PCM_FEATURES_REVIEW']
    m.commit_operation
    m.definitions.purge_unused
    m.materials.purge_unused
    m.layers.purge_unused if m.layers.respond_to?(:purge_unused)
    m.save(skp_path)
    b = m.bounds
    items = m.entities.grep(Sketchup::Group) + m.entities.grep(Sketchup::ComponentInstance)
    { built: made, groups: m.entities.grep(Sketchup::Group).length,
      components: m.entities.grep(Sketchup::ComponentInstance).length,
      faces: items.sum { |g| g.definition.entities.grep(Sketchup::Face).length },
      edges: items.sum { |g| g.definition.entities.grep(Sketchup::Edge).length },
      x: [(b.min.x / M_TO_IN).round(3), (b.max.x / M_TO_IN).round(3)],
      y: [(b.min.y / M_TO_IN).round(3), (b.max.y / M_TO_IN).round(3)],
      z: [(b.min.z / M_TO_IN).round(3), (b.max.z / M_TO_IN).round(3)],
      seconds: (Time.now - t0).round(1), saved: skp_path }.to_json
  end
end
