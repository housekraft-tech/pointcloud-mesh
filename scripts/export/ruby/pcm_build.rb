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
    made = 0
    data['parts'].each do |p|
      pts = p['v'].map { |v| Geom::Point3d.new(v[0]*M_TO_IN, v[1]*M_TO_IN, v[2]*M_TO_IN) }
      mesh = Geom::PolygonMesh.new(pts.length, p['f'].length)
      pts.each { |q| mesh.add_point(q) }
      p['f'].each { |f| mesh.add_polygon(f[0] + 1, f[1] + 1, f[2] + 1) }
      g = m.entities.add_group
      g.entities.add_faces_from_mesh(mesh, Geom::PolygonMesh::AUTO_SOFTEN)
      merge_coplanar(g.entities)
      g.name = p['name']
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
    m.commit_operation
    m.save(skp_path)
    b = m.bounds
    { built: made, groups: m.entities.grep(Sketchup::Group).length,
      faces: m.entities.grep(Sketchup::Group).sum { |g| g.entities.grep(Sketchup::Face).length },
      edges: m.entities.grep(Sketchup::Group).sum { |g| g.entities.grep(Sketchup::Edge).length },
      x: [(b.min.x / M_TO_IN).round(3), (b.max.x / M_TO_IN).round(3)],
      y: [(b.min.y / M_TO_IN).round(3), (b.max.y / M_TO_IN).round(3)],
      z: [(b.min.z / M_TO_IN).round(3), (b.max.z / M_TO_IN).round(3)],
      seconds: (Time.now - t0).round(1), saved: skp_path }.to_json
  end
end
