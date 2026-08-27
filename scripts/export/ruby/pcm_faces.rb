# Draw the faces of a solid that Python already worked out.
#
# The geometry is decided before it gets here: manifold3d unions the runs and
# the pilasters and cuts the niches and openings, then every triangle is merged
# onto its plane, so what arrives is one polygon per wall side with its
# doorways as holes. SketchUp only has to draw them -- native faces, no
# triangles, no Solid Tools to lose the model to.
require 'sketchup.rb'
require 'json'

module PCMFaces
  M = 39.3700787401575
  COLOURS = { 'Walls' => [205, 197, 183], 'Floors' => [168, 168, 160],
              'Ceiling (structural)' => [150, 160, 175],
              'Ceiling (false)' => [186, 196, 206],
              'Ceiling (dropped)' => [176, 186, 196],
              'Beams' => [170, 160, 150], 'Columns' => [150, 140, 130],
              'Site clutter / scaffolding' => [220, 150, 90] }

  def self.pt(axis, off, u, v)
    case axis
    when 0 then Geom::Point3d.new(off*M, u*M, v*M)
    when 1 then Geom::Point3d.new(u*M, off*M, v*M)
    else        Geom::Point3d.new(u*M, v*M, off*M)
    end
  end

  def self.build(json_path, _ = nil)
    t0 = Time.now
    data = JSON.parse(File.read(json_path))
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.entities.clear!
    m.definitions.purge_unused
    m.start_operation('draw the faces', true)
    st = Hash.new(0)

    # Tags are how a SketchUp model is made readable without moving anything:
    # every millimetre stays where it was measured and the designer switches
    # off what they do not want to look at.
    COLOURS.each_key do |t|
      lay = m.layers[t] || m.layers.add(t)
      lay.visible = true
      mat = m.materials[t] || m.materials.add(t)
      mat.color = Sketchup::Color.new(*COLOURS[t])
    end

    walls = m.entities.add_group
    walls.name = 'walls'
    walls.layer = m.layers['Walls']
    walls.material = m.materials['Walls']
    we = walls.entities

    data['faces'].each do |f|
      ax = f['axis']; off = f['offset']
      pts = f['outer'].map { |u, v| pt(ax, off, u, v) }
      next if pts.length < 3
      face = (we.add_face(pts) rescue nil)
      if face.nil?
        st[:face_failed] += 1
        next
      end
      st[:faces] += 1
      # a hole: draw it on the same plane, which splits the face, then erase
      # the piece in the middle -- exactly how you cut a door by hand
      f['holes'].each do |h|
        hp = h.map { |u, v| pt(ax, off, u, v) }
        next if hp.length < 3
        inner = (we.add_face(hp) rescue nil)
        if inner
          inner.erase!
          st[:holes] += 1
        else
          st[:hole_failed] += 1
        end
      end
    end

    # The non-flat quarter goes in as ONE mesh, not 78,392 add_face calls.
    #
    # Adding them one at a time ran for over ten minutes and had not finished:
    # every call re-tests the whole context for intersections. PolygonMesh
    # hands SketchUp the lot in a single operation.
    tris = data['tris'] || []
    unless tris.empty?
      pm = Geom::PolygonMesh.new(tris.length*3, tris.length)
      tris.each do |t|
        i1 = pm.add_point(Geom::Point3d.new(t[0][0]*M, t[0][1]*M, t[0][2]*M))
        i2 = pm.add_point(Geom::Point3d.new(t[1][0]*M, t[1][1]*M, t[1][2]*M))
        i3 = pm.add_point(Geom::Point3d.new(t[2][0]*M, t[2][1]*M, t[2][2]*M))
        pm.add_polygon(i1, i2, i3)
      end
      we.add_faces_from_mesh(pm, Geom::PolygonMesh::AUTO_SOFTEN)
      st[:tris] = tris.length
    end

    data['boxes'].each do |b|
      g = m.entities.add_group
      lo = b['lo']; hi = b['hi']
      f = (g.entities.add_face([lo[0]*M, lo[1]*M, lo[2]*M], [hi[0]*M, lo[1]*M, lo[2]*M],
                               [hi[0]*M, hi[1]*M, lo[2]*M], [lo[0]*M, hi[1]*M, lo[2]*M]) rescue nil)
      if f
        f.reverse! if f.normal.z < 0
        f.pushpull((hi[2]-lo[2])*M)
        g.name = b['name']
        t = b['tag'] || 'Walls'
        lay = m.layers[t] || m.layers.add(t)
        g.layer = lay
        g.material = m.materials[t] if m.materials[t]
        st[t.to_sym] += 1
        st[:boxes] += 1
      else
        g.erase! if g.valid?
      end
    end

    m.commit_operation
    bb = m.bounds
    st.merge(wall_faces: we.grep(Sketchup::Face).length,
             triangles: we.grep(Sketchup::Face).count { |x| x.vertices.length == 3 },
             wall_solid: walls.manifold?,
             wall_m3: (walls.manifold? ? (walls.volume/(M*M*M)).round(2) : nil),
             groups: m.entities.grep(Sketchup::Group).length,
             extent_m: [((bb.max.x-bb.min.x)/M).round(3), ((bb.max.y-bb.min.y)/M).round(3),
                        ((bb.max.z-bb.min.z)/M).round(3)],
             seconds: (Time.now-t0).round(1)).to_json
  end

  def self.save_as(path)
    m = Sketchup.active_model
    "saved=#{m.save(path)} bytes=#{File.exist?(path) ? File.size(path) : -1}"
  end
end
