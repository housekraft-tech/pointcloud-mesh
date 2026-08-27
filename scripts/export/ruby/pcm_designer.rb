# Build the flat the way a designer builds it -- and without Solid Tools.
#
# The first attempt used group.union / group.subtract for the relief and the
# openings. Those replace the group with a new one and return nil on failure,
# so a single nil part-way through the 117 operations left `net` pointing at a
# deleted group, every later call failed into a rescue, and the .skp came out
# with the slabs present and the entire wall network missing.
#
# A designer never opens Solid Tools for this anyway:
#
#   1. draw each wall run's footprint as a rectangle, all in ONE group, so the
#      shared corners weld and the masonry becomes a network
#   2. push/pull each footprint up to its own height
#   3. a pilaster: draw the rectangle ON the wall face, push OUT
#   4. a niche:    draw the rectangle ON the wall face, push IN
#   5. a door or window: same rectangle, pushed THROUGH the thickness
#
# Every one of those is add_face + pushpull inside a single context, which is
# what keeps the relief part of the wall instead of a box sitting next to it.
require 'sketchup.rb'
require 'json'

module PCMDesigner
  M = 39.3700787401575

  # A face on the plane x|y = c, spanning s0..s1 along and v0..v1 up.
  def self.face_on(ents, ax, c, s0, s1, v0, v1)
    a = c * M
    p = if ax == 0
      [[a, s0*M, v0*M], [a, s1*M, v0*M], [a, s1*M, v1*M], [a, s0*M, v1*M]]
    else
      [[s0*M, a, v0*M], [s1*M, a, v0*M], [s1*M, a, v1*M], [s0*M, a, v1*M]]
    end
    ents.add_face(p)
  rescue ArgumentError
    nil
  end

  def self.footprint(ents, lo, hi)
    return nil if (hi[0]-lo[0]).abs < 0.0005 || (hi[1]-lo[1]).abs < 0.0005 ||
                  (hi[2]-lo[2]).abs < 0.0005
    f = ents.add_face([lo[0]*M, lo[1]*M, lo[2]*M], [hi[0]*M, lo[1]*M, lo[2]*M],
                      [hi[0]*M, hi[1]*M, lo[2]*M], [lo[0]*M, hi[1]*M, lo[2]*M])
    return nil unless f
    f.reverse! if f.normal.z < 0
    f.pushpull((hi[2]-lo[2]) * M)
    true
  rescue ArgumentError
    nil
  end

  def self.save_as(path)
    m = Sketchup.active_model
    ok = m.save(path)
    "saved=#{ok} bytes=#{File.exist?(path) ? File.size(path) : -1}"
  end

  def self.build(json_path, skp_path)
    t0 = Time.now
    sched = JSON.parse(File.read(json_path))['parts']
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.entities.clear!
    m.definitions.purge_unused
    m.start_operation('draw the flat', true)
    stat = Hash.new(0)

    walls = m.entities.add_group
    walls.name = 'walls'
    we = walls.entities

    # Draw the plan FIRST, all of it, then push it up once.
    #
    # Pushing each run separately leaves the wall full of internal plates: two
    # runs that meet split each other's faces and the divider stays inside the
    # solid. On screen that reads as a wall made of loose parallel panels,
    # which is exactly what came back. A designer draws the whole footprint,
    # rubs out the lines that are not corners, and pulls the lot up together.
    buckets = {}
    sched.select { |r| r['op'] == 'run' }.each do |r|
      lo, hi = r['lo'], r['hi']
      if (hi[0]-lo[0]).abs < 0.0005 || (hi[1]-lo[1]).abs < 0.0005 ||
         (hi[2]-lo[2]).abs < 0.0005
        stat[:degenerate] += 1
        next
      end
      key = [(lo[2]*100).round, (hi[2]*100).round]     # same storey band
      (buckets[key] ||= []) << r
    end
    buckets.each do |(z0i, z1i), runs|
      z0 = z0i/100.0; z1 = z1i/100.0
      plan = []
      runs.each do |r|
        lo, hi = r['lo'], r['hi']
        f = we.add_face([lo[0]*M, lo[1]*M, z0*M], [hi[0]*M, lo[1]*M, z0*M],
                        [hi[0]*M, hi[1]*M, z0*M], [lo[0]*M, hi[1]*M, z0*M]) rescue nil
        plan << f if f
        stat[:runs] += 1
      end
      # rub out the internal lines: an edge between two faces of the same plan
      # is not a corner of the masonry
      we.grep(Sketchup::Edge).each do |e|
        next unless e.valid? && e.faces.length == 2
        a, b = e.faces
        next unless (a.normal.z.abs > 0.99 && b.normal.z.abs > 0.99)
        next unless (a.vertices.first.position.z - b.vertices.first.position.z).abs < 1e-6
        e.erase!
      end
      # now push what is left -- one move per connected piece of plan
      we.grep(Sketchup::Face).each do |f|
        next unless f.valid? && f.normal.z.abs > 0.99
        next unless (f.vertices.first.position.z - z0*M).abs < 1e-6
        f.reverse! if f.normal.z < 0
        begin
          f.pushpull((z1 - z0) * M)
          stat[:pushed] += 1
        rescue => e
          stat[:push_failed] += 1
        end
      end
    end

    # relief, niches and openings: drawn on the face and pushed
    sched.each do |r|
      op = r['op']
      next unless %w[relief niche opening].include?(op)
      lo, hi = r['lo'], r['hi']
      ax = (hi[0]-lo[0]) < (hi[1]-lo[1]) ? 0 : 1        # thin axis = across
      depth = (hi[ax] - lo[ax])
      s0, s1 = lo[1-ax], hi[1-ax]
      v0, v1 = lo[2], hi[2]
      next if (s1-s0).abs < 0.0005 || (v1-v0).abs < 0.0005 || depth < 0.0005
      # Find the wall face this belongs to, rather than assuming its plane.
      # Assuming cost 24 of 26 openings: a thickness step moves the near face,
      # and a rectangle drawn on a plane with no face on it does nothing.
      cu = (s0 + s1)/2.0; cv = (v0 + v1)/2.0
      probe = ax == 0 ? Geom::Point3d.new(lo[ax]*M, cu*M, cv*M)
                      : Geom::Point3d.new(cu*M, lo[ax]*M, cv*M)
      # A group keeps its geometry in ITS OWN coordinates, and SketchUp moves a
      # group's origin as the geometry inside it changes. Probing with world
      # coordinates therefore looked for the wall face in the wrong place and
      # missed 98 of 117 features -- everything except the few near the origin.
      probe = probe.transform(walls.transformation.inverse)
      best = nil; bestd = 0.35 * M
      we.grep(Sketchup::Face).each do |f|
        next unless f.valid? && f.normal.z.abs < 0.01
        n = f.normal
        next unless (ax == 0 ? n.x.abs : n.y.abs) > 0.99
        d = (f.plane[3] + (ax == 0 ? n.x*probe.x : n.y*probe.y) +
             (ax == 0 ? 0 : 0)).abs
        d = probe.distance_to_plane(f.plane)
        next if d > bestd
        # the face must actually cover the spot, not merely share its plane
        pt = probe.project_to_plane(f.plane)
        next unless f.classify_point(pt) == Sketchup::Face::PointInside ||
                    f.classify_point(pt) == Sketchup::Face::PointOnEdge
        best = f; bestd = d
      end
      if best.nil?
        stat[:"#{op}_missed"] += 1
        next
      end
      cut = face_on(we, ax, best.vertices.first.position[ax]/M, s0, s1, v0, v1)
      if cut.nil?
        stat[:"#{op}_nodraw"] += 1
        next
      end
      begin
        dir = (op == 'relief' ? -1.0 : 1.0) * depth * M
        # push away from the material for a pilaster, into it otherwise
        cut.pushpull(cut.normal[ax] > 0 ? -dir : dir)
        stat[op.to_sym] += 1
      rescue => e
        stat[:"#{op}_failed"] += 1
      end
    end

    sched.select { |r| %w[slab column].include?(r['op']) }.each do |r|
      g = m.entities.add_group
      if footprint(g.entities, r['lo'], r['hi'])
        g.name = r['name']
        stat[r['op'].to_sym] += 1
      else
        g.erase! if g.valid?
      end
    end

    m.commit_operation
    # the save is a separate call from the client: doing it here means the
    # socket is still waiting on this command while SketchUp writes, and the
    # bridge timer fires into the middle of the write
    b = m.bounds
    stat.merge(
      wall_faces: we.grep(Sketchup::Face).length,
      wall_edges: we.grep(Sketchup::Edge).length,
      wall_solid: walls.manifold?,
      wall_volume_m3: (walls.manifold? ? (walls.volume/(M*M*M)).round(2) : nil),
      groups: m.entities.grep(Sketchup::Group).length,
      extent_m: [((b.max.x-b.min.x)/M).round(3), ((b.max.y-b.min.y)/M).round(3),
                 ((b.max.z-b.min.z)/M).round(3)],
      seconds: (Time.now-t0).round(1)).to_json
  end
end
