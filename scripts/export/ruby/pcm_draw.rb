# Draw the building in SketchUp the way a person would: rectangle, push/pull.
#
# Handing SketchUp a mesh is what produced "not flat walls at all, it's a
# triangular mesh" -- 7,732 triangles arrive as 6,012 edges and 2,340 faces,
# and a wall you cannot push/pull. But nothing here was ever a mesh to begin
# with: every wall, pilaster, column, slab and beam is an axis-aligned box, and
# every niche, door and window is a box cut out of one. Given those numbers,
# SketchUp can draw its own geometry -- six clean quads per box, no triangles,
# no diagonals, and a solid the Solid Tools can cut.
#
# What is accounted for, and how:
#   wall / parapet   core runs (one per thickness step along the wall)
#   pilaster, boxed  the "add" boxes on a wall -- unioned into that same wall
#     conduit, beam
#     soffit
#   niche, recess    the "cut" boxes -- subtracted from the wall they sit in
#   door, window,    opening cutters -- subtracted, so the hole goes right
#     arch             through both faces
#   floor, ceiling,  one box per level
#     dropped ceiling
#   beam             its own box, soffit to ceiling
#   column           its own box, floor to ceiling
require 'sketchup.rb'
require 'json'

module PCMDraw
  M = 39.3700787401575     # metres -> inches, SketchUp's internal unit

  # One box, drawn and pushed, not triangulated.
  def self.solid(ents, lo, hi)
    x0, y0, z0 = lo.map { |v| v * M }
    x1, y1, z1 = hi.map { |v| v * M }
    return nil if (x1 - x0).abs < 1e-6 || (y1 - y0).abs < 1e-6 || (z1 - z0).abs < 1e-6
    f = ents.add_face([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0])
    return nil if f.nil?
    f.reverse! if f.normal.z < 0        # push upwards, not into the ground
    f.pushpull(z1 - z0)
    true
  end

  def self.draw(json_path, skp_path)
    t0 = Time.now
    data = JSON.parse(File.read(json_path))
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.entities.clear!
    m.definitions.purge_unused
    m.start_operation('draw the scan', true)

    by_part = {}
    data['parts'].each { |r| (by_part[r['name']] ||= []) << r }

    # The masonry is a NETWORK, not a set of boxes that happen to touch. Drawn
    # into one context, SketchUp welds coincident faces and heals the edges, so
    # a corner is a corner and a tape measure run across three rooms crosses
    # real geometry. Drawn as separate groups, every junction is two surfaces
    # a hair apart and nothing measures cleanly through it.
    made = 0; cuts = 0; failed = []
    wall_kinds = %w[wall parapet column]
    net = m.entities.add_group
    net.name = 'wall network'
    by_part.each do |name, recs|
      next unless wall_kinds.include?(recs.first['kind'])
      recs.select { |r| r['op'] == 'add' }.each do |r|
        solid(net.entities, r['lo'], r['hi']) ? made += 1 : failed << name
      end
    end
    # openings and niches come out of the network as one pass
    by_part.each do |name, recs|
      next unless wall_kinds.include?(recs.first['kind'])
      recs.select { |r| r['op'] == 'cut' }.each do |r|
        cg = m.entities.add_group
        next unless solid(cg.entities, r['lo'], r['hi'])
        begin
          res = net.subtract(cg)
          net = res if res
          cuts += 1
        rescue => e
          cg.erase! if cg.valid?
          failed << "#{name}:cut"
        end
      end
    end
    net.name = 'wall network' if net.valid?

    by_part.each do |name, recs|
      next if wall_kinds.include?(recs.first['kind'])
      kind = recs.first['kind']
      g = m.entities.add_group
      g.name = name
      adds = recs.select { |r| r['op'] == 'add' }
      subs = recs.select { |r| r['op'] == 'cut' }
      adds.each { |r| solid(g.entities, r['lo'], r['hi']) ? made += 1 : failed << name }
      # the cuts: a niche or a doorway is a box taken out of the wall
      subs.each do |r|
        cg = m.entities.add_group
        next unless solid(cg.entities, r['lo'], r['hi'])
        begin
          res = g.subtract(cg)         # Solid Tools; needs SketchUp Pro
          g = res if res
          cuts += 1
        rescue => e
          cg.erase! if cg.valid?
          failed << "#{name}:cut"
        end
      end
      g.name = name if g.valid?
    end
    m.commit_operation
    m.save(skp_path)
    b = m.bounds
    net_faces = net.valid? ? net.entities.grep(Sketchup::Face).length : 0
    { parts: by_part.length, boxes_drawn: made, cuts_made: cuts,
      network_faces: net_faces, network_solid: (net.valid? ? net.manifold? : false),
      groups: m.entities.grep(Sketchup::Group).length,
      faces: m.entities.grep(Sketchup::Group).sum { |x| x.entities.grep(Sketchup::Face).length },
      solids: m.entities.grep(Sketchup::Group).count { |x| x.manifold? },
      failed: failed.uniq.first(8),
      x: [(b.min.x / M).round(3), (b.max.x / M).round(3)],
      z: [(b.min.z / M).round(3), (b.max.z / M).round(3)],
      pro: Sketchup.is_pro?, seconds: (Time.now - t0).round(1) }.to_json
  end
end
