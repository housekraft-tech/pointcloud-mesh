# One wall at a time: draw it, work its relief into it, cut its openings.
#
# Two earlier approaches failed for reasons worth writing down.
#
# Solid Tools on the WHOLE network: group.subtract replaces the group and
# returns nil on failure, so one nil left `net` pointing at a deleted group and
# the entire wall network vanished from the saved file.
#
# Push/pull on a found face: after merging the plan, a wall's face is not one
# face -- it is 258 slivers, none of which the opening's centre lands inside --
# and Face#classify_point rejects even a projected point as "not on plane".
#
# So each wall is built alone: its runs pushed up, its pilasters unioned on,
# its niches and openings subtracted. The solids are small, the booleans are
# reliable, and if one wall fails it is one wall, not the building. They are
# exploded into a single context at the end, which welds the shared faces and
# leaves a real network.
require 'sketchup.rb'
require 'json'

module PCMWall
  M = 39.3700787401575

  def self.box(ents, lo, hi)
    return nil if (hi[0]-lo[0]).abs < 5e-4 || (hi[1]-lo[1]).abs < 5e-4 ||
                  (hi[2]-lo[2]).abs < 5e-4
    f = ents.add_face([lo[0]*M, lo[1]*M, lo[2]*M], [hi[0]*M, lo[1]*M, lo[2]*M],
                      [hi[0]*M, hi[1]*M, lo[2]*M], [lo[0]*M, hi[1]*M, lo[2]*M])
    return nil unless f
    f.reverse! if f.normal.z < 0
    f.pushpull((hi[2]-lo[2]) * M)
    true
  rescue ArgumentError
    nil
  end

  def self.build(json_path, weld = true)
    t0 = Time.now
    sched = JSON.parse(File.read(json_path))['parts']
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.entities.clear!
    m.definitions.purge_unused
    m.start_operation('draw the flat', true)
    st = Hash.new(0)

    by = {}
    sched.each { |r| (by[r['name']] ||= []) << r }
    wall_groups = []

    by.each do |name, recs|
      kind = recs.first['kind']
      unless %w[wall parapet].include?(kind)
        recs.each do |r|
          g = m.entities.add_group
          box(g.entities, r['lo'], r['hi']) ? (g.name = name; st[r['op'].to_sym] += 1)
                                            : (g.erase! if g.valid?)
        end
        next
      end
      # Each run gets its own group first, then they are folded together.
      #
      # Solid Tools refuses unless BOTH operands are manifold, and a group with
      # several overlapping boxes drawn straight into it is not: the boxes
      # leave faces inside each other. Built that way, 107 of 117 relief and
      # opening operations failed. One box per group IS manifold, and a union
      # of two manifold solids is manifold, so folding keeps the invariant all
      # the way up.
      pieces = []
      recs.select { |r| r['op'] == 'run' }.each do |r|
        pg = m.entities.add_group
        # Two runs of the same wall only TOUCH, and Solid Tools will not union
        # solids that merely share a face -- which is why a 12 m wall came back
        # as a 1.8 m fragment. Overlap them by a millimetre along the wall and
        # the union is unambiguous. A millimetre is well inside the 4 mm the
        # boxes already differ from the scan.
        lo = r['lo'].dup; hi = r['hi'].dup
        long_ax = (hi[0]-lo[0]) >= (hi[1]-lo[1]) ? 0 : 1
        lo[long_ax] -= 0.001; hi[long_ax] += 0.001
        r = r.merge('lo' => lo, 'hi' => hi)
        if box(pg.entities, r['lo'], r['hi'])
          pieces << pg
          st[:runs] += 1
        else
          pg.erase! if pg.valid?
          st[:degenerate] += 1
        end
      end
      next if pieces.empty?
      g = pieces.shift
      pieces.each do |pg|
        res = (g.union(pg) rescue nil)
        if res
          g = res
        else
          # disjoint runs cannot be unioned; keep them as their own wall
          pg.name = name
          wall_groups << pg
          st[:run_kept_apart] += 1
        end
      end
      # a pilaster belongs to this wall: union it in
      recs.select { |r| r['op'] == 'relief' }.each do |r|
        t = m.entities.add_group
        next unless box(t.entities, r['lo'], r['hi'])
        res = (g.union(t) rescue nil)
        if res then g = res; st[:relief] += 1
        else t.erase! if t.valid?; st[:relief_failed] += 1 end
      end
      # niches and openings come out of it
      recs.select { |r| %w[niche opening].include?(r['op']) }.each do |r|
        t = m.entities.add_group
        next unless box(t.entities, r['lo'], r['hi'])
        # SketchUp subtracts the RECEIVER from the argument: a.subtract(b) is
        # b - a. Written the natural way round, every cut kept the little
        # cutter box and discarded the wall -- which is why wall_04 came back
        # as a 0.5 x 0.11 x 0.75 m block instead of a 10.6 m wall.
        res = (t.subtract(g) rescue nil)
        if res then g = res; st[r['op'].to_sym] += 1
        else t.erase! if t.valid?; st[:"#{r['op']}_failed"] += 1 end
      end
      if g && g.valid?
        g.name = name
        wall_groups << g
        st[:walls] += 1
        st[:solid_walls] += 1 if g.manifold?
      else
        st[:wall_lost] += 1
      end
    end

    if weld && wall_groups.length > 1
      net = m.entities.add_group(wall_groups.select(&:valid?))
      net.name = 'walls'
      net.entities.grep(Sketchup::Group).each { |x| x.explode if x.valid? }
      st[:network_faces] = net.entities.grep(Sketchup::Face).length
    end

    m.commit_operation
    b = m.bounds
    st.merge(groups: m.entities.grep(Sketchup::Group).length,
             extent_m: [((b.max.x-b.min.x)/M).round(3),
                        ((b.max.y-b.min.y)/M).round(3),
                        ((b.max.z-b.min.z)/M).round(3)],
             seconds: (Time.now-t0).round(1)).to_json
  end

  def self.save_as(path)
    m = Sketchup.active_model
    ok = m.save(path)
    "saved=#{ok} bytes=#{File.exist?(path) ? File.size(path) : -1}"
  end
end
