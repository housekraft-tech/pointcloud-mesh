# Build it in the order a person builds it.
#
#   1. every wall run goes up, exactly where it was measured -- no merging, no
#      snapping, nothing moved
#   2. the runs are welded into ONE network solid, so a junction is a junction
#      and a tape measure crosses it
#   3. the pilasters and boxed conduits are added onto that network
#   4. the niches are cut into it
#   5. the doors, windows and arches are cut through it
#
# The previous attempt merged the geometry first to make the model tidy: 105
# runs became 24 walls, and the walls went from 3.3 mm off the scan to 64 mm.
# Tidiness is not worth that. Here nothing is merged -- the runs are welded,
# which changes no coordinate -- so the model stays as accurate as the
# measurement and still reads as one network.
require 'sketchup.rb'
require 'json'

module PCMNetwork
  M = 39.3700787401575
  BOND = 0.001            # m: solids that merely touch will not union

  def self.box(ents, lo, hi, grow_ax = nil)
    lo = lo.dup; hi = hi.dup
    if grow_ax
      lo[grow_ax] -= BOND; hi[grow_ax] += BOND
    end
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

  def self.solid_of(m, lo, hi, grow_ax = nil)
    g = m.entities.add_group
    if box(g.entities, lo, hi, grow_ax)
      g
    else
      g.erase! if g.valid?
      nil
    end
  end

  # A failed Solid Tools call can still delete its operands, so `net` ends up
  # pointing at a group that no longer exists -- and every operation after it
  # fails too. That is how 24 pilasters went in and then all 60 remaining cuts
  # failed against nothing. If the handle goes stale, take the biggest solid in
  # the model instead: that is the network.
  def self.recover(m, net)
    return net if net && net.valid?
    cands = m.entities.grep(Sketchup::Group).select { |g| g.valid? && g.manifold? }
    cands.max_by { |g| g.volume rescue 0 }
  end

  def self.build(json_path, _ = nil)
    t0 = Time.now
    sched = JSON.parse(File.read(json_path))['parts']
    m = Sketchup.active_model
    m.options['UnitsOptions']['LengthUnit'] = 2
    m.entities.clear!
    m.definitions.purge_unused
    m.start_operation('build the network', true)
    st = Hash.new(0)

    # 1. the walls go up
    solids = []
    sched.select { |r| r['op'] == 'run' }.each do |r|
      long = (r['hi'][0]-r['lo'][0]) >= (r['hi'][1]-r['lo'][1]) ? 0 : 1
      g = solid_of(m, r['lo'], r['hi'], long)     # grown along its length only
      g ? (solids << g; st[:runs] += 1) : st[:degenerate] += 1
    end

    # 2. welded into one network
    net = solids.shift
    solids.each do |g|
      res = (net.union(g) rescue nil)
      if res then net = res; st[:welds] += 1
      else st[:weld_failed] += 1 end
      net = recover(m, net)
    end
    net.name = 'walls' if net && net.valid?

    # 3. what stands proud of the wall is added to it
    sched.select { |r| r['op'] == 'relief' }.each do |r|
      g = solid_of(m, r['lo'], r['hi'])
      next unless g
      res = (net.union(g) rescue nil)
      if res then net = res; st[:relief] += 1
      else (g.erase! if g.valid?); st[:relief_failed] += 1 end
      net = recover(m, net)
    end

    # 4 and 5. what is cut out of it -- niches, then the openings
    %w[niche opening].each do |op|
      sched.select { |r| r['op'] == op }.each do |r|
        g = solid_of(m, r['lo'], r['hi'])
        next unless g
        res = (g.subtract(net) rescue nil)      # a.subtract(b) is b - a
        if res then net = res; st[op.to_sym] += 1
        else (g.erase! if g.valid?); st[:"#{op}_failed"] += 1 end
        net = recover(m, net)
        break unless net
      end
    end
    net.name = 'walls' if net && net.valid?

    # the slabs and columns stay as their own objects, as they would be drawn
    sched.select { |r| %w[slab column].include?(r['op']) }.each do |r|
      g = solid_of(m, r['lo'], r['hi'])
      next unless g
      g.name = r['name']
      st[r['op'].to_sym] += 1
    end

    m.commit_operation
    b = m.bounds
    st.merge(network_faces: (net && net.valid?) ? net.entities.grep(Sketchup::Face).length : 0,
             network_solid: (net && net.valid?) ? net.manifold? : false,
             network_m3: (net && net.valid? && net.manifold?) ?
                         (net.volume/(M*M*M)).round(2) : nil,
             groups: m.entities.grep(Sketchup::Group).length,
             extent_m: [((b.max.x-b.min.x)/M).round(3), ((b.max.y-b.min.y)/M).round(3),
                        ((b.max.z-b.min.z)/M).round(3)],
             seconds: (Time.now-t0).round(1)).to_json
  end

  def self.save_as(path)
    m = Sketchup.active_model
    ok = m.save(path)
    "saved=#{ok} bytes=#{File.exist?(path) ? File.size(path) : -1}"
  end
end
