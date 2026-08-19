"""The modular model, built FROM the cell complex instead of fitted to points.

The old modular model fitted each wall independently, which is why 30% of the
surface it drew had no scanned point within 100 mm, why duplicate walls had to
be hunted with heuristics, and why half its thicknesses were the median rather
than a measurement.

Here the geometry is not fitted at all -- it is taken. Every part is a union of
cells from the complex, so:

  no overlaps   cells are disjoint, so two parts cannot occupy the same space
  no gaps       cells tile the masonry, so the parts partition it exactly
  no guesses    a wall's thickness is the distance between the two planes that
                bound its cells, always measured

Each part is itself a closed solid, which is what a quantity takeoff needs: the
volume of a part is the sum of its cell volumes, exactly, with nothing double
counted at a junction.

Grouping: a solid cell is thinnest along one axis, and that axis is the normal
of the masonry it belongs to. Cells sharing a normal, sharing the same pair of
bounding planes, and touching each other are one wall.
"""
import sys, os, json, numpy as np
from scipy import ndimage
sys.path.insert(0, "scripts/export")
from glb import GLB

Z = np.load("output/cells.npz")
solid = Z['solid']; PL = [Z['px'], Z['py'], Z['pz']]
nx, ny, nz = solid.shape
H = json.load(open("output/fp_walls.json"))['clear_height']
print(f"{solid.sum():,} solid cells in a {nx} x {ny} x {nz} complex")

D = [np.diff(PL[ax]) for ax in (0, 1, 2)]
DD = np.stack(np.meshgrid(D[0], D[1], D[2], indexing='ij'))
vol = DD[0]*DD[1]*DD[2]

# The wall normal is the direction in which the MASONRY is thinnest, not the
# direction in which the individual cell is thinnest. A 200 mm wall spans
# several plane intervals, and a cell inside it can easily be thinner in some
# other direction wherever planes happen to be dense -- so per-cell argmin
# assigned different normals to cells of the same wall and split it into 60 mm
# laminations. Measure instead how far the solid run extends through each cell
# along each axis, and take the shortest.
def run_thickness(ax):
    m = np.moveaxis(solid, ax, 0)
    d = np.moveaxis(np.broadcast_to(DD[ax], solid.shape), ax, 0)
    L = m.shape[0]
    up = np.zeros(m.shape, np.float32)
    acc = np.zeros(m.shape[1:], np.float32)
    for i in range(L):
        acc = np.where(m[i], acc + d[i], 0.0); up[i] = acc
    dn = np.zeros(m.shape, np.float32)
    acc = np.zeros(m.shape[1:], np.float32)
    for i in range(L-1, -1, -1):
        acc = np.where(m[i], acc + d[i], 0.0); dn[i] = acc
    tot = up + dn - np.moveaxis(np.broadcast_to(DD[ax], solid.shape), ax, 0)
    return np.moveaxis(tot, 0, ax)

RT = np.stack([run_thickness(ax) for ax in (0, 1, 2)])
RT[:, ~solid] = 1e9
amin = RT.argmin(0)                       # the wall normal for each cell
thick = RT.min(0)
print(f"masonry run thickness: median {np.median(thick[solid])*1000:.0f} mm, "
      f"90th {np.percentile(thick[solid], 90)*1000:.0f} mm")

# Connect in 3D, not per plane-interval. Planes are dense enough that a 200 mm
# wall spans several intervals along its own normal, so grouping one interval at
# a time sliced every wall into ~60 mm laminations: 7,549 parts with a median
# "thickness" of 60 mm. A wall is all of its cells, through the full thickness.
parts = []
claimed = np.zeros_like(solid)
for ax in (0, 1, 2):
    mine = solid & (amin == ax)
    if not mine.any(): continue
    # 6-connectivity, not 26: face-touching only. Diagonal connectivity welds
    # walls together where they merely graze at a junction.
    st = ndimage.generate_binary_structure(3, 1)
    # Consolidate first. The solid labelling is patchy along a wall -- a cell
    # here and there fails the vote -- and face-connectivity then shatters one
    # wall into dozens of parts. Closing within the wall's own plane (never
    # across its normal, which would fuse parallel walls) knits them back.
    k = np.ones((3, 3, 3), bool); k[tuple([slice(None)]*3)] = True
    ksh = [3, 3, 3]; ksh[ax] = 1
    mine = ndimage.binary_closing(mine, np.ones(ksh, bool))
    mine &= solid          # closing must never admit a cell that is not solid
    lab, nb = ndimage.label(mine, st)
    for c in range(1, nb+1):
        cells = np.argwhere(lab == c)
        if len(cells) < 4: continue
        # Split by contiguity ALONG THE NORMAL. A component that reaches around
        # a corner spans two separate positions in its own normal direction, and
        # measuring across the gap called one wall 1706 mm thick.
        ks = np.sort(np.unique(cells[:, ax]))
        brk = np.flatnonzero(np.diff(ks) > 1)
        runs = np.split(ks, brk+1)
        for run in runs:
            sub = cells[np.isin(cells[:, ax], run)]
            if len(sub) < 4: continue
            if PL[ax][run[-1]+1]-PL[ax][run[0]] > 0.60: continue   # not masonry
            parts.append((ax, sub))
            claimed[sub[:, 0], sub[:, 1], sub[:, 2]] = True
# Nothing may be dropped. The filters above discard small and over-thick
# fragments, and those cells are still masonry -- the scan sits on them. Left
# out, the modular model missed 22.5% of the scan where the cell complex it is
# built from missed 7.2%. Every unclaimed solid cell is gathered up here, so the
# parts partition the masonry exactly rather than approximately.
# Absorbed into whichever part they touch, not emitted as parts of their own.
# Left as separate parts they took the count from 634 to 1,640 -- a fragment
# beside a wall belongs to that wall, it is not a component of the building.
left = solid & ~claimed
if left.any():
    owner = np.zeros(solid.shape, np.int32)
    for pi, (ax_, cs) in enumerate(parts, 1):
        owner[cs[:, 0], cs[:, 1], cs[:, 2]] = pi
    st2 = ndimage.generate_binary_structure(3, 1)
    n0 = int(left.sum())
    for _ in range(60):
        if not left.any(): break
        grown = ndimage.grey_dilation(owner, footprint=st2)
        take = left & (grown > 0)
        if not take.any(): break
        owner[take] = grown[take]; left &= ~take
    add = {}
    for pi in range(1, len(parts)+1):
        # solid only: the dilation that grows ownership runs into empty cells
        # too, and absorbing one carried the 1e9 no-solid sentinel into the
        # thickness, reporting a wall 5e11 mm thick.
        extra = np.argwhere((owner == pi) & ~claimed & solid)
        if len(extra): add[pi-1] = extra
    for k, extra in add.items():
        ax_, cs = parts[k]
        parts[k] = (ax_, np.vstack([cs, extra]))
    print(f"  {n0:,} cells unclaimed by the filters, absorbed into the part "
          f"each one touches ({len(add)} parts grew); {int(left.sum()):,} orphaned")
    if left.any():
        lab2, nb2 = ndimage.label(left, st2)
        for c in range(1, nb2+1):
            cs = np.argwhere(lab2 == c)
            ax2 = int(np.bincount(amin[cs[:, 0], cs[:, 1], cs[:, 2]],
                                  minlength=3).argmax())
            parts.append((ax2, cs))
print(f"{len(parts):,} connected masonry parts")

o_of = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
def classify(ax, cells):
    """Thickness is the typical depth of the masonry, not the extent of the
    part. Once slivers are merged a part is no longer a single clean slab, and
    measuring corner to corner along its normal reported an 11.4 m wall. Take
    the median run of cells through the thickness instead, which is what a
    tape measure would read."""
    o = o_of[ax]
    # The same physical measure the normal was chosen by: how far the solid run
    # extends through each cell. Taking the median over the part's own columns
    # instead reported 59 mm against a true 275, because a part picks up thin
    # single-cell columns along its edges and they dominate the median.
    t = float(np.median(thick[cells[:, 0], cells[:, 1], cells[:, 2]]))
    e0 = PL[o[0]][cells[:, o[0]].min()], PL[o[0]][cells[:, o[0]].max()+1]
    e1 = PL[o[1]][cells[:, o[1]].min()], PL[o[1]][cells[:, o[1]].max()+1]
    d0, d1 = e0[1]-e0[0], e1[1]-e1[0]
    if ax == 2:
        return "SLAB", t, max(d0, d1), min(d0, d1)
    # for a vertical wall one of the two spans is height, the other is length
    horiz, vert = (d0, d1) if o[1] == 2 else (d1, d0)
    if horiz <= 0.55 and vert >= 1.5: return "COLUMN", t, horiz, vert
    if vert <= 0.60 and horiz >= 0.60: return "BEAM", t, horiz, vert
    return "WALL", t, horiz, vert

G = GLB()
def yup(x0, x1, y0, y1, z0, z1): return (x0, z0, -y1), (x1, z1, -y0)

def greedy(mask):
    m = mask.copy(); out = []; R, C = m.shape
    for a in range(R):
        if not m[a].any(): continue
        b = 0
        while b < C:
            if not m[a, b]: b += 1; continue
            w = 1
            while b+w < C and m[a, b+w]: w += 1
            h = 1
            while a+h < R and m[a+h, b:b+w].all(): h += 1
            m[a:a+h, b:b+w] = False; out.append((a, b, h, w)); b += w
    return out

# Merge slivers into the part they touch.
#
# The grouping was producing 911 parts, but half the masonry sits in the largest
# 22 of them and 440 parts are under 5 litres holding 0.66 m3 between them --
# 1.8% of the volume in 48% of the parts. Those are not components of the
# building, they are the ragged edge of the labelling. Each is absorbed into its
# largest neighbour, so no cell is lost and the count reflects structure.
MIN_PART_L = float(os.environ.get("MIN_PART_L", 150))

def part_volume(cells):
    return float(vol[cells[:, 0], cells[:, 1], cells[:, 2]].sum())*1000

owner = np.full(solid.shape, -1, np.int32)
for pi, (ax_, cs) in enumerate(parts):
    owner[cs[:, 0], cs[:, 1], cs[:, 2]] = pi
vols = [part_volume(cs) for (ax_, cs) in parts]
alive = [True]*len(parts)
st1 = ndimage.generate_binary_structure(3, 1)
n0 = len(parts)
for _ in range(40):
    order = sorted((v, i) for i, v in enumerate(vols)
                   if alive[i] and v < MIN_PART_L)
    if not order: break
    merged = 0
    for v, pi in order:
        if not alive[pi] or vols[pi] >= MIN_PART_L: continue
        cs = parts[pi][1]
        m = np.zeros(solid.shape, bool)
        m[cs[:, 0], cs[:, 1], cs[:, 2]] = True
        ring = ndimage.binary_dilation(m, st1) & ~m
        nb = owner[ring]
        nb = nb[(nb >= 0) & (nb != pi)]
        if not len(nb): continue
        cand = [(vols[q], q) for q in set(nb.tolist()) if alive[q]]
        if not cand: continue
        _, best = max(cand)
        parts[best] = (parts[best][0], np.vstack([parts[best][1], cs]))
        owner[cs[:, 0], cs[:, 1], cs[:, 2]] = best
        vols[best] += vols[pi]
        alive[pi] = False; merged += 1
    if not merged: break
parts = [pt for i, pt in enumerate(parts) if alive[i]]
print(f"merged slivers under {MIN_PART_L:.0f} L: {n0} parts -> {len(parts)}")

seq = {}; rows = []
for (ax, cells) in parts:
    kind, t, span, other = classify(ax, cells)
    seq[kind] = seq.get(kind, 0)+1
    nm = f"{kind}_{seq[kind]:03d}"
    o = o_of[ax]
    first = len(G.parts); v = 0.0
    for i in sorted(set(cells[:, ax].tolist())):     # one slab of the wall
        sub = cells[cells[:, ax] == i]
        mask = np.zeros((len(PL[o[0]])-1, len(PL[o[1]])-1), bool)
        mask[sub[:, o[0]], sub[:, o[1]]] = True
        for (a, b, h, w) in greedy(mask):
            q0 = {ax: PL[ax][i], o[0]: PL[o[0]][a], o[1]: PL[o[1]][b]}
            q1 = {ax: PL[ax][i+1], o[0]: PL[o[0]][a+h], o[1]: PL[o[1]][b+w]}
            G.add_box(f"{nm}_{len(G.parts)-first:03d}",
                      *yup(q0[0], q1[0], q0[1], q1[1], q0[2], q1[2]))
            v += (q1[0]-q0[0])*(q1[1]-q0[1])*(q1[2]-q0[2])
    G.parent(f"{nm}_t{t*1000:.0f}_L{span*1000:.0f}_v{v*1000:.0f}L", first)
    rows.append(dict(name=nm, kind=kind, axis=int(ax),
                     thickness_mm=round(t*1000), span_mm=round(span*1000),
                     other_mm=round(other*1000), volume_l=round(v*1000, 1),
                     boxes=len(G.parts)-first))

from collections import Counter
print("parts by kind:", dict(Counter(r['kind'] for r in rows)))
tw = [r for r in rows if r['kind'] == "WALL"]
if tw:
    th = np.array([r['thickness_mm'] for r in tw], float)
    vv = np.array([r['volume_l'] for r in tw], float)
    o = np.argsort(th); c = np.cumsum(vv[o])
    med = th[o][np.searchsorted(c, c[-1]/2)]
    print(f"WALL thickness: {med:.0f} mm by volume, {np.median(th):.0f} mm by count, "
          f"range {th.min():.0f}-{th.max():.0f}")
    print("  (by volume is the honest one -- there are many small parts and a "
          "plain median lets them outvote the walls that hold the masonry)")
print(f"total masonry volume {sum(r['volume_l'] for r in rows)/1000:.2f} m3")
sz, np_, tris = G.write("output/model/modular_cells.glb")
json.dump(dict(clear_height_mm=round(H*1000), parts=rows),
          open("output/model/modular_cells.json", "w"), indent=1)
print(f"wrote output/model/modular_cells.glb  {sz/1e6:.2f} MB, {np_} boxes, "
      f"{tris:,} triangles, {len(rows)} named parts")
