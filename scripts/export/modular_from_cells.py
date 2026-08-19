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

# A wall is the cells one PLANE PAIR filled -- not a blob found by connectivity.
#
# cellcomplex.py records, for every cell it made solid, which two planes enclose
# it. That pair IS the wall: a slab between the face you see on one side and the
# face you see on the other, with a thickness that is exact rather than a median.
# Grouping by connectivity instead has to guess where one wall stops and the next
# starts, and it guessed badly -- 384 parts, walls shattered into confetti,
# because the chosen normal flips at junctions and around openings.
#
# Connectivity is still used, but only WITHIN a pair, to separate two different
# walls that happen to lie between the same two planes at opposite ends of the
# flat.
o_of = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
prov = Z['prov']
parts = []
claimed = np.zeros_like(solid)
ids = np.unique(prov[solid & (prov != 0)])
print(f"{len(ids):,} distinct plane pairs account for "
      f"{int((solid & (prov != 0)).sum()):,} of {int(solid.sum()):,} solid cells")
for pid in ids:
    ax = int(pid)//1000000
    i = (int(pid) % 1000000)//1000
    j = int(pid) % 1000
    m = solid & (prov == pid)
    if m.sum() < 4: continue
    flat2 = m.any(axis=ax)                       # its footprint in the wall plane
    lab, nb = ndimage.label(flat2, np.ones((3, 3), bool))
    for c in range(1, nb+1):
        keep2 = lab == c
        if keep2.sum() < 3: continue
        sel = np.zeros_like(m)
        idx = np.argwhere(keep2)
        for k in range(i, j):
            cc = np.zeros((len(idx), 3), int); cc[:, ax] = k
            o = o_of[ax]; cc[:, o[0]] = idx[:, 0]; cc[:, o[1]] = idx[:, 1]
            ok2 = m[cc[:, 0], cc[:, 1], cc[:, 2]]
            cc = cc[ok2]
            if len(cc): sel[cc[:, 0], cc[:, 1], cc[:, 2]] = True
        cells = np.argwhere(sel)
        if len(cells) < 4: continue
        parts.append((ax, cells, PL[ax][j]-PL[ax][i]))
        claimed |= sel
print(f"{len(parts):,} walls from plane pairs")

# cells labelled solid by enclosure have no pair; group those by connectivity
rest = solid & ~claimed
if rest.any():
    st0 = ndimage.generate_binary_structure(3, 1)
    lab0, nb0 = ndimage.label(rest, st0)
    for c in range(1, nb0+1):
        cs = np.argwhere(lab0 == c)
        if len(cs) < 4: continue
        ax0 = int(np.bincount(amin[cs[:, 0], cs[:, 1], cs[:, 2]], minlength=3).argmax())
        parts.append((ax0, cs, float(np.median(thick[cs[:, 0], cs[:, 1], cs[:, 2]]))))
        claimed[cs[:, 0], cs[:, 1], cs[:, 2]] = True
    print(f"  plus {nb0} groups from cells solid by enclosure alone")

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
for pi, (ax_, cs, _t) in enumerate(parts):
    owner[cs[:, 0], cs[:, 1], cs[:, 2]] = pi
vols = [part_volume(cs) for (ax_, cs, _t) in parts]
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
        parts[best] = (parts[best][0], np.vstack([parts[best][1], cs]),
                       parts[best][2])
        owner[cs[:, 0], cs[:, 1], cs[:, 2]] = best
        vols[best] += vols[pi]
        alive[pi] = False; merged += 1
    if not merged: break
parts = [pt for i, pt in enumerate(parts) if alive[i]]
print(f"merged slivers under {MIN_PART_L:.0f} L: {n0} parts -> {len(parts)}")

# Rejoin one wall that was claimed by several plane pairs.
#
# Thinnest-first assignment means a narrower pair can win over part of a wall --
# a band where skirting, a slight bulge or a beam shifts the face plane -- so
# the wall arrives as vertical fragments. WALL_048 was z 621-861 and WALL_049
# z 901-1180: the same wall, two pairs, two parts. They are collinear and they
# overlap in plan; nothing else in a building does that.
def part_geom(ax, cells):
    o = o_of[ax]
    hax = o[0] if o[0] != 2 else o[1]
    c = (PL[ax][cells[:, ax].min()] + PL[ax][cells[:, ax].max()+1])/2
    return c, PL[hax][cells[:, hax].min()], PL[hax][cells[:, hax].max()+1], hax

BRIDGE = 1.20            # a doorway-sized gap is still one wall
nb0 = len(parts)
while True:
    hit = None
    G_ = [part_geom(ax, cs)+(t,) for (ax, cs, t) in parts]
    for x in range(len(parts)):
        for y in range(x+1, len(parts)):
            if parts[x][0] != parts[y][0]: continue
            cx, ax0, bx, hx, tx = G_[x]
            cy, ay0, by, hy, ty = G_[y]
            if hx != hy: continue
            if abs(cx-cy) > max(0.04, 0.30*min(tx, ty)): continue
            gap = max(ax0, ay0) - min(bx, by)
            if gap > BRIDGE: continue
            hit = (x, y); break
        if hit: break
    if not hit: break
    x, y = hit
    ax_, csx, tx = parts[x]; _, csy, ty = parts[y]
    keep_t = tx if len(csx) >= len(csy) else ty
    parts[x] = (ax_, np.vstack([csx, csy]), keep_t)
    parts.pop(y)
print(f"rejoined walls split across plane pairs: {nb0} -> {len(parts)}")

seq = {}; rows = []
for (ax, cells, tpair) in parts:
    kind, t, span, other = classify(ax, cells)
    if tpair: t = tpair          # exact: the gap between the two planes
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
    o = o_of[ax]
    ctr = (PL[ax][cells[:, ax].min()] + PL[ax][cells[:, ax].max()+1])/2
    # the horizontal axis of a vertical wall is whichever of the two is not z
    hax = o[0] if o[0] != 2 else o[1]
    a0 = float(PL[hax][cells[:, hax].min()]); a1 = float(PL[hax][cells[:, hax].max()+1])
    z0 = float(PL[2][cells[:, 2].min()]); z1 = float(PL[2][cells[:, 2].max()+1])
    rows.append(dict(name=nm, kind=kind, axis=int(ax),
                     centre_mm=round(float(ctr)*1000),
                     a_mm=round(a0*1000), b_mm=round(a1*1000),
                     z0_mm=round(z0*1000), z1_mm=round(z1*1000),
                     thickness_mm=round(t*1000), span_mm=round(span*1000),
                     other_mm=round(other*1000), volume_l=round(v*1000, 1),
                     boxes=len(G.parts)-first))

# ---------------------------------------------------- openings in the walls
# An opening is a void in a wall with masonry OVER it. That is the same test
# that scored 9 doors out of 9 on the fitted model, but it can now be asked of
# the wall itself rather than of a raster: the wall's own cells say exactly
# where it is solid, so the void is whatever is left inside its extent.
#
# Sill and head come straight off the plane positions bounding the void, so they
# are measured, not fitted.
def openings_in(ax, cells, tname):
    o = o_of[ax]
    hax = o[0] if o[0] != 2 else o[1]
    h0, h1 = cells[:, hax].min(), cells[:, hax].max()+1
    z0, z1 = cells[:, 2].min(), cells[:, 2].max()+1
    if h1-h0 < 2 or z1-z0 < 3: return []
    m = np.zeros((h1-h0, z1-z0), bool)
    m[cells[:, hax]-h0, cells[:, 2]-z0] = True
    top = PL[2][z1]
    out = []
    voids = []
    for c in range(h1-h0):
        col = m[c]
        if not col.any(): continue
        hi = np.flatnonzero(col).max()
        k = 0
        while k < hi:
            if col[k]: k += 1; continue
            e = k
            while e+1 <= hi and not col[e+1]: e += 1
            # masonry above the void is what makes it an opening rather than
            # simply the end of the wall
            if e < hi and col[e+1:hi+1].any():
                voids.append((c, PL[2][z0+k], PL[2][z0+e+1]))
            k = e+1
    if not voids: return []
    voids.sort()
    run = [voids[0]]
    for v in voids[1:]:
        if v[0] <= run[-1][0]+1 and abs(v[1]-run[-1][1]) < 0.35 \
           and abs(v[2]-run[-1][2]) < 0.35:
            run.append(v)
        else:
            out.append(run); run = [v]
    out.append(run)
    res = []
    for r in out:
        c0 = r[0][0]; c1 = r[-1][0]
        wmm = (PL[hax][h0+c1+1]-PL[hax][h0+c0])
        if wmm < 0.55 or wmm > 4.50: continue
        sill = float(np.median([v[1] for v in r]))
        head = float(np.median([v[2] for v in r]))
        if head-sill < 0.75: continue
        # A head at 861 mm is not a doorway, it is a gap under a shelf or a
        # ragged patch in the labelling. Doors and balconies in this building
        # head out between 2.0 and 2.4 m; allow 1.70 up so a low one still
        # counts, but no lower.
        if head < 1.70: continue
        kind = ("window" if sill >= 0.60 else
                "door" if wmm <= 1.15 else "opening")
        res.append(dict(kind=kind, wall=tname,
                        width_mm=round(wmm*1000), sill_mm=round(sill*1000),
                        head_mm=round(head*1000),
                        a_mm=round(float(PL[hax][h0+c0])*1000),
                        b_mm=round(float(PL[hax][h0+c1+1])*1000),
                        arch_mm=round((top-head)*1000)))
    return res

# the floor plane, so sills read from the floor rather than from the origin
FLOOR = float(PL[2][1]) if len(PL[2]) > 2 else 0.0
OPEN = []
for r, (ax, cells, tpair) in zip(rows, parts):
    if r['kind'] not in ("WALL",): continue
    for op in openings_in(ax, cells, r['name']):
        op['sill_mm'] = round(op['sill_mm'] - FLOOR*1000)
        # a sill within a skirting's height of the floor IS the floor; the
        # residual is the z-plane spacing, not a step to walk over
        if op['sill_mm'] < 150: op['sill_mm'] = 0
        op['head_mm'] = round(op['head_mm'] - FLOOR*1000)
        op['axis'] = int(ax)
        op['centre_mm'] = r['centre_mm']
        op['thickness_mm'] = r['thickness_mm']
        OPEN.append(op)
from collections import Counter
print(f"openings found in the walls: {dict(Counter(o['kind'] for o in OPEN))}"
      f"   (ground truth 9 doors, 4 balconies, 2 openings)")
if OPEN:
    wd = [o['width_mm'] for o in OPEN]
    print(f"  widths {min(wd)}-{max(wd)} mm, "
          f"heads {min(o['head_mm'] for o in OPEN)}-{max(o['head_mm'] for o in OPEN)} mm")

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
# Draw each opening: a leaf in the void and the arch over it. These are not
# masonry -- they are what the masonry leaves behind -- so they are separate
# named parts and their volume is not counted as material.
LEAF = 0.040
oseq = {}
for op in OPEN:
    oseq[op['kind']] = oseq.get(op['kind'], 0)+1
    nm = f"{op['kind'].upper()}_{oseq[op['kind']]:02d}"
    ax = op['axis']; c = op['centre_mm']/1000
    a, b = op['a_mm']/1000, op['b_mm']/1000
    s0 = op['sill_mm']/1000 + FLOOR; h0 = op['head_mm']/1000 + FLOOR
    first = len(G.parts)
    q0 = {ax: c-LEAF/2}; q1 = {ax: c+LEAF/2}
    o = o_of[ax]; hax = o[0] if o[0] != 2 else o[1]
    q0[hax] = a; q1[hax] = b; q0[2] = s0; q1[2] = h0
    G.add_box(f"{nm}_leaf_w{op['width_mm']}_h{op['head_mm']-op['sill_mm']}",
              *yup(q0[0], q1[0], q0[1], q1[1], q0[2], q1[2]))
    G.parent(f"{nm}_w{op['width_mm']}_sill{op['sill_mm']}_head{op['head_mm']}", first)
sz, np_, tris = G.write("output/model/modular_cells.glb")
json.dump(dict(clear_height_mm=round(H*1000), parts=rows, openings=OPEN),
          open("output/model/modular_cells.json", "w"), indent=1)
print(f"wrote output/model/modular_cells.glb  {sz/1e6:.2f} MB, {np_} boxes, "
      f"{tris:,} triangles, {len(rows)} named parts")
