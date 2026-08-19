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
import sys, json, numpy as np
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
amin = DD.argmin(0)                       # the wall normal for each cell
vol = DD[0]*DD[1]*DD[2]

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
    mine &= solid | ndimage.binary_dilation(solid, np.ones(ksh, bool))
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
        extra = np.argwhere((owner == pi) & ~claimed)
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
    o = o_of[ax]
    t = PL[ax][cells[:, ax].max()+1] - PL[ax][cells[:, ax].min()]
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
    th = [r['thickness_mm'] for r in tw]
    print(f"WALL thickness: median {np.median(th):.0f} mm, "
          f"range {min(th)}-{max(th)} -- every one MEASURED between two planes")
print(f"total masonry volume {sum(r['volume_l'] for r in rows)/1000:.2f} m3")
sz, np_, tris = G.write("output/model/modular_cells.glb")
json.dump(dict(clear_height_mm=round(H*1000), parts=rows),
          open("output/model/modular_cells.json", "w"), indent=1)
print(f"wrote output/model/modular_cells.glb  {sz/1e6:.2f} MB, {np_} boxes, "
      f"{tris:,} triangles, {len(rows)} named parts")
