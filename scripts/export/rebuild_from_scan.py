"""Rebuild the modular shell from the LiDAR alone.

Nothing here reads the drawing or the bbox data. The scan is the only source
of geometry, because the registered wall lines and the scan's own openings
disagreed badly enough that 10 of 11 openings had no host wall within 300 mm.

How a wall is found -- by measuring its CROSS-SECTION where it actually is,
rather than by projecting the whole flat onto one axis:

  at every 50 mm station along the plan, look across the perpendicular
  direction for a SHORT run of occupied cells. A short run is a wall seen
  edge-on: its length is that wall's thickness right there, and its midpoint is
  that wall's centreline right there. A long run is a wall seen face-on, which
  belongs to the other axis, so it is skipped.

  stations whose centrelines agree are the same wall. Its thickness is then a
  measurement per station, not one number for the whole flat.

This replaces a projection-and-peak-picking pass that merged parallel walls
into single 500 mm slabs and invented duplicate walls beside real ones.

Openings, arches, niches and columns all follow from those walls, so every part
is bound to a wall and none can float.
"""
import sys, json, numpy as np, laspy
from scipy import ndimage
SC = "/private/tmp/claude-501/-Users-vallerikoushik-Documents-pointcloud-latest-pointcloud-mesh/437552a3-658f-4215-b1c0-d1640f83e863/scratchpad"
sys.path.insert(0, SC)
from glb import GLB

CELL = 0.05
MIN_PTS = 6            # a cell is real material, not a stray return
MAX_T_CELLS = 8         # 400 mm: longer than this is a wall seen face-on
BRIDGE = 0.30           # occlusion gaps this short are bridged unconditionally
OPEN_MIN, OPEN_MAX = 0.55, 4.50
MIN_LEN = 0.45          # a wall shorter than this is not a wall

H = json.load(open("output/fp_walls.json"))['clear_height']
mins = np.array(json.load(open("output/fp_walls.json"))['mins'])
# The aligned scan with the connectivity declutter, not the v6 structural
# cleanup -- v6 deletes parapets and any half-height wall outright, because it
# keeps low points only beside a floor-to-ceiling column of points.
with laspy.open("output/mujammel_aligned_z0.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
import os
if os.path.exists("output/keep_mask.npy"):
    _k = np.load("output/keep_mask.npy")
    if len(_k) == len(P):
        P = P[_k]
        print(f"declutter: {(~_k).sum():,} free-standing points dropped")
z = P[:, 2]
XY = P[:, :2] - mins
nx = int(np.ceil((XY[:, 0].max() + CELL) / CELL))
ny = int(np.ceil((XY[:, 1].max() + CELL) / CELL))
ij = np.floor(XY / CELL).astype(np.int64)
ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
flat = ij[:, 0] * ny + ij[:, 1]
def occ(m): return (np.bincount(flat[m & ok], minlength=nx*ny).reshape(nx, ny)) >= MIN_PTS
body = occ((z > 1.10) & (z < 1.85))
above = occ((z > 2.10) & (z < H - 0.10))
wallr = ndimage.binary_closing(body, np.ones((3, 3), bool))
print(f"grid {nx}x{ny} @ {CELL*1000:.0f} mm   clear height {H*1000:.0f} mm")

# ---------------------------------------------------------- cross-sections
def sections(axis):
    """(centre cell, thickness cells, station) for every wall cross-section."""
    n_perp = nx if axis == 0 else ny
    n_along = ny if axis == 0 else nx
    out = []
    for a in range(n_along):
        col = wallr[:, a] if axis == 0 else wallr[a, :]
        i = 0
        while i < n_perp:
            if not col[i]: i += 1; continue
            j = i
            while j+1 < n_perp and col[j+1]: j += 1
            L = j-i+1
            if L <= MAX_T_CELLS: out.append(((i+j+1)/2.0, L, a))
            i = j+1
    return out

def zmeasure(axis, c, a, b):
    """Sill and head of a void, from the raw points standing inside it.
    head = underside of the lintel; sill = top of the wall below (0 for a door)."""
    m = ok & (np.abs(XY[:, axis]-c) < 0.09) & (XY[:, 1-axis] > a+0.05) & (XY[:, 1-axis] < b-0.05)
    if m.sum() < 40: return None, None
    zz = z[m]; hz = zz[zz > 1.95]
    if len(hz) < 25: return None, None
    head = float(np.percentile(hz, 3))
    lz = zz[zz < 1.95]
    sill = float(np.percentile(lz, 97)) if len(lz) >= 50 else 0.0
    if sill < 0.30: sill = 0.0
    if head - sill < 0.75: return None, None
    return sill, head

walls = []
for axis in (0, 1):
    secs = sections(axis)
    n_along = ny if axis == 0 else nx
    # Group cross-sections into wall lines. Bucketing by rounded centreline and
    # merging each bucket into the one below chains ALL consecutive buckets
    # together, so unrelated parallel walls collapsed into one and a 300 mm wall
    # running 4.7 m went missing entirely. Cluster by centreline proximity
    # instead: a wall wanders by at most a cell over its length, and two real
    # walls are never one cell apart.
    n_perp = nx if axis == 0 else ny
    S = np.zeros((n_perp+2, n_along), bool)
    for (cc, L, a) in secs: S[int(round(cc)), a] = True
    comp, ncomp = ndimage.label(S, np.ones((3, 3), bool))
    lines = []
    for ci in range(1, ncomp+1):
        cells = np.argwhere(comp == ci)
        lines.append((float(np.median(cells[:, 0])), set(cells[:, 1].tolist())))
    lines.sort()
    groups = []
    for (cen, st) in lines:
        if groups and abs(cen-groups[-1][0]) <= 1.0:
            groups[-1] = (groups[-1][0], groups[-1][1] | st)
        else:
            groups.append((cen, st))
    for (gcen, gst) in groups:
        recs = [(cc, L, a) for (cc, L, a) in secs if a in gst and abs(cc-gcen) <= 1.5]
        if len(recs) < MIN_LEN/CELL: continue
        occ_a = np.zeros(n_along, bool); th = np.zeros(n_along); ctr = np.zeros(n_along)
        for (cc, L, a) in recs:
            if not occ_a[a] or L > th[a]/CELL: th[a] = L*CELL; ctr[a] = cc*CELL
            occ_a[a] = True
        c = float(np.median(ctr[occ_a])); t = float(np.median(th[occ_a]))
        ci = int(round(c/CELL))
        lint = above[max(ci-2, 0):ci+3, :].any(axis=0) if axis == 0 else \
               above[:, max(ci-2, 0):ci+3].any(axis=1)
        # walk along: solid runs are wall, gaps are openings or the wall's end
        segs = []; cur = None; ops = []; kk = 0
        while kk < n_along:
            if occ_a[kk]:
                if cur is None: cur = kk
                kk += 1; continue
            j = kk
            while j+1 < n_along and not occ_a[j+1]: j += 1
            glen = (j-kk+1)*CELL
            bridged = False
            if cur is not None and j+1 < n_along:
                if glen <= BRIDGE:
                    bridged = True                                  # occlusion
                elif OPEN_MIN <= glen <= OPEN_MAX and lint[kk:j+1].mean() >= 0.35:
                    s, h = zmeasure(axis, c, kk*CELL, (j+1)*CELL)
                    if s is not None:
                        ops.append(dict(a=kk*CELL, b=(j+1)*CELL, sill=s, head=h,
                                        arch=H-h, width=round(glen*1000)))
                        bridged = True                              # an opening IS wall
            if not bridged and cur is not None:
                segs.append((cur, kk-1)); cur = None
            kk = j+1
        if cur is not None: segs.append((cur, n_along-1))
        for (s0, s1) in segs:
            L = (s1-s0+1)*CELL
            if L < MIN_LEN: continue
            a, b = s0*CELL, (s1+1)*CELL
            sel = occ_a.copy(); sel[:s0] = False; sel[s1+1:] = False
            walls.append(dict(axis=axis, c=c, t=float(np.median(th[sel])), lo=a, hi=b,
                              length=L, ops=[o for o in ops
                                             if o['a'] >= a-1e-6 and o['b'] <= b+1e-6]))

# a wall can be picked up twice where two centreline buckets straddle it
walls.sort(key=lambda w: -w['length'])
keep = []
for w in walls:
    if any(v['axis'] == w['axis'] and abs(v['c']-w['c']) < 0.12 and
           min(v['hi'], w['hi'])-max(v['lo'], w['lo']) > 0.5*w['length'] for v in keep):
        continue
    keep.append(w)
walls = keep
# ---------------------------------------------------- rejoin collinear walls
# A wall interrupted by a door arrives here as two fragments, because the
# cross-section walk sees no material across the opening. Left alone that both
# inflates the wall count -- structure the building does not have -- and loses
# the opening, since a gap only becomes an opening when it falls INSIDE a wall.
# Rejoin collinear neighbours: a short gap is occlusion, and a door-sized gap
# with an arch over it is an opening. "Connected like honey unless there is a
# doorway."
def lintel_frac(axis, c, a, b):
    ci = int(round(c/CELL))
    k0, k1 = int(a/CELL), int(np.ceil(b/CELL))
    if k1 <= k0: return 0.0
    sl = above[max(ci-2, 0):ci+3, k0:k1] if axis == 0 else above[k0:k1, max(ci-2, 0):ci+3]
    return float(sl.any(axis=axis).mean()) if sl.size else 0.0

def solid_frac(axis, c, a, b):
    """How much of this gap is still WALL at body height.

    A gap in a wall line is not automatically an opening. The cross-section walk
    discards any section thicker than MAX_T_CELLS, so wherever a column makes the
    wall locally fatter the walk drops those stations and the wall arrives here
    split in two -- with a "gap" that is in fact solid masonry. Measured: gaps of
    650-2300 mm between perfectly collinear fragments holding 47,000+ points
    spread floor to ceiling. Those must be rejoined with NO opening; treating
    them as doorways is what was inventing walls around the columns."""
    ci = int(round(c/CELL))
    k0, k1 = int(np.ceil(a/CELL)), int(b/CELL)
    if k1 <= k0: return 1.0
    sl = wallr[max(ci-3, 0):ci+4, k0:k1] if axis == 0 else wallr[k0:k1, max(ci-3, 0):ci+4]
    return float(sl.any(axis=axis).mean()) if sl.size else 0.0

def rejoin(walls, dcmax=0.15, ovmax=0.08):
    joined = 0
    while True:
        walls.sort(key=lambda w: (w['axis'], round(w['c'], 3), w['lo']))
        hit = None
        for i in range(len(walls)-1):
            a, b = walls[i], walls[i+1]
            # Two tolerances, because the two merges are different claims.
            # Across a COLUMN the wall genuinely gets thicker, which moves the
            # centreline (it is the midpoint of the cross-section), so 150 mm
            # of drift is expected and merging is still correct. Across a
            # DOORWAY nothing about the wall changes, so a centreline that has
            # moved means these are two different walls -- merging them anyway
            # forces one plane through both and was pushing the worst face
            # error from 35 mm to 115 mm.
            if a['axis'] != b['axis']: continue
            dc = abs(a['c']-b['c'])
            if dc > 0.12: continue
            gap = b['lo']-a['hi']
            if gap < -0.02:
                # Overlapping, not merely adjacent: the same stretch of wall
                # found twice. Three walls sat on one line here -- span
                # -0.19..5.01 with 1.53..3.33 and -0.19..0.83 inside it -- and
                # skipping overlaps left all three. Absorb the shorter into the
                # longer; the union is the wall.
                if dc <= ovmax:
                    hit = (i, None); break
                continue
            if gap <= BRIDGE:
                hit = (i, None); break
            # solid all the way across -- the walk lost it at a column, not a door
            if gap <= 3.00 and solid_frac(a['axis'], a['c'], a['hi'], b['lo']) >= 0.60:
                hit = (i, None); break
            if dc <= 0.08 and OPEN_MIN <= gap <= OPEN_MAX and \
               lintel_frac(a['axis'], a['c'], a['hi'], b['lo']) >= 0.35:
                sh = zmeasure(a['axis'], a['c'], a['hi'], b['lo'])
                if sh[0] is not None:
                    hit = (i, dict(a=a['hi'], b=b['lo'], sill=sh[0], head=sh[1],
                                   arch=H-sh[1], width=round(gap*1000)))
                    break
        if hit is None: break
        i, op = hit
        a, b = walls[i], walls[i+1]
        a['ops'] = a['ops'] + ([op] if op else []) + b['ops']
        # the union: with an absorbed wall b sits INSIDE a, and assigning
        # b's end outright was truncating a to the shorter of the two
        a['lo'] = min(a['lo'], b['lo']); a['hi'] = max(a['hi'], b['hi'])
        a['length'] = a['hi']-a['lo']
        a['t'] = max(a['t'], b['t'])
        walls.pop(i+1); joined += 1
    return joined

nj = rejoin(walls)
print(f"rejoined {nj} collinear fragments into their parent walls")
nop = sum(len(w['ops']) for w in walls)
ts = [w['t']*1000 for w in walls]
print(f"{len(walls)} walls  ({sum(1 for w in walls if w['axis']==0)} X-const, "
      f"{sum(1 for w in walls if w['axis']==1)} Y-const)")
print(f"thickness measured per wall: median {np.median(ts):.0f} mm, "
      f"range {min(ts):.0f}-{max(ts):.0f}")
print(f"{nop} openings, each one a gap in a wall with an arch over it")

# ------------------------------------------- snap faces onto the scanned surfaces
# The raster locates a wall to the nearest 50 mm cell, which is far coarser than
# the scan. Each wall's faces are re-measured from the raw points inside that
# wall's own extent, so the model sits ON the scanned surface rather than near it.
def refine(w):
    ax = w['axis']
    m = ok & (np.abs(XY[:, ax]-w['c']) < 0.28) & (XY[:, 1-ax] > w['lo']+0.05) & \
        (XY[:, 1-ax] < w['hi']-0.05) & (z > 0.40) & (z < H-0.20)
    if m.sum() < 400: return False
    off = XY[m, ax] - w['c']
    nb = 113                                        # 5 mm bins across +/-282 mm
    h = np.bincount(np.clip(((off+0.282)/0.005).astype(int), 0, nb-1),
                    minlength=nb).astype(float)
    h = np.convolve(h, np.ones(3)/3, mode='same')
    pk = [i for i in range(1, nb-1) if h[i] >= h[i-1] and h[i] > h[i+1]
          and h[i] > h.max()*0.25]
    pk.sort(key=lambda i: -h[i]); sel = []
    for i in pk:
        if all(abs(i-j) >= 24 for j in sel): sel.append(i)   # faces >= 120 mm apart
        if len(sel) == 2: break
    if not sel: return False
    if len(sel) == 2:
        a, b = sorted((-0.282+sel[0]*0.005, -0.282+sel[1]*0.005))
        w['c'] = w['c'] + (a+b)/2; w['t'] = b-a; w['measured'] = True
    else:
        f = -0.282 + sel[0]*0.005                   # only one face was ever scanned
        # The wall BODY is solid, so it returns nothing; the room side returns
        # points at every distance. Count over a whole shell rather than one
        # ring -- a single ring is noisy, and getting this backwards puts the
        # wall a full thickness off, which was the only large error left.
        pos = ((off > f+0.06) & (off < f+0.30)).sum()
        neg = ((off < f-0.06) & (off > f-0.30)).sum()
        d = 1.0 if pos < neg else -1.0              # push away from the open side
        w['c'] = w['c'] + f + d*DEF_T/2; w['t'] = DEF_T; w['measured'] = False
    return True

DEF_T = 0.200
for w in walls: w['measured'] = False
nref = sum(1 for w in walls if refine(w))
mt = [w['t']*1000 for w in walls if w['measured']]
DEF_T = float(np.median(mt))/1000 if mt else 0.200
for w in walls:
    if not w['measured']: w['t'] = DEF_T
print(f"faces snapped to raw points on {nref}/{len(walls)} walls; "
      f"{len(mt)} measured from both faces")
if mt: print(f"  thickness: median {np.median(mt):.0f} mm, "
             f"range {min(mt):.0f}-{max(mt):.0f}  (single-faced walls use the median)")

# ------------------------------------------------------- close the corners
REACH = 0.45; nclosed = 0
for w in walls:
    for end in ('lo', 'hi'):
        best = None
        for v in walls:
            if v['axis'] == w['axis']: continue
            if not (v['lo']-0.30 <= w['c'] <= v['hi']+0.30): continue
            d = v['c'] - w[end]
            if (end == 'hi' and -0.05 <= d <= REACH) or (end == 'lo' and -REACH <= d <= 0.05):
                if best is None or abs(d) < abs(best): best = d
        if best is not None and abs(best) > 1e-6:
            w[end] += best; nclosed += 1
            w['length'] = w['hi']-w['lo']
print(f"closed {nclosed} wall ends onto a perpendicular wall")

DC2 = 0.03      # after refine a true duplicate sits at dc ~ 0
# Rejoin AGAIN, now that the walls sit on the raw points.
#
# The first pass runs on raster centrelines, where two records of one wall still
# differ by a few centimetres and so look like neighbours rather than the same
# thing. refine() is what snaps each wall onto its measured faces -- and that is
# exactly when duplicates collapse onto an identical centreline. Ten pairs were
# reaching the output with dc = 0 and spans overlapping by up to 10.2 m, two of
# them the same span to the millimetre. Deduplicating before the step that makes
# duplicates identical could never have caught them.
nj2 = rejoin(walls, dcmax=DC2, ovmax=DC2)
print(f"rejoined {nj2} more once the walls sat on the raw points; "
      f"{len(walls)} walls remain")

# ------------------------------------------------- openings, one last pass
# Openings were only ever found as a gap BETWEEN two fragments of a wall, which
# means a door was invisible unless the wall walk happened to split there. It
# usually did not: rescanning every wall end to end turned up 3 voids in 52
# walls, because the walk had already bridged the rest.
#
# So stop looking at walls and look for the thing itself. A door or a balcony is
# masonry overhead with nothing underneath -- that is what an opening IS, and it
# shows up in the rasters directly: `above & ~wallr`. There are 23 such regions
# of door or balcony proportions, against 8 found the old way. Each is then
# attached to the collinear wall it belongs to, and a candidate that matches no
# wall is dropped, which is what filters the spurious ones.
hole = above & ~wallr
hlab, hn = ndimage.label(hole, np.ones((3, 3), bool))

def zmeasure_cells(region_id):
    """Sill and head measured from the candidate's OWN cells.

    The wall-centred version sampled a fixed +/-90 mm slab at the wall's
    centreline, which is the wrong place: the region has its own position and
    width, and a lintel that sits off-centre or a slab wider than 180 mm both
    read as empty. Using the region's cells fixed 13 of 23 candidates."""
    sel = np.zeros(nx*ny, bool)
    sel[np.flatnonzero((hlab == region_id).ravel())] = True
    m = ok & sel[flat]
    if m.sum() < 40: return None, None
    zz = z[m]; hz = zz[zz > 1.95]
    if len(hz) < 25: return None, None
    head = float(np.percentile(hz, 3))
    lz = zz[zz < 1.95]
    sill = float(np.percentile(lz, 97)) if len(lz) >= 50 else 0.0
    if sill < 0.30: sill = 0.0
    if head - sill < 0.75: return None, None
    return sill, head
cands = []
for i in range(1, hn+1):
    cl = np.argwhere(hlab == i)
    if len(cl) < 6: continue
    i0, j0 = cl.min(0); i1, j1 = cl.max(0)
    di, dj = (i1-i0+1)*CELL, (j1-j0+1)*CELL
    L, T = max(di, dj), min(di, dj)
    if T > 0.60 or not (OPEN_MIN <= L <= OPEN_MAX): continue
    ax = 0 if dj > di else 1                       # axis-0 wall: gap runs along Y
    c = ((i0+i1+1)/2*CELL) if ax == 0 else ((j0+j1+1)/2*CELL)
    a = (j0*CELL) if ax == 0 else (i0*CELL)
    b = ((j1+1)*CELL) if ax == 0 else ((i1+1)*CELL)
    cands.append((ax, c, a, b, L, i))

nfound = 0; nowall = 0; ndup = 0; nzm = 0
for (ax, c, a, b, L, rid) in cands:
    best = None
    for w in walls:
        if w['axis'] != ax or abs(w['c']-c) > 0.25: continue
        if w['lo'] > a+0.15 or w['hi'] < b-0.15: continue
        if best is None or abs(w['c']-c) < abs(best['c']-c): best = w
    if best is None:
        nowall += 1
        if os.environ.get("OP_DEBUG"):
            near=[(abs(w['c']-c),w['lo'],w['hi']) for w in walls if w['axis']==ax
                  and abs(w['c']-c)<=0.40]
            print(f"   NOWALL ax{ax} c={c:+.2f} {a:+.2f}..{b:+.2f} L={L*1000:.0f} "
                  f"near={sorted(near)[:2]}")
        continue
    if any(a < e['b']-0.05 and b > e['a']+0.05 for e in best['ops']):
        ndup += 1; continue
    sh = zmeasure_cells(rid)
    if sh[0] is None:
        nzm += 1
        if os.environ.get("OP_DEBUG"):
            mm = ok & (np.abs(XY[:, ax]-best['c'])<0.09) & (XY[:,1-ax]>a+0.05) & \
                 (XY[:,1-ax]<b-0.05)
            zz=z[mm]
            print(f"   ZMEAS  ax{ax} c={best['c']:+.2f} {a:+.2f}..{b:+.2f} "
                  f"L={L*1000:.0f} pts={mm.sum()} hi={int((zz>1.95).sum())}")
        continue
    best['ops'].append(dict(a=a, b=b, sill=sh[0], head=sh[1],
                            arch=H-sh[1], width=round((b-a)*1000)))
    nfound += 1
# One physical opening can land on two collinear wall fragments -- the same
# doorway seen from either side of a split. Keep it once, on the longer wall.
ndd = 0
for i, w in enumerate(walls):
    for v in walls[i+1:]:
        if v['axis'] != w['axis'] or abs(v['c']-w['c']) > 0.40: continue
        keep, drop = (w, v) if w['length'] >= v['length'] else (v, w)
        for o in list(drop['ops']):
            # Same width as well as same place. A 600 mm door overlapping a
            # 3350 mm opening is two different things, not one counted twice,
            # and collapsing them loses the door.
            same = [e for e in keep['ops']
                    if min(o['b'], e['b'])-max(o['a'], e['a']) > 0.20
                    and abs((e['b']-e['a'])-(o['b']-o['a'])) <= 0.30*(o['b']-o['a'])]
            if same:
                drop['ops'].remove(o); ndd += 1

for w in walls: w['ops'].sort(key=lambda o: o['a'])
nop = sum(len(w['ops']) for w in walls)
if ndd: print(f"  dropped {ndd} openings counted twice on collinear fragments")
print(f"openings: {len(cands)} lintel-with-nothing-under-it candidates, "
      f"{nfound} new, {ndup} already known, {nzm} no sill/head, "
      f"{nowall} matched no wall")
print(f"  {nop} openings in total")

# ------------------------------------- columns, from the THICKNESS profile
# A column is not a feature of a wall's face, it is the wall being thicker where
# a pillar sits in it -- your rule that thickness varies in steps rather than
# continuously. So it is measured as thickness, station by station.
#
# It cannot be measured off the raster. The scan only ever sees the two faces
# and nothing between them, so a 200 mm wall shows up as two separate 1-cell
# runs, and raster thickness reads 50 mm for almost every wall. Both faces are
# located per station in the raw points instead.
STEP_MIN = 25          # mm of extra thickness before it counts as a step;
                       # steps here go below 50 mm, so this stays low
BIN = 0.10             # along-wall station for a local thickness reading

def thickness_profile(w):
    """Local wall thickness every 100 mm along the wall, from raw points."""
    ax = w['axis']
    m = ok & (np.abs(XY[:, ax]-w['c']) < 0.28) & (XY[:, 1-ax] > w['lo']) & \
        (XY[:, 1-ax] < w['hi']) & (z > 0.40) & (z < H-0.20)
    if m.sum() < 800: return None, None
    off = XY[m, ax]-w['c']; al = XY[m, 1-ax]
    nb = max(int(round((w['hi']-w['lo'])/BIN)), 1)
    kb = np.clip(((al-w['lo'])/BIN).astype(int), 0, nb-1)
    prof = np.full(nb, np.nan)
    order = np.argsort(kb, kind='stable')
    ks = kb[order]; os_ = off[order]
    for grp in np.split(np.arange(len(ks)), np.flatnonzero(np.diff(ks))+1):
        d = os_[grp]
        if len(d) < 60: continue
        h, e = np.histogram(d, bins=56, range=(-0.28, 0.28))
        pk = np.flatnonzero(h > h.max()*0.28)
        if len(pk) < 2: continue
        lo_, hi_ = e[pk[0]], e[pk[-1]+1]
        t = hi_-lo_
        if 0.08 <= t <= 0.55: prof[ks[grp[0]]] = t
    fin = np.isfinite(prof)
    if fin.sum() < 4: return None, None
    return prof, float(np.median(prof[fin]))

COLS = []
for wi, w in enumerate(walls, 1):
    prof, base = thickness_profile(w)
    if prof is None: continue
    w['t_base'] = base
    extra = (prof-base)*1000
    hot = np.isfinite(extra) & (extra >= STEP_MIN)
    q = 0
    while q < len(hot):
        if not hot[q]: q += 1; continue
        j = q
        while j+1 < len(hot) and hot[j+1]: j += 1
        L = (j-q+1)*BIN
        if 0.15 <= L <= 2.0:
            d = float(np.median(extra[q:j+1]))
            if STEP_MIN <= d <= 400:
                COLS.append(dict(kind="COLUMN", wall=wi, axis=int(w['axis']),
                                 c=float(w['c']), t=float(base+d/1000),
                                 a=float(w['lo']+q*BIN), b=float(w['lo']+(j+1)*BIN),
                                 z0=0.0, z1=float(H), length_mm=round(L*1000),
                                 depth_mm=round(d), base_mm=round(base*1000)))
        q = j+1
if COLS:
    dd = [r['depth_mm'] for r in COLS]; bb = [r['base_mm'] for r in COLS]
    print(f"columns from thickness steps: {len(COLS)}   step median {np.median(dd):.0f} mm "
          f"(range {min(dd)}-{max(dd)}), wall base median {np.median(bb):.0f} mm")
    print(f"  site rule: thickness steps at pillars, about 75 mm")
else:
    print("columns from thickness steps: none found")

# ------------------------------- what is on each wall FACE, resolved in height
# The previous pass averaged the face offset over every height at once. That
# collapses the vertical dimension, so a beam running along a wall -- proud near
# the ceiling, wall set back below it -- was invisible, and every feature came
# back spanning 350-2550 mm because its height was taken from the points in the
# along-band rather than from where the step actually is.
#
# Each face now gets a RELIEF MAP: face offset as a function of (along, height).
# A step is a rectangle in that map, so it has real bounds both ways:
#
#   proud, hugging the ceiling, running most of the wall  -> BEAM
#   proud, floor to ceiling                               -> COLUMN
#   proud, bounded                                        -> PILASTER
#   recessed                                              -> NICHE
Z0 = 0.10
def relief(w, f):
    """(along, height) map of how far the face steps in or out, in mm.
    +ve = recessed into the wall, -ve = proud of it."""
    ax = w['axis']; inward = np.sign(w['c']-f)
    depth_lim = min(0.18, w['t']-0.06)
    m = ok & (XY[:, 1-ax] > w['lo']+0.03) & (XY[:, 1-ax] < w['hi']-0.03) & \
        (z > Z0) & (z < H-0.06)
    if m.sum() < 2000: return None, None, None
    d = (XY[m, ax]-f)*inward
    keep = (d > -0.30) & (d < depth_lim)     # room side out to 300 mm, and into
    if keep.sum() < 2000: return None, None, None   # the wall only as far as its
    d = d[keep]                                     # own far face, never past it
    al = XY[m, 1-ax][keep]; zz = z[m][keep]
    na = int(round((w['hi']-w['lo'])/CELL)); nz = int((H-0.06-Z0)/CELL)+1
    if na < 4 or nz < 4: return None, None, None
    ka = np.clip(((al-w['lo'])/CELL).astype(int), 0, na-1)
    kz = np.clip(((zz-Z0)/CELL).astype(int), 0, nz-1)
    k = ka*nz + kz
    srt = np.argsort(k, kind='stable'); ks = k[srt]; ds = d[srt]
    R = np.full(na*nz, np.nan)
    for grp in np.split(np.arange(len(ks)), np.flatnonzero(np.diff(ks))+1):
        R[ks[grp[0]]] = np.median(ds[grp])
    R = R.reshape(na, nz)
    fin = np.isfinite(R)
    if fin.sum() < 40: return None, None, None
    return (R-np.median(R[fin]))*1000, fin, (na, nz)

FEAT = []
for wi, w in enumerate(walls, 1):
    faces = [('A', w['c']-w['t']/2)] + ([('B', w['c']+w['t']/2)] if w['measured'] else [])
    for side, f in faces:
        rel, fin, shp = relief(w, f)
        if rel is None: continue
        na, nz = shp
        for kind, mask in (("recess", fin & (rel > 45)), ("proud", fin & (rel < -45))):
            mask = ndimage.binary_opening(mask, np.ones((2, 2), bool))
            lab, n = ndimage.label(mask, np.ones((3, 3), bool))
            for li in range(1, n+1):
                cells = np.argwhere(lab == li)
                if len(cells) < 12: continue                 # under 0.03 m2
                a0c, z0c = cells.min(0); a1c, z1c = cells.max(0)
                L = (a1c-a0c+1)*CELL; Hh = (z1c-z0c+1)*CELL
                if L < 0.20 or Hh < 0.20 or L > 6.0: continue
                dpt = float(np.median(np.abs(rel[lab == li])))
                if not (45 <= dpt <= 350): continue
                # A recess cannot be deeper than the wall it is cut into --
                # that would be a hole, and it means the base plane was taken
                # off the wrong side of the face.
                if kind == "recess" and dpt > w['t']*1000-50: continue
                zlo = Z0+z0c*CELL; zhi = min(Z0+(z1c+1)*CELL, H)
                a = w['lo']+a0c*CELL; b = w['lo']+(a1c+1)*CELL
                # A wall butting into this face is a T-junction, not a step in
                # it: the other wall's cross-section reads as a proud region
                # ~200 mm long and a full wall thick, which is why "pilasters"
                # were coming out 257 mm deep against a ~75 mm site rule.
                # A junction is one wall thick. Anything longer running along
                # the face is a real step, and a beam necessarily crosses the
                # walls it spans, so only short regions are tested.
                if kind == "proud" and L < 0.60 and any(
                        v['axis'] != w['axis'] and a-0.10 <= v['c'] <= b+0.10 and
                        v['lo']-0.25 <= w['c'] <= v['hi']+0.25 for v in walls):
                    continue
                if kind == "recess": tag = "NICHE"
                elif zhi >= H-0.25 and Hh <= 1.20: tag = "BEAM"
                # Anything else proud is the wall being locally thicker, which
                # is a column -- and columns are measured properly above, from
                # the thickness profile, not from one face's relief. Dropping
                # them here is what stops the same pillar being reported twice,
                # once per face, as a "pilaster".
                else: continue
                FEAT.append(dict(kind=tag, wall=wi, face=side, axis=int(w['axis']),
                                 f=float(f), c=float(w['c']), t=float(w['t']),
                                 a=float(a), b=float(b), z0=float(zlo), z1=float(zhi),
                                 length_mm=round(L*1000), height_mm=round(Hh*1000),
                                 depth_mm=round(dpt)))

def dedupe(arr):
    """A step that thickens the wall shows on BOTH faces -- one feature, not two.
    Only same-kind pairs are merged: a recess on one face and a proud step on the
    other are different features, not two views of one."""
    out = []
    for r in sorted(arr, key=lambda q: -(q['b']-q['a'])*(q['z1']-q['z0'])):
        if any(q['kind'] == r['kind'] and q['axis'] == r['axis'] and
               abs(q['c']-r['c']) < 0.35 and
               min(q['b'], r['b'])-max(q['a'], r['a']) > 0.10 and
               min(q['z1'], r['z1'])-max(q['z0'], r['z0']) > 0.10 for q in out): continue
        out.append(r)
    return out
FEAT = dedupe(FEAT) + COLS
from collections import Counter
print("wall features:", dict(Counter(r['kind'] for r in FEAT)))
for t in ("BEAM", "COLUMN", "NICHE"):
    g = [r for r in FEAT if r['kind'] == t]
    if g:
        d = [r['depth_mm'] for r in g]
        print(f"  {t:<9} {len(g):>3}   depth median {np.median(d):>3.0f} mm "
              f"(range {min(d)}-{max(d)}), site rule is a ~75 mm rectangular step")

# ----------------------------------------------------------------- build
G = GLB()
def yup(axis, c, t, A, B, z0, z1):
    if axis == 0: xl, xh, yl, yh = c-t/2, c+t/2, A, B
    else:         xl, xh, yl, yh = A, B, c-t/2, c+t/2
    return (xl+mins[0], z0, -(yh+mins[1])), (xh+mins[0], z1, -(yl+mins[1]))
def kindof(o):
    return "window" if o['sill'] >= 0.60 else "door" if o['width'] <= 1050 else "opening"

# Each wall is a PARENT node owning its own parts: every solid panel, every
# arch, every column, niche and beam that sits on it. Panels are no longer
# merged into one mesh per wall -- a pier beside a door and the spandrel under a
# window are different pieces of masonry, and a cutlist needs them separately.
narch = 0; npanel = 0
byWall = {}
for r in FEAT: byWall.setdefault(r.get('wall'), []).append(r)

def emit_feature(r, wi):
    nseq[r['kind']] = nseq.get(r['kind'], 0)+1
    tag = f"{r['kind']}_{wi:02d}_{nseq[r['kind']]:02d}"
    if r['kind'] == "COLUMN":
        # A thickness step straddles the wall's centreline: the pillar IS the
        # wall, locally fatter, so it is drawn full-thickness on the centreline
        # rather than hung off one face.
        G.add_box(f"{tag}_t{round(r['t']*1000)}_step{r['depth_mm']}"
                  f"_L{r['length_mm']}",
                  *yup(r['axis'], r['c'], r['t'], r['a'], r['b'], r['z0'], r['z1']))
        return
    out = 1.0 if r['f'] > r['c'] else -1.0
    d = r['depth_mm']/1000
    off = (r['t']/2 - d/2) if r['kind'] == "NICHE" else (r['t']/2 + d/2)
    G.add_box(f"{tag}_d{r['depth_mm']}_L{r['length_mm']}_h{r['height_mm']}",
              *yup(r['axis'], r['c']+out*off, d, r['a'], r['b'], r['z0'], r['z1']))

nseq = {}
LEAF = 0.040
ndoor = 0; nwin = 0
for wi, w in enumerate(walls, 1):
    first = len(G.parts)
    axis, c, t = w['axis'], w['c'], w['t']
    ops = sorted(w['ops'], key=lambda o: o['a'])
    xs = sorted({w['lo'], w['hi']} | {v for o in ops for v in (o['a'], o['b'])})
    for q in range(len(xs)-1):
        A, B = xs[q], xs[q+1]
        if B-A < 0.02: continue
        cov = [o for o in ops if o['a'] < B-1e-6 and o['b'] > A+1e-6]
        if not cov:
            npanel += 1
            G.add_box(f"PANEL_{wi:02d}_{npanel:03d}_L{(B-A)*1000:.0f}_t{t*1000:.0f}",
                      *yup(axis, c, t, A, B, 0.0, H))
        elif cov[0]['sill'] > 0.02:
            npanel += 1
            G.add_box(f"SILL_{wi:02d}_{npanel:03d}_L{(B-A)*1000:.0f}"
                      f"_h{cov[0]['sill']*1000:.0f}",
                      *yup(axis, c, t, A, B, 0.0, cov[0]['sill']))
    for o in ops:
        narch += 1
        k = kindof(o)
        G.add_box(f"ARCH_{wi:02d}_{narch:02d}_{k}_w{o['width']}"
                  f"_head{o['head']*1000:.0f}_drop{o['arch']*1000:.0f}",
                  *yup(axis, c, t, o['a'], o['b'], o['head'], H))
        if k == "door":
            ndoor += 1
            G.add_box(f"DOOR_{wi:02d}_{ndoor:02d}_w{o['width']}"
                      f"_h{(o['head']-o['sill'])*1000:.0f}",
                      *yup(axis, c, LEAF, o['a'], o['b'], o['sill'], o['head']))
        elif k == "window":
            nwin += 1
            G.add_box(f"WINDOW_{wi:02d}_{nwin:02d}_w{o['width']}"
                      f"_sill{o['sill']*1000:.0f}_h{(o['head']-o['sill'])*1000:.0f}",
                      *yup(axis, c, LEAF, o['a'], o['b'], o['sill'], o['head']))
    for r in byWall.get(wi, []):
        emit_feature(r, wi)
    G.parent(f"WALL_{wi:02d}_{'X' if axis==0 else 'Y'}"
             f"_L{w['length']*1000:.0f}_t{t*1000:.0f}", first)
for r in byWall.get(None, []):
    emit_feature(r, 0)
print(f"door leaves {ndoor}   windows {nwin}   panels {npanel}")
fx0, fx1 = float(XY[:, 0].min()+mins[0]), float(XY[:, 0].max()+mins[0])
fy0, fy1 = float(XY[:, 1].min()+mins[1]), float(XY[:, 1].max()+mins[1])
G.add_box("FLOOR", (fx0, -0.05, -fy1), (fx1, 0.0, -fy0))
sz, parts, tris = G.write("output/model/shell_fp.glb")

kinds = {}
for w in walls:
    for o in w['ops']: kinds[kindof(o)] = kinds.get(kindof(o), 0)+1
print(f"\nopenings: {kinds}   (ground truth 9 doors, 4 balconies, 2 openings)")
alla = [o['arch'] for w in walls for o in w['ops']]
if alla: print(f"arch drop median {np.median(alla)*1000:.0f} mm, "
               f"range {min(alla)*1000:.0f}-{max(alla)*1000:.0f}")
print(f"wrote output/model/shell_fp.glb  {sz/1e6:.2f} MB, {parts} parts, {tris:,} triangles")
json.dump(dict(clear_height_mm=round(H*1000),
               walls=[dict(id=i+1, axis=w['axis'],
                           centre_mm=round((w['c']+mins[w['axis']])*1000),
                           lo_mm=round((w['lo']+mins[1-w['axis']])*1000),
                           hi_mm=round((w['hi']+mins[1-w['axis']])*1000),
                           length_mm=round(w['length']*1000),
                           thickness_mm=round(w['t']*1000), both_faces=bool(w['measured']),
                           openings=[dict(kind=kindof(o), width_mm=o['width'],
                           a_mm=round(o['a']*1000), b_mm=round(o['b']*1000),
                                          sill_mm=round(o['sill']*1000),
                                          head_mm=round(o['head']*1000),
                                          arch_mm=round(o['arch']*1000)) for o in w['ops']])
                      for i, w in enumerate(walls)],
               features=FEAT),
          open("output/model/shell_fp.json", "w"), indent=1)
print("wrote output/model/shell_fp.json")
