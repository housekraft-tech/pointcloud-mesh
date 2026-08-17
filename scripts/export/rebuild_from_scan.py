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
MIN_PTS = 12
MAX_T_CELLS = 8         # 400 mm: longer than this is a wall seen face-on
BRIDGE = 0.30           # occlusion gaps this short are bridged unconditionally
OPEN_MIN, OPEN_MAX = 0.55, 3.60
MIN_LEN = 0.60          # a wall shorter than this is not a wall

H = json.load(open("output/fp_walls.json"))['clear_height']
mins = np.array(json.load(open("output/fp_walls.json"))['mins'])
with laspy.open("output/mujammel_structural_v6.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64); z = P[:, 2]
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
    # bucket by centreline; a wall wanders by at most a cell along its length
    buckets = {}
    for (cc, L, a) in secs: buckets.setdefault(int(round(cc)), []).append((cc, L, a))
    for key in sorted(buckets):
        for nb in (key-1,):                       # merge with the bucket below
            if nb in buckets: buckets[key] = buckets.pop(nb) + buckets[key]
    for key in sorted(buckets):
        recs = buckets[key]
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
                elif OPEN_MIN <= glen <= OPEN_MAX and lint[kk:j+1].mean() >= 0.50:
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

# ------------------------------------------------------ niches and columns
NICHE = []; COL = []
for wi, w in enumerate(walls, 1):
    ax = w['axis']
    for side, f in (('A', w['c']-w['t']/2), ('B', w['c']+w['t']/2)):
        near = ok & (np.abs(XY[:, ax]-f) < 0.14) & (XY[:, 1-ax] > w['lo']) & \
               (XY[:, 1-ax] < w['hi']) & (z > 0.30) & (z < H-0.15)
        if near.sum() < 3000: continue
        al = XY[near, 1-ax]; offs = XY[near, ax]-f
        n = int(round((w['hi']-w['lo'])/CELL))
        if n < 4: continue
        k = np.floor((al-w['lo'])/CELL).astype(int)
        good = (k >= 0) & (k < n)
        prof = np.full(n, np.nan)
        srt = np.argsort(k[good]); ks = k[good][srt]; os_ = offs[good][srt]
        for grp in np.split(np.arange(len(ks)), np.flatnonzero(np.diff(ks))+1):
            prof[ks[grp[0]]] = np.median(os_[grp])
        rel = (prof-np.nanmedian(prof))*1000 * np.sign(w['c']-f)     # +ve = into wall
        fin = np.isfinite(rel)
        for kind, mask in (("niche", fin & (rel > 45)), ("column", fin & (rel < -45))):
            q = 0
            while q < n:
                if not mask[q]: q += 1; continue
                j = q
                while j+1 < n and mask[j+1]: j += 1
                L = (j-q+1)*CELL
                if 0.20 <= L <= 3.0:
                    dpt = float(np.nanmedian(np.abs(rel[q:j+1])))
                    if 45 <= dpt <= 350:
                        a0 = w['lo']+q*CELL; a1 = w['lo']+(j+1)*CELL
                        m2 = near & (XY[:, 1-ax] >= a0) & (XY[:, 1-ax] < a1)
                        zr = z[m2]
                        z0, z1 = ((float(np.percentile(zr, 2)), float(np.percentile(zr, 98)))
                                  if m2.sum() > 50 else (0.0, H))
                        (NICHE if kind == "niche" else COL).append(
                            dict(wall=wi, face=side, axis=ax, f=float(f), c=float(w['c']),
                                 t=float(w['t']), a=float(a0), b=float(a1),
                                 length_mm=round(L*1000), depth_mm=round(dpt),
                                 z0=float(max(z0, 0.0)), z1=float(min(z1, H))))
                q = j+1
def dedupe(arr):            # the same step is seen from both faces of a wall
    out = []
    for r in sorted(arr, key=lambda q: -(q['b']-q['a'])):
        if any(q['axis'] == r['axis'] and abs(q['c']-r['c']) < 0.35 and
               min(q['b'], r['b'])-max(q['a'], r['a']) > 0.10 for q in out): continue
        out.append(r)
    return out
NICHE, COL = dedupe(NICHE), dedupe(COL)
print(f"niches {len(NICHE)}   columns {len(COL)}   (paired across both faces, counted once)")

# ----------------------------------------------------------------- build
G = GLB()
def yup(axis, c, t, A, B, z0, z1):
    if axis == 0: xl, xh, yl, yh = c-t/2, c+t/2, A, B
    else:         xl, xh, yl, yh = A, B, c-t/2, c+t/2
    return (xl+mins[0], z0, -(yh+mins[1])), (xh+mins[0], z1, -(yl+mins[1]))
def kindof(o):
    return "window" if o['sill'] >= 0.60 else "door" if o['width'] <= 1050 else "opening"

narch = 0
for wi, w in enumerate(walls, 1):
    axis, c, t = w['axis'], w['c'], w['t']
    ops = sorted(w['ops'], key=lambda o: o['a'])
    acc = G.new_group()
    xs = sorted({w['lo'], w['hi']} | {v for o in ops for v in (o['a'], o['b'])})
    for q in range(len(xs)-1):
        A, B = xs[q], xs[q+1]
        if B-A < 0.02: continue
        cov = [o for o in ops if o['a'] < B-1e-6 and o['b'] > A+1e-6]
        if not cov:
            G.add_box("", *yup(axis, c, t, A, B, 0.0, H), acc)
        elif cov[0]['sill'] > 0.02:
            G.add_box("", *yup(axis, c, t, A, B, 0.0, cov[0]['sill']), acc)
    G.add_group(f"WALL_{wi:02d}_{'X' if axis==0 else 'Y'}"
                f"_L{w['length']*1000:.0f}_t{t*1000:.0f}", acc)
    for o in ops:
        narch += 1
        G.add_box(f"ARCH_{wi:02d}_{kindof(o)}_w{o['width']}"
                  f"_head{o['head']*1000:.0f}_drop{o['arch']*1000:.0f}",
                  *yup(axis, c, t, o['a'], o['b'], o['head'], H))
for i, r in enumerate(COL, 1):
    out = 1.0 if r['f'] > r['c'] else -1.0; d = r['depth_mm']/1000
    G.add_box(f"COLUMN_{i:02d}_d{r['depth_mm']}_L{r['length_mm']}",
              *yup(r['axis'], r['c']+out*(r['t']/2+d/2), d, r['a'], r['b'], r['z0'], r['z1']))
for i, r in enumerate(NICHE, 1):
    out = 1.0 if r['f'] > r['c'] else -1.0; d = r['depth_mm']/1000
    G.add_box(f"NICHE_{i:02d}_d{r['depth_mm']}_L{r['length_mm']}",
              *yup(r['axis'], r['c']+out*(r['t']/2-d/2), d, r['a'], r['b'], r['z0'], r['z1']))
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
                                          sill_mm=round(o['sill']*1000),
                                          head_mm=round(o['head']*1000),
                                          arch_mm=round(o['arch']*1000)) for o in w['ops']])
                      for i, w in enumerate(walls)],
               niches=NICHE, columns=COL),
          open("output/model/shell_fp.json", "w"), indent=1)
print("wrote output/model/shell_fp.json")
