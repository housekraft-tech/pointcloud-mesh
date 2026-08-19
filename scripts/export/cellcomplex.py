"""Reconstruct the shell as a CELL COMPLEX, not as a pile of boxes.

The old model fitted each wall independently, which is why a third of the
surface it drew had no scanned point within 100 mm: nothing in a list of boxes
prevents two walls occupying one space, or stops a wall being drawn 195 mm thick
because that is the median when only one of its faces was ever seen.

This follows the PolyFit / Kinetic-Surface-Reconstruction line instead: detect
planes, partition space into cells, label each cell solid or empty, and take the
boundary between them. The output is watertight because it is a union of
volumes, and three failures become impossible rather than merely rare:

  assumed thickness   a wall's interior is a CELL between two DETECTED planes.
                      Its thickness is measured, never defaulted.
  duplicate walls     a cell is solid or it is not. Two walls cannot occupy it.
  invented surface    surface exists only between a solid cell and an empty one,
                      and solidity has to be voted for by the points.

The general method needs kinetic partitioning because planes point in arbitrary
directions. A bare shell is rectilinear, which collapses that: with axis-aligned
planes the complex is the outer product of three sorted plane lists and every
cell is a box. No CGAL, no SageMath.
"""
import sys, json, os, numpy as np, laspy
from scipy import ndimage
sys.path.insert(0, "scripts/export")
from glb import GLB

G          = 0.02      # support grid
MERGE      = 0.015     # planes closer than this are the same plane
MIN_AREA   = float(os.environ.get("MIN_AREA", 0.15))   # m2 before a plane is real
MAX_WALL   = 0.45      # a cell thinner than this, between two faces, is masonry
FACE_SUP   = float(os.environ.get("FACE_SUP", 0.15))  # face scanned this much = masonry
CUT        = 0.12
SLAB       = 0.15      # assumed depth of the floor and ceiling slabs

J = json.load(open("output/fp_walls.json")); H = J['clear_height']
with laspy.open("output/mujammel_aligned_z0.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
if os.path.exists("output/keep_mask.npy"):
    k = np.load("output/keep_mask.npy")
    if len(k) == len(P): P = P[k]
P = P[P[:, 2] < H-CUT]
lo = P.min(0); hi = P.max(0)
print(f"{len(P):,} points, extent "
      f"{hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {hi[2]-lo[2]:.2f} m")

n = np.ceil((hi-lo)/G).astype(int) + 1
idx = np.clip(((P-lo)/G).astype(np.int64), 0, n-1)
flat = (idx[:, 0]*n[1] + idx[:, 1])*n[2] + idx[:, 2]
occ = np.zeros(int(n.prod()), bool); occ[flat] = True
OCC = occ.reshape(n)
print(f"occupancy {n[0]}x{n[1]}x{n[2]} @ {G*1000:.0f} mm, {OCC.sum():,} cells")

# ---- planes: a slab of the grid whose scanned AREA is large ----------------
def planes(ax):
    other = [a for a in (0, 1, 2) if a != ax]
    area = OCC.sum(axis=tuple(other)) * G*G          # m2 per slab
    # A fixed area threshold, not a fraction of the largest slab. The relative
    # term put the cut at 0.77 m2, and the peak count falls off a cliff there:
    # 0.77 -> 67 planes, 0.60 -> 124. Real wall faces were sitting just under it,
    # and a wall whose faces are not both detected never becomes a cell.
    thr = MIN_AREA
    pk = [i for i in range(1, len(area)-1)
          if area[i] >= thr and area[i] >= area[i-1] and area[i] > area[i+1]]
    out = []
    for i in pk:
        c = lo[ax] + i*G
        m = np.abs(P[:, ax]-c) < 0.025               # refine to the raw points
        if m.sum() < 500: continue
        out.append((float(np.median(P[m, ax])), float(area[i])))
    out.sort()
    keep = []
    for (c, a) in out:                               # regularize coplanar
        if keep and c-keep[-1][0] < MERGE:
            if a > keep[-1][1]: keep[-1] = (c, a)
        else: keep.append((c, a))
    return [c for (c, a) in keep]

PL = []
for ax in (0, 1, 2):
    q = planes(ax)
    # No slab planes below the floor or above the ceiling. Adding them forces
    # the slabs solid and then DRAWS their outer faces -- 570 m2 of underside
    # and roof that the scanner can never see, which took unsupported surface
    # from 2.1% to 53%. The floor's top face is scanned and gets drawn as the
    # boundary of the masonry above it; its underside is outside the domain.
    # This is what CGAL handles by labelling the bounding box external rather
    # than extracting boundary there.
    q = [lo[ax]-1e-3] + q + [hi[ax]+1e-3]            # bound the volume
    q = sorted(set(round(v, 4) for v in q))
    PL.append(np.array(q))
    print(f"  axis {'XYZ'[ax]}: {len(q)-2} planes detected")

nx, ny, nz = [len(q)-1 for q in PL]
print(f"cell complex: {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")

def gidx(ax, v):                                     # world -> support grid
    return int(np.clip(round((v-lo[ax])/G), 0, n[ax]-1))

# ---- how much of each cell face is actually scanned ------------------------
# Precomputed for every face in the complex, not per query: for each plane take
# the thin slab of occupancy at it, collapse along the plane normal, then
# aggregate to cell resolution. A face carrying scanned surface is masonry and
# nothing propagates across it.
def cell_edges(ax):
    return np.array([int(np.clip(round((v-lo[ax])/G), 0, n[ax])) for v in PL[ax]])
EDG = [cell_edges(ax) for ax in (0, 1, 2)]

def face_support_all(ax):
    o = [a for a in (0, 1, 2) if a != ax]
    e0, e1 = EDG[o[0]], EDG[o[1]]
    nb, nc = len(e0)-1, len(e1)-1
    out = np.zeros((len(PL[ax]), nb, nc), np.float32)
    for pi, v in enumerate(PL[ax]):
        i = int(np.clip(round((v-lo[ax])/G), 0, n[ax]-1))
        sl = [slice(None)]*3
        sl[ax] = slice(max(i-1, 0), i+2)
        blk = OCC[tuple(sl)].any(axis=ax).astype(np.float32)   # (o0, o1) at grid res
        cs = blk.cumsum(0).cumsum(1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        b0 = np.clip(e0, 0, cs.shape[0]-1); c0 = np.clip(e1, 0, cs.shape[1]-1)
        S = (cs[np.ix_(b0[1:], c0[1:])] - cs[np.ix_(b0[:-1], c0[1:])]
             - cs[np.ix_(b0[1:], c0[:-1])] + cs[np.ix_(b0[:-1], c0[:-1])])
        area = np.maximum(np.outer(np.diff(b0), np.diff(c0)), 1)
        out[pi] = S/area
    return out

SUP = [face_support_all(ax) for ax in (0, 1, 2)]
for ax in (0, 1, 2):
    print(f"  axis {'XYZ'[ax]}: face support computed, "
          f"{(SUP[ax] >= FACE_SUP).sum():,} faces carry scanned surface")

# ---- label cells ----------------------------------------------------------
# A room and a wall interior both hold no points -- the scanner sees neither.
# What separates them is whether the scanner ever saw THROUGH the space. Rays
# cast from the recovered trajectory carve out everything that was looked
# through; what is left is either masonry or the world outside the building.
# Flooding inward from the bounding box separates those two: anything enclosed
# and never seen through is solid.
#
# This replaces the old rule, which called a cell solid only when BOTH of its
# faces were >=25% scanned and it was under 450 mm thick. That is true of an
# internal partition and false of every external wall, which is why recall was
# 50%.
FS = np.load("output/freespace.npy")
FM = json.load(open("output/freespace_meta.json"))
assert FM['G'] == G, "free space grid must match"
flo = np.array(FM['lo'])

FREE_FRAC = float(os.environ.get("FREE_FRAC", 0.95))   # only a genuinely open cell
# Vectorised over every cell at once with a 3D summed-area table -- a Python
# loop over half a million cells, each taking a numpy slice, is minutes.
CS = np.zeros((FS.shape[0]+1, FS.shape[1]+1, FS.shape[2]+1), np.int32)
CS[1:, 1:, 1:] = FS
np.cumsum(CS, axis=0, out=CS); np.cumsum(CS, axis=1, out=CS)
np.cumsum(CS, axis=2, out=CS)
E = []
for ax in (0, 1, 2):
    e = np.clip(np.round((PL[ax]-flo[ax])/G).astype(int), 0, FS.shape[ax])
    e[1:] = np.maximum(e[1:], e[:-1]+1)
    E.append(np.clip(e, 0, FS.shape[ax]))
a0, a1 = E[0][:-1][:, None, None], E[0][1:][:, None, None]
b0, b1 = E[1][:-1][None, :, None], E[1][1:][None, :, None]
c0, c1 = E[2][:-1][None, None, :], E[2][1:][None, None, :]
def blk(i, j, k): return CS[i, j, k]
tot = (blk(a1,b1,c1)-blk(a0,b1,c1)-blk(a1,b0,c1)-blk(a1,b1,c0)
       +blk(a0,b0,c1)+blk(a0,b1,c0)+blk(a1,b0,c0)-blk(a0,b0,c0))
vol = np.maximum((a1-a0)*(b1-b0)*(c1-c0), 1)
free = (tot/vol) >= FREE_FRAC
del CS
print(f"seen through: {free.sum():,} of {nx*ny*nz:,} cells are open space")

# Flood in from the bounding box through everything not carved -- but a face
# carrying scanned surface is masonry and blocks it. Without that gate the
# unknown cells form one connected network joining the outside to every wall
# interior, the flood reaches everything, and 119 cells survive as solid.
unknown = ~free
outside = np.zeros_like(unknown)
outside[0, :, :] |= unknown[0, :, :];   outside[-1, :, :] |= unknown[-1, :, :]
outside[:, 0, :] |= unknown[:, 0, :];   outside[:, -1, :] |= unknown[:, -1, :]
outside[:, :, 0] |= unknown[:, :, 0];   outside[:, :, -1] |= unknown[:, :, -1]
OPEN = [SUP[ax] < FACE_SUP for ax in (0, 1, 2)]     # face is passable
while True:
    prev = outside.sum()
    for ax in (0, 1, 2):
        for sgn in (+1, -1):
            src = [slice(None)]*3; dst = [slice(None)]*3; fsl = [slice(None)]*3
            L = (nx, ny, nz)[ax]
            if sgn > 0:
                src[ax] = slice(0, L-1); dst[ax] = slice(1, L); fsl[ax] = slice(1, L)
            else:
                src[ax] = slice(1, L); dst[ax] = slice(0, L-1); fsl[ax] = slice(1, L)
            gate = np.moveaxis(OPEN[ax][1:L], 0, ax)
            outside[tuple(dst)] |= (outside[tuple(src)] & unknown[tuple(dst)] & gate)
    if outside.sum() == prev: break
enclosed = unknown & ~outside
print(f"outside the building: {outside.sum():,} cells")

# Union the two signals. They fail on opposite cases, so neither alone is
# enough: face-support needs BOTH faces scanned, which is true of an internal
# partition and false of every external wall; free-space needs the cell to be
# properly enclosed, which the flood breaks wherever a face is under-scanned.
D = [np.diff(PL[ax]) for ax in (0, 1, 2)]
DD = np.stack(np.meshgrid(D[0], D[1], D[2], indexing='ij'))
amin = DD.argmin(0); dmin = DD.min(0)
# A wall is a CONTIGUOUS SLAB and the per-cell vote does not know that: where
# the scan is thin a cell fails the threshold and the wall comes out perforated,
# which then shatters into dozens of parts under face-connectivity. Closing
# within the plane of each slab knits those breaks back.
#
# Closing ONLY -- never binary_fill_holes. A doorway is a hole in a wall slab,
# and filling holes bricks up every door: precision fell from 6.4% to 34.9%
# unsupported when I tried it. Never across the normal either, which would fuse
# parallel walls.
def consolidate(m3, ax, k=3):
    out = np.moveaxis(m3, ax, 0).copy()
    ker = np.ones((k, k), bool)
    for i in range(out.shape[0]):
        if out[i].any(): out[i] = ndimage.binary_closing(out[i], ker)
    return np.moveaxis(out, 0, ax)

thin = np.zeros((nx, ny, nz), bool)
for ax in (0, 1, 2):
    both = np.minimum(SUP[ax][:-1], SUP[ax][1:]) >= FACE_SUP   # (cells_ax, o0, o1)
    both = np.moveaxis(both, 0, ax)
    thin |= consolidate((amin == ax) & (dmin <= MAX_WALL) & both, ax)
print(f"  masonry by two scanned faces : {thin.sum():,} cells")
print(f"  masonry by being enclosed    : {enclosed.sum():,} cells")
print(f"  overlap                      : {(thin & enclosed).sum():,}")
solid = thin | enclosed
print(f"labelled solid: {solid.sum():,} of {nx*ny*nz:,} cells")

# a solid sliver thicker than any masonry is an unscanned void, not a wall
solid &= dmin <= MAX_WALL
print(f"after dropping cells thicker than {MAX_WALL*1000:.0f} mm "
      f"in every axis: {solid.sum():,}")

# ---- boundary faces between solid and empty -------------------------------
# Only faces where a solid cell meets an empty one. The faces BETWEEN two solid
# cells are interior masonry and must not be drawn -- emitting every cell as a
# box would put 138k triangles of invisible surface inside the walls, which is
# the very thing this is meant to stop.
def greedy(mask):
    m = mask.copy(); out = []; R, C = m.shape
    for i in range(R):
        if not m[i].any(): continue
        j = 0
        while j < C:
            if not m[i, j]: j += 1; continue
            w = 1
            while j+w < C and m[i, j+w]: w += 1
            h = 1
            while i+h < R and m[i+h, j:j+w].all(): h += 1
            m[i:i+h, j:j+w] = False; out.append((i, j, h, w)); j += w
    return out

Gl = GLB()
dims = (nx, ny, nz)
nq = 0
for ax in (0, 1, 2):
    o = [a for a in (0, 1, 2) if a != ax]
    for sgn in (-1, +1):
        sh = np.zeros_like(solid)
        d, srr = [slice(None)]*3, [slice(None)]*3
        if sgn > 0: d[ax], srr[ax] = slice(0, dims[ax]-1), slice(1, dims[ax])
        else:       d[ax], srr[ax] = slice(1, dims[ax]), slice(0, dims[ax]-1)
        sh[tuple(d)] = solid[tuple(srr)]
        exposed = solid & ~sh
        quads = []
        for pi in range(dims[ax]):
            sl = [slice(None)]*3; sl[ax] = pi
            mask = exposed[tuple(sl)]
            if not mask.any(): continue
            plane = PL[ax][pi + (1 if sgn > 0 else 0)]
            for (i, j, h, w) in greedy(mask):
                b0, b1 = PL[o[0]][i], PL[o[0]][i+h]
                c0, c1 = PL[o[1]][j], PL[o[1]][j+w]
                v = []
                for (bb, cc) in ((b0, c0), (b1, c0), (b1, c1), (b0, c1)):
                    q = [0, 0, 0]; q[ax] = plane; q[o[0]] = bb; q[o[1]] = cc
                    v.append((q[0], q[2], -q[1]))          # into the glTF frame
                quads.append(v)
        if not quads: continue
        nrm = [0.0, 0.0, 0.0]
        nrm[{0: 0, 1: 2, 2: 1}[ax]] = float(sgn) * (-1.0 if ax == 1 else 1.0)
        Gl.add_quads(f"SHELL_{'XYZ'[ax]}{'+' if sgn>0 else '-'}", quads, tuple(nrm))
        nq += len(quads)
np.savez_compressed("output/cells.npz", solid=solid, free=free,
                    px=PL[0], py=PL[1], pz=PL[2],
                    supx=SUP[0], supy=SUP[1], supz=SUP[2])
print("wrote output/cells.npz  (solid labels + plane positions + face support)")
sz, parts, tris = Gl.write("output/model/cellcomplex.glb")
print(f"{nq:,} boundary quads -> {tris:,} triangles")
print(f"wrote output/model/cellcomplex.glb  {sz/1e6:.2f} MB, {parts} parts")
