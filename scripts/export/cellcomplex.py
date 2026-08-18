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
MIN_AREA   = 0.60      # m2 of scanned surface before a plane is real
MAX_WALL   = 0.45      # a cell thinner than this, between two faces, is masonry
FACE_SUP   = 0.25      # fraction of a cell face that must be scanned to block
CUT        = 0.12

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
    thr = max(MIN_AREA, area.max()*0.04)
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
    q = [lo[ax]-1e-3] + q + [hi[ax]+1e-3]            # bound the volume
    q = sorted(set(round(v, 4) for v in q))
    PL.append(np.array(q))
    print(f"  axis {'XYZ'[ax]}: {len(q)-2} planes detected")

nx, ny, nz = [len(q)-1 for q in PL]
print(f"cell complex: {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")

def gidx(ax, v):                                     # world -> support grid
    return int(np.clip(round((v-lo[ax])/G), 0, n[ax]-1))

# ---- how much of each cell face is actually scanned ------------------------
def face_support(ax, pi, b0, b1, c0, c1):
    """Fraction of this face's area carrying scanned surface."""
    o = [a for a in (0, 1, 2) if a != ax]
    i = gidx(ax, PL[ax][pi])
    sl = [slice(None)]*3
    sl[ax] = slice(max(i-1, 0), i+2)
    sl[o[0]] = slice(gidx(o[0], b0), max(gidx(o[0], b1), gidx(o[0], b0)+1))
    sl[o[1]] = slice(gidx(o[1], c0), max(gidx(o[1], c1), gidx(o[1], c0)+1))
    blk = OCC[tuple(sl)]
    if blk.size == 0: return 0.0
    return float(blk.any(axis=ax).mean())

# ---- label cells ----------------------------------------------------------
# A room and a wall interior both hold no points -- the scanner sees neither.
# What separates them is thickness. A cell thin in one axis, with scanned
# surface on BOTH of its bounding faces, is the inside of a wall.
solid = np.zeros((nx, ny, nz), bool)
thin_ax = np.full((nx, ny, nz), -1, np.int8)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            d = [PL[0][i+1]-PL[0][i], PL[1][j+1]-PL[1][j], PL[2][k+1]-PL[2][k]]
            ax = int(np.argmin(d))
            if d[ax] > MAX_WALL: continue
            b, c = [a for a in (0, 1, 2) if a != ax]
            rng = {0: (PL[0][i], PL[0][i+1]), 1: (PL[1][j], PL[1][j+1]),
                   2: (PL[2][k], PL[2][k+1])}
            pi = (i, j, k)[ax]
            s0 = face_support(ax, pi,   *rng[b], *rng[c])
            s1 = face_support(ax, pi+1, *rng[b], *rng[c])
            if min(s0, s1) >= FACE_SUP:
                solid[i, j, k] = True; thin_ax[i, j, k] = ax
print(f"labelled solid: {solid.sum():,} of {nx*ny*nz:,} cells")

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
sz, parts, tris = Gl.write("output/model/cellcomplex.glb")
print(f"{nq:,} boundary quads -> {tris:,} triangles")
print(f"wrote output/model/cellcomplex.glb  {sz/1e6:.2f} MB, {parts} parts")
