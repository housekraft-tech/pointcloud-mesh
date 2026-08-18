"""A surface mesh of the scan itself -- cleaned, and snapped off the lattice.

Poisson proper needs open3d, which cannot be installed here (no pip), so this
takes the other standard route: occupancy volume, then the isosurface between
filled and empty. Three things make it accurate rather than merely blocky:

  CLEAN    a voxel needs real support, and a blob needs to be part of something.
           Isolated specks -- a few stray returns hanging in mid-air -- are
           dropped by size, so what is left is surface the scanner actually saw.

  SNAP     a voxel isosurface puts every face on the lattice, so a wall lands on
           the nearest 30 mm plane no matter where it really is. Each face is
           moved to the MEAN POSITION OF THE POINTS BEHIND IT instead, which is
           sub-voxel and independent of the grid. This is what makes parallel
           faces come out parallel and a wall come out its true thickness.

  MERGE    coplanar faces are combined into the largest rectangles that fit, so
           a flat wall costs a handful of quads rather than thousands.

Edges stay axis-aligned steps at the voxel size: that is inherent to any voxel
isosurface, and the fix for it is not a finer grid but a different algorithm
(dual contouring, or fitting planes and intersecting them, which is what the
modular model does).
"""
import sys, json, numpy as np, laspy
from scipy import ndimage
SC = "scripts/export"
sys.path.insert(0, SC)
from glb import GLB

VOX = 0.030            # lattice pitch
MINP = 4               # points before a voxel counts as surface
MIN_BLOB = 60          # voxels before a connected blob counts as structure
CUT = 0.12             # ceiling slab removed from the top down

H = json.load(open("output/fp_walls.json"))['clear_height']
with laspy.open("output/mujammel_structural_v6.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
P = P[P[:, 2] < H - CUT]
lo = P.min(0) - VOX
n = np.ceil((P.max(0) + VOX - lo)/VOX).astype(int) + 1
idx = np.floor((P - lo)/VOX).astype(np.int64)
flat = (idx[:, 0]*n[1] + idx[:, 1])*n[2] + idx[:, 2]
cnt = np.bincount(flat, minlength=int(n.prod()))
V = (cnt >= MINP).reshape(n)
print(f"volume {n[0]}x{n[1]}x{n[2]} @ {VOX*1000:.0f} mm   "
      f"{V.sum():,} voxels with >={MINP} points")

# ---- clean: drop specks that are not part of anything ----
lab, nb = ndimage.label(V, np.ones((3, 3, 3), bool))
sizes = np.bincount(lab.ravel()); sizes[0] = 0
keep = np.zeros(len(sizes), bool); keep[sizes >= MIN_BLOB] = True
dropped = int(V.sum() - keep[lab].sum())
V = keep[lab]
print(f"cleaned: {nb:,} blobs, dropped {dropped:,} voxels in blobs under "
      f"{MIN_BLOB} ({dropped/max(dropped+V.sum(),1)*100:.1f}% of surface)")
V = ndimage.binary_closing(V, np.ones((3, 3, 3), bool))
print(f"after closing pinholes: {V.sum():,} voxels")

# ---- sub-voxel face positions: the mean point position inside each voxel ----
# Computed only for occupied voxels, so the whole volume is never held as floats.
occ_ids = np.flatnonzero(V.ravel())
pos = np.searchsorted(occ_ids, flat)
np.clip(pos, 0, len(occ_ids)-1, out=pos)
good = occ_ids[pos] == flat
pg = pos[good]
w = np.bincount(pg, minlength=len(occ_ids)).astype(np.float64)
w[w == 0] = 1.0
MEAN = np.stack([np.bincount(pg, weights=P[good, k], minlength=len(occ_ids))/w
                 for k in range(3)], axis=1)
lookup = np.full(int(n.prod()), -1, np.int64)
lookup[occ_ids] = np.arange(len(occ_ids))
LK = lookup.reshape(n)
print(f"sub-voxel snapping: {len(occ_ids):,} voxel means computed")

def greedy(mask):
    """Largest-rectangle decomposition of a 2D boolean slice."""
    m = mask.copy(); out = []
    R, C = m.shape
    for i in range(R):
        if not m[i].any(): continue
        j = 0
        while j < C:
            if not m[i, j]: j += 1; continue
            wd = 1
            while j+wd < C and m[i, j+wd]: wd += 1
            h = 1
            while i+h < R and m[i+h, j:j+wd].all(): h += 1
            m[i:i+h, j:j+wd] = False
            out.append((i, j, h, wd)); j += wd
    return out

G = GLB()
def Wpt(ix, iy, iz): return (lo[0]+ix*VOX, lo[2]+iz*VOX, -(lo[1]+iy*VOX))
nq = 0; snapped = 0; resid = []
for axis in (0, 1, 2):
    for sgn in (-1, +1):
        sh = np.zeros_like(V)
        d, s = [slice(None)]*3, [slice(None)]*3
        if sgn > 0: d[axis], s[axis] = slice(0, n[axis]-1), slice(1, n[axis])
        else:       d[axis], s[axis] = slice(1, n[axis]), slice(0, n[axis]-1)
        sh[tuple(d)] = V[tuple(s)]
        exp = V & ~sh
        quads = []
        for k in range(n[axis]):
            sl = [slice(None)]*3; sl[axis] = k
            mask = exp[tuple(sl)]
            if not mask.any(): continue
            lk = LK[tuple(sl)]
            for (i, j, h, wd) in greedy(mask):
                # snap this face to the points behind it, not to the lattice
                ids = lk[i:i+h, j:j+wd].ravel()
                ids = ids[ids >= 0]
                lat = lo[axis] + (k + (1 if sgn > 0 else 0))*VOX
                if len(ids):
                    mu = float(MEAN[ids, axis].mean())
                    # the face is half a voxel off the points' mean, on the
                    # outward side; clamp so a bad cell cannot fling it away
                    cand = mu + sgn*VOX*0.5
                    if abs(cand-lat) <= VOX: 
                        resid.append(abs(cand-lat)); lat = cand; snapped += 1
                pv = (lat - lo[axis])/VOX
                if axis == 0:
                    a = [(pv, i, j), (pv, i+h, j), (pv, i+h, j+wd), (pv, i, j+wd)]
                elif axis == 1:
                    a = [(i, pv, j), (i, pv, j+wd), (i+h, pv, j+wd), (i+h, pv, j)]
                else:
                    a = [(i, j, pv), (i+h, j, pv), (i+h, j+wd, pv), (i, j+wd, pv)]
                quads.append([Wpt(*v) for v in a])
        if not quads: continue
        nrm = [0.0, 0.0, 0.0]
        nrm[{0: 0, 1: 2, 2: 1}[axis]] = float(sgn) * (-1.0 if axis == 1 else 1.0)
        G.add_quads(f"SCAN_{'XYZ'[axis]}{'+' if sgn>0 else '-'}", quads, tuple(nrm))
        nq += len(quads)
sz, parts, tris = G.write("output/model/scan_surface.glb")
print(f"{nq:,} merged quads -> {tris:,} triangles; {snapped:,} faces snapped "
      f"off the lattice by a median {np.median(resid)*1000:.1f} mm")
print(f"wrote output/model/scan_surface.glb  {sz/1e6:.2f} MB, {parts} parts")
