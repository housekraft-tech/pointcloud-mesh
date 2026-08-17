"""A watertight surface mesh of the scan itself, to overlay on the model.

Poisson reconstruction needs open3d, and this environment has no pip, so this
builds the equivalent by the other standard route: occupancy volume, then the
isosurface between filled and empty voxels. It is blockier than Poisson at the
voxel size but it is the SCAN, not an interpretation of it -- which is what an
overlay is for. Coplanar faces are merged into the largest rectangles that fit
(greedy meshing), so a flat wall costs a handful of quads instead of thousands.

The top of the ceiling is cut, same as the point cloud, so you can see in.
"""
import sys, json, numpy as np, laspy
from scipy import ndimage
SC = "/private/tmp/claude-501/-Users-vallerikoushik-Documents-pointcloud-latest-pointcloud-mesh/437552a3-658f-4215-b1c0-d1640f83e863/scratchpad"
sys.path.insert(0, SC)
from glb import GLB

VOX = 0.06
MINP = 2
CUT = 0.12

H = json.load(open("output/fp_walls.json"))['clear_height']
with laspy.open("output/mujammel_structural_v6.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
P = P[P[:, 2] < H - CUT]
lo = P.min(0) - VOX
n = np.ceil((P.max(0) + VOX - lo)/VOX).astype(int) + 1
idx = np.floor((P - lo)/VOX).astype(np.int64)
flat = (idx[:, 0]*n[1] + idx[:, 1])*n[2] + idx[:, 2]
V = (np.bincount(flat, minlength=int(n.prod())) >= MINP).reshape(n)
print(f"volume {n[0]}x{n[1]}x{n[2]} @ {VOX*1000:.0f} mm   {V.sum():,} filled voxels")
V = ndimage.binary_closing(V, np.ones((3, 3, 3), bool))
print(f"after closing pinholes: {V.sum():,}")

def greedy(mask):
    """Largest-rectangle decomposition of a 2D boolean slice."""
    m = mask.copy(); out = []
    R, C = m.shape
    for i in range(R):
        row = m[i]
        if not row.any(): continue
        j = 0
        while j < C:
            if not m[i, j]: j += 1; continue
            w = 1
            while j+w < C and m[i, j+w]: w += 1
            h = 1
            while i+h < R and m[i+h, j:j+w].all(): h += 1
            m[i:i+h, j:j+w] = False
            out.append((i, j, h, w)); j += w
    return out

G = GLB()
# three.js frame: Y up, and scan Y runs backwards -> (x, z, -y)
def W(ix, iy, iz): return (lo[0]+ix*VOX, lo[2]+iz*VOX, -(lo[1]+iy*VOX))
nq = 0
for axis in (0, 1, 2):
    for sgn in (-1, +1):
        sh = np.zeros_like(V)
        if sgn > 0:
            sl_dst = [slice(None)]*3; sl_src = [slice(None)]*3
            sl_dst[axis] = slice(0, n[axis]-1); sl_src[axis] = slice(1, n[axis])
        else:
            sl_dst = [slice(None)]*3; sl_src = [slice(None)]*3
            sl_dst[axis] = slice(1, n[axis]); sl_src[axis] = slice(0, n[axis]-1)
        sh[tuple(sl_dst)] = V[tuple(sl_src)]
        exp = V & ~sh
        quads = []
        for k in range(n[axis]):
            sl = [slice(None)]*3; sl[axis] = k
            mask = exp[tuple(sl)]
            if not mask.any(): continue
            plane = k + (1 if sgn > 0 else 0)
            for (i, j, h, w) in greedy(mask):
                if axis == 0:      # slice is (y,z)
                    a = [(plane, i, j), (plane, i+h, j), (plane, i+h, j+w), (plane, i, j+w)]
                elif axis == 1:    # slice is (x,z)
                    a = [(i, plane, j), (i, plane, j+w), (i+h, plane, j+w), (i+h, plane, j)]
                else:              # slice is (x,y)
                    a = [(i, j, plane), (i+h, j, plane), (i+h, j+w, plane), (i, j+w, plane)]
                quads.append([W(*v) for v in a])
        if not quads: continue
        nrm = [0.0, 0.0, 0.0]
        # scan axes -> three.js axes: x->x, y->-z, z->y
        nrm[{0: 0, 1: 2, 2: 1}[axis]] = float(sgn) * (-1.0 if axis == 1 else 1.0)
        G.add_quads(f"SCAN_{'XYZ'[axis]}{'+' if sgn>0 else '-'}", quads, tuple(nrm))
        nq += len(quads)
sz, parts, tris = G.write("output/model/scan_surface.glb")
print(f"{nq:,} merged quads -> {tris:,} triangles")
print(f"wrote output/model/scan_surface.glb  {sz/1e6:.2f} MB, {parts} parts")
