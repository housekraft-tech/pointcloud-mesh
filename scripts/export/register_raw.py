"""Find the rigid transform from the raw scanner frame to the aligned frame.

The trajectory comes out of mujammelexport.las, but every other output lives in
the aligned frame of mujammel_aligned_z0.las. Both are the same flat, so the
transform is a yaw, a translation and a height shift.

Yaw is found by the sharpness of the axis histograms: a rectilinear building
projected along its own axes gives tall narrow spikes at each wall, and any
other angle smears them. Translation then follows from a 2D cross-correlation
of the two occupancy rasters.
"""
import numpy as np, laspy, json
from scipy import ndimage
from numpy.fft import rfft2, irfft2

CELL = 0.05
J = json.load(open("output/fp_walls.json")); H = J['clear_height']

def load(path, n=4_000_000):
    with laspy.open(path) as r:
        P = []
        for pts in r.chunk_iterator(4_000_000):
            P.append(np.column_stack([pts.x, pts.y, pts.z]).astype(np.float64))
    P = np.concatenate(P)
    rng = np.random.default_rng(0)
    if len(P) > n: P = P[rng.choice(len(P), n, replace=False)]
    return P

A = load("output/mujammel_aligned_z0.las")           # target frame
R = load("mujammelexport.las")                       # source frame
print(f"aligned {len(A):,} pts, raw {len(R):,} pts")

# height: the floor is the densest horizontal slab
def floor_z(P):
    h, e = np.histogram(P[:, 2], bins=1200)
    return float(e[np.argmax(h)])
fzA, fzR = floor_z(A), floor_z(R)
print(f"floor: aligned z={fzA:+.3f}  raw z={fzR:+.3f}  -> dz={fzA-fzR:+.3f}")
R = R[(R[:, 2] > fzR+0.30) & (R[:, 2] < fzR+H-0.30)]
A2 = A[(A[:, 2] > fzA+0.30) & (A[:, 2] < fzA+H-0.30)]
print(f"storey slice: aligned {len(A2):,}, raw {len(R):,}")

def sharpness(XY):
    """How spiky the two axis histograms are -- max when walls are axis-aligned."""
    s = 0.0
    for k in (0, 1):
        h, _ = np.histogram(XY[:, k], bins=np.arange(XY[:, k].min(),
                                                     XY[:, k].max()+CELL, CELL))
        h = h/max(h.sum(), 1)
        s += (h**2).sum()
    return s

best = None
for deg in np.arange(0, 90, 0.25):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    XY = np.column_stack([R[:, 0]*c - R[:, 1]*s, R[:, 0]*s + R[:, 1]*c])
    v = sharpness(XY)
    if best is None or v > best[1]: best = (deg, v)
yaw = best[0]
# refine
for deg in np.arange(yaw-0.25, yaw+0.25, 0.02):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    XY = np.column_stack([R[:, 0]*c - R[:, 1]*s, R[:, 0]*s + R[:, 1]*c])
    v = sharpness(XY)
    if v > best[1]: best = (deg, v)
yaw = best[0]
print(f"yaw = {yaw:.2f} deg  (histogram sharpness {best[1]:.5f})")

# A rectilinear building looks identical under 90 degree turns, so the
# sharpness peak fixes the angle only modulo 90. Try all four and let the
# overlap decide -- picking the wrong quadrant was giving 34% IoU.

def raster(XY, lo, n):
    ij = np.floor((XY-lo)/CELL).astype(int)
    m = (ij[:, 0] >= 0) & (ij[:, 0] < n[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < n[1])
    g = np.zeros(n, np.float32)
    np.add.at(g, (ij[m, 0], ij[m, 1]), 1.0)
    return (g > 3).astype(np.float32)

loA = A2[:, :2].min(0); nA = np.ceil((A2[:, :2].max(0)-loA)/CELL).astype(int)+1
gA_r = raster(A2[:, :2], loA, nA); gA_b = gA_r > 0

best4 = None
for quad in range(4):
    ang = yaw + 90*quad
    a = np.radians(ang); c, s = np.cos(a), np.sin(a)
    RXq = np.column_stack([R[:, 0]*c - R[:, 1]*s, R[:, 0]*s + R[:, 1]*c])
    loR = RXq.min(0); nR = np.ceil((RXq.max(0)-loR)/CELL).astype(int)+1
    N = np.maximum(nA, nR)+8
    gA = np.zeros(N, np.float32); gR = np.zeros(N, np.float32)
    gA[:nA[0], :nA[1]] = gA_r
    gR[:nR[0], :nR[1]] = raster(RXq, loR, nR)
    cc = irfft2(rfft2(gA)*np.conj(rfft2(gR)), s=N)
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    sh = np.array([pk[0] if pk[0] < N[0]//2 else pk[0]-N[0],
                   pk[1] if pk[1] < N[1]//2 else pk[1]-N[1]])*CELL
    t = (loA - loR) + sh
    RT = RXq + t
    ij = np.floor((RT-loA)/CELL).astype(int)
    m = (ij[:, 0] >= 0) & (ij[:, 0] < nA[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < nA[1])
    g = np.zeros(nA, np.float32); np.add.at(g, (ij[m, 0], ij[m, 1]), 1.0)
    gT = g > 3
    iou = (gT & gA_b).sum()/max((gT | gA_b).sum(), 1)
    rec = (gT & gA_b).sum()/max(gA_b.sum(), 1)
    print(f"  yaw {ang:7.2f} deg -> IoU {iou*100:5.1f}%  recall {rec*100:5.1f}%")
    if best4 is None or iou > best4[0]: best4 = (iou, ang, t, gT)
iou, yaw, (tx, ty), gT = best4
print(f"chosen yaw {yaw:.2f} deg, dx={tx:+.3f} dy={ty:+.3f} dz={fzA-fzR:+.3f}")

T = dict(yaw_deg=float(yaw), tx=float(tx), ty=float(ty), dz=float(fzA-fzR))
json.dump(T, open("output/raw_to_aligned.json", "w"), indent=1)

inter = (gT & gA_b).sum(); union = (gT | gA_b).sum(); gA2 = gA_b
print(f"overlap after transform: IoU {inter/max(union,1)*100:.1f}%, "
      f"{inter:,} of {gA2.sum():,} aligned cells matched "
      f"({inter/max(gA2.sum(),1)*100:.1f}%)")
print("wrote output/raw_to_aligned.json")
