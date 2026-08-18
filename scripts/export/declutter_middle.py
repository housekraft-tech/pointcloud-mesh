"""Remove only what stands free in the middle of a room.

The old rule kept a point below 1300 mm only if it sat beside a PROVEN wall
cell, and a cell was proven by a floor-to-ceiling column of points. A balcony
parapet has no such column, so the rule deleted it in full -- 8 wall-like runs,
1.6-2.1 m long, standing to 1270 mm, 0.5 m2 of real wall.

This uses your actual rule instead: loose items are put in the middle of rooms,
deliberately clear of the walls. So the test is CONNECTIVITY, not height.

  everything low is grouped into connected regions in plan
  a region that touches the wall network is structure -- a parapet, a plinth,
    a wall base, a step -- and is kept whole, however short it is
  a region that touches nothing is a loose object, and only those go

A parapet survives because it meets the walls at its ends, which is exactly
what makes it a parapet and not a chair.
"""
import numpy as np, laspy, json
from scipy import ndimage

CELL = 0.05
SEED = 2.00            # material above this is structure -- a person cannot be
                       # here, but a wall, column, beam and arch all are
BAND = 2.00            # everything below this is tested for being free-standing
FLOOR = 0.05           # always kept
REACH = 2              # cells: how close a region must come to count as joined

H = json.load(open("output/fp_walls.json"))['clear_height']
with laspy.open("output/mujammel_aligned_z0.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64); z = P[:, 2]
print(f"{len(P):,} points from the aligned scan (nothing filtered yet)")

lo = P[:, :2].min(0) - CELL
n = np.ceil((P[:, :2].max(0) + CELL - lo)/CELL).astype(int) + 1
k = np.floor((P[:, :2] - lo)/CELL).astype(np.int64)
flat = k[:, 0]*n[1] + k[:, 1]
def occ(m, thr=6):
    return (np.bincount(flat[m], minlength=int(n.prod())).reshape(n)) >= thr

# The wall network seeds ABOVE head height on purpose. Seeding at 1400 mm would
# let a standing person seed the network and then keep themselves, since they
# carry material there too. Nothing human reaches 2000 mm; every wall, column,
# beam and arch does.
walls = occ((z > SEED) & (z < H-0.15))
walls = ndimage.binary_closing(walls, np.ones((3, 3), bool))
near_wall = ndimage.binary_dilation(walls, np.ones((2*REACH+1, 2*REACH+1), bool))
print(f"wall network: {walls.sum():,} cells above {SEED*1000:.0f} mm")

low = occ((z > FLOOR) & (z < BAND))
lab, nb = ndimage.label(low, np.ones((3, 3), bool))
touch = np.zeros(nb+1, bool)
touch[np.unique(lab[near_wall & (lab > 0)])] = True
touch[0] = True                                   # unlabelled = nothing to drop
clutter_cells = low & ~touch[lab]
print(f"low material: {nb:,} regions, {int(touch[1:].sum()):,} joined to the "
      f"wall network and kept, {int((~touch[1:]).sum()):,} standing free")

# report what is going, so nothing large disappears unexamined
cl, nc = ndimage.label(clutter_cells, np.ones((3, 3), bool))
rows = []
for i in range(1, nc+1):
    c = np.argwhere(cl == i)
    if len(c) < 4: continue
    i0, j0 = c.min(0); i1, j1 = c.max(0)
    m = (cl.ravel()[flat] == i) & (z > FLOOR) & (z < BAND)
    rows.append((len(c), (i1-i0+1)*CELL, (j1-j0+1)*CELL,
                 float(z[m].max()) if m.any() else 0.0,
                 i0*CELL+lo[0], j0*CELL+lo[1]))
rows.sort(reverse=True)
print(f"\n{'cells':>6} {'size (m)':>12} {'top mm':>7} {'looks like':<12} position")
nhum = 0
for (cn, di, dj, tz, x, y) in rows[:16]:
    # a person: small footprint, standing between waist and head height
    human = max(di, dj) <= 0.90 and 1.30 <= tz <= 1.95
    if human: nhum += 1
    print(f"{cn:>6} {di:>5.2f} x {dj:<4.2f} {tz*1000:>7.0f} "
          f"{('person' if human else 'object'):<12} ({x:+.2f}, {y:+.2f})")
print(f"{len(rows)} free-standing regions of 4+ cells, "
      f"{nhum} of them person-shaped")

drop = clutter_cells.ravel()[flat] & (z > FLOOR) & (z < BAND)
keep = ~drop
np.save("output/keep_mask.npy", keep)
print(f"\nremoving {drop.sum():,} points ({drop.mean()*100:.2f}%), "
      f"all of them free-standing and below {BAND*1000:.0f} mm")
print(f"keeping  {keep.sum():,}  ->  output/keep_mask.npy")
