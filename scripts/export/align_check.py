"""How well does the model sit on the scan? Two numbers, both from raw points.

  OFFSET   for every model wall face, the distance from that face to the
           scanned surface it represents. This is the "is it aligned" number.
  COVERAGE what fraction of the scan's wall cells lie under some model wall,
           and what fraction of model wall lies over scanned wall. This is the
           "is anything missing or invented" number.

Faces, not centrelines: the scan only ever sees faces, so comparing centrelines
would bake in half a wall thickness before measuring anything.
"""
import json, numpy as np, laspy
from scipy import ndimage

CELL = 0.05
M = json.load(open("output/model/shell_fp.json"))
H = M['clear_height_mm']/1000
mins = np.array(json.load(open("output/fp_walls.json"))['mins'])
with laspy.open("output/mujammel_structural_v6.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64); z = P[:, 2]
XY = P[:, :2] - mins
nx = int(np.ceil((XY[:, 0].max()+CELL)/CELL)); ny = int(np.ceil((XY[:, 1].max()+CELL)/CELL))
ij = np.floor(XY/CELL).astype(np.int64)
ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
flat = ij[:, 0]*ny + ij[:, 1]
body = (np.bincount(flat[ok & (z > 1.10) & (z < 1.85)], minlength=nx*ny).reshape(nx, ny)) >= 12
scan = ndimage.binary_closing(body, np.ones((3, 3), bool))

print(f"{'#':>3} {'ax':>2} {'len':>6} {'thk':>4} {'face A':>8} {'face B':>8} {'pts':>8}")
offs = []; unver = []
model = np.zeros((nx, ny), bool)
for w in M['walls']:
    ax = w['axis']
    c = w['centre_mm']/1000 - mins[ax]
    a = w['lo_mm']/1000 - mins[1-ax]; b = w['hi_mm']/1000 - mins[1-ax]
    t = w['thickness_mm']/1000
    row = []
    for f in (c-t/2, c+t/2):
        m = ok & (np.abs(XY[:, ax]-f) < 0.12) & (XY[:, 1-ax] > a+0.05) & \
            (XY[:, 1-ax] < b-0.05) & (z > 0.40) & (z < H-0.20)
        if m.sum() < 200: row.append(None); continue
        d = XY[m, ax]-f
        # the scanned surface nearest this face: the mode of the offset
        h, e = np.histogram(d, bins=240, range=(-0.12, 0.12))   # 1 mm bins
        row.append(float((e[h.argmax()]+e[h.argmax()+1])/2))
    for v in row:
        # Only a face the scan actually saw can be checked. On a single-faced
        # wall the far face is inferred from the median thickness and there is
        # no surface behind it to compare against, so scoring it would be
        # scoring an assumption.
        if v is not None:
            (offs if w.get('both_faces') else unver).append(abs(v)*1000)
    npts = 0
    i0 = int(max((c-t/2)/CELL, 0)); i1 = int((c+t/2)/CELL)+1
    j0 = int(max(a/CELL, 0)); j1 = int(b/CELL)+1
    if ax == 0: model[i0:i1, j0:j1] = True
    else:       model[j0:j1, i0:i1] = True
    print(f"{w['id']:>3} {ax:>2} {w['length_mm']:>5}m {w['thickness_mm']:>4} "
          f"{('-' if row[0] is None else f'{row[0]*1000:+.0f}mm'):>8} "
          f"{('-' if row[1] is None else f'{row[1]*1000:+.0f}mm'):>8}")

offs = np.array(offs); unver = np.array(unver)
print(f"\nFACE OFFSET, model face -> scanned surface")
print(f"  {len(offs)} faces on walls seen from BOTH sides -- these are checkable")
print(f"  median {np.median(offs):.0f} mm | mean {offs.mean():.0f} mm | "
      f"90th pct {np.percentile(offs,90):.0f} mm | worst {offs.max():.0f} mm")
for lim in (5, 10, 20, 50):
    print(f"    within {lim:>3} mm: {(offs<=lim).mean()*100:>5.1f}%")
print(f"  {len(unver)} faces on single-faced walls: thickness assumed, not measured")

md = ndimage.binary_dilation(model, np.ones((3, 3), bool))
sd = ndimage.binary_dilation(scan, np.ones((3, 3), bool))
print(f"\nCOVERAGE (50 mm cells, 1-cell tolerance)")
print(f"  scan wall covered by model : {(scan & md).sum()/scan.sum()*100:>5.1f}%"
      f"   ({scan.sum():,} scan cells)")
print(f"  model wall backed by scan  : {(model & sd).sum()/model.sum()*100:>5.1f}%"
      f"   ({model.sum():,} model cells)")
miss = scan & ~md
lab, n = ndimage.label(miss, np.ones((3, 3), bool))
big = [(lab == i).sum() for i in range(1, n+1)]
big = sorted([b for b in big if b >= 20], reverse=True)
print(f"  scanned wall the model MISSES: {miss.sum():,} cells "
      f"({miss.sum()*CELL*CELL:.1f} m2) in {len(big)} runs of 20+ cells")
if big: print(f"    biggest missing runs (cells): {big[:10]}")
