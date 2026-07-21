"""house_major_walls.py
-------------------
STAGE 1 of the house rebuild: the major (structural) walls, from horizontal
slices of the mesh.

Why slices rather than the per-run plane fits: a plane fit describes whatever
triangles were grouped together, so a corner blob or a two-run object produces
a wall that is not there. A horizontal slice asks a different and much harder-
to-fool question -- "is there solid material at this (x, y), at this height?"
-- and a MAJOR wall is the material that is still there at every height.

That persistence test is what separates structure from contents. A wardrobe,
a counter and a bed all appear in the low slices and vanish above; a wall
appears in all of them. Nothing about furniture has to be recognised for this
to work.

Output is axis-aligned wall centrelines with real extents, in the building's
own grid frame, ready for Stage 4 to extrude.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\house_major_walls.py \\
      <detailed_modular.obj> <features.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.viz_deviation_plan import parse_groups, wall_grid_angle

CELL = 0.03           # m: slice raster pitch
BANDS = 6             # horizontal slices between floor and ceiling
BAND_H = 0.10         # m: thickness of each slice
MARGIN = 0.25         # m: skip this much above the floor (skirting)
TOP_MARGIN = 0.55     # m: and this much below the ceiling -- the wall crop
                      # thins out at the slab and that band fails everything
PERSIST = 0.60        # fraction of bands a cell must be solid in to be structure
MIN_LEN = 0.60        # m: shorter than this is not a wall run
MIN_FILL = 0.55       # occupied fraction along a candidate run
BAND_FILL = 0.60      # a run counts as present in a band when this much
                      # of its LENGTH has material in that band
MERGE_GAP = 0.35      # m: gaps this small along a line are doorless jogs, not ends
CLOSE_PX = 7          # cells (~0.21 m): closes a slice of a surface mesh into a
                      # continuous footprint and unites a wall's two faces


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def slice_masks(V, z0, z1, lo, nx, ny, R, pivot):
    """Occupancy raster per height band, in the grid frame."""
    masks = []
    # Stop well below the ceiling: the wall crop thins out near the slab, and
    # that sparse top band was dragging every wall below the persistence bar.
    zs = np.linspace(z0 + MARGIN, z1 - TOP_MARGIN, BANDS)
    for zc in zs:
        sel = V[(V[:, 2] >= zc - BAND_H / 2) & (V[:, 2] <= zc + BAND_H / 2)]
        g = np.zeros((ny, nx), np.uint8)
        if len(sel):
            P = (sel[:, :2] - pivot) @ R.T + pivot
            ij = ((P - lo) / CELL).astype(np.int32)
            ok = ((ij[:, 0] >= 0) & (ij[:, 0] < nx) &
                  (ij[:, 1] >= 0) & (ij[:, 1] < ny))
            ij = ij[ok]
            g[ij[:, 1], ij[:, 0]] = 1
        # A slice of a SURFACE mesh is a dotted line: vertices do not land on
        # the same (x, y) at different heights, so a raw cell-by-cell
        # persistence test finds almost nothing (it returned 4 walls). Close
        # each band into a continuous footprint first -- the kernel is about
        # one wall thickness, which also merges the two faces of a wall into a
        # single band -- and only then ask what persists.
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                             np.ones((CLOSE_PX, CLOSE_PX), np.uint8))
        masks.append(g)
        log(f"  band z={zc:5.2f} m: {int(g.sum()):6d} cells")
    return np.array(masks), zs


def extract_runs(persist, lo):
    """Wall runs by directional opening + connected components.

    Opening with a long horizontal element keeps only material that continues
    horizontally for at least MIN_LEN, which IS a wall running along x; the
    vertical element does the same for y. Each surviving blob is then one run,
    and its bounding box gives length and thickness directly. The earlier
    row-by-row scan fragmented walls into pieces and recovered only 28 m of the
    building's ~110 m.
    """
    n_px = max(3, int(MIN_LEN / CELL))
    out = []
    for axis, k in (("x", np.ones((1, n_px), np.uint8)),
                    ("y", np.ones((n_px, 1), np.uint8))):
        m = cv2.morphologyEx(persist, cv2.MORPH_OPEN, k)
        ncc, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, ncc):
            x, y, w, h, area = stats[i]
            if axis == "x":
                length, thick = w * CELL, h * CELL
                p0 = (lo[0] + x * CELL, lo[1] + (y + h / 2) * CELL)
                p1 = (lo[0] + (x + w) * CELL, lo[1] + (y + h / 2) * CELL)
            else:
                length, thick = h * CELL, w * CELL
                p0 = (lo[0] + (x + w / 2) * CELL, lo[1] + y * CELL)
                p1 = (lo[0] + (x + w / 2) * CELL, lo[1] + (y + h) * CELL)
            if length < MIN_LEN:
                continue
            # a blob as thick as it is long is a junction, not a run
            if thick > max(0.45, 0.5 * length):
                continue
            if area * CELL ** 2 < 0.5 * MIN_LEN * 0.06:
                continue
            out.append(dict(p0=[round(float(v), 3) for v in p0],
                            p1=[round(float(v), 3) for v in p1], axis=axis,
                            length_m=round(float(length), 3),
                            thickness_m=round(float(np.clip(thick, 0.06, 0.40)), 3)))
    return out


def keep_persistent(segs, masks, lo, zs):
    """Apply the structure test PER WALL, not per cell.

    Mesh coverage is patchy band to band (10k cells each, 18k in union), so
    demanding that every individual cell survive in 60% of bands let one hole
    delete a whole wall -- it recovered 30 m of the building's ~110 m. What
    actually defines structure is that THE RUN is there at every height, so
    the footprint comes from the union and each run is then scored by how many
    bands it appears in. Furniture still fails: a counter or a wardrobe simply
    is not present in the upper bands.
    """
    out = []
    for s in segs:
        p0 = ((np.array(s["p0"]) - lo) / CELL).astype(int)
        p1 = ((np.array(s["p1"]) - lo) / CELL).astype(int)
        t = max(1, int(s["thickness_m"] / CELL / 2))
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        x0, x1 = max(0, x0 - t), x1 + t + 1
        y0, y1 = max(0, y0 - t), y1 + t + 1
        hits = 0
        for m in masks:
            sub = m[y0:y1, x0:x1]
            if not sub.size:
                continue
            # Coverage ALONG THE RUN, not area density over the box. A wall
            # fills one cell-column of a 7-wide box, so a perfectly solid
            # 6.7 m wall scored 0.14 by area and was thrown away as 0/6 bands.
            # Asking what fraction of its LENGTH has material scores it ~1.0.
            cover = (sub.any(axis=1) if s["axis"] == "y"
                     else sub.any(axis=0)).mean()
            if cover >= BAND_FILL:
                hits += 1
        s["bands"] = f"{hits}/{len(masks)}"
        # Rather than discard a run that fails the full-height test, measure
        # how high it actually goes. The runs failing here are balcony
        # PARAPETS -- real building fabric that genuinely stops at waist
        # height -- and a house rebuilt without them is missing its balustrades.
        present = [i for i, m in enumerate(masks)
                   if (m[y0:y1, x0:x1].any(axis=1) if s["axis"] == "y"
                       else m[y0:y1, x0:x1].any(axis=0)).mean() >= BAND_FILL] \
            if masks[0][y0:y1, x0:x1].size else []
        s["top_z"] = round(float(zs[max(present)] + BAND_H / 2), 3) if present \
            else None
        s["kind"] = "wall" if hits / len(masks) >= PERSIST else "parapet"
        if s["kind"] == "wall" or (present and len(present) >= 2):
            out.append(s)
    return out


def runs_along(mask, axis, lo):
    """Wall runs on one grid axis.

    axis=0 -> walls running along x (horizontal lines, found row by row)
    axis=1 -> walls running along y
    A wall is a line index whose occupancy stands out, and along it the
    contiguous stretches of material are the runs.
    """
    m = mask if axis == 0 else mask.T
    out = []
    n_line, n_along = m.shape
    minc = int(MIN_LEN / CELL)
    for i in range(n_line):
        row = m[i]
        if row.sum() < minc:
            continue
        # contiguous stretches, tolerating small gaps (a jog, not an end)
        idx = np.flatnonzero(row)
        if not len(idx):
            continue
        splits = np.flatnonzero(np.diff(idx) > int(MERGE_GAP / CELL)) + 1
        for seg in np.split(idx, splits):
            if len(seg) < minc:
                continue
            a0, a1 = seg[0], seg[-1]
            fill = row[a0:a1 + 1].mean()
            if fill < MIN_FILL:
                continue
            out.append((i, a0, a1, float(fill), len(seg)))
    return out


def merge_parallel(runs, axis, lo):
    """Adjacent line indices describing one wall -> a single run with thickness."""
    runs = sorted(runs)
    used = [False] * len(runs)
    merged = []
    for i, r in enumerate(runs):
        if used[i]:
            continue
        grp = [r]; used[i] = True
        for j in range(i + 1, len(runs)):
            if used[j]:
                continue
            s = runs[j]
            if s[0] - grp[-1][0] > 1:
                if s[0] - grp[-1][0] > int(0.40 / CELL):
                    break
                continue
            # overlapping extent?
            if min(grp[-1][2], s[2]) - max(grp[-1][1], s[1]) > 0:
                grp.append(s); used[j] = True
        lines = [g[0] for g in grp]
        a0 = int(np.median([g[1] for g in grp]))
        a1 = int(np.median([g[2] for g in grp]))
        thick = (max(lines) - min(lines) + 1) * CELL
        centre = (min(lines) + max(lines)) / 2.0
        length = (a1 - a0 + 1) * CELL
        if length < MIN_LEN:
            continue
        if axis == 0:      # line index is y, extent is x
            p0 = (lo[0] + a0 * CELL, lo[1] + centre * CELL)
            p1 = (lo[0] + a1 * CELL, lo[1] + centre * CELL)
        else:              # line index is x, extent is y
            p0 = (lo[0] + centre * CELL, lo[1] + a0 * CELL)
            p1 = (lo[0] + centre * CELL, lo[1] + a1 * CELL)
        merged.append(dict(p0=[round(v, 3) for v in p0],
                           p1=[round(v, 3) for v in p1],
                           axis="x" if axis == 0 else "y",
                           length_m=round(length, 3),
                           thickness_m=round(float(np.clip(thick, 0.06, 0.40)), 3)))
    return merged


def main(obj_path, features_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(features_path))
    z_floor = float(np.median([r["z_floor"] for r in feats["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in feats["rooms"]]))
    log(f"floor z={z_floor:.3f}  ceiling z={z_ceil:.3f}  "
        f"({(z_ceil-z_floor)*1000:.0f} mm)")

    walls = parse_groups(obj_path, "wall")
    cols = parse_groups(obj_path, "column")
    log(f"{len(walls)} wall objects, {len(cols)} columns")
    V = np.vstack(list(walls.values()) + list(cols.values()))
    log(f"{len(V):,} vertices to slice")

    theta = wall_grid_angle(walls)
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    pivot = np.vstack(list(walls.values()))[:, :2].mean(0)
    G = (V[:, :2] - pivot) @ R.T + pivot
    lo = G.min(0) - 0.2
    hi = G.max(0) + 0.2
    nx = int((hi[0] - lo[0]) / CELL) + 1
    ny = int((hi[1] - lo[1]) / CELL) + 1
    log(f"grid frame {np.degrees(-theta):+.2f} deg, raster {nx}x{ny} @ {CELL} m")

    masks, zs = slice_masks(V, z_floor, z_ceil, lo, nx, ny, R, pivot)
    # footprint = union of the bands (holes in one band must not delete a wall)
    persist = (masks.sum(0) >= 1).astype(np.uint8)
    persist = cv2.morphologyEx(persist, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    persist = cv2.morphologyEx(persist, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    log(f"union footprint: {int(persist.sum()):,} cells "
        f"({persist.sum()*CELL**2:.1f} m2)")

    segs = extract_runs(persist, lo)
    log(f"{len(segs)} candidate runs from the footprint")
    segs = keep_persistent(segs, masks, lo, zs)
    segs.sort(key=lambda s: -s["length_m"])
    nw=sum(1 for s in segs if s["kind"]=="wall")
    npar=sum(1 for s in segs if s["kind"]=="parapet")
    log(f"{nw} full-height walls + {npar} parapets / low walls")
    tot = sum(s["length_m"] for s in segs)
    log(f"total wall length {tot:.1f} m, thickness median "
        f"{np.median([s['thickness_m'] for s in segs])*1000:.0f} mm")
    for s in segs[:14]:
        log(f"  {s['kind']:8} {s['axis']}-run  L {s['length_m']:5.2f} m  "
            f"t {s['thickness_m']*1000:3.0f} mm  top {s['top_z']}  bands {s['bands']}")

    json.dump(dict(grid_angle_deg=round(float(np.degrees(-theta)), 3),
                   pivot=[round(float(v), 4) for v in pivot],
                   z_floor=round(z_floor, 4), z_ceiling=round(z_ceil, 4),
                   cell_m=CELL, walls=segs),
              open(out / "major_walls.json", "w"), indent=1)
    log(f"wrote {out/'major_walls.json'}")

    # quick look
    vis = np.full((ny, nx, 3), 255, np.uint8)
    vis[persist > 0] = (205, 215, 225)
    for s in segs:
        p0 = ((np.array(s["p0"]) - lo) / CELL).astype(int)
        p1 = ((np.array(s["p1"]) - lo) / CELL).astype(int)
        cv2.line(vis, tuple(p0), tuple(p1), (30, 30, 30),
                 max(1, int(s["thickness_m"] / CELL)))
    cv2.imwrite(str(out / "major_walls.png"), cv2.flip(vis, 0))
    log(f"wrote {out/'major_walls.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
