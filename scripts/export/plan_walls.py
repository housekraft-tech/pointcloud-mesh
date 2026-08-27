"""Walls from the rooms, the way a plan is actually organised.

Every earlier attempt built walls from FACES: find a plane, find the plane
opposite, call the space between them a wall. That is how the measurement
works, but it is not how a building is arranged, and it shows -- a wall splits
wherever a pier thickens it or another wall crosses it, and a ten-room flat
comes out with thirty-two walls in it.

A room is the primary thing. The free-space carve already knows which air the
scanner saw through, so the rooms are simply the connected pieces of air at
head height, and the masonry is what is left inside the building outline. Cut
that into the fewest rectangles that cover it and you have the wall network:
one rectangle per wall, junctions shared, nothing split by a pier.

Resolution is the carve's 30 mm, which is coarse -- so every rectangle's faces
are snapped back onto the planes the face-settling measured to a few hundredths
of a millimetre. Structure from the rooms, position from the measurement.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from freespace_carve import load_grid


def rectangles(mask, min_cells=8):
    """Greedy maximal rectangles covering a boolean mask."""
    todo = mask.copy()
    out = []
    while todo.any():
        # start from the longest run in the densest row or column
        best = None
        rows = todo.sum(1); cols = todo.sum(0)
        if rows.max() >= cols.max():
            i = int(np.argmax(rows))
            js = np.flatnonzero(todo[i])
            j0 = js[0]
            j1 = j0
            while j1 + 1 < todo.shape[1] and todo[i, j1 + 1]:
                j1 += 1
            i0 = i1 = i
            while i0 - 1 >= 0 and todo[i0 - 1, j0:j1 + 1].all():
                i0 -= 1
            while i1 + 1 < todo.shape[0] and todo[i1 + 1, j0:j1 + 1].all():
                i1 += 1
            best = (i0, i1 + 1, j0, j1 + 1)
        else:
            j = int(np.argmax(cols))
            is_ = np.flatnonzero(todo[:, j])
            i0 = is_[0]; i1 = i0
            while i1 + 1 < todo.shape[0] and todo[i1 + 1, j]:
                i1 += 1
            j0 = j1 = j
            while j0 - 1 >= 0 and todo[i0:i1 + 1, j0 - 1].all():
                j0 -= 1
            while j1 + 1 < todo.shape[1] and todo[i0:i1 + 1, j1 + 1].all():
                j1 += 1
            best = (i0, i1 + 1, j0, j1 + 1)
        i0, i1, j0, j1 = best
        todo[i0:i1, j0:j1] = False
        if (i1 - i0) * (j1 - j0) >= min_cells:
            out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--free", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--measured", required=True, help="boxes_schedule.json, for the planes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-wall", type=float, default=0.06, help="m")
    ap.add_argument("--snap", type=float, default=0.05, help="m: face snap radius")
    a = ap.parse_args()

    g = load_grid(a.free)
    man = json.load(open(a.manifest))
    zf = man["floor_z"]; zc = zf + man["modal_ceiling_height_mm"]/1000.0
    n, lo, G = g["n"], g["lo"], float(g["G"])
    seen = g["seen"].reshape(n)

    # air at head height, and again lower down: furniture blocks one slice but
    # rarely both, and a room is air at any of them
    ks = [int((zf + h - lo[2])/G) for h in (0.9, 1.4, 1.9)]
    ks = [k for k in ks if 0 <= k < n[2]]
    free = np.zeros(n[:2], bool)
    for k in ks:
        free |= seen[:, :, k]
    # Do NOT close the free space. Closing it by even one cell dilates the air
    # INTO the walls: masonry came out 8.1 m2 where 65 m of 200 mm wall needs
    # about 13. The leaks belong to the carve -- a ray slipping through a
    # doorway at a shallow angle marks a few cells inside a wall as seen -- so
    # they are repaired on the MASONRY instead, where closing fills them back
    # in rather than eating the wall.
    inside = ndimage.binary_fill_holes(free)
    inside = ndimage.binary_closing(inside, np.ones((9, 9)))
    inside = ndimage.binary_fill_holes(inside)
    masonry = inside & ~free
    masonry = ndimage.binary_closing(masonry, np.ones((5, 5)))   # heal ray leaks
    masonry = ndimage.binary_opening(masonry, np.ones((3, 3)))   # drop specks

    lab, nr = ndimage.label(free)
    sizes = ndimage.sum(free, lab, range(1, nr+1))*G*G
    rooms = int((sizes > 1.5).sum())
    print(f"rooms found: {rooms} (of {nr} air pockets over 1.5 m2), "
          f"floor area {sizes[sizes > 1.5].sum():.1f} m2")
    print(f"masonry in plan: {masonry.sum()*G*G:.1f} m2")

    rects = rectangles(masonry, min_cells=int(round(0.5*a.min_wall/(G*G))))
    print(f"{len(rects)} rectangles cover it")

    # the measured planes, to snap onto
    meas = json.load(open(a.measured))["parts"]
    planes = {0: [], 1: []}
    for r in meas:
        if r["op"] != "run":
            continue
        ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
        planes[ax] += [r["lo"][ax], r["hi"][ax]]
    for k in planes:
        planes[k] = np.unique(np.round(planes[k], 4))

    def snap(v, ax):
        if len(planes[ax]) == 0:
            return v
        i = int(np.argmin(np.abs(planes[ax] - v)))
        return float(planes[ax][i]) if abs(planes[ax][i] - v) <= a.snap else float(v)

    out, snapped = [], []
    for i, (i0, i1, j0, j1) in enumerate(rects):
        x0 = lo[0] + i0*G; x1 = lo[0] + i1*G
        y0 = lo[1] + j0*G; y1 = lo[1] + j1*G
        w, d = x1-x0, y1-y0
        if min(w, d) < a.min_wall or max(w, d) < 0.30:
            continue
        ax = 0 if w < d else 1
        before = (x0, x1) if ax == 0 else (y0, y1)
        if ax == 0:
            x0n, x1n = snap(x0, 0), snap(x1, 0)
        else:
            y0n, y1n = snap(y0, 1), snap(y1, 1)
        if ax == 0:
            snapped += [abs(x0n-x0), abs(x1n-x1)]; x0, x1 = x0n, x1n
        else:
            snapped += [abs(y0n-y0), abs(y1n-y1)]; y0, y1 = y0n, y1n
        if abs((x1-x0) if ax == 0 else (y1-y0)) < a.min_wall:
            continue
        out.append(dict(name=f"wall_{len(out):02d}", kind="wall", op="run",
                        lo=[round(x0, 5), round(y0, 5), round(zf-0.02, 5)],
                        hi=[round(x1, 5), round(y1, 5), round(zc, 5)]))
    if snapped:
        s = np.array(snapped)*1000
        print(f"faces snapped onto the measured planes: median {np.median(s):.1f} mm, "
              f"90th {np.percentile(s, 90):.1f}, worst {s.max():.1f}")
    # carry the openings and the slabs across unchanged
    keep = [r for r in meas if r["op"] in ("opening", "slab", "column")]
    json.dump(dict(parts=out + keep), open(a.out, "w"))
    L = sum(max(r["hi"][0]-r["lo"][0], r["hi"][1]-r["lo"][1]) for r in out)
    print(f"{len(out)} walls, {L:.1f} m of wall, plus {len(keep)} openings/slabs -> {a.out}")


if __name__ == "__main__":
    main()
