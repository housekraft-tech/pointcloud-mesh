"""Drop duplicate rooms, recover floor that belongs to none.

Two defects in CAGE's raw output propagate into every later stage:

  * duplicates -- polygons 9 and 10 are the same room at IoU 0.93, and 8 sits
    95.7% inside 1. CAGE's own remove_rooms_with_iou missed both.
  * unclaimed floor -- 14.1 m2, 13% of the free floor, sits in no room at all:
    a whole balcony down the left side and the corridor strip above the living
    room.

Both are fixed here, once, in the polygon set, so the snapper, the wall graph,
the openings and the 3D model all inherit the correction instead of each
working around it.

Output has the same schema as the CAGE result files, so the rest of the chain
runs against it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

DUP_IOU = 0.50        # same room emitted twice
CONTAIN = 0.90        # one polygon swallowed by another
MIN_RECOVER_M2 = 1.0  # ignore slivers
WALL_LEVEL = 0.35     # density above this is wall, not floor


def valid(p):
    q = Polygon(p)
    return q if q.is_valid else q.buffer(0)


def dedup(polys):
    keep, dropped = list(range(len(polys))), []
    shp = [valid(p) for p in polys]
    for i in list(keep):
        for j in list(keep):
            if i >= j or i not in keep or j not in keep:
                continue
            a, b = shp[i], shp[j]
            inter = a.intersection(b).area
            if inter <= 0:
                continue
            union = a.union(b).area
            iou = inter / union if union else 0
            small = min(a.area, b.area)
            contained = inter / small if small else 0
            if iou > DUP_IOU or contained > CONTAIN:
                loser = i if a.area < b.area else j
                keep.remove(loser)
                dropped.append({"poly": loser, "against": j if loser == i else i,
                                "iou": round(iou, 3),
                                "contained": round(contained, 3)})
    return keep, dropped


def recover(polys, density, px_m2):
    """Free floor claimed by no room, as new axis-aligned rooms."""
    occ = density > 0.02
    filled = ndimage.binary_fill_holes(
        ndimage.binary_closing(occ, np.ones((5, 5), bool)))
    free = filled & ~ndimage.binary_dilation(density > WALL_LEVEL,
                                             np.ones((3, 3), bool))
    mask = np.zeros(density.shape, np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.array(p, np.int32)], 1)
    un = free & ~mask.astype(bool)
    # open first: a one-pixel seam along a room edge is not a room
    un = ndimage.binary_opening(un, np.ones((3, 3), bool))
    lab, n = ndimage.label(un, np.ones((3, 3), bool))

    new, notes = [], []
    for k in range(1, n + 1):
        m = lab == k
        area = m.sum() * px_m2
        if area < MIN_RECOVER_M2:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 4:
            x, y, w, h = cv2.boundingRect(c)
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        new.append(pts.tolist())
        notes.append({"area_m2": round(float(area), 2), "corners": len(pts)})
    return new, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_run = a.out or f"{a.run}_clean"

    res = json.loads((BASE / "polygons" / f"{a.run}.json").read_text())
    key = Path(res["density"]).stem.replace("density_", "")
    v = json.loads((BASE / "scores" / "variants.json").read_text())["variants"][key]
    mn, mx = np.array(v["min_coords"]), np.array(v["max_coords"])
    px_m2 = ((mx[0] - mn[0]) / 256) * ((mx[1] - mn[1]) / 256)

    polys = res["polys_refined"]
    print(f"{len(polys)} polygons in", a.run, flush=True)

    keep, dropped = dedup(polys)
    for d in dropped:
        print(f"  dropped {d['poly']} (IoU {d['iou']} / {100*d['contained']:.0f}% "
              f"inside {d['against']})", flush=True)
    kept = [polys[i] for i in keep]
    print(f"  {len(kept)} after dedup", flush=True)

    density = cv2.imread(str(BASE / "density" / f"density_{key}.png"),
                         0).astype(np.float32) / 255
    new, notes = recover(kept, density, px_m2)
    for nt in notes:
        print(f"  recovered {nt['area_m2']:5.2f} m2 region "
              f"({nt['corners']} corners)", flush=True)

    final = kept + new
    print(f"  {len(final)} rooms total "
          f"({len(kept)} kept + {len(new)} recovered)", flush=True)

    (BASE / "polygons" / f"{out_run}.json").write_text(json.dumps({
        "density": res["density"], "backbone": res.get("backbone"),
        "n_decoded": len(final), "n_refined": len(final),
        "polys_decoded": final, "polys_refined": final,
        "cleaning": {"source": a.run, "dropped": dropped,
                     "recovered": notes,
                     "dup_iou": DUP_IOU, "contain": CONTAIN,
                     "min_recover_m2": MIN_RECOVER_M2},
    }, indent=1))
    print("wrote", BASE / "polygons" / f"{out_run}.json")


if __name__ == "__main__":
    main()
