"""Score snapped rooms against the architect plan, before vs after.

Both polygon sets live in the same scan frame, so they get the SAME rigid
placement -- fitted once on the unsnapped prediction and reused. Re-fitting per
set would let the snap buy accuracy by sliding the whole plan, which is exactly
the thing we want to rule out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_compare import gt_polygons, pred_polygons, best_alignment, match  # noqa: E402

BASE = HERE.parent
FORCE = (2, False)


def place(polys, k, flip, t):
    out = []
    for p in polys:
        a = np.array(p.exterior.coords)
        if flip:
            a[:, 1] = -a[:, 1]
        th = np.deg2rad(90 * k)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out.append(Polygon(a @ R.T + t))
    return out


def summarise(label, polys, gts):
    pairs = match(polys, gts)
    ok = [v for _, _, v in pairs if v >= 0.5]
    from shapely.ops import unary_union
    u, g = unary_union(polys), unary_union([x for _, x in gts])
    return {
        "set": label,
        "n": len(polys),
        "area_m2": round(sum(p.area for p in polys), 2),
        "footprint_iou": round(u.intersection(g).area / u.union(g).area, 3),
        "matched_iou50": len(ok),
        "mean_iou_matched": round(float(np.mean(ok)), 3) if ok else 0.0,
        "mean_iou_all": round(float(np.mean([v for _, _, v in pairs])), 3),
        "per_room": [{"gt": gts[j][0], "iou": round(v, 3),
                      "gt_m2": round(gts[j][1].area, 2),
                      "pred_m2": round(polys[i].area, 2)}
                     for i, j, v in sorted(pairs, key=lambda x: -x[2])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="interior")
    a = ap.parse_args()

    gts = gt_polygons()
    raw, _ = pred_polygons(BASE / "polygons" / f"{a.run}.json",
                           BASE / "scores" / "variants.json")
    iou, k, flip, t, raw_aligned = best_alignment(raw, gts, force=FORCE)
    print(f"placement fitted on the UNSNAPPED set: rot{90*k} flip={flip} "
          f"t=({t[0]:.3f}, {t[1]:.3f})", flush=True)

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    snapped = []
    for r in snap["rooms"]:
        p = Polygon(r["polygon_m"])
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type != "Polygon":
            p = max(p.geoms, key=lambda g: g.area)
        snapped.append(p)
    snap_aligned = place(snapped, k, flip, t)

    before = summarise("cage_raw", raw_aligned, gts)
    after = summarise(f"cage_snapped_{a.mode}", snap_aligned, gts)

    print(f"\n{'':22s} {'n':>3s} {'area m2':>8s} {'fpIoU':>6s} {'m@.5':>5s} "
          f"{'meanIoU(all)':>12s}")
    for r in (before, after):
        print(f"{r['set']:22s} {r['n']:3d} {r['area_m2']:8.2f} "
              f"{r['footprint_iou']:6.3f} {r['matched_iou50']:5d} "
              f"{r['mean_iou_all']:12.3f}")

    delta = {k2: round(after[k2] - before[k2], 3)
             for k2 in ("footprint_iou", "mean_iou_all", "area_m2")}
    delta["matched_iou50"] = after["matched_iou50"] - before["matched_iou50"]
    print("\ndelta:", json.dumps(delta))

    out = BASE / "snapped" / f"score_{a.run}_{a.mode}.json"
    out.write_text(json.dumps({"placement": {"rot90": k, "flip": bool(flip),
                                             "t": [float(x) for x in t]},
                               "before": before, "after": after,
                               "delta": delta,
                               "snap_report": snap["report"]}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
