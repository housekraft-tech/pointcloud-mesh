"""Score snapped rooms against the hand tape-measured survey, in millimetres.

IoU against the architect plan cannot resolve this question: moving a room edge
20 mm changes a 3.5 m room's area by well under 1%, so the metric is blind to
exactly the thing the snap is for. The manual survey is in millimetres, so it
can see it.

Rooms are matched to survey entries by dimension similarity (Hungarian), not by
name -- CAGE emits no names. A match is only reported when both principal
dimensions agree within MATCH_TOL_MM, so a wrong pairing is dropped rather than
scored.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from eval_compare import pred_polygons  # noqa: E402

# From scripts/experiments/validate_vs_manual.py -- the ACE Scanner survey PDF.
MANUAL = {
    "Kitchen":         dict(w=2210, l=4991),
    "Utility":         dict(w=1521, l=1537),
    "Living":          dict(w=4947, l=6540),
    "Common washroom": dict(w=1451, l=2369),
    "Foyer":           dict(w=1462, l=1893),
    "MBR":             dict(w=3103, l=5823),
    "MBR washroom":    dict(w=1470, l=2415),
    "KBR":             dict(w=2951, l=4790),
    "KBR washroom":    dict(w=1472, l=2356),
    "GBR":             dict(w=3390, l=3398),
}
MATCH_TOL_MM = 400.0     # generous: this gates pairing, not accuracy


def dims_mm(poly):
    x, y = poly.exterior.xy
    wx = (max(x) - min(x)) * 1000
    wy = (max(y) - min(y)) * 1000
    return min(wx, wy), max(wx, wy)


def score(polys, label):
    names = list(MANUAL)
    D = np.zeros((len(polys), len(names)))
    dd = [dims_mm(p) for p in polys]
    for i, (w, l) in enumerate(dd):
        for j, n in enumerate(names):
            m = MANUAL[n]
            D[i, j] = abs(w - m["w"]) + abs(l - m["l"])
    ri, ci = linear_sum_assignment(D)

    rows, errs = [], []
    for i, j in zip(ri, ci):
        w, l = dd[i]
        m = MANUAL[names[j]]
        ew, el = w - m["w"], l - m["l"]
        ok = max(abs(ew), abs(el)) <= MATCH_TOL_MM
        rows.append({"room": names[j], "poly": int(i), "matched": bool(ok),
                     "survey_w": m["w"], "survey_l": m["l"],
                     "pred_w": round(w, 1), "pred_l": round(l, 1),
                     "dw_mm": round(ew, 1), "dl_mm": round(el, 1)})
        if ok:
            errs += [abs(ew), abs(el)]
    errs = np.array(errs) if errs else np.array([np.nan])
    return {"set": label, "n_matched": int(len(errs) // 2),
            "median_abs_err_mm": round(float(np.median(errs)), 1),
            "mean_abs_err_mm": round(float(np.mean(errs)), 1),
            "p90_abs_err_mm": round(float(np.percentile(errs, 90)), 1),
            "max_abs_err_mm": round(float(np.max(errs)), 1),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--modes", nargs="+", default=["interior", "nearest"])
    a = ap.parse_args()

    raw, _ = pred_polygons(BASE / "polygons" / f"{a.run}.json",
                           BASE / "scores" / "variants.json")
    sets = {"cage_raw": raw}
    for mode in a.modes:
        s = json.loads((BASE / "snapped" /
                        f"snapped_{a.run}_{mode}.json").read_text())
        polys = []
        for r in s["rooms"]:
            p = Polygon(r["polygon_m"])
            if not p.is_valid:
                p = p.buffer(0)
            if p.geom_type != "Polygon":
                p = max(p.geoms, key=lambda g: g.area)
            polys.append(p)
        sets[f"snapped_{mode}"] = polys

    out = [score(v, k) for k, v in sets.items()]
    print(f"{'set':20s} {'matched':>7s} {'median':>8s} {'mean':>8s} "
          f"{'p90':>8s} {'max':>8s}   (mm vs tape)")
    for r in out:
        print(f"{r['set']:20s} {r['n_matched']:7d} {r['median_abs_err_mm']:8.1f} "
              f"{r['mean_abs_err_mm']:8.1f} {r['p90_abs_err_mm']:8.1f} "
              f"{r['max_abs_err_mm']:8.1f}")

    best = max(out, key=lambda r: (r["n_matched"], -r["median_abs_err_mm"]))
    print(f"\nper-room detail for {best['set']}:")
    print(f"{'room':18s} {'survey w x l':>16s} {'ours w x l':>16s} "
          f"{'dw':>8s} {'dl':>8s}")
    for r in sorted(best["rows"], key=lambda r: not r["matched"]):
        flag = "" if r["matched"] else "   (unmatched)"
        print(f"{r['room']:18s} {r['survey_w']:7.0f} x{r['survey_l']:7.0f} "
              f"{r['pred_w']:7.0f} x{r['pred_l']:7.0f} "
              f"{r['dw_mm']:+8.1f} {r['dl_mm']:+8.1f}{flag}")

    (BASE / "snapped" / f"survey_{a.run}.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", BASE / "snapped" / f"survey_{a.run}.json")


if __name__ == "__main__":
    main()
