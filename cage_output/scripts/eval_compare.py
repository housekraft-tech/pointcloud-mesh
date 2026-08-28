"""Score CAGE's predicted room polygons against the architect-plan ground truth.

The prediction lives in density-image pixels, the ground truth in architect-
drawing pixels. Both have a known metres-per-pixel scale, so the only unknown
is a rigid placement: we search the 8 dihedral orientations and solve
translation by centroid, keeping whichever maximises total IoU. Scale is never
fitted -- if the metric scale were wrong the IoU would collapse, which is
exactly the property we want to test.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")


def gt_polygons():
    d = json.loads((ROOT / "data" / "bbox_data.json").read_text())
    s = d["dimensions"][0]["scale"]
    out = []
    for a in d["areas"]:
        for b in a.get("bboxes", []):
            x1, y1, x2, y2 = b["x1"] * s, -b["y1"] * s, b["x2"] * s, -b["y2"] * s
            out.append((a["area_name"], Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
    return out


def pred_polygons(result_path, variants_path):
    res = json.loads(Path(result_path).read_text())
    var = json.loads(Path(variants_path).read_text())
    key = Path(res["density"]).stem.replace("density_", "")
    v = var["variants"][key]
    mn = np.array(v["min_coords"])
    mx = np.array(v["max_coords"])
    sx, sy = (mx[0] - mn[0]) / 256.0, (mx[1] - mn[1]) / 256.0
    polys = []
    for p in res["polys_refined"]:
        arr = np.array(p, dtype=float)
        # generate_density writes density[y, x]; undo to metric x,y
        xy = np.stack([mn[0] + arr[:, 0] * sx, mn[1] + arr[:, 1] * sy], axis=1)
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        polys.append(poly if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda g: g.area))
    return polys, v


def transform(poly, k, flip, t):
    a = np.array(poly.exterior.coords)
    if flip:
        a[:, 1] = -a[:, 1]
    th = np.deg2rad(90 * k)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = a @ R.T + t
    return Polygon(a)


def best_alignment(preds, gts, force=None):
    """force=(k, flip) pins the orientation. Every variant is built from the
    same density frame, so the scan->drawing rotation is one physical fact;
    letting each variant pick its own lets a partial prediction score a
    spurious optimum at the wrong orientation."""
    gt_union = unary_union([g for _, g in gts])
    gc = np.array(gt_union.centroid.coords[0])
    best = None
    orients = [force] if force is not None else list(product(range(4), [False, True]))
    for k, flip in orients:
        moved = [transform(p, k, flip, np.zeros(2)) for p in preds]
        pc = np.array(unary_union(moved).centroid.coords[0])
        base = gc - pc
        # coarse-to-fine translation search; a too-narrow window would
        # understate CAGE by scoring it at the wrong offset
        best_t = base
        for step, span in ((0.25, 3.0), (0.05, 0.3)):
            local = None
            centre = best_t
            for dx in np.arange(-span, span + 1e-9, step):
                for dy in np.arange(-span, span + 1e-9, step):
                    t = centre + np.array([dx, dy])
                    u = unary_union([Polygon(np.array(m.exterior.coords) + t)
                                     for m in moved])
                    inter = u.intersection(gt_union).area
                    union = u.union(gt_union).area
                    iou = inter / union if union else 0.0
                    if local is None or iou > local[0]:
                        local = (iou, t)
            best_t = local[1]
            if best is None or local[0] > best[0]:
                cand = [Polygon(np.array(m.exterior.coords) + best_t) for m in moved]
                best = (local[0], k, flip, best_t, cand)
    return best


def match(preds, gts):
    C = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, (_, g) in enumerate(gts):
            inter = p.intersection(g).area
            union = p.union(g).area
            C[i, j] = inter / union if union else 0.0
    ri, ci = linear_sum_assignment(-C)
    return [(int(i), int(j), float(C[i, j])) for i, j in zip(ri, ci)]


def main(result_path, force=(2, False)):
    gts = gt_polygons()
    preds, vmeta = pred_polygons(result_path, HERE / "variants.json")
    gt_area = sum(g.area for _, g in gts)
    pr_area = sum(p.area for p in preds)

    iou, k, flip, t, aligned = best_alignment(preds, gts, force=force)
    pairs = match(aligned, gts)

    matched = [p for p in pairs if p[2] >= 0.5]
    name = Path(result_path).stem
    report = {
        "result": name,
        "n_pred": len(preds), "n_gt": len(gts),
        "pred_total_area_m2": round(pr_area, 2),
        "gt_total_area_m2": round(gt_area, 2),
        "area_error_pct": round(100 * (pr_area - gt_area) / gt_area, 1),
        "global_iou": round(iou, 3),
        "orientation": {"rot90": k, "flip": bool(flip), "t": [round(float(x), 3) for x in t]},
        "n_matched_iou50": len(matched),
        "mean_iou_matched": round(float(np.mean([p[2] for p in matched])), 3) if matched else 0.0,
        "rooms": [],
    }
    for i, j, v in sorted(pairs, key=lambda x: -x[2]):
        report["rooms"].append({
            "gt_name": gts[j][0],
            "gt_area_m2": round(gts[j][1].area, 2),
            "pred_area_m2": round(aligned[i].area, 2),
            "iou": round(v, 3),
            "n_corners": len(aligned[i].exterior.coords) - 1,
        })
    unmatched_gt = set(range(len(gts))) - {j for _, j, _ in pairs}
    report["missed_gt"] = [gts[j][0] for j in sorted(unmatched_gt)]

    print(json.dumps(report, indent=1))
    (HERE / "results" / f"{name}_score.json").write_text(json.dumps(report, indent=1))
    # keep the aligned polygons for plotting
    np.save(HERE / "results" / f"{name}_aligned.npy",
            np.array([np.array(p.exterior.coords) for p in aligned], dtype=object),
            allow_pickle=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1])
