"""Check every detected opening against the architect's own opening schedule.

Matching is by POSITION, not by size: our openings are transformed into the
drawing frame using the placement already fitted for the room comparison
(never re-fitted here), then paired to plan openings by centre distance. A
size-only match would flatter the result by pairing whatever numbers happen to
agree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
HERE = Path(__file__).resolve().parent
BASE = HERE.parent

# fitted once on the unsnapped rooms in score_snapped.py; reused verbatim
PLACEMENT = {"rot90": 2, "flip": False, "t": [9.719, -7.331]}
MATCH_MAX_M = 1.2


def place(pts):
    a = np.asarray(pts, dtype=float).reshape(-1, 2).copy()
    if PLACEMENT["flip"]:
        a[:, 1] = -a[:, 1]
    th = np.deg2rad(90 * PLACEMENT["rot90"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return a @ R.T + np.array(PLACEMENT["t"])


def plan_openings():
    d = json.loads((ROOT / "data" / "bbox_data.json").read_text())
    s = d["dimensions"][0]["scale"]
    out = []
    for o in d["global_openings"]:
        w = o.get("door_width")
        if w is None:
            w = max(abs(o["x2"] - o["x1"]), abs(o["y2"] - o["y1"]))
        c = o.get("center")
        if c is None:
            c = [0.5 * (o["x1"] + o["x2"]), 0.5 * (o["y1"] + o["y2"])]
        cx, cy = c
        out.append({
            "type": o["product_name"].lower().replace(" ", "_"),
            "centre": [cx * s, -cy * s],
            "width_mm": w * s * 1000,
            "height_mm": o.get("height", 0) * 10,
            "sill_mm": o.get("elevation", 0) * 10,
        })
    return out


KIND_MAP = {"door": "door", "balcony_door": "balcony_door",
            "wide_opening": "balcony_door", "window": "window",
            "high_level_void": None}


def main():
    ours = json.loads((BASE / "openings" /
                       "openings_robust_c95_swin.json").read_text())["openings"]
    plan = plan_openings()

    mine = []
    for o in ours:
        c = place([[(o["p0"][0] + o["p1"][0]) / 2,
                    (o["p0"][1] + o["p1"][1]) / 2]])[0]
        mine.append({**o, "centre": c.tolist()})

    C = np.full((len(mine), len(plan)), 1e6)
    for i, m in enumerate(mine):
        for j, p in enumerate(plan):
            d = np.hypot(m["centre"][0] - p["centre"][0],
                         m["centre"][1] - p["centre"][1])
            if d <= MATCH_MAX_M:
                C[i, j] = d
    ri, ci = linear_sum_assignment(C)

    rows, dw, dh = [], [], []
    matched_plan = set()
    for i, j in zip(ri, ci):
        if C[i, j] >= 1e6:
            continue
        m, p = mine[i], plan[j]
        ew = 1000 * m["width_m"] - p["width_mm"]
        eh = 1000 * m["height_m"] - p["height_mm"]
        es = 1000 * m["sill_m"] - p["sill_mm"]
        kind_ok = KIND_MAP.get(m["kind"]) == p["type"]
        rows.append({"ours": m["kind"], "plan": p["type"],
                     "dist_mm": round(1000 * C[i, j]),
                     "our_w": round(1000 * m["width_m"]),
                     "plan_w": round(p["width_mm"]),
                     "dw_mm": round(ew), "our_h": round(1000 * m["height_m"]),
                     "plan_h": round(p["height_mm"]), "dh_mm": round(eh),
                     "our_sill": round(1000 * m["sill_m"]),
                     "plan_sill": round(p["sill_mm"]), "ds_mm": round(es),
                     "kind_ok": kind_ok})
        matched_plan.add(j)
        dw.append(abs(ew)); dh.append(abs(eh))

    print(f"matched {len(rows)} of {len(mine)} detected "
          f"against {len(plan)} plan openings (<= {MATCH_MAX_M} m apart)\n")
    print(f"{'ours':15s} {'plan':13s} {'pos':>5s} "
          f"{'our w':>6s} {'plan w':>7s} {'dw':>6s} "
          f"{'our h':>6s} {'plan h':>7s} {'dh':>6s} {'ds':>6s} type")
    for r in sorted(rows, key=lambda r: -abs(r["dw_mm"])):
        print(f"{r['ours']:15s} {r['plan']:13s} {r['dist_mm']:5d} "
              f"{r['our_w']:6d} {r['plan_w']:7d} {r['dw_mm']:+6d} "
              f"{r['our_h']:6d} {r['plan_h']:7d} {r['dh_mm']:+6d} "
              f"{r['ds_mm']:+6d} {'ok' if r['kind_ok'] else 'MISMATCH'}")

    if dw:
        dw, dh = np.array(dw), np.array(dh)
        print(f"\nwidth  |err|: median {np.median(dw):6.0f} mm  "
              f"mean {dw.mean():6.0f}  max {dw.max():6.0f}")
        print(f"height |err|: median {np.median(dh):6.0f} mm  "
              f"mean {dh.mean():6.0f}  max {dh.max():6.0f}")
        print(f"type agreement: {sum(r['kind_ok'] for r in rows)}/{len(rows)}")

    unmatched_ours = [m["kind"] for i, m in enumerate(mine) if i not in set(ri[C[ri, ci] < 1e6])]
    unmatched_plan = [p["type"] for j, p in enumerate(plan) if j not in matched_plan]
    print(f"\ndetected but not in plan: {unmatched_ours}")
    print(f"in plan but not detected: {unmatched_plan}")

    (BASE / "openings" / "validation.json").write_text(json.dumps(
        {"placement": PLACEMENT, "rows": rows,
         "width_median_abs_mm": float(np.median(dw)) if len(dw) else None,
         "height_median_abs_mm": float(np.median(dh)) if len(dh) else None,
         "n_matched": len(rows), "n_ours": len(mine), "n_plan": len(plan),
         "unmatched_ours": unmatched_ours,
         "unmatched_plan": unmatched_plan}, indent=1))
    print("\nwrote", BASE / "openings" / "validation.json")


if __name__ == "__main__":
    main()
