"""Before/after figure for the plane snap."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from eval_compare import pred_polygons  # noqa: E402

RUN, MODE = "robust_c95_swin", "strict"
ACC, INK, MUT = "#0A6E79", "#101A1F", "#7C8F97"


def main():
    raw, meta = pred_polygons(BASE / "polygons" / f"{RUN}.json",
                              BASE / "scores" / "variants.json")
    snap = json.loads((BASE / "snapped" / f"snapped_{RUN}_{MODE}.json").read_text())
    survey = json.loads((BASE / "snapped" / f"survey_{RUN}.json").read_text())
    sets = {s["set"]: s for s in survey}

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=.22)

    # --- panel 1: geometry, raw vs snapped
    ax = fig.add_subplot(gs[0, 0])
    for p in raw:
        ax.add_patch(MplPoly(np.array(p.exterior.coords), closed=True,
                             facecolor="none", edgecolor="#C2513F", lw=1.3,
                             ls=(0, (4, 2)), zorder=2))
    for r in snap["rooms"]:
        ax.add_patch(MplPoly(np.array(r["polygon_m"]), closed=True,
                             facecolor=ACC, alpha=.13, edgecolor=ACC, lw=1.7, zorder=3))
    ax.set_aspect("equal"); ax.axis("off")
    ax.autoscale_view()
    ax.set_title("Room boundaries\ndashed = CAGE at 62 mm/px · solid = snapped to measured planes",
                 fontsize=9.5)

    # --- panel 2: how far each edge moved
    ax2 = fig.add_subplot(gs[0, 1])
    moves = [abs(e["moved_mm"]) for r in snap["rooms"] for e in r["edges"]
             if e["moved_mm"] is not None]
    ax2.hist(moves, bins=np.arange(0, 210, 12), color=ACC, alpha=.85,
             edgecolor="white", linewidth=.7)
    ax2.axvline(62.4, color="#C2513F", lw=1.4, ls="--")
    ax2.text(65, ax2.get_ylim()[1] * .93, "one density pixel\n(62 mm)",
             fontsize=8, color="#C2513F", va="top")
    ax2.set_xlabel("edge correction (mm)", fontsize=9)
    ax2.set_ylabel("edges", fontsize=9)
    ax2.set_title(f"{len(moves)} of {sum(r['n_edges'] for r in snap['rooms'])} edges re-solved\n"
                  f"onto a face measured to {snap['report']['median_face_stderr_um']:.0f} µm",
                  fontsize=9.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # --- panel 3: error against the tape survey
    ax3 = fig.add_subplot(gs[0, 2])
    # "before" must be the SAME polygon, not whatever the raw set happened to
    # pair with that survey room -- otherwise the comparison is between two
    # different assignments and the bars mean nothing.
    from compare_to_survey import MANUAL, dims_mm
    rows = [r for r in sets["snapped_strict"]["rows"] if r["matched"]]
    rows.sort(key=lambda r: max(abs(r["dw_mm"]), abs(r["dl_mm"])))
    names = [r["room"] for r in rows]
    y = np.arange(len(rows))
    before = []
    for r in rows:
        raw_idx = snap["rooms"][r["poly"]]["id"]
        w, l = dims_mm(raw[raw_idx])
        m = MANUAL[r["room"]]
        before.append(max(abs(w - m["w"]), abs(l - m["l"])))
    after = [max(abs(r["dw_mm"]), abs(r["dl_mm"])) for r in rows]
    print("per-room worst error (mm), same polygon before and after:")
    for n, b, a2 in zip(names, before, after):
        print(f"  {n:18s} {b:8.1f}  ->  {a2:6.1f}")
    print(f"  {'MEDIAN':18s} {np.median(before):8.1f}  ->  {np.median(after):6.1f}")
    ax3.barh(y - .19, before, height=.36, color="#C2513F", alpha=.75, label="CAGE raw")
    ax3.barh(y + .19, after, height=.36, color=ACC, label="after snap")
    ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=8.5)
    ax3.set_xlabel("worst dimension error vs tape survey (mm)", fontsize=9)
    ax3.axvline(10, color=MUT, lw=1, ls=":")
    ax3.text(11, len(rows) - .4, "10 mm survey tolerance", fontsize=8, color=MUT)
    ax3.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax3.set_title(f"median {sets['cage_raw']['median_abs_err_mm']:.0f} mm  →  "
                  f"{sets['snapped_strict']['median_abs_err_mm']:.0f} mm",
                  fontsize=9.5)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    fig.savefig(BASE / "snapped" / "snap_summary.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    print("wrote", BASE / "snapped" / "snap_summary.png")


if __name__ == "__main__":
    main()
