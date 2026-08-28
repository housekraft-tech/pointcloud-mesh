"""Aligned prediction vs architect-plan ground truth, one panel per variant."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_compare import gt_polygons, pred_polygons, best_alignment, match  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANTS = ["raw", "robust", "robust_c99", "robust_c95", "robust_c90"]
TITLE = {"raw": "raw (no clip, loose bbox)", "robust": "robust bbox, no clip",
         "robust_c99": "clip p99", "robust_c95": "clip p95", "robust_c90": "clip p90"}


def panel(ax, name, suffix="_r50"):
    gts = gt_polygons()
    preds, _ = pred_polygons(HERE / "results" / f"{name}{suffix}.json",
                             HERE / "variants.json")
    iou, k, flip, t, aligned = best_alignment(preds, gts, force=(2, False))
    pairs = match(aligned, gts)
    iou_of = {i: v for i, _, v in pairs}

    for _, g in gts:
        x, y = g.exterior.xy
        ax.plot(x, y, color="#444", lw=1.4, zorder=1)

    for i, p in enumerate(aligned):
        v = iou_of.get(i, 0.0)
        col = "#1a9850" if v >= 0.5 else ("#fdae61" if v >= 0.25 else "#d73027")
        ax.add_patch(MplPoly(np.array(p.exterior.coords), closed=True,
                             facecolor=col, alpha=0.45, edgecolor=col,
                             lw=1.8, zorder=2))

    n_ok = sum(1 for _, _, v in pairs if v >= 0.5)
    ax.set_title(f"{TITLE.get(name, name)}\n{len(preds)} rooms · "
                 f"{n_ok}/15 matched@IoU0.5 · footprint IoU {iou:.2f}",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    return iou, n_ok


def main(suffix="_r50", out="results/vs_gt_r50.png"):
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 4.6))
    for ax, name in zip(axes, VARIANTS):
        panel(ax, name, suffix)
    fig.suptitle("CAGE (Structured3D pretrained) on the Koushik handheld-LiDAR scan\n"
                 "grey = architect plan (15 spaces) · green ≥0.5 IoU · "
                 "orange 0.25–0.5 · red <0.25", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / out, dpi=130)
    print("wrote", HERE / out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
