"""Render strict persistent-wall candidates over the registered LiDAR slice."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from extract_persistent_wall_graph import _drawing_wall_points  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", default="cage_output/metrology/mujammel_persistent_wall_graph_2m.json"
    )
    parser.add_argument("--density", default="cage_output/slices/density_s02_mid.npy")
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--drawing", default="floorplan_original.png")
    parser.add_argument(
        "--drawing-transform",
        default="output2/koushik_all/skeleton_3d/annotated/drawing_transform.json",
    )
    parser.add_argument("--min-length-mm", type=float, default=800.0)
    parser.add_argument("--max-plan-mm", type=float, default=120.0)
    parser.add_argument(
        "--out", default="cage_output/metrology/mujammel_persistent_wall_graph_2m.png"
    )
    args = parser.parse_args()

    graph = json.loads(resolve(args.graph).read_text())
    density = np.load(resolve(args.density))
    meta = json.loads(resolve(args.frame_meta).read_text())["variants"]["s02_mid"]
    min_x, min_y = meta["min_coords"]
    max_x, max_y = meta["max_coords"]
    drawing = _drawing_wall_points(
        resolve(args.drawing), resolve(args.drawing_transform), resolve(args.frame_meta)
    )

    chosen = [
        wall for wall in graph["walls"]
        if wall["status"] == "pass"
        and wall["length_mm"] >= args.min_length_mm
        and wall["plan_distance_mm"] <= args.max_plan_mm
    ]
    fig, axis = plt.subplots(figsize=(13, 13), dpi=180)
    axis.set_facecolor("#050607")
    axis.imshow(np.power(density, 0.32), origin="lower",
                extent=[min_x, max_x, min_y, max_y], cmap="gray",
                vmin=0, vmax=1, interpolation="nearest")
    axis.scatter(drawing[:, 0], drawing[:, 1], s=0.12, c="#38bdf8", alpha=0.22,
                 linewidths=0, label="architect wall-stroke prior")
    for wall in chosen:
        low, high = wall["faces_m"]
        start, stop = wall["span_m"]
        if wall["axis"] == "x":
            rect = (low, start, high - low, stop - start)
            tx, ty = wall["center_m"], 0.5 * (start + stop)
        else:
            rect = (start, low, stop - start, high - low)
            tx, ty = 0.5 * (start + stop), wall["center_m"]
        axis.add_patch(Rectangle(rect[:2], rect[2], rect[3], fill=False,
                                 edgecolor="#4ade80", linewidth=1.25, zorder=4))
        axis.text(tx, ty, f"{wall['wall_id']}\n{wall['thickness_mm']:.0f}",
                  color="#4ade80", fontsize=4.4, ha="center", va="center",
                  bbox={"boxstyle": "round,pad=0.12", "fc": "#050607",
                        "ec": "#4ade80", "alpha": 0.82, "lw": 0.4}, zorder=5)
    axis.set_title(
        f"Strict persistent wall candidates: {len(chosen)}\n"
        f"dual scan pass, length ≥ {args.min_length_mm:.0f} mm, "
        f"drawing distance ≤ {args.max_plan_mm:.0f} mm",
        color="#e5e7eb", fontsize=11,
    )
    axis.set_xlim(min_x, max_x); axis.set_ylim(min_y, max_y); axis.set_aspect("equal")
    axis.tick_params(colors="#94a3b8", labelsize=7)
    axis.set_xlabel("registered X (m)", color="#cbd5e1")
    axis.set_ylabel("registered Y (m)", color="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    fig.tight_layout()
    destination = resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"rendered {len(chosen)} candidates to {destination}")


if __name__ == "__main__":
    main()
