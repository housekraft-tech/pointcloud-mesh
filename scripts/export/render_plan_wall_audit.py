"""Render the plan-wall metrology result over a waist-height LiDAR slice.

This is a survey QA artifact, not a presentation drawing.  Grey rectangles are
the architect-plan wall locations; coloured lines/rectangles are the measured
finished faces selected from the point cloud.  The overlay makes it difficult
for a repeatable cabinet, wardrobe, or niche return to be silently accepted as
the structural wall behind it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
STATUS_COLOURS = {"pass": "#36d399", "review": "#fbbf24", "fail": "#fb7185"}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _rectangle(wall: dict, faces: list[float]) -> tuple[float, float, float, float]:
    along0, along1 = wall["along_m"]
    face0, face1 = min(faces), max(faces)
    if wall["axis"] == "x":
        return face0, along0, face1 - face0, along1 - along0
    return along0, face0, along1 - along0, face1 - face0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements", default="cage_output/metrology/mujammel_plan_walls.json"
    )
    parser.add_argument("--density", default="cage_output/slices/density_s02_mid.npy")
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument(
        "--out", default="cage_output/metrology/mujammel_plan_wall_audit.png"
    )
    args = parser.parse_args()

    report = json.loads(_resolve(args.measurements).read_text())
    frame = json.loads(_resolve(args.frame_meta).read_text())
    density = np.load(_resolve(args.density))
    meta = frame["variants"]["s02_mid"]
    min_x, min_y = meta["min_coords"]
    max_x, max_y = meta["max_coords"]

    fig, axis = plt.subplots(figsize=(13, 13), dpi=180)
    axis.set_facecolor("#050607")
    axis.imshow(
        np.power(density, 0.32),
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    for wall in report["walls"]:
        nominal = _rectangle(wall, wall["nominal_faces_m"])
        axis.add_patch(
            Rectangle(
                nominal[:2], nominal[2], nominal[3],
                facecolor="#94a3b8", edgecolor="#cbd5e1",
                linewidth=0.55, alpha=0.15, zorder=2,
            )
        )

        colour = STATUS_COLOURS[wall["status"]]
        faces = wall["faces_m"]
        if len(faces) == 2:
            measured = _rectangle(wall, faces)
            axis.add_patch(
                Rectangle(
                    measured[:2], measured[2], measured[3],
                    fill=False, edgecolor=colour, linewidth=1.45, zorder=4,
                )
            )
        elif len(faces) == 1:
            along0, along1 = wall["along_m"]
            if wall["axis"] == "x":
                axis.plot([faces[0], faces[0]], [along0, along1], color=colour,
                          linewidth=1.45, zorder=4)
            else:
                axis.plot([along0, along1], [faces[0], faces[0]], color=colour,
                          linewidth=1.45, zorder=4)

        along_mid = 0.5 * sum(wall["along_m"])
        across_mid = (
            wall["center_m"]
            if wall["center_m"] is not None
            else wall["nominal_center_m"]
        )
        if wall["axis"] == "x":
            text_x, text_y = across_mid, along_mid
        else:
            text_x, text_y = along_mid, across_mid
        label = wall["wall_id"].removeprefix("wall_")
        if wall["thickness_mm"] is not None:
            label += f"\n{wall['thickness_mm']:.0f}"
        axis.text(
            text_x, text_y, label, color=colour, fontsize=5.3,
            ha="center", va="center", weight="bold", zorder=5,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#050607", "ec": colour,
                  "alpha": 0.84, "lw": 0.45},
        )

    summary = report["summary"]
    title = (
        "Mujammel shared-wall metrology audit\n"
        f"green pass {summary['pass']}  |  amber review {summary['review']}  |  "
        f"red fail {summary['fail']}  |  labels show wall ID and measured thickness (mm)"
    )
    axis.set_title(title, color="#e5e7eb", fontsize=11, pad=12)
    axis.set_xlabel("registered X (m)", color="#cbd5e1")
    axis.set_ylabel("registered Y (m)", color="#cbd5e1")
    axis.tick_params(colors="#94a3b8", labelsize=7)
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    fig.tight_layout()

    destination = _resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
