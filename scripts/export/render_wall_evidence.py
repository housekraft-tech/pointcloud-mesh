"""Render dual-walk wall elevations with preliminary and fitted opening boxes."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))
from snap_to_planes import load_frame  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def wall_points(xyz: np.ndarray, wall: dict, face_index: int, floor: float, ceiling: float):
    axis = 0 if wall["axis"] == "x" else 1
    along_axis = 1 - axis
    face = float(wall["faces_m"][face_index])
    outward = -1.0 if face_index == 0 else 1.0
    along = xyz[:, along_axis]
    z = xyz[:, 2]
    depth = (xyz[:, axis] - face) * outward
    lo, hi = map(float, wall["span_m"])
    mask = ((along >= lo) & (along <= hi) & (z >= floor) & (z <= ceiling)
            & (depth >= -0.28) & (depth <= 0.38))
    return along[mask], z[mask], depth[mask]


def add_boxes(axis, wall_id: str, priors: list[dict], fitted: dict[str, dict]) -> None:
    for index, opening in enumerate(priors):
        if opening.get("wall_id") != wall_id:
            continue
        a, b = opening["stable_along_m"]
        z0, z1 = opening["stable_z_m"]
        axis.add_patch(Rectangle((a, z0), b-a, z1-z0, fill=False,
                                 edgecolor="#dc2626", linestyle="--", linewidth=1.0))
        axis.text(a, z1, str(index + 1), color="#dc2626", fontsize=7, va="bottom")
        measured = fitted.get(opening.get("name", ""))
        if measured and measured.get("measured_along_m"):
            a, b = measured["measured_along_m"]
            z0, z1 = measured["measured_z_m"]
            colour = "#16a34a" if measured["metric_status"].startswith("pass") else "#f59e0b"
            axis.add_patch(Rectangle((a, z0), b-a, z1-z0, fill=False,
                                     edgecolor=colour, linewidth=1.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walls", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--manifest", default="output_final/mujammel_asbuilt_v2/feature_manifest.json")
    parser.add_argument("--validated", default="cage_output/metrology/mujammel_openings_validated.json")
    parser.add_argument("--wall-ids", required=True, help="comma-separated wall ids")
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument("--primary-transform", default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json")
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--max-points", type=int, default=6_000_000)
    parser.add_argument("--out-dir", default="cage_output/metrology/wall_evidence")
    args = parser.parse_args()

    graph = json.loads(resolve(args.walls).read_text())
    wall_map = {wall["wall_id"]: wall for wall in graph["walls"]}
    ids = [value.strip() for value in args.wall_ids.split(",") if value.strip()]
    priors = json.loads(resolve(args.manifest).read_text())["openings"]
    validated = json.loads(resolve(args.validated).read_text())["openings"]
    fitted = {item.get("name", ""): item for item in validated}
    print("loading dual-walk elevation evidence", flush=True)
    primary, _, _, _ = load_frame(args.primary_las, args.max_points,
                                   args.primary_transform, args.frame_meta)
    reference, _, _, _ = load_frame(args.reference_las, args.max_points,
                                     args.reference_transform, args.frame_meta)
    out_dir = resolve(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for wall_id in ids:
        wall = wall_map[wall_id]
        fig, axes = plt.subplots(2, 2, figsize=(15, 7), dpi=180, sharex=True, sharey=True)
        for row, (label, xyz) in enumerate((("Mujammel", primary), ("Koushik repeat", reference))):
            for face_index in (0, 1):
                ax = axes[row, face_index]
                along, z, depth = wall_points(xyz, wall, face_index,
                                              graph["floor_z_m"], graph["ceiling_z_m"])
                if len(along) > 220_000:
                    take = np.linspace(0, len(along)-1, 220_000, dtype=int)
                    along, z, depth = along[take], z[take], depth[take]
                ax.scatter(along, z, c=np.clip(depth * 1000, -180, 180), s=0.25,
                           cmap="coolwarm", vmin=-180, vmax=180, alpha=0.38,
                           linewidths=0, rasterized=True)
                add_boxes(ax, wall_id, priors, fitted)
                ax.set_title(f"{label} — face {face_index} — {len(along):,} points")
                ax.grid(alpha=0.12); ax.set_ylabel("stable z (m)")
                ax.set_xlabel("along wall (m)")
        fig.suptitle(f"{wall_id}: depth-coloured wall evidence | red dashed=old box, green=10 mm repeat")
        fig.tight_layout()
        destination = out_dir / f"{wall_id}.png"
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {destination}", flush=True)
    del primary, reference; gc.collect()


if __name__ == "__main__":
    main()
