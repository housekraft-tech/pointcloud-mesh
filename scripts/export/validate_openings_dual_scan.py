"""Refit preliminary openings to jamb, header and sill planes in two LiDAR walks.

The input boxes are location priors only.  Each physical boundary is searched
for independently in each registered scan.  A release opening requires the
two walks to agree within 10 mm on both jambs and the header (and on the sill
for a window).  Anything else remains explicit review evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cage_output" / "scripts"))
from snap_to_planes import load_frame  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def robust_peak(
    values: np.ndarray,
    seed: float,
    search_m: float = 0.22,
    bin_m: float = 0.005,
    min_points: int = 18,
) -> dict | None:
    """Fit a narrow planar return near *seed*, rejecting flat background density."""
    values = values[np.isfinite(values) & (np.abs(values - seed) <= search_m)]
    if len(values) < min_points:
        return None
    edges = np.arange(seed - search_m, seed + search_m + bin_m, bin_m)
    counts, edges = np.histogram(values, edges)
    if not len(counts) or int(counts.max()) < 4:
        return None
    # A physical plane is a compact peak; wall-face background beside a jamb
    # is approximately uniform in this one-dimensional coordinate.
    smooth = np.convolve(counts.astype(float), np.array([0.25, 0.5, 0.25]), mode="same")
    index = int(np.argmax(smooth))
    background = float(np.median(smooth[smooth > 0])) if np.any(smooth > 0) else 0.0
    prominence = float(smooth[index] / max(background, 1.0))
    centre = float((edges[index] + edges[index + 1]) / 2)
    selected = values[np.abs(values - centre) <= 0.022]
    if len(selected) < min_points or prominence < 1.45:
        return None
    value = float(np.median(selected))
    mad = float(np.median(np.abs(selected - value)))
    sigma = max(1.4826 * mad, 0.001)
    return {
        "value_m": value,
        "n": int(len(selected)),
        "sigma_mm": round(sigma * 1000, 2),
        "stderr_mm": round(sigma / np.sqrt(len(selected)) * 1000, 3),
        "prominence": round(prominence, 2),
    }


def fit_opening(xyz: np.ndarray, opening: dict, wall: dict, floor_z: float) -> dict:
    axis = 0 if wall["axis"] == "x" else 1
    along_axis = 1 - axis
    low_face, high_face = map(float, wall["faces_m"])
    along0, along1 = map(float, opening["stable_along_m"])
    z0, z1 = map(float, opening["stable_z_m"])
    is_door = z0 <= floor_z + 0.16 or "door" in opening.get("kind", "").lower()

    cross = xyz[:, axis]
    along = xyz[:, along_axis]
    z = xyz[:, 2]
    inside_cross = (cross >= low_face - 0.055) & (cross <= high_face + 0.055)

    # Jamb returns are perpendicular to the wall and exist within the opening
    # height.  Exclude the floor/header bands and fit their constant-along plane.
    vertical_margin = min(0.20, max(0.08, 0.18 * (z1 - z0)))
    jamb_band = inside_cross & (z >= z0 + vertical_margin) & (z <= z1 - vertical_margin)
    left = robust_peak(along[jamb_band], along0)
    right = robust_peak(along[jamb_band], along1)

    def jamb_support(fit: dict | None) -> dict | None:
        if fit is None:
            return None
        selected = jamb_band & (np.abs(along - fit["value_m"]) <= 0.024)
        if selected.sum() < 18:
            return None
        cross_span = float(np.quantile(cross[selected], 0.95) - np.quantile(cross[selected], 0.05))
        z_span = float(np.quantile(z[selected], 0.95) - np.quantile(z[selected], 0.05))
        fit["cross_span_mm"] = round(cross_span * 1000, 1)
        fit["vertical_span_mm"] = round(z_span * 1000, 1)
        # A jamb must turn through a meaningful part of the wall thickness and
        # extend vertically.  A narrow strip on the main wall face is not a jamb.
        if cross_span < min(0.050, 0.35 * (high_face - low_face)) \
                or z_span < max(0.28, 0.30 * (z1 - z0)):
            return None
        return fit

    left, right = jamb_support(left), jamb_support(right)

    # Headers and sills are horizontal returns through the wall thickness.  Stay
    # away from the jambs so their vertical points do not dominate the histogram.
    side_margin = min(0.16, max(0.05, 0.12 * (along1 - along0)))
    horizontal_band = inside_cross & (along >= along0 + side_margin) & (along <= along1 - side_margin)
    header = robust_peak(z[horizontal_band], z1, search_m=0.24)
    sill = ({"value_m": float(floor_z), "source": "measured_floor_plane"}
            if is_door else robust_peak(z[horizontal_band], z0, search_m=0.24))

    def horizontal_support(fit: dict | None) -> dict | None:
        if fit is None or fit.get("source") == "measured_floor_plane":
            return fit
        selected = horizontal_band & (np.abs(z - fit["value_m"]) <= 0.024)
        if selected.sum() < 18:
            return None
        cross_span = float(np.quantile(cross[selected], 0.95) - np.quantile(cross[selected], 0.05))
        along_span = float(np.quantile(along[selected], 0.95) - np.quantile(along[selected], 0.05))
        fit["cross_span_mm"] = round(cross_span * 1000, 1)
        fit["horizontal_span_mm"] = round(along_span * 1000, 1)
        if cross_span < min(0.050, 0.35 * (high_face - low_face)) \
                or along_span < max(0.22, 0.30 * (along1 - along0)):
            return None
        return fit

    header, sill = horizontal_support(header), horizontal_support(sill)

    fits = {"left": left, "right": right, "sill": sill, "header": header}
    complete = left is not None and right is not None and header is not None and sill is not None
    if complete and left["value_m"] > right["value_m"]:
        left, right = right, left
        fits["left"], fits["right"] = left, right
    return {"fits": fits, "complete": bool(complete)}


def combine(
    opening: dict,
    primary: dict,
    reference: dict,
    tolerance_mm: float,
    axis_bias_mm: dict[str, float],
    z_bias_mm: float,
) -> dict:
    item = {key: value for key, value in opening.items() if key not in {"cut_lo", "cut_hi"}}
    item["primary_fit"] = primary
    item["reference_fit"] = reference
    if not primary["complete"] or not reference["complete"]:
        item["metric_status"] = "review_incomplete_boundary_fit"
        return item

    deltas = {}
    averaged = {}
    for boundary in ("left", "right", "sill", "header"):
        first = primary["fits"][boundary]["value_m"]
        second = reference["fits"][boundary]["value_m"]
        deltas[boundary] = round((first - second) * 1000, 2)
        averaged[boundary] = float((first + second) / 2)
    width_delta = (deltas["right"] - deltas["left"])
    height_delta = (deltas["header"] - deltas["sill"])
    item["repeat_boundary_delta_mm"] = deltas
    along_axis = "y" if opening.get("opening_axis") == "x" else "x"
    corrected = {
        "left": round(deltas["left"] - axis_bias_mm[along_axis], 2),
        "right": round(deltas["right"] - axis_bias_mm[along_axis], 2),
        "sill": round(deltas["sill"] - z_bias_mm, 2),
        "header": round(deltas["header"] - z_bias_mm, 2),
    }
    item["registration_bias_mm"] = {"along": round(axis_bias_mm[along_axis], 3),
                                    "z": round(z_bias_mm, 3)}
    item["repeat_boundary_residual_mm"] = corrected
    item["repeat_width_delta_mm"] = round(width_delta, 2)
    item["repeat_height_delta_mm"] = round(height_delta, 2)
    item["measured_along_m"] = [averaged["left"], averaged["right"]]
    item["measured_z_m"] = [averaged["sill"], averaged["header"]]
    item["measured_width_mm"] = round((averaged["right"] - averaged["left"]) * 1000, 1)
    item["measured_height_mm"] = round((averaged["header"] - averaged["sill"]) * 1000, 1)
    item["prior_boundary_move_mm"] = {
        "left": round((averaged["left"] - opening["stable_along_m"][0]) * 1000, 1),
        "right": round((averaged["right"] - opening["stable_along_m"][1]) * 1000, 1),
        "sill": round((averaged["sill"] - opening["stable_z_m"][0]) * 1000, 1),
        "header": round((averaged["header"] - opening["stable_z_m"][1]) * 1000, 1),
    }
    values = list(corrected.values()) + [width_delta, height_delta]
    item["metric_status"] = ("pass_two_walk_10mm" if max(abs(value) for value in values) <= tolerance_mm
                             else "review_two_walk_disagreement")
    return item


def merge_fragmented_window_priors(openings: list[dict]) -> list[dict]:
    """Merge adjacent legacy panes when they describe one framed wall opening.

    The earlier mesh pass fragmented a single window by its internal frame and
    assigned each fragment a different height.  A wall opening is bounded by
    the outer jamb/header/sill returns; mullions belong to a later window-frame
    component, not to the masonry cut.
    """
    consumed: set[int] = set()
    output: list[dict] = []
    for index, first in enumerate(openings):
        if index in consumed or first.get("kind", "").lower() != "window":
            if index not in consumed:
                output.append(first)
            continue
        cluster = [index]
        for other_index, second in enumerate(openings[index + 1 :], index + 1):
            if second.get("kind", "").lower() != "window" \
                    or second.get("wall_id") != first.get("wall_id"):
                continue
            gap = max(first["stable_along_m"][0], second["stable_along_m"][0]) \
                - min(first["stable_along_m"][1], second["stable_along_m"][1])
            z_overlap = min(first["stable_z_m"][1], second["stable_z_m"][1]) \
                - max(first["stable_z_m"][0], second["stable_z_m"][0])
            if gap <= 0.45 and z_overlap >= 0.20:
                cluster.append(other_index)
        if len(cluster) == 1:
            output.append(first)
            continue
        parts = [openings[value] for value in cluster]
        consumed.update(cluster)
        along = [min(item["stable_along_m"][0] for item in parts),
                 max(item["stable_along_m"][1] for item in parts)]
        # Expand the inconsistent pane-height priors just enough to seed the
        # physical outer sill/header peaks seen in the elevation evidence.
        z = [min(item["stable_z_m"][0] for item in parts) - 0.15,
             max(item["stable_z_m"][1] for item in parts) + 0.15]
        merged = {
            **first,
            "name": "merged_window_" + "__".join(item.get("name", str(value))
                                                     for value, item in zip(cluster, parts)),
            "source": "merged_fragmented_legacy_prior",
            "stable_along_m": along,
            "stable_z_m": z,
            "width_mm": round((along[1] - along[0]) * 1000, 1),
            "height_mm": round((z[1] - z[0]) * 1000, 1),
            "aliases": [item.get("name", "") for item in parts],
            "prior_semantics": "single masonry opening; internal mullions excluded",
        }
        output.append(merged)
    return output


def render(items: list[dict], destination: Path) -> None:
    labels, boundary_error, dimension_error, colours = [], [], [], []
    for index, item in enumerate(items, 1):
        labels.append(f"{index:02d} {item.get('kind', 'opening')}\n{item.get('wall_id', '')}")
        deltas = item.get("repeat_boundary_delta_mm", {})
        boundary_error.append(max([abs(value) for value in deltas.values()] or [0]))
        dimension_error.append(max(abs(item.get("repeat_width_delta_mm", 0)),
                                   abs(item.get("repeat_height_delta_mm", 0))))
        colours.append("#16a34a" if item.get("metric_status", "").startswith("pass") else "#dc2626")
    width = max(10, 0.65 * len(items))
    fig, axis = plt.subplots(figsize=(width, 5), dpi=170)
    x = np.arange(len(items)); bar_width = 0.38
    axis.bar(x - bar_width / 2, boundary_error, bar_width, label="worst boundary repeat", color=colours, alpha=0.9)
    axis.bar(x + bar_width / 2, dimension_error, bar_width, label="worst size repeat", color="#64748b")
    axis.axhline(10, color="#111827", linestyle="--", linewidth=1, label="10 mm gate")
    axis.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    axis.set_ylabel("two-walk disagreement (mm)")
    axis.set_title("Opening metrology gate — inherited boxes are priors only")
    axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="output_final/mujammel_asbuilt_v2/feature_manifest.json")
    parser.add_argument("--walls", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--primary-las", default="mujammelexport.las")
    parser.add_argument("--primary-transform", default="output2/koushik_all/skeleton_3d/annotated/scan_transform.json")
    parser.add_argument("--reference-las", default="koushikexport.las")
    parser.add_argument("--reference-transform", default=None)
    parser.add_argument("--frame-meta", default="cage_output/scores/variants.json")
    parser.add_argument("--max-points", type=int, default=6_000_000)
    parser.add_argument("--tolerance-mm", type=float, default=10.0)
    parser.add_argument("--repeat-graph", default="cage_output/metrology/mujammel_persistent_wall_graph_8m.json")
    parser.add_argument("--out", default="cage_output/metrology/mujammel_openings_validated.json")
    parser.add_argument("--plot", default="cage_output/metrology/mujammel_openings_validated.png")
    args = parser.parse_args()

    manifest = json.loads(resolve(args.manifest).read_text())
    graph = json.loads(resolve(args.walls).read_text())
    repeat_graph = json.loads(resolve(args.repeat_graph).read_text())
    walls = {wall["wall_id"]: wall for wall in graph["walls"]}
    openings = merge_fragmented_window_priors(
        [item for item in manifest["openings"] if item.get("wall_id") in walls]
    )

    print("loading primary opening evidence", flush=True)
    primary_xyz, _, _, _ = load_frame(args.primary_las, args.max_points,
                                      args.primary_transform, args.frame_meta)
    primary = [fit_opening(primary_xyz, item, walls[item["wall_id"]], graph["floor_z_m"])
               for item in openings]
    del primary_xyz; gc.collect()
    print("loading reference opening evidence", flush=True)
    reference_xyz, _, _, _ = load_frame(args.reference_las, args.max_points,
                                        args.reference_transform, args.frame_meta)
    reference = [fit_opening(reference_xyz, item, walls[item["wall_id"]], graph["floor_z_m"])
                 for item in openings]
    del reference_xyz; gc.collect()

    axis_bias_mm = {}
    for axis in ("x", "y"):
        values = [value for wall in graph["walls"] if wall["axis"] == axis
                  for value in wall.get("face_repeatability_mm", [])]
        axis_bias_mm[axis] = float(np.median(values)) if values else 0.0
    z_bias_mm = 1000 * float(np.mean([
        repeat_graph["primary"]["floor_z_m"] - repeat_graph["reference"]["floor_z_m"],
        repeat_graph["primary"]["ceiling_z_m"] - repeat_graph["reference"]["ceiling_z_m"],
    ]))
    items = [combine(item, first, second, args.tolerance_mm, axis_bias_mm, z_bias_mm)
             for item, first, second in zip(openings, primary, reference)]
    passed = [item for item in items if item["metric_status"].startswith("pass")]
    result = {
        "schema": "dual-walk-opening-metrology-v1",
        "tolerance_mm": args.tolerance_mm,
        "registration_bias_mm": {**axis_bias_mm, "z": z_bias_mm},
        "role": "only pass openings may cut release geometry",
        "summary": {"input_priors": len(items), "pass": len(passed), "review": len(items) - len(passed)},
        "openings": items,
    }
    destination = resolve(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    render(items, resolve(args.plot))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
