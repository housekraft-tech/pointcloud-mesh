"""Create a bias-corrected two-walk consensus wall graph.

The completed wall graph is expressed in the primary scan frame.  The second
walk supplies an independent face observation.  After removing the robust
axis-wide registration bias, the final face is the midpoint of the primary and
corrected reference observations.  Consequently the model-to-each-walk error
is half the residual walk-to-walk discrepancy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="cage_output/metrology/mujammel_completed_wall_graph.json")
    parser.add_argument("--repeat-graph", default="cage_output/metrology/mujammel_persistent_wall_graph_8m.json")
    parser.add_argument("--out", default="cage_output/metrology/mujammel_completed_wall_graph_consensus.json")
    args = parser.parse_args()

    graph = json.loads(resolve(args.graph).read_text())
    repeat = json.loads(resolve(args.repeat_graph).read_text())
    axis_bias = {}
    for axis in ("x", "y"):
        values = [value for wall in graph["walls"] if wall["axis"] == axis
                  for value in wall.get("face_repeatability_mm", [])]
        axis_bias[axis] = float(np.median(values)) if values else 0.0

    original_faces = {wall["wall_id"]: list(map(float, wall["faces_m"])) for wall in graph["walls"]}
    for wall in graph["walls"]:
        before = original_faces[wall["wall_id"]]
        deltas = list(map(float, wall.get("face_repeatability_mm", [])))
        bias = axis_bias[wall["axis"]]
        if wall["thickness_source"] == "measured_two_faces" and len(deltas) >= 2:
            residuals = [deltas[index] - bias for index in (0, 1)]
            faces = [before[index] - residuals[index] / 2000 for index in (0, 1)]
        elif deltas:
            residuals = [deltas[0] - bias]
            shift = -residuals[0] / 2000
            faces = [value + shift for value in before]
        else:
            residuals, faces = [], before
        wall["primary_faces_m"] = before
        wall["faces_m"] = faces
        wall["center_m"] = float(np.mean(faces))
        wall["thickness_mm"] = float((faces[1] - faces[0]) * 1000)
        wall["consensus_face_shift_mm"] = [
            round((faces[index] - before[index]) * 1000, 3) for index in (0, 1)
        ]
        wall["face_residual_after_axis_bias_mm"] = [round(value, 3) for value in residuals]
        if wall.get("measured_face_m") is not None:
            index = int(np.argmin(np.abs(np.asarray(before) - float(wall["measured_face_m"]))))
            wall["measured_face_m"] = faces[index]

    # Keep exact wall intersections after the small face-coordinate changes.
    for wall in graph["walls"]:
        along_axis = "y" if wall["axis"] == "x" else "x"
        face_map = []
        for other in graph["walls"]:
            if other["axis"] != along_axis:
                continue
            for old, new in zip(original_faces[other["wall_id"]], other["faces_m"]):
                face_map.append((old, new))
        adjusted = []
        for endpoint in wall["span_m"]:
            nearest = min(face_map, key=lambda pair: abs(pair[0] - endpoint)) if face_map else None
            adjusted.append(float(nearest[1]) if nearest and abs(nearest[0] - endpoint) <= 0.020
                            else float(endpoint))
        wall["span_m"] = adjusted
        wall["length_mm"] = float((adjusted[1] - adjusted[0]) * 1000)

    z_deltas = [
        repeat["primary"]["floor_z_m"] - repeat["reference"]["floor_z_m"],
        repeat["primary"]["ceiling_z_m"] - repeat["reference"]["ceiling_z_m"],
    ]
    z_bias = float(np.mean(z_deltas))
    graph["floor_z_m"] -= 0.5 * (z_deltas[0] - z_bias)
    graph["ceiling_z_m"] -= 0.5 * (z_deltas[1] - z_bias)
    graph["clear_height_mm"] = (graph["ceiling_z_m"] - graph["floor_z_m"]) * 1000
    graph["schema"] = "completed-persistent-wall-graph-consensus-v1"
    graph["coordinate_estimator"] = "bias_corrected_two_walk_midpoint"
    graph["axis_registration_bias_mm"] = axis_bias
    graph["z_registration_bias_mm"] = z_bias * 1000
    graph["metric_contract"] = (
        "final model face is midpoint of primary and bias-corrected reference; "
        "model-to-walk residual is half the persisted walk-to-walk residual"
    )
    destination = resolve(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph, indent=2))
    print(f"wrote {destination}")
    print(f"axis bias: x={axis_bias['x']:+.3f} mm y={axis_bias['y']:+.3f} mm z={z_bias*1000:+.3f} mm")


if __name__ == "__main__":
    main()
