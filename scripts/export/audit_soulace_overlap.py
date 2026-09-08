"""Read-only, registered comparison of Soulace CAD against LAS and Poisson.

The existing CAD is never fitted or moved to improve its score. Display clouds
are reduced; distance queries use every LAS return. Low support is NOT proof of
an invented wall: unseen backs, contacts, glass and occlusion also cause it.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[2]
STEMS = ["Soulace_L0_ground", "Soulace_L1_first", "Soulace_L2_second"]
RAW_COLOUR = "#008d9f"
MODEL_COLOUR = "#dc542f"
POISSON_COLOUR = "#8b5bb5"
DIST_COLOURS = ["#14854c", "#26a7a5", "#df980b", "#cc3449"]
HEIGHTS = [0.9, 1.5, 2.5]


def say(message):
    print(message, flush=True)


def stats(distance):
    distance = np.asarray(distance, float) * 1000.0
    if not len(distance):
        return None
    return {"samples": len(distance), "median_mm": round(float(np.median(distance)), 2),
            "p90_mm": round(float(np.percentile(distance, 90)), 2),
            **{f"within_{t}mm_pct": round(float((distance <= t).mean() * 100), 2)
               for t in [10, 30, 100]},
            "beyond_100mm_pct": round(float((distance > 100).mean() * 100), 2)}


def rotate_xy(points, yaw):
    angle = math.radians(-yaw)
    c, s = math.cos(angle), math.sin(angle)
    for start in range(0, len(points), 2_000_000):
        q = points[start:start + 2_000_000]
        x = q[:, 0].copy()
        y = q[:, 1].copy()
        q[:, 0] = x * c - y * s
        q[:, 1] = x * s + y * c


def raw_points(path, yaw, floor_z):
    with laspy.open(path) as reader:
        points = np.empty((reader.header.point_count, 3), np.float64)
        start = 0
        for chunk in reader.chunk_iterator(2_000_000):
            n = len(chunk)
            points[start:start + n, 0] = chunk.x
            points[start:start + n, 1] = chunk.y
            points[start:start + n, 2] = chunk.z
            start += n
    shift = float(np.percentile(points[:, 2], 0.5))
    points[:, 2] -= shift + floor_z
    rotate_xy(points, yaw)
    return points, shift


def uniform_surface(mesh, indices, count, rng):
    areas = mesh.area_faces[indices]
    selected = rng.choice(indices, size=count, p=areas / areas.sum())
    tri = mesh.triangles[selected]
    r = rng.random((count, 2))
    root = np.sqrt(r[:, 0])
    q = ((1 - root)[:, None] * tri[:, 0]
         + (root * (1 - r[:, 1]))[:, None] * tri[:, 1]
         + (root * r[:, 1])[:, None] * tri[:, 2])
    return q, selected


def section_segments(vertices, faces, height):
    """Exact horizontal triangle intersections, bounded-memory implementation."""
    segments = []
    for start in range(0, len(faces), 500_000):
        f = faces[start:start + 500_000]
        z = vertices[f, 2] - height
        hit = (z.min(axis=1) <= 0) & (z.max(axis=1) > 0)
        if not hit.any():
            continue
        tri, z = vertices[f[hit]], z[hit]
        edge_points = np.zeros((len(tri), 3, 3), dtype=np.float64)
        crossing = np.zeros((len(tri), 3), bool)
        for e, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
            cross = (z[:, a] <= 0) != (z[:, b] <= 0)
            crossing[:, e] = cross
            frac = -z[cross, a] / (z[cross, b] - z[cross, a])
            edge_points[cross, e] = tri[cross, a] + frac[:, None] * (tri[cross, b] - tri[cross, a])
        valid = crossing.sum(axis=1) == 2
        pair = edge_points[valid][crossing[valid]].reshape(-1, 2, 3)
        length = np.linalg.norm(pair[:, 1] - pair[:, 0], axis=1)
        segments.append(pair[length > 1e-6])
    return np.concatenate(segments) if segments else np.empty((0, 2, 3))


def thin(points, count, rng):
    if len(points) <= count:
        return points
    return points[rng.choice(len(points), count, replace=False)]


def line_samples(segments, spacing=0.06):
    if not len(segments):
        return np.empty((0, 3))
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    segments, lengths = segments[lengths > 1e-9], lengths[lengths > 1e-9]
    if not len(segments):
        return np.empty((0, 3))
    cumulative = np.cumsum(lengths)
    count = max(1, int(math.ceil(cumulative[-1] / spacing)))
    distance = (np.arange(count) + .5) * cumulative[-1] / count
    index = np.searchsorted(cumulative, distance)
    previous = np.r_[0.0, cumulative[:-1]]
    fraction = (distance - previous[index]) / lengths[index]
    return segments[index, 0] + fraction[:, None] * (segments[index, 1] - segments[index, 0])


def render_plan(level, height, raw_xy, poisson_lines, model_lines, q, distances, bounds, out):
    figure, axes = plt.subplots(1, 3, figsize=(21, 8.3), dpi=170)
    titles = ["LiDAR returns + model", "Poisson section + model", "Model-to-LiDAR distance"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=14, loc="left", pad=12)
        ax.set_aspect("equal")
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(color="#e3e5e8", linewidth=0.4, alpha=0.6)
        ax.tick_params(labelsize=9)
    for ax in [axes[0], axes[2]]:
        ax.scatter(raw_xy[:, 0], raw_xy[:, 1], s=0.45, color=RAW_COLOUR, alpha=0.55, rasterized=True)
    axes[1].add_collection(LineCollection(poisson_lines[:, :, :2], colors=POISSON_COLOUR,
                                          linewidths=0.6, alpha=0.8))
    for ax in axes[:2]:
        ax.add_collection(LineCollection(model_lines[:, :, :2], colors=MODEL_COLOUR,
                                         linewidths=0.95, alpha=0.85))
    colours = np.array(DIST_COLOURS)[np.digitize(distances, [0.010, 0.030, 0.100])]
    axes[2].scatter(q[:, 0], q[:, 1], c=colours, s=3.0, linewidths=0, rasterized=True)
    figure.suptitle(f"SOULACE  L{level}  |  section {height:.2f} m above floor  |  fixed source coordinates", x=0.03, ha="left", fontsize=17)
    legend = [Line2D([], [], color=RAW_COLOUR, marker=".", linestyle="none", label="LiDAR: +/-25 mm slice"),
              Line2D([], [], color=MODEL_COLOUR, label="Existing full CAD"),
              Line2D([], [], color=POISSON_COLOUR, label="Exact Poisson mesh section")]
    legend += [Line2D([], [], color=c, marker="o", linestyle="none", markersize=5, label=t)
               for c, t in zip(DIST_COLOURS, ["<=10 mm", "10-30 mm", "30-100 mm", ">100 mm: investigate"])]
    figure.legend(handles=legend, loc="lower center", ncol=7, frameon=False, fontsize=10)
    figure.subplots_adjust(top=0.88, bottom=0.14, left=0.045, right=0.99, wspace=0.20)
    figure.savefig(out, facecolor="white")
    plt.close(figure)


def render_wall(level, item, mesh, raw, q, d, axis_index, out, rng):
    along_index = 1 - axis_index
    lo, hi = mesh.bounds
    face_coordinate = []
    for sign in [-1, 1]:
        ids = np.where(mesh.face_normals[:, axis_index] * sign > 0.95)[0]
        coords = mesh.triangles_center[ids, axis_index]
        rounded = np.round(coords, 3)
        u, inv = np.unique(rounded, return_inverse=True)
        area = np.bincount(inv, weights=mesh.area_faces[ids])
        face_coordinate.append(float(u[np.argmax(area)]))
    nearby = ((raw[:, along_index] >= lo[along_index] - 0.2)
              & (raw[:, along_index] <= hi[along_index] + 0.2)
              & (raw[:, axis_index] >= min(face_coordinate) - 0.35)
              & (raw[:, axis_index] <= max(face_coordinate) + 0.35)
              & (raw[:, 2] >= 0.05) & (raw[:, 2] <= hi[2] + 0.05))
    local = raw[nearby]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), dpi=180)
    plan = thin(local[np.abs(local[:, 2] - 1.5) <= 0.04], 15000, rng)
    axes[0].scatter(plan[:, along_index], plan[:, axis_index], s=1, c=RAW_COLOUR, alpha=.6)
    for face in face_coordinate:
        axes[0].plot([lo[along_index], hi[along_index]], [face, face], color=MODEL_COLOUR, lw=1.3)
    axes[0].set_title("Plan: LiDAR traces vs modeled faces", loc="left")
    axes[0].set_xlabel("Along wall (m)"); axes[0].set_ylabel("Across wall (m)")
    axes[0].set_ylim(min(face_coordinate) - .25, max(face_coordinate) + .25)
    axes[0].set_box_aspect(1)
    for index, sign in enumerate([-1, 1]):
        ax = axes[index + 1]
        ids = np.where(mesh.face_normals[:, axis_index] * sign > .95)[0]
        tris = mesh.triangles[ids][:, :, [along_index, 2]]
        ax.add_collection(PolyCollection(tris, color=MODEL_COLOUR, alpha=.14, edgecolor="none"))
        p = thin(local[np.abs(local[:, axis_index] - face_coordinate[index]) <= .03], 30000, rng)
        ax.scatter(p[:, along_index], p[:, 2], s=.6, c=RAW_COLOUR, alpha=.55, rasterized=True)
        bad = (d > .10) & (np.abs(q[:, axis_index] - face_coordinate[index]) <= .008)
        ax.scatter(q[bad, along_index], q[bad, 2], s=3, c=DIST_COLOURS[3], alpha=.5, rasterized=True)
        side = item["faces"]["negative" if sign == -1 else "positive"]
        ax.set_title(f"{'Negative' if sign == -1 else 'Positive'} face: {side['within_30mm_pct']:.0f}% within 30 mm", loc="left")
        ax.set_xlabel("Along wall (m)"); ax.set_ylabel("Height above floor (m)")
        ax.set_xlim(lo[along_index] - .12, hi[along_index] + .12); ax.set_ylim(0, hi[2] + .1)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes:
        ax.grid(color="#e1e4e8", lw=.4)
    fig.suptitle(f"SOULACE L{level} / {item['name']} | nearest raw-return evidence, not proof of absence", x=.04, ha="left", fontsize=15)
    fig.text(.5, .025, "Cyan = LiDAR near face   |   pale orange = CAD surface   |   red = modeled sample >100 mm from any LAS return", ha="center", fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.91))
    fig.savefig(out, facecolor="white"); plt.close(fig)


def analyze_level(level, out):
    rng = np.random.default_rng(4700 + level)
    stem = STEMS[level]
    source = ROOT / "output_final" / f"soulace_L{level}"
    export = ROOT / "output_final" / "soulace_asbuilt_v2"
    manifest = json.loads((export / f"{stem}_asbuilt_manifest.json").read_text())
    payload = json.loads((export / f"{stem}_asbuilt.build.json").read_text())["parts"]
    yaw, floor_z = float(manifest["source_yaw_deg"]), float(manifest["source_floor_z_m"])
    wall_info = {w["wall"]: w for w in manifest["walls"]}
    meshes = {p["name"]: trimesh.Trimesh(np.asarray(p["v"]), np.asarray(p["f"]), process=False) for p in payload}
    say(f"L{level}: read full LAS, registered yaw={yaw} and floor={floor_z}")
    raw, z_shift = raw_points(source / "lidar" / f"L{level}.las", yaw, floor_z)
    say(f"L{level}: index all {len(raw):,} raw returns (no reference decimation)")
    tree = cKDTree(raw, leafsize=32, compact_nodes=False)
    records, samples = [], {}
    for p in payload:
        if not (p["kind"].startswith("wall") or p["kind"] == "column"):
            continue
        mesh = meshes[p["name"]]
        axis_index = 0 if wall_info.get(p["name"], {}).get("axis") == "x" else 1
        indices = np.where(np.abs(mesh.face_normals[:, axis_index]) > .95)[0] if p["kind"].startswith("wall") else np.where(np.abs(mesh.face_normals[:, 2]) < .05)[0]
        area = float(mesh.area_faces[indices].sum())
        q, fi = uniform_surface(mesh, indices, max(300, int(area * 250)), rng)
        take = (q[:, 2] >= .15) & (q[:, 2] <= mesh.bounds[1, 2] - .10)
        q, fi = q[take], fi[take]
        d, _ = tree.query(q, workers=8)
        normal_sign = mesh.face_normals[fi, axis_index]
        faces = {label: stats(d[normal_sign * sign > .95]) for sign, label in [(-1, "negative"), (1, "positive")]}
        rec = {"level": level, "name": p["name"], "kind": p["kind"], "area_m2": round(area, 3),
               "axis": "x" if axis_index == 0 else "y", "model_to_lidar": stats(d), "faces": faces}
        values = [s["within_30mm_pct"] for s in faces.values() if s]
        rec["best_face_within_30mm_pct"] = max(values, default=0)
        rec["worst_face_within_30mm_pct"] = min(values, default=0)
        records.append(rec); samples[p["name"]] = (q, d)
    say(f"L{level}: {len(records)} walls/columns scored; now exact Poisson sections")
    with np.load(source / "poisson" / "poisson.npz") as z:
        pv = z["V"].astype(np.float32)
        pf = z["T"]
    rotate_xy(pv, yaw); pv[:, 2] -= floor_z
    preview = thin(pv[(pv[:, 2] > .1) & (pv[:, 2] < 3.0)], 2500, rng)
    poisson_check = stats(tree.query(thin(pv, 50000, rng), workers=8)[0])
    all_bounds = np.array([m.bounds for name, m in meshes.items() if name in wall_info])
    blo, bhi = all_bounds[:, 0].min(0), all_bounds[:, 1].max(0)
    bounds = [float(blo[0] - .35), float(blo[1] - .35), float(bhi[0] + .35), float(bhi[1] + .35)]
    sections = []
    for height in HEIGHTS:
        pp = section_segments(pv, pf, height)
        mm = np.concatenate([section_segments(m.vertices, m.faces, height) for name, m in meshes.items() if name in wall_info or name.startswith("column")])
        nearby = raw[np.abs(raw[:, 2] - height) <= .025]
        rp = thin(nearby, 100000, rng)
        qs = line_samples(mm)
        dd = tree.query(qs, workers=8)[0]
        render_plan(level, height, rp[:, :2], pp, mm, qs, dd, bounds, out / f"Soulace_L{level}_overlay_{height:.1f}m.png")
        sections.append({"height_m": height, "raw_returns_in_slice": len(nearby), "section_to_raw": stats(dd)})
        np.savez_compressed(out / f"L{level}_section_{height:.1f}.npz", raw=rp, poisson=pp, model=mm, samples=qs, distance=dd)
    # A Poisson vertex sample near raw returns checks coordinate consistency,
    # not accuracy of the Poisson surface itself.
    scene = o3d.t.geometry.RaycastingScene()
    combined = trimesh.util.concatenate(list(meshes.values()))
    scene.add_triangles(o3d.core.Tensor(np.asarray(combined.vertices, np.float32)), o3d.core.Tensor(np.asarray(combined.faces, np.uint32)))
    subset = thin(raw, 250000, rng)
    subset = subset[(subset[:, 2] > .15) & (subset[:, 2] < 3.0)]
    reverse = scene.compute_distance(o3d.core.Tensor(subset.astype(np.float32))).numpy()
    ranked = sorted([r for r in records if r["name"] in wall_info], key=lambda r: r["best_face_within_30mm_pct"])
    selected = ranked[:2] + ranked[-1:]
    for rec in selected:
        q, d = samples[rec["name"]]
        render_wall(level, rec, meshes[rec["name"]], raw, q, d, 0 if rec["axis"] == "x" else 1,
                    out / f"Soulace_L{level}_{rec['name']}_evidence.png", rng)
    rp = thin(subset, 6500, rng)
    score_by_name = {r["name"]: r["model_to_lidar"] for r in records}
    model_display = [{"name": p["name"], "kind": p["kind"], "v": np.round(p["v"], 4).tolist(), "f": p["f"], "score": score_by_name.get(p["name"])}
                     for p in payload if p["kind"].startswith("wall") or p["kind"] == "column"]
    report = {"level": level, "las_points": len(raw), "transform": {"yaw_deg": yaw, "las_z_percentile_0_5_m": z_shift,
              "source_floor_z_m": floor_z, "icp_or_scale_adjustment": False},
              "poisson_vertex_to_lidar_frame_check": poisson_check, "scan_to_full_model_interior_including_clutter": stats(reverse),
              "sections": sections, "walls": records, "illustrated_walls": [r["name"] for r in selected]}
    (out / f"Soulace_L{level}_overlap.json").write_text(json.dumps(report, indent=2))
    display = {"level": level, "raw": np.round(rp, 3).ravel().tolist(), "poisson": np.round(preview, 3).astype(float).ravel().tolist(),
               "model": model_display, "stats": report["sections"][1]["section_to_raw"]}
    # round through Python scalars; float32 -> JSON otherwise exposes binary tails.
    display["poisson"] = [round(v, 3) for v in display["poisson"]]
    (out / f"L{level}_display.json").write_text(json.dumps(display, separators=(",", ":")))
    say(f"L{level}: complete. Weakest examples: {[r['name'] for r in ranked[:3]]}; frame median {poisson_check['median_mm']} mm")
    del raw, tree, pv, pf, samples, scene
    gc.collect()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--out", default="output_final/soulace_overlap_audit_20260903")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]
    reports = []
    for level in levels:
        reports.append(json.loads((out / f"Soulace_L{level}_overlap.json").read_text()) if args.assemble_only else analyze_level(level, out))
    fields = ["level", "name", "kind", "area_m2", "median_mm", "p90_mm", "within_10mm_pct", "within_30mm_pct", "beyond_100mm_pct", "best_face_within_30mm_pct", "worst_face_within_30mm_pct"]
    with (out / "wall_overlap_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in reports:
            for wall in r["walls"]:
                flat = {**wall, **wall["model_to_lidar"]}
                writer.writerow({k: flat[k] for k in fields})
    summary = {"project": "Soulace", "model": "full asbuilt_v2 / Soulace_review.skp", "levels": reports,
               "limitations": ["No Egrance/Elegance data found in workspace.",
                   "Distances use every LAS point as reference; display points are sampled.",
                   "Only documented yaw/floor transformations; no per-wall fitting or ICP.",
                   "Model surface queries exclude wall tops/bottoms but include unseen backs and contacts.",
                   "Low support is not proof of hallucination. Free-space/visibility and glass need separate checks.",
                   "Poisson is derived from the same scan, can extrapolate, and is not independent validation.",
                   "Time bins in earlier releases are not independent repeat surveys.",
                   "Neither manifold solids nor overlap demonstrate a certified +/-10 mm room tolerance."]}
    (out / "overlap_audit.json").write_text(json.dumps(summary, indent=2))
    display = {"levels": [json.loads((out / f"L{i}_display.json").read_text()) for i in levels]}
    for entry, report in zip(display["levels"], reports):
        by_name = {r["name"]: r["model_to_lidar"] for r in report["walls"]}
        for part in entry["model"]:
            part["score"] = by_name.get(part["name"])
    (out / "viewer_data.json").write_text(json.dumps(display, separators=(",", ":")))
    template = ROOT / "scripts/export/templates/soulace_overlap_inspector.html"
    if template.exists():
        visual = Path("C:/Users/PC/.codex/visualizations/2026/08/28/01a0484a-abd0-7193-bf04-202d1e39e82b/soulace-scan-overlay.html")
        content = template.read_text(encoding="utf-8").replace("__AUDIT_JSON__", json.dumps(display, separators=(",", ":")))
        if len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("Inline visual exceeds 1 MB; reduce display-only points")
        visual.write_text(content, encoding="utf-8")
        say(f"Interactive overlay: {visual} ({visual.stat().st_size:,} bytes)")
    say(f"Audit complete: {out}")


if __name__ == "__main__":
    main()
