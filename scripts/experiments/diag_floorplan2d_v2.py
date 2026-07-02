"""Diagnostic v2: same real-data 2D run as diag_floorplan2d.py, but with an
EXPERIMENTAL wall-run grouping that fixes the two failure modes v1 exposed:

1. u-gap splitting: collinear planes from different physical walls no longer
   merge into one apartment-spanning run -- a run breaks wherever there is a
   > u_gap_m hole in along-wall coverage.
2. corridor-safe relief: max_relief_m drops 1.0 -> 0.35 so two parallel walls
   ~0.5-1.0 m apart (corridor / closet) stay separate runs.
3. rescue thresholds: full_height_frac 0.5 -> 0.4 and min_run_length_m
   1.0 -> 0.5 so occluded/short partitions (bathroom stubs) survive.

Everything else (snap, resolve_corners, pair_thickness, polygonize, render)
is the committed pipeline code, unchanged. Writes floorplan2d_diag_v2.png/json.
"""
import json
import sys
import time

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

import cv2
from types import SimpleNamespace

from scripts.recon.io_las import load_scan
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import (
    _plane_axis_stats, _greedy_chain, extract_columns_beams,
)
from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.floorplan2d import build_room_polygons

OUT_DIR = WT + r"\output\koushik_iso"
PPM = 60


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _chain_by_u_interval(members, u_gap_m):
    """Split a list of plane-stat dicts into groups of overlapping/near
    along-wall (u) intervals: sorted by u_min, a new group starts when the
    next member's u_min is more than u_gap_m past the running u_max."""
    members = sorted(members, key=lambda s: s["u_min"])
    groups, cur, cur_max = [], [], None
    for s in members:
        if cur and s["u_min"] - cur_max > u_gap_m:
            groups.append(cur)
            cur, cur_max = [], None
        cur.append(s)
        cur_max = s["u_max"] if cur_max is None else max(cur_max, s["u_max"])
    if cur:
        groups.append(cur)
    return groups


def group_wall_runs_v2(
    verticals, xyz,
    merge_offset_m=0.15,
    max_relief_m=0.35,
    min_run_length_m=0.5,
    angle_tol_deg=10.0,
    full_height_frac=0.4,
    u_gap_m=1.2,
    min_run_inliers=1200,
):
    xyz = np.asarray(xyz, dtype=float)
    ex, ey = np.eye(3)[:, 0], np.eye(3)[:, 1]
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))

    stats = []
    for i, p in enumerate(verticals):
        s = _plane_axis_stats(p, xyz, ex, ey)
        if s["align"] < cos_tol:
            continue
        s["src_index"] = i
        stats.append(s)
    if not stats:
        return [], set()

    max_z_span = max(s["z_max"] - s["z_min"] for s in stats)
    stats = [s for s in stats if (s["z_max"] - s["z_min"]) >= full_height_frac * max_z_span]

    runs, used = [], set()
    for axis_name in ("x", "y"):
        group = [s for s in stats if s["axis"] == axis_name]
        for offset_cluster in _greedy_chain(group, key=lambda s: s["offset"], max_gap=max_relief_m):
            # NEW: break the offset-cluster wherever along-wall coverage has a hole
            for useg in _chain_by_u_interval(offset_cluster, u_gap_m):
                step_groups = _greedy_chain(useg, key=lambda s: s["offset"], max_gap=merge_offset_m)
                steps = []
                for sg in step_groups:
                    total_n = sum(x["n_inliers"] for x in sg)
                    offset_avg = sum(x["offset"] * x["n_inliers"] for x in sg) / total_n
                    steps.append(WallStep(
                        offset_m=offset_avg,
                        u_min_m=min(x["u_min"] for x in sg),
                        u_max_m=max(x["u_max"] for x in sg),
                        z_min_m=min(x["z_min"] for x in sg),
                        z_max_m=max(x["z_max"] for x in sg),
                    ))
                main_step = max(steps, key=lambda st: st.u_max_m - st.u_min_m)
                if (main_step.u_max_m - main_step.u_min_m) < min_run_length_m:
                    continue
                if sum(x["n_inliers"] for x in useg) < min_run_inliers:
                    continue  # too few points to be a real wall segment
                main_abs = main_step.offset_m
                axis_vec, u_vec = useg[0]["axis_vec"], useg[0]["u_vec"]
                p0 = axis_vec * main_abs + u_vec * main_step.u_min_m
                p1 = axis_vec * main_abs + u_vec * main_step.u_max_m
                rebased = sorted(
                    (WallStep(st.offset_m - main_abs, st.u_min_m, st.u_max_m,
                              st.z_min_m, st.z_max_m) for st in steps),
                    key=lambda st: st.offset_m,
                )
                runs.append(dict(
                    direction=axis_name,
                    normal=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
                    offset_m=main_abs,
                    p0=(float(p0[0]), float(p0[1])),
                    p1=(float(p1[0]), float(p1[1])),
                    steps=rebased,
                ))
                used.update(s["src_index"] for s in useg)
    return runs, used


def main():
    scan = load_scan(OUT_DIR + r"\isolated.las")
    log(f"loaded {scan.n:,} points")

    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    R = dominant_axes(estimate_normals(scan.xyz[sub]))
    scan = axis_align(scan, R)
    xyz = scan.xyz

    z = xyz[:, 2]
    z_floor = float(np.percentile(z, 1.0))
    z_ceiling = float(np.percentile(z, 99.0))
    storey = z_ceiling - z_floor

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, used_runs = group_wall_runs_v2(verticals, xyz)
    cols, beams, used_cb = extract_columns_beams(
        verticals, xyz, z_floor, z_ceiling, return_used_indices=True
    )
    used = used_runs | used_cb
    unclaimed_idx = [i for i in range(len(verticals)) if i not in used]
    log(f"v2 grouping: runs={len(runs)} columns={len(cols)} beams={len(beams)} "
        f"unclaimed={len(unclaimed_idx)}/{len(verticals)}")

    walls = snap_walls(runs, np.eye(3))
    walls = resolve_corners(walls)
    walls = pair_thickness(walls, xyz)

    wall_ns = [
        SimpleNamespace(
            p0=tuple(w["p0"]), p1=tuple(w["p1"]),
            thickness_m=float(w.get("thickness_m", 0.1)),
            length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))),
        )
        for w in walls
    ]
    rooms = {}
    for eps in (0.05, 0.40):
        polys = build_room_polygons(wall_ns, epsilon_m=eps)
        big = [p for p in polys if p.area >= 1.0]
        rooms[eps] = (polys, big)
        log(f"polygonize eps={eps}: {len(polys)} cells, {len(big)} >=1m2, "
            f"areas={sorted(round(p.area, 1) for p in big)}")

    # ---------- render (same style as v1) ----------
    xy = xyz[:, :2]
    xmin, ymin = np.percentile(xy, 0.2, axis=0) - 0.5
    xmax, ymax = np.percentile(xy, 99.8, axis=0) + 0.5
    W, H = int((xmax - xmin) * PPM), int((ymax - ymin) * PPM)

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    cols_i = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows_i = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)
    density = np.zeros((H, W), dtype=np.float64)
    np.add.at(density, (rows_i, cols_i), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 110).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])

    overlay = img.copy()
    for poly in rooms[0.40][1]:
        pts = np.array([to_px(c) for c in poly.exterior.coords], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (140, 60, 0))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for poly in rooms[0.40][1]:
        pts = np.array([to_px(c) for c in poly.exterior.coords], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 160, 40), 2)
        c = poly.centroid
        cv2.putText(img, f"{poly.area:.1f}m2", to_px((c.x, c.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1, cv2.LINE_AA)

    n_tall = 0
    tall_details = []
    for i in unclaimed_idx:
        p = verticals[i]
        pts = xyz[p.inlier_idx]
        zspan = pts[:, 2].max() - pts[:, 2].min()
        tall = zspan >= 0.6 * storey
        n_tall += int(tall)
        if tall:
            tall_details.append({
                "n_inliers": int(len(p.inlier_idx)),
                "z_span": round(float(zspan), 2),
                "normal": [round(float(v), 2) for v in p.normal],
            })
        color = (40, 40, 255) if tall else (30, 30, 130)
        pc = np.clip(((pts[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
        pr = np.clip(((ymax - pts[:, 1]) * PPM).astype(int), 0, H - 1)
        img[pr, pc] = color

    for b in beams:
        cv2.line(img, to_px(b.p0), to_px(b.p1), (140, 140, 0), 1)
    for w in walls:
        cv2.line(img, to_px(w["p0"]), to_px(w["p1"]), (0, 230, 0), 2)
        mid = ((w["p0"][0] + w["p1"][0]) / 2, (w["p0"][1] + w["p1"][1]) / 2)
        label = f'{np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"])):.2f}m/' \
                f'{w["thickness_m"]*1000:.0f}mm({w["thickness_source"][0]})'
        cv2.putText(img, label, to_px(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 255, 200), 1, cv2.LINE_AA)
    for c in cols:
        pts = np.array([to_px(p) for p in c.footprint], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 165, 255), 2)

    banner = (f"V2 EXPERIMENT (u-gap split + rescue): {len(verticals)} vert -> {len(walls)} walls | "
              f"unclaimed {len(unclaimed_idx)} ({n_tall} tall) | "
              f"rooms>=1m2: {len(rooms[0.40][1])} (eps 0.40) / {len(rooms[0.05][1])} (eps 0.05)")
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_png = OUT_DIR + r"\floorplan2d_diag_v2.png"
    cv2.imwrite(out_png, img)
    log(f"wrote {out_png}")

    summary = {
        "grouping": "v2: u_gap_m=1.2, max_relief_m=0.35, full_height_frac=0.4, "
                    "min_run_length_m=0.5, min_run_inliers=1200",
        "planes_vertical": len(verticals),
        "wall_runs": len(walls), "columns": len(cols), "beams": len(beams),
        "unclaimed_vertical": len(unclaimed_idx), "unclaimed_tall": n_tall,
        "unclaimed_tall_details": tall_details,
        "rooms_eps040": sorted(round(p.area, 2) for p in rooms[0.40][1]),
        "rooms_eps005": sorted(round(p.area, 2) for p in rooms[0.05][1]),
        "walls": [
            {"direction": w["direction"],
             "p0": [round(float(v), 3) for v in w["p0"]],
             "p1": [round(float(v), 3) for v in w["p1"]],
             "length_m": round(float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))), 3),
             "thickness_m": round(w["thickness_m"], 3),
             "thickness_source": w["thickness_source"],
             "n_steps": len(w["steps"])}
            for w in walls
        ],
    }
    with open(OUT_DIR + r"\floorplan2d_diag_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
