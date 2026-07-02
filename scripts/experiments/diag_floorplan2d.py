"""Diagnostic: run the NEW recon pipeline stages (planes -> wall runs ->
regularize -> room polygonization) on the real isolated koushik cloud and
render one image showing:
  - gray: point density background
  - dim red: unclaimed vertical planes (dropped by the wall/column/beam filters)
  - bright red: TALL unclaimed planes (>=60% storey height -- candidate missed walls)
  - green: regularized wall-run centerlines (what the pipeline keeps today)
  - orange: column footprints
  - blue fill: closed room polygons from shapely polygonize (the actual 2D
    floorplan stage, never before run on real data)
Writes floorplan2d_diag.png + floorplan2d_diag.json into output/koushik_iso/.
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
from scripts.recon.structure import group_wall_runs, extract_columns_beams
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.floorplan2d import build_room_polygons

OUT_DIR = WT + r"\output\koushik_iso"
PPM = 60  # pixels per metre


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    scan = load_scan(OUT_DIR + r"\isolated.las")
    log(f"loaded {scan.n:,} points")

    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    normals = estimate_normals(scan.xyz[sub])
    R = dominant_axes(normals)
    scan = axis_align(scan, R)
    xyz = scan.xyz
    log("axis aligned")

    z = xyz[:, 2]
    z_floor = float(np.percentile(z, 1.0))
    z_ceiling = float(np.percentile(z, 99.0))
    storey = z_ceiling - z_floor
    log(f"z_floor={z_floor:.3f} z_ceiling={z_ceiling:.3f}")

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, used_runs = group_wall_runs(verticals, xyz, np.eye(3), return_used_indices=True)
    cols, beams, used_cb = extract_columns_beams(
        verticals, xyz, z_floor, z_ceiling, return_used_indices=True
    )
    used = used_runs | used_cb
    unclaimed_idx = [i for i in range(len(verticals)) if i not in used]
    log(f"runs={len(runs)} columns={len(cols)} beams={len(beams)} "
        f"unclaimed_vertical={len(unclaimed_idx)}/{len(verticals)}")

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
        log(f"polygonize eps={eps}: {len(polys)} cells, {len(big)} of >=1 m^2, "
            f"areas={sorted(round(p.area, 1) for p in big)}")

    # ---------- render ----------
    xy = xyz[:, :2]
    xmin, ymin = np.percentile(xy, 0.2, axis=0) - 0.5
    xmax, ymax = np.percentile(xy, 99.8, axis=0) + 0.5
    W = int((xmax - xmin) * PPM)
    H = int((ymax - ymin) * PPM)

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    cols_i = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows_i = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)
    density = np.zeros((H, W), dtype=np.float64)
    np.add.at(density, (rows_i, cols_i), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 110).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])

    # room fill first (under the linework), from the looser polygonize pass
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

    # unclaimed vertical planes: dim red all, bright red if tall (candidate walls)
    n_tall = 0
    for i in unclaimed_idx:
        p = verticals[i]
        pts = xyz[p.inlier_idx]
        zspan = pts[:, 2].max() - pts[:, 2].min()
        tall = zspan >= 0.6 * storey
        n_tall += int(tall)
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

    banner = (f"NEW PIPELINE 2D DIAG: {len(verticals)} vert planes -> {len(walls)} walls, "
              f"{len(cols)} cols, {len(beams)} beams | unclaimed {len(unclaimed_idx)} "
              f"({n_tall} tall=candidate missed walls, bright red) | "
              f"rooms>=1m2 closed: {len(rooms[0.40][1])} (eps 0.40) / {len(rooms[0.05][1])} (eps 0.05)")
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_png = OUT_DIR + r"\floorplan2d_diag.png"
    cv2.imwrite(out_png, img)
    log(f"wrote {out_png}")

    summary = {
        "points": int(scan.n),
        "z_floor": z_floor, "z_ceiling": z_ceiling,
        "planes_total": len(planes), "planes_vertical": len(verticals),
        "wall_runs": len(walls), "columns": len(cols), "beams": len(beams),
        "unclaimed_vertical": len(unclaimed_idx), "unclaimed_tall": n_tall,
        "rooms_eps040": sorted(round(p.area, 2) for p in rooms[0.40][1]),
        "rooms_eps005": sorted(round(p.area, 2) for p in rooms[0.05][1]),
        "walls": [
            {"direction": w["direction"], "p0": list(map(float, w["p0"])),
             "p1": list(map(float, w["p1"])),
             "length_m": round(float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))), 3),
             "thickness_m": round(w["thickness_m"], 3),
             "thickness_source": w["thickness_source"],
             "n_steps": len(w["steps"])}
            for w in walls
        ],
    }
    with open(OUT_DIR + r"\floorplan2d_diag.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
