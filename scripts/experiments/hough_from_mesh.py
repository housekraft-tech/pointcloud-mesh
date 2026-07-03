"""hough_from_mesh.py
------------------
Recreate the wall lines by SLICING the output Poisson mesh top-down and
running Hough on the slice (the HouseKraft slice-then-detect idea), using the
accurate mesh as the base instead of noisy raw points.

The saved poisson_mesh_clean.obj is Y-up (Blender convention), so a top-down
horizontal slice is a constant-Y plane and the floor plane is X-Z. We slice
at a few wall heights, rasterize the cross-section polylines into a top-down
binary, morphologically close it, then HoughLinesP -> Manhattan-snapped
straight wall lines (+ SVG).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\hough_from_mesh.py <mesh.obj> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.hough_vectorize import snap_and_merge

PPM = 80


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(mesh_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading {mesh_path} ...")
    mesh = trimesh.load(str(mesh_path), process=False)
    v = np.asarray(mesh.vertices)
    spans = v.max(axis=0) - v.min(axis=0)
    up = int(np.argmin(spans))                 # height = shortest-span axis
    plane = [a for a in (0, 1, 2) if a != up]  # the two floor-plane axes
    up_lo, up_hi = v[:, up].min(), v[:, up].max()
    storey = up_hi - up_lo
    floor = up_lo + 0.10 * storey
    # HouseKraft-style: project the mesh's own VERTICES within a wall-height
    # BAND down to the floor plane (dense + Poisson-denoised), not a razor-thin
    # geometric slice (which is near-empty). This is the mesh's top-down wall
    # footprint.
    band = (v[:, up] >= floor + 0.5) & (v[:, up] <= floor + 2.0)
    vb = v[band][:, plane]
    log(f"up axis = {'XYZ'[up]} (span {storey:.2f}m); {int(band.sum()):,} mesh verts in wall band")
    if vb.shape[0] < 100:
        log("too few band vertices"); return
    minx, miny = vb.min(axis=0) - 0.4
    maxx, maxy = vb.max(axis=0) + 0.4
    W = int((maxx - minx) * PPM) + 1
    H = int((maxy - miny) * PPM) + 1
    cols = np.clip(((vb[:, 0] - minx) * PPM).astype(int), 0, W - 1)
    rows = np.clip(((maxy - vb[:, 1]) * PPM).astype(int), 0, H - 1)
    raster = np.zeros((H, W), np.uint8)
    raster[rows, cols] = 255
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    segs = vb  # for the log line below
    cv2.imwrite(str(out_dir / "mesh_slice_raster.png"), raster)
    log(f"mesh top-down band: {int((raster>0).sum()):,} occupied px -> raster {W}x{H}")

    lines = cv2.HoughLinesP(raster, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.25 * PPM))
    n_raw = 0 if lines is None else len(lines)
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.2 * PPM),
                                      coord_tol_px=int(0.12 * PPM))
    log(f"Hough on mesh slice: {n_raw} raw -> {len(hsegs)} H + {len(vsegs)} V wall lines")

    canvas = cv2.merge([raster // 4] * 3)
    for a0, a1, y in hsegs:
        cv2.line(canvas, (a0, y), (a1, y), (0, 235, 0), 2, cv2.LINE_AA)
    for a0, a1, x in vsegs:
        cv2.line(canvas, (x, a0), (x, a1), (0, 235, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Hough from MESH top-down slice: {len(hsegs)+len(vsegs)} straight wall lines "
                        f"({len(segs)} slice segments)",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "hough_from_mesh.png"), canvas)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for a0, a1, y in hsegs:
        svg.append(f'<line x1="{a0}" y1="{y}" x2="{a1}" y2="{y}" stroke="black" stroke-width="3"/>')
    for a0, a1, x in vsegs:
        svg.append(f'<line x1="{x}" y1="{a0}" x2="{x}" y2="{a1}" stroke="black" stroke-width="3"/>')
    svg.append("</svg>")
    (out_dir / "hough_from_mesh.svg").write_text("\n".join(svg))
    log(f"wrote hough_from_mesh.png + .svg + mesh_slice_raster.png -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
