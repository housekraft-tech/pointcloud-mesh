"""multi_slice_mesh.py
-------------------
The HouseKraft multi-height slice flow on the Poisson output mesh: load the
mesh ONCE, take several top-down slices at different heights, Hough each into
wall lines, then merge across heights. Because a wall's presence varies with
height, combining slices reveals structure a single slice can't:

  present at ALL heights    -> solid full-height WALL
  only LOW (below ~1.4m)    -> RAILING / half-wall (the balcony signature)
  only HIGH (above ~1.8m)   -> BEAM soffit / header over an opening

Each slice = mesh vertices in a thin band around that height projected
top-down (dense + Poisson-denoised), rasterized, HoughLinesP, Manhattan-
snapped. Outputs: one panel per height, a combined overlay colour-coded by
height, and a classified floorplan (wall / railing / beam).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\multi_slice_mesh.py <mesh.obj> <out_dir>
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
HALF_BAND = 0.15           # +-band around each slice height (m)
HEIGHTS_ABOVE_FLOOR = [0.30, 0.75, 1.20, 1.65, 2.10, 2.45]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hough_lines_at(vb, minx, maxy, W, H):
    """Rasterize band vertices (Nx2 in floor plane) -> Hough wall lines."""
    raster = np.zeros((H, W), np.uint8)
    if vb.shape[0]:
        cols = np.clip(((vb[:, 0] - minx) * PPM).astype(int), 0, W - 1)
        rows = np.clip(((maxy - vb[:, 1]) * PPM).astype(int), 0, H - 1)
        raster[rows, cols] = 255
        raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    lines = cv2.HoughLinesP(raster, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.25 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.2 * PPM),
                                      coord_tol_px=int(0.12 * PPM))
    return raster, hsegs, vsegs


def main(mesh_path, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "slices").mkdir(parents=True, exist_ok=True)
    log(f"loading {mesh_path} (once) ...")
    mesh = trimesh.load(str(mesh_path), process=False)
    v = np.asarray(mesh.vertices)
    spans = v.max(axis=0) - v.min(axis=0)
    up = int(np.argmin(spans))
    plane = [a for a in (0, 1, 2) if a != up]
    up_lo, up_hi = v[:, up].min(), v[:, up].max()
    floor = up_lo + 0.10 * (up_hi - up_lo)
    log(f"loaded {len(v):,} verts; up={'XYZ'[up]} span {spans[up]:.2f}m; floor~{floor:.2f}")

    # shared canvas extent from a broad wall band
    broad = (v[:, up] >= floor + 0.3) & (v[:, up] <= floor + 2.4)
    P = v[broad][:, plane]
    minx, miny = P.min(axis=0) - 0.4
    maxx, maxy = P.max(axis=0) + 0.4
    W = int((maxx - minx) * PPM) + 1
    H = int((maxy - miny) * PPM) + 1

    per_height = []   # (h_above, hsegs, vsegs)
    for ha in HEIGHTS_ABOVE_FLOOR:
        z = floor + ha
        band = (v[:, up] >= z - HALF_BAND) & (v[:, up] <= z + HALF_BAND)
        vb = v[band][:, plane]
        raster, hsegs, vsegs = hough_lines_at(vb, minx, maxy, W, H)
        per_height.append((ha, hsegs, vsegs))
        panel = cv2.merge([raster // 4] * 3)
        for a0, a1, y in hsegs:
            cv2.line(panel, (a0, y), (a1, y), (0, 235, 0), 2)
        for a0, a1, x in vsegs:
            cv2.line(panel, (x, a0), (x, a1), (0, 235, 0), 2)
        cv2.putText(panel, f"slice @ floor+{ha:.2f}m  {int(band.sum()):,} verts  "
                           f"{len(hsegs)+len(vsegs)} lines", (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "slices" / f"slice_{ha:.2f}m.png"), panel)
        log(f"slice @ {ha:.2f}m: {int(band.sum()):,} verts -> {len(hsegs)+len(vsegs)} lines")

    # ---- combined overlay: every height's lines colour-coded by height ----
    combo = np.zeros((H, W, 3), np.uint8)
    for i, (ha, hsegs, vsegs) in enumerate(per_height):
        t = i / max(len(per_height) - 1, 1)
        col = cv2.applyColorMap(np.uint8([[int(t * 255)]]), cv2.COLORMAP_TURBO)[0, 0].tolist()
        for a0, a1, y in hsegs:
            cv2.line(combo, (a0, y), (a1, y), col, 1, cv2.LINE_AA)
        for a0, a1, x in vsegs:
            cv2.line(combo, (x, a0), (x, a1), col, 1, cv2.LINE_AA)
    cv2.putText(combo, f"{len(HEIGHTS_ABOVE_FLOOR)} mesh slices, coloured by height "
                       f"(blue=floor -> red=ceiling)", (10, 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "slices_by_height.png"), combo)

    # ---- classify each merged wall line by which heights it appears at ----
    # bucket lines onto a coarse (orientation, coord) grid; count heights hit.
    def key(axis, coord):
        return (axis, int(round(coord / (0.15 * PPM))))
    votes = {}   # key -> set of height indices, plus a representative segment
    for i, (ha, hsegs, vsegs) in enumerate(per_height):
        for a0, a1, y in hsegs:
            k = key("h", y); votes.setdefault(k, [set(), ("h", a0, a1, y)])[0].add(i)
        for a0, a1, x in vsegs:
            k = key("v", x); votes.setdefault(k, [set(), ("v", a0, a1, x)])[0].add(i)
    low_i = {i for i, (ha, _, _) in enumerate(per_height) if ha <= 1.35}
    high_i = {i for i, (ha, _, _) in enumerate(per_height) if ha >= 1.80}
    classed = np.zeros((H, W, 3), np.uint8)
    counts = {"wall": 0, "railing": 0, "beam": 0}
    for k, (hset, seg) in votes.items():
        n = len(hset)
        hits_low = bool(hset & low_i); hits_high = bool(hset & high_i)
        if n >= max(3, len(per_height) // 2):
            typ, col = "wall", (235, 235, 235)
        elif hits_low and not hits_high:
            typ, col = "railing", (255, 200, 0)      # low-only
        elif hits_high and not hits_low:
            typ, col = "beam", (60, 60, 255)         # high-only
        else:
            typ, col = "wall", (180, 180, 180)
        counts[typ] = counts.get(typ, 0) + 1
        ax, a0, a1, c = seg
        if ax == "h":
            cv2.line(classed, (a0, c), (a1, c), col, 3, cv2.LINE_AA)
        else:
            cv2.line(classed, (c, a0), (c, a1), col, 3, cv2.LINE_AA)
    for i, (lab, col) in enumerate([("WALL (all heights)", (235, 235, 235)),
                                    ("RAILING (low only)", (255, 200, 0)),
                                    ("BEAM (high only)", (60, 60, 255))]):
        cv2.rectangle(classed, (10, 40 + i * 22 - 9), (24, 40 + i * 22 + 1), col, -1)
        cv2.putText(classed, lab, (30, 40 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(classed, f"mesh multi-slice classified: wall={counts['wall']} "
                         f"railing={counts.get('railing',0)} beam={counts.get('beam',0)}",
               (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "multi_slice_classified.png"), classed)
    log(f"classified: {counts} -> {out_dir}")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
