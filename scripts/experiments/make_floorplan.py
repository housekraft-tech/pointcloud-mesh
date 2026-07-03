"""make_floorplan.py
-----------------
Turn a wall skeleton (or any clean wall mask/skeleton PNG) into a PROPER
architectural floorplan:

  1. Hough the skeleton -> straight segments
  2. snap near-axis to exact Manhattan H/V, merge collinear runs
  3. EXTEND endpoints to close corners & T-junctions (a wall stub reaching
     toward a perpendicular line snaps to meet it) -> rooms actually close
  4. draw walls at real thickness on white -> clean CAD-style plan (+SVG)

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\make_floorplan.py <skeleton_or_mask.png> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.hough_vectorize import snap_and_merge

WALL_THICK_PX = 6
SNAP_PX = 30           # extend an endpoint up to this far to meet a perpendicular line
MIN_LEN_PX = 16        # drop stub segments shorter than this
PARALLEL_TOL_PX = 16   # merge parallel overlapping lines within this (wall thickness)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def merge_parallels(segs, tol=PARALLEL_TOL_PX):
    """Collapse doubled parallel lines (both faces of a thick wall) that
    overlap and sit within a wall-thickness of each other into one centerline."""
    segs = sorted(segs, key=lambda s: (s[2], s[0]))
    out = []
    for a0, a1, c in segs:
        placed = False
        for m in out:
            if abs(m[2] - c) <= tol and a0 <= m[1] and a1 >= m[0]:   # near + overlap
                m[0] = min(m[0], a0); m[1] = max(m[1], a1); m[2] = (m[2] + c) // 2
                placed = True
                break
        if not placed:
            out.append([a0, a1, c])
    return out


def close_junctions(hsegs, vsegs, snap=SNAP_PX):
    """Extend H endpoints to any V line they nearly reach (and vice-versa),
    so corners meet and T-junctions close."""
    v_x = [x for _, _, x in vsegs]
    h_y = [y for _, _, y in hsegs]
    # extend horizontals to verticals
    for hs in hsegs:
        a0, a1, y = hs
        for xi, (vy0, vy1, vx) in enumerate(vsegs):
            if vy0 - snap <= y <= vy1 + snap:                 # V spans this row (± slack)
                if 0 < vx - a1 <= snap: hs[1] = vx            # extend right end
                if 0 < a0 - vx <= snap: hs[0] = vx            # extend left end
    # extend verticals to horizontals
    for vs in vsegs:
        b0, b1, x = vs
        for (hx0, hx1, hy) in hsegs:
            if hx0 - snap <= x <= hx1 + snap:
                if 0 < hy - b1 <= snap: vs[1] = hy
                if 0 < b0 - hy <= snap: vs[0] = hy
    return hsegs, vsegs


def main(mask_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"cannot read {mask_path}"); return
    _, binm = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    H, W = binm.shape
    # if the input is a THICK wall footprint (not a 1px skeleton), skeletonize
    # first so Hough sees clean centerlines instead of doubled parallel edges.
    if (binm > 0).mean() > 0.02:
        try:
            from skimage.morphology import skeletonize
            binm = skeletonize(binm > 0).astype(np.uint8) * 255
            log("input was thick -> skeletonized to centerlines")
        except Exception:
            pass

    lines = cv2.HoughLinesP(binm, 1, np.pi / 180, threshold=28,
                            minLineLength=22, maxLineGap=16)
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=18, coord_tol_px=8)
    n0 = len(hsegs) + len(vsegs)
    hsegs, vsegs = merge_parallels(hsegs), merge_parallels(vsegs)   # collapse doubled faces
    hsegs, vsegs = close_junctions(hsegs, vsegs)                    # close corners/T-junctions
    # drop stubs (short leftover fragments), but keep any that got extended
    hsegs = [s for s in hsegs if s[1] - s[0] >= MIN_LEN_PX]
    vsegs = [s for s in vsegs if s[1] - s[0] >= MIN_LEN_PX]
    log(f"{0 if lines is None else len(lines)} raw -> {n0} snapped -> "
        f"{len(hsegs)} H + {len(vsegs)} V (parallels merged, stubs dropped, junctions closed)")

    # ---- render proper walls at thickness on white ----
    plan = np.full((H, W, 3), 255, np.uint8)
    t = WALL_THICK_PX
    for a0, a1, y in hsegs:
        cv2.line(plan, (a0, y), (a1, y), (30, 30, 30), t, cv2.LINE_AA)
    for a0, a1, x in vsegs:
        cv2.line(plan, (x, a0), (x, a1), (30, 30, 30), t, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "floorplan.png"), plan)

    # green-on-dark version too (matches other outputs)
    dark = np.full((H, W, 3), 18, np.uint8)
    for a0, a1, y in hsegs:
        cv2.line(dark, (a0, y), (a1, y), (0, 235, 0), t, cv2.LINE_AA)
    for a0, a1, x in vsegs:
        cv2.line(dark, (x, a0), (x, a1), (0, 235, 0), t, cv2.LINE_AA)
    cv2.putText(dark, f"proper floorplan: {len(hsegs)+len(vsegs)} walls, junctions closed",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "floorplan_dark.png"), dark)

    # SVG
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<rect width="{W}" height="{H}" fill="white"/><g stroke="#1e1e1e" stroke-width="{t}" '
           f'stroke-linecap="square">']
    for a0, a1, y in hsegs:
        svg.append(f'<line x1="{a0}" y1="{y}" x2="{a1}" y2="{y}"/>')
    for a0, a1, x in vsegs:
        svg.append(f'<line x1="{x}" y1="{a0}" x2="{x}" y2="{a1}"/>')
    svg.append("</g></svg>")
    (out_dir / "floorplan.svg").write_text("\n".join(svg))
    log(f"wrote floorplan.png + floorplan_dark.png + floorplan.svg -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
