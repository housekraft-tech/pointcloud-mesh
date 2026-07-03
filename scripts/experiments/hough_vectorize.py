"""hough_vectorize.py
------------------
Vectorize the CLEAN free-space floorplan into straight wall lines with Hough.

Earlier Hough attempts (rescue2 etc) ran on noisy raw-point binary images and
struggled. The fix is to feed Hough the ACCURATE top-down flow's output --
the free-space carved wall mask (walls_mask.png), which is already clean,
solid, and gap-free. Skeletonize it to wall centerlines, run HoughLinesP,
snap near-axis segments to exact Manhattan H/V, and merge collinear runs ->
a clean CAD-style vector floorplan (+ SVG export).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\hough_vectorize.py <walls_mask.png> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snap_and_merge(lines, angle_tol_deg=8, merge_gap_px=14, coord_tol_px=6):
    """Snap near-horizontal/vertical segments to exact axes, then merge
    collinear ones that share a row/col and overlap or nearly touch."""
    hsegs, vsegs = [], []
    for x1, y1, x2, y2 in lines:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if min(ang, 180 - ang) <= angle_tol_deg:            # horizontal
            yy = int(round((y1 + y2) / 2))
            hsegs.append([min(x1, x2), max(x1, x2), yy])
        elif abs(ang - 90) <= angle_tol_deg:                # vertical
            xx = int(round((x1 + x2) / 2))
            vsegs.append([min(y1, y2), max(y1, y2), xx])

    def merge(segs):
        segs = sorted(segs, key=lambda s: (s[2], s[0]))
        out = []
        for a0, a1, c in segs:
            placed = False
            for m in out:
                if abs(m[2] - c) <= coord_tol_px and a0 <= m[1] + merge_gap_px and a1 >= m[0] - merge_gap_px:
                    m[0] = min(m[0], a0); m[1] = max(m[1], a1)
                    m[2] = int(round((m[2] + c) / 2))
                    placed = True
                    break
            if not placed:
                out.append([a0, a1, c])
        return out

    return merge(hsegs), merge(vsegs)


def main(mask_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log(f"could not read {mask_path}"); return
    _, binm = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    log(f"mask {binm.shape}, {int((binm>0).mean()*100)}% wall")

    # centerlines: skeletonize the thick wall mask so Hough sees 1px lines
    if skeletonize is not None:
        skel = skeletonize(binm > 0).astype(np.uint8) * 255
    else:
        skel = cv2.ximgproc.thinning(binm) if hasattr(cv2, "ximgproc") else binm
    cv2.imwrite(str(out_dir / "hv_skeleton.png"), skel)

    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=30,
                            minLineLength=25, maxLineGap=12)
    n_raw = 0 if lines is None else len(lines)
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4))
    log(f"Hough: {n_raw} raw segments -> {len(hsegs)} H + {len(vsegs)} V merged wall lines")

    # render clean vector floorplan (lines over faint mask)
    canvas = cv2.merge([binm // 5] * 3)
    for a0, a1, y in hsegs:
        cv2.line(canvas, (a0, y), (a1, y), (0, 235, 0), 2, cv2.LINE_AA)
    for a0, a1, x in vsegs:
        cv2.line(canvas, (x, a0), (x, a1), (0, 235, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Hough-vectorized floorplan from free-space walls: "
                        f"{len(hsegs)+len(vsegs)} straight wall lines",
               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "hough_vector_floorplan.png"), canvas)

    # SVG export (CAD-ready straight lines)
    H, W = binm.shape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>']
    for a0, a1, y in hsegs:
        svg.append(f'<line x1="{a0}" y1="{y}" x2="{a1}" y2="{y}" stroke="black" stroke-width="3"/>')
    for a0, a1, x in vsegs:
        svg.append(f'<line x1="{x}" y1="{a0}" x2="{x}" y2="{a1}" stroke="black" stroke-width="3"/>')
    svg.append("</svg>")
    (out_dir / "hough_vector_floorplan.svg").write_text("\n".join(svg))
    log(f"wrote hough_vector_floorplan.png + .svg + hv_skeleton.png -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
