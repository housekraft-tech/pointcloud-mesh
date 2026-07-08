"""header_detect.py
----------------
Test the hypothesis that the missing "arch" is a LINTEL / HEADER: material that
exists only near the CEILING and BRIDGES a gap in the mid-height wall footprint
(i.e. spans the top of an opening/passage where there is no wall below).

Outputs a plan:
  grey  = mid-height wall footprint (ws)
  blue  = near-ceiling material that BRIDGES a footprint gap = candidate headers
  red   = near-ceiling material off the walls (room-centre ceiling/soffit blobs)

and prints, for each header span, its length and soffit height (bottom of the
lintel) so we know how far down to extrude it.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\header_detect.py <isolated.las> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]

    def raster(m):
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[m] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[m]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1
        return o

    # near-ceiling band (the lintel/header lives here)
    top = raster((z > zc - 0.6) & (z < zc - 0.08))
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # material OFF the mid footprint (dilated a little so wall thickness is ignored)
    wall_halo = cv2.dilate(ws, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.14 / CELL) | 1,) * 2))
    off = (top & (wall_halo == 0)).astype(np.uint8)

    # a HEADER bridges a footprint gap: its component, when dilated, must touch
    # the wall network on >=2 sides (span an opening). room-centre ceiling blobs
    # touch on 0-1 sides / are compact.
    header = np.zeros_like(off)
    ceiling_blob = np.zeros_like(off)
    lbl, n = ndimage.label(off, structure=np.ones((3, 3)))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.20 / CELL) | 1,) * 2)
    spans = []
    for k in range(1, n + 1):
        comp = (lbl == k).astype(np.uint8)
        ys, xs = np.where(comp)
        if xs.size < 4:
            continue
        du = (xs.max() - xs.min() + 1) * CELL; dv = (ys.max() - ys.min() + 1) * CELL
        elong = max(du, dv); thin = min(du, dv)
        touch = cv2.dilate(comp, grow) & ws
        # count distinct wall regions it touches
        tl, tn = ndimage.label(touch, structure=np.ones((3, 3)))
        bridges = (tn >= 2) and (elong >= 0.5) and (thin <= 1.2)
        if bridges:
            header |= comp
            # soffit height = lowest z of near-wall top material in this span
            cxs = xmin + xs * CELL; cys = ymax - ys * CELL
            zc_lo = []
            for cx, cy in zip(cxs[::7], cys[::7]):
                col = (np.abs(x - cx) < 0.06) & (np.abs(y - cy) < 0.06) & (z > zf + 1.2)
                if col.sum() >= 5:
                    zc_lo.append(np.percentile(z[col], 5))
            soffit = float(np.median(zc_lo)) if zc_lo else zc - 0.4
            spans.append((elong, soffit))
        else:
            ceiling_blob |= comp

    vis = np.full((H, W, 3), 255, np.uint8)
    vis[ws > 0] = (150, 150, 150)
    vis[ceiling_blob > 0] = (0, 0, 255)       # red = ceiling/soffit blob (not a header)
    vis[header > 0] = (255, 60, 0)            # blue = header spans
    cv2.imwrite(str(out_dir / "headers_plan.png"), vis)
    log(f"{len(spans)} header spans (blue). ceiling blobs in red.")
    for i, (l, s) in enumerate(sorted(spans, reverse=True)):
        log(f"  header {i}: span {l:.2f}m  soffit z={s:.2f}  (drop {zc - s:.2f}m below ceiling)")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
