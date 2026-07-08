"""door_leaf_detect.py
-------------------
The scan was taken with some doors OPEN, so an open door LEAF (a ~4cm-thick,
~0.8m-wide, ~2.0m-tall panel) appears stuck to the wall near an entrance and
would be reconstructed as a spurious little wall / would block the opening.

Discriminator: a real WALL runs FLOOR->CEILING; an open door leaf stops at
~2.0m (well below the 2.52m ceiling). So a door leaf is a THIN, door-sized,
elongated panel that has material at chest height but NOT near the ceiling, and
sits next to a doorway. Highlight candidates so we can exclude them.

Plan output:
  grey  = full-height wall footprint
  green = half-height thin door-sized panels = OPEN DOOR LEAVES (to remove)
  red   = other half-height blobs (furniture) -- left alone

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\door_leaf_detect.py <isolated.las> <out_dir>
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

    mid = raster((z > zf + 0.6) & (z < zf + 1.5))          # chest / knee band
    ceil = raster((z > zc - 0.6) & (z < zc - 0.1))         # near-ceiling band
    ceil_d = cv2.dilate(ceil, np.ones((int(0.15 / CELL) | 1,) * 2, np.uint8))
    half = cv2.morphologyEx((mid & (ceil_d == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    half = (half & (ws == 0)).astype(np.uint8)             # exclude the walls themselves

    leaves = np.zeros_like(half); furniture = np.zeros_like(half)
    lbl, n = ndimage.label(half, structure=np.ones((3, 3)))
    wall_near = cv2.dilate(ws, np.ones((int(0.30 / CELL) | 1,) * 2, np.uint8))
    ncand = 0
    for k in range(1, n + 1):
        comp = (lbl == k).astype(np.uint8)
        ys, xs = np.where(comp)
        if xs.size < 3:
            continue
        du = (np.ptp(xs) + 1) * CELL; dv = (np.ptp(ys) + 1) * CELL
        length = max(du, dv); thick = min(du, dv)
        area = xs.size * CELL * CELL
        elong = length / max(thick, CELL)
        near_wall = bool((cv2.dilate(comp, np.ones((3, 3), np.uint8)) & wall_near).any())
        # door leaf: door-width long, thin, elongated, touching a wall (hinge)
        if 0.5 <= length <= 1.3 and thick <= 0.22 and elong >= 2.2 and area <= 0.35 and near_wall:
            leaves |= comp; ncand += 1
        else:
            furniture |= comp

    vis = np.full((H, W, 3), 255, np.uint8)
    vis[ws > 0] = (150, 150, 150)
    vis[furniture > 0] = (0, 0, 255)         # red = furniture / other
    vis[leaves > 0] = (0, 180, 0)            # green = open door leaves
    cv2.imwrite(str(out_dir / "door_leaves_plan.png"), vis)
    log(f"{ncand} open-door-leaf candidates (green). ceiling floor {zc:.2f}, door top ~{zf+2.05:.2f}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
