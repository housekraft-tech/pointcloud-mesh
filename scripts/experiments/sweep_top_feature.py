"""sweep_top_feature.py
--------------------
Diagnose the wall-TOP arch / column that the mid-level footprint misses.

The skeleton_3d + volumetric flows mask every height by the MID-LEVEL wall
footprint (ws), so any material that exists only near the ceiling (an arch head,
a dropped bulkhead, a column that stops partway) is erased before it can show.
This script bypasses the mask and looks at RAW occupancy:

  * X and Y vertical-section sweeps (raw, unmasked) -> see the height profile
  * a plan overlay: mid-band footprint (grey) vs TOP-only extra material (red)
    -> pinpoints exactly where a top-only feature sits and that it is top-only
  * a per-height occupancy curve -> at what z the extra material starts/stops

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\sweep_top_feature.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint

STEP_CELLS = 1        # sweep EVERY grid column (2cm) -> no aliasing of thin walls
DZ = 0.02


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def montage(imgs, cols=10, cell=140, pad=2):
    rows = (len(imgs) + cols - 1) // cols
    grid = np.zeros((rows * (cell + pad), cols * (cell + pad), 3), np.uint8)
    for i, im in enumerate(imgs):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        s = min(cell / w, cell / h)
        t = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((cell, cell, 3), np.uint8)
        y0 = (cell - t.shape[0]) // 2; x0 = (cell - t.shape[1]) // 2
        canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        r, c = divmod(i, cols)
        grid[r * (cell + pad):r * (cell + pad) + cell, c * (cell + pad):c * (cell + pad) + cell] = canvas
    return grid


def raw_volume(R):
    """Raw per-height occupancy, NO footprint mask, so top-only features survive."""
    x, y, z = R["x"], R["y"], R["z"]
    zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    levels = np.arange(zf, zc + DZ, DZ)
    nz = len(levels)
    vol = np.zeros((H, W, nz), np.uint8)
    for k, zk in enumerate(levels):
        band = (z >= zk - DZ * 0.6) & (z <= zk + DZ * 0.6)
        if band.sum() < 10:
            continue
        cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
        vol[rr, cc, k] = 1
    return vol, levels


def sweep(vol, axis, out_dir, name, spacing_cm):
    d = out_dir / name; d.mkdir(parents=True, exist_ok=True)
    n = vol.shape[axis]
    idxs = np.arange(0, n, STEP_CELLS)         # EVERY column (2cm) -> nothing skipped
    imgs = []
    for j, k in enumerate(idxs):
        sl = np.take(vol, k, axis=axis)
        if axis != 2:                          # vertical section -> height up the image
            sl = np.flipud(sl.T)
        img = (sl > 0).astype(np.uint8) * 255
        cv2.imwrite(str(d / f"{name}_{j:03d}.png"), img)
        imgs.append(img)
    # MAX-PROJECT every section onto one image so a thin feature at ANY plane shows
    stack = np.max(np.stack(imgs), axis=0)
    cv2.imwrite(str(out_dir / f"maxproj_{name}.png"), stack)
    cv2.imwrite(str(out_dir / f"montage_{name}.png"), montage(imgs, cols=20, cell=90))
    log(f"{name}: {len(idxs)} sections @ {spacing_cm:.1f}cm -> montage + maxproj")


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    zf, zc = R["z_floor"], R["z_ceiling"]
    x, y, z = R["x"], R["y"], R["z"]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    storey = zc - zf
    log(f"floor {zf:.2f}  ceiling {zc:.2f}  storey {storey:.2f}m")

    vol, levels = raw_volume(R)
    log(f"raw volume {vol.shape}")

    # vertical-section sweeps (raw) @ native 2cm -> no aliasing of thin walls
    sweep(vol, 1, out_dir, "x_left_to_right", CELL * STEP_CELLS * 100)   # Y-Z elevations
    sweep(vol, 0, out_dir, "y_front_to_back", CELL * STEP_CELLS * 100)   # X-Z elevations

    # ---- plan overlay: MID-band footprint vs TOP-only material ----
    def band_plan(z0, z1):
        m = (z >= z0) & (z < z1)
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[m] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[m]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1
        return cv2.dilate(o, np.ones((3, 3), np.uint8))

    mid = band_plan(zf + 0.9, zf + 1.6)          # chest height = the footprint band
    top = band_plan(zc - 0.35, zc - 0.05)        # just under the ceiling
    top_only = ((top > 0) & (mid == 0)).astype(np.uint8)
    top_only = cv2.morphologyEx(top_only, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    overlay = np.full((H, W, 3), 255, np.uint8)
    overlay[mid > 0] = (150, 150, 150)           # mid footprint = grey
    overlay[top_only > 0] = (0, 0, 255)          # top-only extra = red (BGR)
    cv2.imwrite(str(out_dir / "top_vs_mid_plan.png"), overlay)
    log(f"top-only material cells: {int(top_only.sum())}  (red in top_vs_mid_plan.png)")

    # ---- per-height occupancy curve: where does material start/stop ----
    counts = vol.reshape(-1, vol.shape[2]).sum(0).astype(float)
    counts /= max(counts.max(), 1)
    ch = 400; cw = 900
    curve = np.full((ch, cw, 3), 255, np.uint8)
    for i in range(len(levels) - 1):
        x0 = int(i / (len(levels) - 1) * (cw - 40)) + 20
        x1 = int((i + 1) / (len(levels) - 1) * (cw - 40)) + 20
        y0 = ch - 20 - int(counts[i] * (ch - 40))
        y1 = ch - 20 - int(counts[i + 1] * (ch - 40))
        cv2.line(curve, (x0, y0), (x1, y1), (180, 60, 30), 2)
    for frac, lab in [(0.0, "floor"), (0.5, "mid"), (0.9, "top"), (1.0, "ceil")]:
        xx = int(frac * (cw - 40)) + 20
        cv2.line(curve, (xx, 20), (xx, ch - 20), (210, 210, 210), 1)
        cv2.putText(curve, f"{lab} z={zf+frac*storey:.2f}", (xx - 30, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.imwrite(str(out_dir / "height_occupancy_curve.png"), curve)
    log("wrote height_occupancy_curve.png")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
