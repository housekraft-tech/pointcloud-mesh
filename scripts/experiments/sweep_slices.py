"""sweep_slices.py
---------------
Tomographic SWEEPS through the wall volume in all three directions, with each
slice skeletonised to clean centrelines:

  Z sweep  (100)  base -> top      horizontal plan slices
  Y sweep  (100)  front -> back    vertical sections (X-Z elevations)
  X sweep  (100)  left -> right    vertical sections (Y-Z elevations)

Stacking any one direction reconstructs the 3D; the three together are the CT-
style multi-view. Saves every slice + a 10x10 montage per direction.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\sweep_slices.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct
from scripts.experiments.volumetric_3d import wall_footprint, build_volume

N = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def montage(imgs, cols=10, cell=120, pad=2):
    rows = (len(imgs) + cols - 1) // cols
    thumbs = []
    for im in imgs:
        h, w = im.shape[:2]
        s = min(cell / w, cell / h)
        t = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((cell, cell), np.uint8)
        y0 = (cell - t.shape[0]) // 2; x0 = (cell - t.shape[1]) // 2
        canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        thumbs.append(canvas)
    grid = np.zeros((rows * (cell + pad), cols * (cell + pad)), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * (cell + pad):r * (cell + pad) + cell, c * (cell + pad):c * (cell + pad) + cell] = t
    return grid


def skel_slice(sl):
    sl = (sl > 0).astype(np.uint8)
    if sl.sum() < 3:
        return sl * 255
    return skeletonize(sl > 0).astype(np.uint8) * 255


def sweep(vol, axis, out_dir, name):
    d = out_dir / name; d.mkdir(parents=True, exist_ok=True)
    n = vol.shape[axis]
    idxs = np.linspace(0, n - 1, N).astype(int)
    imgs = []
    for j, k in enumerate(idxs):
        sl = np.take(vol, k, axis=axis)
        if axis != 2:                          # vertical section -> put height up the image
            sl = np.flipud(sl.T)
        img = skel_slice(sl)
        cv2.imwrite(str(d / f"{name}_{j:03d}.png"), img)
        imgs.append(img)
    cv2.imwrite(str(out_dir / f"montage_{name}.png"), montage(imgs))
    log(f"{name}: {N} slices -> {d}  (+ montage_{name}.png)")


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    vol, _ = build_volume(R, ws)               # H(y) x W(x) x nz(z)
    log(f"volume {vol.shape}")
    sweep(vol, 2, out_dir, "z_base_to_top")    # horizontal plan slices
    sweep(vol, 0, out_dir, "y_front_to_back")  # vertical sections along Y
    sweep(vol, 1, out_dir, "x_left_to_right")  # vertical sections along X
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
