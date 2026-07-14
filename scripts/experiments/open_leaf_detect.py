"""open_leaf_detect.py
-------------------
The SLAM scan was taken with some doors OPEN. An open leaf (~0.8m wide, ~2.0m
tall, ~4cm thick) butts against a wall / swings into the room and shows up in the
output as a spurious thin panel (or a wall thickening). It must be REMOVED and a
CLOSED door put at the real entrance.

Detect open leaves and tie them to entrances using the WALK-PATH (the operator
physically walked through every used doorway):
  grey   = wall footprint
  yellow = walk-path (operator trajectory)
  green  = open-leaf candidate NEAR a walk-path wall-crossing (high confidence)
  red    = thin sub-ceiling panel not near the path (furniture / unsure)

A leaf = a thin (<=~0.14m), door-width (0.55-1.1m) panel with material at chest
height but NONE near the ceiling (a real wall/pier is full height).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\open_leaf_detect.py <isolated.las> <out_dir>
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

    # walk-path
    traj = np.zeros((0, 2))
    try:
        import laspy
        from scripts.recon.trajectory import approx_trajectory
        las_o = laspy.read(las)
        if "gps_time" in las_o.point_format.dimension_names:
            xyz = np.column_stack([np.asarray(las_o.x), np.asarray(las_o.y), np.asarray(las_o.z)])
            traj = np.asarray(approx_trajectory(np.asarray(las_o.gps_time), xyz, dt_s=0.25))[:, :2]
    except Exception as e:
        log(f"trajectory unavailable: {e}")
    path_mask = np.zeros((H, W), np.uint8)
    for a, b in zip(traj[:-1], traj[1:]):
        pa = (int((a[0] - xmin) / CELL), int((ymax - a[1]) / CELL))
        pb = (int((b[0] - xmin) / CELL), int((ymax - b[1]) / CELL))
        cv2.line(path_mask, pa, pb, 1, 1)
    # where the path passes THROUGH a wall = an entrance
    entrances = cv2.dilate(path_mask, np.ones((int(0.25 / CELL) | 1,) * 2, np.uint8)) & \
        cv2.dilate(ws, np.ones((int(0.25 / CELL) | 1,) * 2, np.uint8))

    # half-height panels: material at chest, none near ceiling
    mid = raster((z > zf + 0.6) & (z < zf + 1.5))
    ceil = cv2.dilate(raster((z > zc - 0.6) & (z < zc - 0.1)), np.ones((int(0.15 / CELL) | 1,) * 2, np.uint8))
    half = cv2.morphologyEx((mid & (ceil == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    leaves_g = np.zeros_like(half); leaves_r = np.zeros_like(half)
    lbl, n = ndimage.label(half, structure=np.ones((3, 3)))
    ng = nr = 0
    ent_d = cv2.dilate(entrances, np.ones((int(0.6 / CELL) | 1,) * 2, np.uint8))
    for k in range(1, n + 1):
        comp = (lbl == k).astype(np.uint8)
        ys, xs = np.where(comp)
        du = (np.ptp(xs) + 1) * CELL; dv = (np.ptp(ys) + 1) * CELL
        length = max(du, dv); thick = min(du, dv); area = xs.size * CELL * CELL
        if not (0.55 <= length <= 1.15 and thick <= 0.16 and length / max(thick, CELL) >= 2.0 and area <= 0.30):
            continue
        near_ent = bool((comp & ent_d).any())
        if near_ent:
            leaves_g |= comp; ng += 1
        else:
            leaves_r |= comp; nr += 1

    vis = np.full((H, W, 3), 255, np.uint8)
    vis[ws > 0] = (150, 150, 150)
    vis[path_mask > 0] = (0, 210, 235)         # yellow-ish walk path
    vis[leaves_r > 0] = (0, 0, 255)            # red
    vis[leaves_g > 0] = (0, 180, 0)            # green = open leaf near entrance
    cv2.imwrite(str(out_dir / "open_leaves_plan.png"), vis)
    log(f"open-leaf candidates: {ng} near-entrance (green), {nr} other (red); {len(traj)} path pts")
    log(f"done -> {out_dir}/open_leaves_plan.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
