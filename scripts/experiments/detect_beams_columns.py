"""detect_beams_columns.py
-----------------------
Verify (before modelling) that DOWN-STAND BEAMS and COLUMNS actually exist in the
scan and where -- the 3D features the 2D floorplan omits.

  DOWN-STAND BEAM: material near the ceiling that projects BELOW the slab, runs
  in a thin line, is NOT on a wall footprint, and has OPEN space below it (no
  chest-height wall). (A header/lintel is the same but sits over a doorway.)

  COLUMN (free-standing): FULL-HEIGHT material (points near floor AND ceiling),
  compact + roughly square, NOT part of the wall network. (Distinct from a WC/
  sink which is not full height, and from furniture which is not full height.)

  WALL PIER / PILASTER: a compact full-height blob that TOUCHES the wall network
  = a column-like thickening of a wall.

Plan output (per scan):
  grey  = wall footprint
  blue  = down-stand beam candidates
  green = free-standing column candidates
  amber = wall pier / pilaster candidates

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\detect_beams_columns.py <isolated.las> <out_dir>
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

    lo = raster((z > zf + 0.25) & (z < zf + 0.75))         # near floor
    hi = raster((z > zc - 0.75) & (z < zc - 0.25))         # near ceiling
    mid = raster((z > zf + 0.9) & (z < zf + 1.5))          # chest height
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.10 / CELL) | 1,) * 2)
    fullh = cv2.morphologyEx((cv2.dilate(lo, dk) & cv2.dilate(hi, dk)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    halo = cv2.dilate(ws, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.14 / CELL) | 1,) * 2))

    # ---- DOWN-STAND BEAMS: near-ceiling material, off the walls, thin/linear,
    # with OPEN space below (no chest-height material) = a beam crossing a room. ----
    top = cv2.morphologyEx(raster((z > zc - 0.6) & (z < zc - 0.08)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    beam_src = (top & (halo == 0) & (cv2.dilate(mid, dk) == 0)).astype(np.uint8)
    beam_src = cv2.morphologyEx(beam_src, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    beams = np.zeros((H, W), np.uint8); nbeam = 0
    lbl, n = ndimage.label(beam_src, structure=np.ones((3, 3)))
    for k in range(1, n + 1):
        ys, xs = np.where(lbl == k)
        du = (np.ptp(xs) + 1) * CELL; dv = (np.ptp(ys) + 1) * CELL
        length = max(du, dv); thick = min(du, dv)
        if length >= 0.8 and thick <= 0.6 and length / max(thick, CELL) >= 2.0:
            beams |= (lbl == k).astype(np.uint8); nbeam += 1

    # ---- COLUMNS / PIERS: compact full-height blobs. Fat (survive erosion) so a
    # thin wall isn't a column. Free-standing (off wall) vs pier (touches wall). ----
    fat = cv2.erode(fullh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.14 / CELL) | 1,) * 2))
    fat = cv2.dilate(fat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.16 / CELL) | 1,) * 2))
    cols = np.zeros((H, W), np.uint8); piers = np.zeros((H, W), np.uint8)
    ncol = npier = 0
    lbl2, n2 = ndimage.label(fat, structure=np.ones((3, 3)))
    for k in range(1, n2 + 1):
        comp = (lbl2 == k)
        ys, xs = np.where(comp)
        du = (np.ptp(xs) + 1) * CELL; dv = (np.ptp(ys) + 1) * CELL
        area = xs.size * CELL * CELL
        if not (0.10 <= min(du, dv) and max(du, dv) <= 0.9 and area <= 0.7):   # compact, column-sized
            continue
        touches = bool((cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) & (ws > 0)).any())
        if touches:
            piers |= comp.astype(np.uint8); npier += 1
        else:
            cols |= comp.astype(np.uint8); ncol += 1

    vis = np.full((H, W, 3), 255, np.uint8)
    vis[ws > 0] = (150, 150, 150)
    vis[piers > 0] = (0, 170, 235)        # amber-ish (BGR) = wall pier
    vis[beams > 0] = (235, 60, 0)         # blue = down-stand beam
    vis[cols > 0] = (0, 180, 0)           # green = free-standing column
    cv2.imwrite(str(out_dir / "beams_columns_plan.png"), vis)
    log(f"down-stand beams: {nbeam} (blue)   free-standing columns: {ncol} (green)   wall piers: {npier} (amber)")
    log(f"done -> {out_dir}/beams_columns_plan.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
