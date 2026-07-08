"""opening_diag.py
--------------
For every wall segment, render the (u = along-wall, z = height) MATERIAL map with
the search band and any detected empty openings boxed. Lets us see why windows /
balcony doors are or aren't found. Green box = passes current filters; yellow =
an empty region that was rejected (and why, printed).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\opening_diag.py <isolated.las> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint
from skimage.morphology import skeletonize
from scripts.experiments.hough_vectorize import snap_and_merge


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las); ws = wall_footprint(R)
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax = R["xmin"], R["ymax"]
    # single-sided recovery so exterior walls are present (mirror skeleton_3d)
    H, W = R["H"], R["W"]

    def raster(m):
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[m] - xmin) / CELL).astype(int), 0, W - 1); rr = np.clip(((ymax - y[m]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1; return o
    lo = raster((z > zf + 0.25) & (z < zf + 0.75)); hi = raster((z > zc - 0.75) & (z < zc - 0.25))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.10 / CELL) | 1,) * 2)
    fh = cv2.morphologyEx((cv2.dilate(lo, dk) & cv2.dilate(hi, dk)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fat = cv2.dilate(cv2.erode(fh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.28 / CELL) | 1,) * 2)),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.36 / CELL) | 1,) * 2))
    fh = ((fh > 0) & (fat == 0)).astype(np.uint8)
    wallmask = ((ws > 0) | (fh > 0)).astype(np.uint8)
    skel = skeletonize(wallmask > 0).astype(np.uint8)
    ppm = 1.0 / CELL
    lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, threshold=20, minLineLength=int(0.4 * ppm), maxLineGap=int(0.3 * ppm))
    segs = []
    if lines is not None:
        hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
        segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL])) for a0, a1, yr in hs] + \
               [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL])) for a0, a1, xc in vs]
    xy = np.column_stack([x, y])
    UR = 0.02; z0w = zf - 0.05; z1w = zc + 0.05
    tiles = []
    for si, (p0, p1) in enumerate(segs):
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.20) & (u >= 0) & (u <= L)
        if near.sum() < 120:
            continue
        uu, zz = u[near], z[near]
        nu2 = max(int(L / UR) + 1, 4); nz2 = max(int((z1w - z0w) / UR) + 1, 4)
        mat = np.zeros((nz2, nu2), np.uint8)
        mat[np.clip(((zz - z0w) / UR).astype(int), 0, nz2 - 1), np.clip((uu / UR).astype(int), 0, nu2 - 1)] = 1
        mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        core = np.zeros_like(mat); core[int((zf + 0.08 - z0w) / UR):int((zc - 0.03 - z0w) / UR), :] = 1
        empty = cv2.morphologyEx(((core > 0) & (mat == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ncE, lblE, st, _ = cv2.connectedComponentsWithStats(empty, 8)
        img = np.zeros((nz2, nu2, 3), np.uint8)
        img[mat > 0] = (90, 90, 90)
        img[(core > 0) & (mat == 0)] = (40, 40, 90)
        exterior = int((ws[max(0, int((ymax - p0[1]) / CELL))] == 0).sum() > 0)  # rough
        for i in range(1, ncE):
            bx, by, bw, bh, ar = st[i]
            wm = bw * UR; hm = bh * UR; sill = (z0w + by * UR) - zf
            flank = bx > 1 and bx + bw < nu2 - 1
            passes = (0.5 <= wm <= 3.0) and hm >= 0.6 and flank
            col = (0, 200, 0) if passes else (0, 200, 200)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), col, 1)
            if hm >= 0.4 and wm >= 0.4:
                why = "OK" if passes else ("wide" if wm > 3.0 else "narrow" if wm < 0.5 else "short" if hm < 0.6 else "edge" if not flank else "?")
                log(f"seg{si:02d} L{L:.1f}: empty w={wm:.2f} h={hm:.2f} sill={sill:.2f} flank={flank} -> {why}")
        img = np.flipud(img)
        img = cv2.resize(img, (nu2 * 3, nz2 * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"s{si} L{L:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(img)
    # montage
    cell = 240; cols = 6
    rows = (len(tiles) + cols - 1) // cols
    g = np.zeros((rows * (cell + 3), cols * (cell + 3), 3), np.uint8)
    for i, im in enumerate(tiles):
        h, w = im.shape[:2]; s = min(cell / w, cell / h)
        t2 = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        c = np.zeros((cell, cell, 3), np.uint8); y0 = (cell - t2.shape[0]) // 2; x0 = (cell - t2.shape[1]) // 2
        c[y0:y0 + t2.shape[0], x0:x0 + t2.shape[1]] = t2
        r, cc = divmod(i, cols); g[r * (cell + 3):r * (cell + 3) + cell, cc * (cell + 3):cc * (cell + 3) + cell] = c
    cv2.imwrite(str(out_dir / "opening_diag_montage.png"), g)
    log(f"{len(tiles)} segment tiles -> opening_diag_montage.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
