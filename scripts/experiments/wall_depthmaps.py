"""wall_depthmaps.py
-----------------
Per-wall (u = along-wall, z = height) DEPTH MAP, using the same grid the model's
intrusion/extrusion detector uses, so we can SEE the true depth of every feature
and where it really sits in height -- the clear depth distinction the flat views
can't show.

Colour (relative to each wall side's DOMINANT/base plane):
  grey  = wall at base depth
  red   = INTRUSION (recessed, wall thinner here)   -- darker = deeper
  blue  = EXTRUSION (proud, wall thicker here)
  black = no data
Floor / ceiling gridlines drawn; z increases UP.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\wall_depthmaps.py <isolated.las> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint
from scripts.experiments.hough_vectorize import snap_and_merge

UB = 0.04; ZB = 0.05; MIN_DEPTH = 0.030


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax = R["xmin"], R["ymax"]; xy = np.column_stack([x, y])
    skel = skeletonize(ws > 0).astype(np.uint8); ppm = 1 / CELL
    lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, threshold=20,
                            minLineLength=int(0.4 * ppm), maxLineGap=int(0.3 * ppm))
    hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
    segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL])) for a0, a1, yr in hs] + \
           [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL])) for a0, a1, xc in vs]

    tiles = []
    for si, (p0, p1) in enumerate(segs):
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.6:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.22) & (u >= 0) & (u <= L)
        if near.sum() < 200:
            continue
        uu, zz, pp = u[near], z[near], perp[near]
        for side in (+1.0, -1.0):
            sel = (pp * side) > 0.004
            if sel.sum() < 200:
                continue
            sp = pp[sel] * side; sz = zz[sel]; su = uu[sel]
            zb = np.arange(zf + 0.10, zc - 0.02, ZB); nz = len(zb)
            nu = max(int(L / UB), 2)
            ui = np.clip((su / UB).astype(int), 0, nu - 1)
            zi = np.clip(((sz - (zf + 0.10)) / ZB).astype(int), 0, nz - 1)
            depth = np.full((nz, nu), np.nan)
            for a in range(nu):
                ma = ui == a
                if not ma.any():
                    continue
                spa, zia = sp[ma], zi[ma]
                for b in np.unique(zia):
                    cm = zia == b
                    if cm.sum() >= 2:
                        depth[b, a] = np.percentile(spa[cm], 15)
            valid = ~np.isnan(depth)
            if valid.sum() < 20:
                continue
            base = float(np.nanmedian(depth[valid]))
            delta = np.where(valid, depth - base, 0.0)      # + intrusion, - extrusion
            # colour map
            img = np.zeros((nz, nu, 3), np.uint8)
            for b in range(nz):
                for a in range(nu):
                    if not valid[b, a]:
                        continue
                    d = delta[b, a]
                    if d >= MIN_DEPTH:                        # intrusion -> red
                        s = min(d / 0.10, 1.0)
                        img[b, a] = (60, 60, int(120 + 135 * s))
                    elif d <= -MIN_DEPTH:                     # extrusion -> blue
                        s = min(-d / 0.10, 1.0)
                        img[b, a] = (int(120 + 135 * s), 60, 60)
                    else:
                        img[b, a] = (110, 110, 110)           # base wall
            img = np.flipud(img)                              # z up
            sc = 6
            img = cv2.resize(img, (nu * sc, nz * sc), interpolation=cv2.INTER_NEAREST)
            # floor(0) already bottom; mark ceiling line near top (grid tops at zc-0.02)
            cv2.putText(img, f"s{si:02d}{'A' if side > 0 else 'B'} L{L:.1f} th~{(np.percentile(sp,95)):.2f}",
                        (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(img)
    # montage
    cell = 150; cols = 8
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * (cell + 4), cols * (cell + 4), 3), np.uint8)
    for i, im in enumerate(tiles):
        h, w = im.shape[:2]; s = min(cell / w, cell / h)
        t = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        c = np.zeros((cell, cell, 3), np.uint8)
        c[:t.shape[0], :t.shape[1]] = t
        r, cc = divmod(i, cols)
        grid[r * (cell + 4):r * (cell + 4) + cell, cc * (cell + 4):cc * (cell + 4) + cell] = c
    cv2.imwrite(str(out_dir / "wall_depthmaps.png"), grid)
    log(f"{len(tiles)} wall-side depth maps (red=intrusion, blue=extrusion, grey=base) -> wall_depthmaps.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
