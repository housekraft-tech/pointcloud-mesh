"""arch_depthmap.py
----------------
For each long wall segment, build a (u = along-wall, z = height) map of the wall
FACE DEPTH (perpendicular setback of the nearest material). Recessed heads/jambs
of cased openings and horizontal shadow-gap reveals show as depth STEPS; a
through-opening shows as a hole. This is the ground-truth measurement of the
"arch" feature: where (in u and z) it starts, how tall it is, how deep it recesses.

Saves a colour heatmap per wall + an annotated one for the wall with the
strongest head recess.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\arch_depthmap.py <isolated.las> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint
from skimage.morphology import skeletonize
from scipy import ndimage as _ndi


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wall_segments(R, ws):
    from scripts.experiments.hough_vectorize import snap_and_merge
    xmin, ymax = R["xmin"], R["ymax"]
    skel = skeletonize(ws > 0).astype(np.uint8)
    ppm = 1.0 / CELL
    lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, threshold=20,
                            minLineLength=int(0.4 * ppm), maxLineGap=int(0.3 * ppm))
    segs = []
    if lines is not None:
        hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
        segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL])) for a0, a1, yr in hs] + \
               [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL])) for a0, a1, xc in vs]
    return segs


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    xy = np.column_stack([x, y])
    segs = wall_segments(R, ws)
    log(f"{len(segs)} wall segments")

    UB = 0.03; ZB = 0.03                 # 3cm along-wall x 3cm height cells
    best = None
    for si, (p0, p1) in enumerate(segs):
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 1.0:
            continue
        dd /= L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.25) & (u >= 0) & (u <= L)
        if near.sum() < 400:
            continue
        uu, zz, pp = u[near], z[near], perp[near]
        for side in (+1.0, -1.0):
            sel = (pp * side) > 0.005
            if sel.sum() < 400:
                continue
            su, sz, sp = uu[sel], zz[sel], pp[sel] * side
            nu = max(int(L / UB), 4); nz = max(int((zc - zf) / ZB), 4)
            depth = np.full((nz, nu), np.nan)
            ui = np.clip((su / UB).astype(int), 0, nu - 1)
            zi = np.clip(((sz - zf) / ZB).astype(int), 0, nz - 1)
            for a in range(nu):
                for b in range(nz):
                    m = (ui == a) & (zi == b)
                    if m.sum() >= 2:
                        depth[b, a] = np.percentile(sp[m], 15)   # nearest face
            valid = ~np.isnan(depth)
            if valid.sum() < 40:
                continue
            face = np.nanpercentile(depth, 15)
            recess = np.where(valid, depth - face, 0.0)
            # score = amount of coherent recess in the TOP third (head region)
            top = zi.max() if False else nz
            headband = recess[int(nz * 0.6):, :]
            score = float(np.nansum((headband > 0.03)))
            # render heatmap (height up)
            img = np.zeros((nz, nu, 3), np.uint8)
            rc = np.clip(recess / 0.10, 0, 1)                    # 0..10cm -> colour
            img[..., 2] = (rc * 255).astype(np.uint8)            # red = recessed
            img[~valid] = (60, 60, 60)                           # grey = no data (hole/opening)
            img = np.flipud(img)
            img = cv2.resize(img, (nu * 4, nz * 8), interpolation=cv2.INTER_NEAREST)
            tag = f"seg{si:02d}_{'A' if side > 0 else 'B'}_L{L:.1f}"
            cv2.imwrite(str(out_dir / f"depth_{tag}.png"), img)
            if best is None or score > best[0]:
                best = (score, tag, L, nu, nz, recess, valid)
    if best:
        score, tag, L, nu, nz, recess, valid = best
        # annotate the strongest head-recess wall
        img = np.zeros((nz, nu, 3), np.uint8)
        rc = np.clip(recess / 0.10, 0, 1)
        img[..., 2] = (rc * 255).astype(np.uint8)
        img[~valid] = (60, 60, 60)
        img = cv2.resize(np.flipud(img), (nu * 6, nz * 12), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{tag}  red=recess(0-10cm)  grey=hole/opening  z up", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / "ARCH_strongest.png"), img)
        # numeric summary of the head band
        head = recess[int(nz * 0.55):, :]
        hv = head[head > 0.03]
        log(f"strongest head recess: {tag}")
        if hv.size:
            log(f"  head-band recess depth: median {np.median(hv)*100:.1f}cm  p90 {np.percentile(hv,90)*100:.1f}cm  cells>3cm {hv.size}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
