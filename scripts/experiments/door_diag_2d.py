"""door_diag_2d.py
---------------
2D visualisation of the room-neck DOOR detector so we can see hallucinations.

  * watershed rooms  -> translucent colours, numbered
  * walls (footprint) -> dark grey
  * wall segments     -> thin cyan centrelines
  * detected doors    -> RED box at the placed doorway (width labelled)
  * room-room contact -> yellow dots (where two rooms touch after dilation)

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\door_diag_2d.py <las> <out.png>
"""
import sys
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint
from skimage.morphology import skeletonize
from scripts.experiments.hough_vectorize import snap_and_merge

WALL_T = 0.11


def build_segs(R, ws):
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]

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
    hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
    segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL])) for a0, a1, yr in hs] + \
           [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL])) for a0, a1, xc in vs]
    return segs


def main(las, out_png):
    R = reconstruct(las); ws = wall_footprint(R)
    mk = R["mk"]; labs = list(R["room_labels"])
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]
    segs = build_segs(R, ws)

    vis = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(3)
    for A in labs:
        col = tuple(int(c) for c in rng.integers(90, 230, 3))
        vis[mk == A] = col
    vis[ws > 0] = (40, 40, 40)
    for p0, p1 in segs:
        a = (int((p0[0] - xmin) / CELL), int((ymax - p0[1]) / CELL))
        b = (int((p1[0] - xmin) / CELL), int((ymax - p1[1]) / CELL))
        cv2.line(vis, a, b, (255, 200, 0), 1)
    for A in labs:
        ys, xs = np.where(mk == A)
        if xs.size:
            cv2.putText(vis, str(A), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # ---- DOORWAY = a GAP in a wall (between two jambs) with an interior space on
    # both perpendicular sides. Walk each wall segment; where wall material is
    # ABSENT for a 2-3.5ft run, flanked by wall, and free on both sides -> a door.
    # Independent of watershed rooms (which are over-segmented). ----
    wsd = cv2.dilate(ws.astype(np.uint8), np.ones((3, 3)))
    inside = ndimage.binary_fill_holes((R["free"] > 0) | (R["occ"] > 0))

    def solid(pt):
        c = int((pt[0] - xmin) / CELL); r = int((ymax - pt[1]) / CELL)
        return 0 <= r < H and 0 <= c < W and wsd[r, c]

    def interior(pt):
        c = int((pt[0] - xmin) / CELL); r = int((ymax - pt[1]) / CELL)
        return 0 <= r < H and 0 <= c < W and inside[r, c]

    q2px = lambda p: (int((p[0] - xmin) / CELL), int((ymax - p[1]) / CELL))
    # each segment as (horizontal?, line-coord, lo, hi)
    S = []
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.35:
            continue
        hor = abs(dd[0]) >= abs(dd[1])
        if hor:
            S.append((True, (p0[1] + p1[1]) / 2, min(p0[0], p1[0]), max(p0[0], p1[0])))
        else:
            S.append((False, (p0[0] + p1[0]) / 2, min(p0[1], p1[1]), max(p0[1], p1[1])))
    doors = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            hi, ci, lo_i, ho_i = S[i]; hj, cj, lo_j, ho_j = S[j]
            if hi != hj or abs(ci - cj) > 0.22:            # same orientation + same line (jambs colinear)
                continue
            if ho_i <= lo_j:
                gap = lo_j - ho_i; gc = (ho_i + lo_j) / 2
            elif ho_j <= lo_i:
                gap = lo_i - ho_j; gc = (ho_j + lo_i) / 2
            else:
                continue                                    # overlap, same wall
            if not (0.5 < gap < 1.2):                        # a 2-4ft gap between two walls = doorway
                continue
            cl = (ci + cj) / 2
            cen = np.array([gc, cl]) if hi else np.array([cl, gc])
            nn = np.array([0, 1.0]) if hi else np.array([1.0, 0])
            doors.append((cen, gap, interior(cen + nn * 0.5) and interior(cen - nn * 0.5)))
    # dedupe near-duplicates
    kept = []
    for cen, gap, inside2 in doors:
        if any(np.linalg.norm(cen - k[0]) < 0.4 for k in kept):
            continue
        kept.append((cen, gap, inside2))
    ndoor = 0
    for cen, gap, inside2 in kept:
        px, py = q2px(cen)
        if inside2:
            cv2.circle(vis, (px, py), 8, (0, 0, 255), 2)
            cv2.putText(vis, f"{gap:.1f}", (px + 7, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            ndoor += 1
        else:
            cv2.circle(vis, (px, py), 6, (255, 0, 255), 2)   # one side exterior = entrance/balcony, not interior door
    cv2.putText(vis, f"rooms={len(labs)} colinear-gap doors(red)={ndoor}  magenta=exterior-side gap",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out_png, vis)
    print(f"wrote {out_png}  rooms={len(labs)} doors={ndoor}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
