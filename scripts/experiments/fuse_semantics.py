"""fuse_semantics.py
-----------------
Architecture A: SEMANTICS from the architect drawing (RF-DETR elements model,
which only works on real drawings) fused onto the LiDAR metric GEOMETRY.

  1. run the elements model on the architect drawing -> room-type boxes
  2. reconstruct the LiDAR plan -> watershed room centres (metric, validated)
  3. auto-align the two plans (same flat, unknown rotation/mirror/scale):
     normalise both centroid sets, search 4 rotations x 2 mirrors, pick the
     orientation whose nearest-neighbour matching cost is lowest
  4. transfer each drawing room's TYPE onto the nearest LiDAR room
  5. render the LiDAR floorplan with the fused room-type labels

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\fuse_semantics.py <drawing.png> <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.rfdetr_infer import (load, run_model, ELEMENTS_CLASSES,
                                              ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS, ROOM_CLASSES)
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm(P):
    c = P.mean(0)
    s = P.std() + 1e-9
    return (P - c) / s


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def align(Dpx, Lpx):
    """Find the orientation (rot in 0/90/180/270, mirror y/n) mapping drawing
    room centres Dpx onto LiDAR room centres Lpx best. Uses BIJECTIVE (Hungarian)
    assignment so a wrong orientation that piles many rooms onto one match is
    penalised -- nearest-neighbour can't tell 0 from 180 here. Returns
    (cost, rot, mir, P_normalised)."""
    from scipy.optimize import linear_sum_assignment
    Dn, Ln = _norm(Dpx), _norm(Lpx)
    best = None
    for rot in (0, 90, 180, 270):
        for mir in (1.0, -1.0):
            P = (Dn * np.array([mir, 1.0])) @ _rot(np.deg2rad(rot)).T
            C = np.sqrt(((P[:, None] - Ln[None]) ** 2).sum(-1))   # |D| x |L|
            ri, ci = linear_sum_assignment(C)
            cost = C[ri, ci].mean()
            log(f"   orient rot={rot:3d} mirror={'Y' if mir < 0 else 'N'}  cost={cost:.3f}")
            if best is None or cost < best[0]:
                best = (cost, rot, mir, P)
    return best


def main(drawing, las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. drawing room detections
    sess = load("elements")
    if sess is None:
        log("elements model missing"); return
    bgr = cv2.imread(str(drawing))
    dets = run_model(sess, bgr, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    rd = [d for d in dets if d["cls"] in ROOM_CLASSES]
    log(f"drawing: {len(dets)} dets, {len(rd)} room-type boxes: "
        + ", ".join(sorted({d['name'] for d in rd})))
    if len(rd) < 3:
        log("too few room detections to align"); return
    Dpx = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in (d["box"] for d in rd)])
    Dname = [d["name"] for d in rd]

    # 2. LiDAR room centres (metric + pixel)
    R = reconstruct(las)
    xmin, ymax = R["xmin"], R["ymax"]
    Lpx_img, Lmetric = [], []
    for Lr in R["room_labels"]:
        m = (R["mk"] == Lr)
        if m.sum() < (1.0 / (CELL * CELL)):
            continue
        rdist = R["distm"] * m
        cy, cx = np.unravel_index(int(np.argmax(rdist)), rdist.shape)
        Lpx_img.append([cx, cy])
        Lmetric.append([xmin + cx * CELL, ymax - cy * CELL])
    Lpx_img = np.array(Lpx_img); Lmetric = np.array(Lmetric)
    log(f"LiDAR: {len(Lmetric)} room centres")

    # 3. align drawing->LiDAR (use metric centres; y already up)
    cost, rot, mir, Pn = align(Dpx, Lmetric)
    Ln = _norm(Lmetric)
    log(f"best alignment: rot={rot} mirror={'yes' if mir < 0 else 'no'} cost={cost:.3f}")

    # 4. transfer: each LiDAR room gets the nearest drawing room's type
    fused = []
    for i, ln in enumerate(Ln):
        dist = ((Pn - ln) ** 2).sum(1)
        j = int(np.argmin(dist))
        gate = np.sqrt(dist[j])
        fused.append((Dname[j] if gate < 0.6 else "?", gate))

    # 5. render the LiDAR plan with fused labels
    H, W = R["H"], R["W"]
    canvas = np.full((H, W, 3), 255, np.uint8)

    def m2px(px, py):
        return int((px - xmin) / CELL), int((ymax - py) / CELL)
    tpx = max(5, int(round(0.14 / CELL)))
    for p0, p1 in R["walls"]:
        d = p1 - p0; n = np.linalg.norm(d); u = d / n if n > 0 else d
        cv2.line(canvas, m2px(*(p0 - u * 0.14)), m2px(*(p1 + u * 0.14)), (0, 0, 0), tpx, cv2.LINE_8)
    FT = cv2.FONT_HERSHEY_SIMPLEX
    for (cx, cy), (name, gate) in zip(Lpx_img, fused):
        col = (150, 0, 160) if name != "?" else (150, 150, 150)
        (tw, th), _ = cv2.getTextSize(name, FT, 0.5, 1)
        tx = int(np.clip(cx - tw // 2, 2, W - tw - 2))
        cv2.rectangle(canvas, (tx - 2, cy - th - 2), (tx + tw + 2, cy + 4), (255, 255, 255), -1)
        cv2.putText(canvas, name, (tx, cy), FT, 0.5, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(cx), int(cy)), 3, col, -1)
    cv2.putText(canvas, f"LiDAR geometry + RF-DETR room types (drawing->LiDAR, rot {rot})",
                (10, 22), FT, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    p = out_dir / "fused_room_types.png"
    cv2.imwrite(str(p), canvas)
    log(f"wrote {p}")
    for (cx, cy), (name, gate) in zip(Lmetric, fused):
        log(f"   room @ ({cx:+.1f},{cy:+.1f}) m  ->  {name}  (match dist {gate:.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
