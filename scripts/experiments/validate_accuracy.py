"""validate_accuracy.py
--------------------
Quantify how ACCURATE the reconstructed walls are vs the raw scan (ground truth
of where material actually is). Slice the reconstructed wall solid at chest
height, rasterise it, and compare to the scan's wall occupancy at the same
height:

  * boundary deviation: distance from each scan wall pixel to the nearest
    reconstructed wall face (median / p90) -- how well faces sit on real walls
  * coverage: fraction of scan wall captured (recall)
  * invented: reconstructed wall with NO scan support nearby (precision)

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\validate_accuracy.py <isolated.las> <model.obj> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
import trimesh
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(las, obj, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    x, y, z = R["x"], R["y"], R["z"]; zf, zc = R["z_floor"], R["z_ceiling"]
    xmin, ymax, H, W = R["xmin"], R["ymax"], R["H"], R["W"]

    def raster_pts(m):
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[m] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[m]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1
        return o

    zmid = zf + 1.2
    scan = raster_pts((z > zmid - 0.30) & (z < zmid + 0.30))
    scan = cv2.morphologyEx(scan, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # reconstructed walls: slice the 'walls' geometry at chest height
    sc = trimesh.load(obj)
    walls = None
    if isinstance(sc, trimesh.Scene):
        walls = sc.geometry.get("walls")
        if walls is None:                       # fall back: biggest geometry
            walls = max(sc.geometry.values(), key=lambda g: len(g.faces))
    else:
        walls = sc
    # rasterise all wall triangles crossing the chest-height plane
    v = walls.vertices; f = walls.faces
    tri = v[f]
    zc0 = tri[:, :, 2].min(1); zc1 = tri[:, :, 2].max(1)
    cross = (zc0 <= zmid) & (zc1 >= zmid)
    recon = np.zeros((H, W), np.uint8)
    for t in tri[cross]:
        pix = np.column_stack([((t[:, 0] - xmin) / CELL), ((ymax - t[:, 1]) / CELL)]).astype(np.int32)
        cv2.fillConvexPoly(recon, pix, 1)
    recon = cv2.morphologyEx(recon, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # deviation: scan wall -> nearest recon wall
    dt_recon = ndimage.distance_transform_edt(recon == 0) * CELL * 100   # cm
    dt_scan = ndimage.distance_transform_edt(scan == 0) * CELL * 100
    sp = scan > 0; rp = recon > 0
    dev = dt_recon[sp]
    med = float(np.median(dev)); p90 = float(np.percentile(dev, 90))
    recall = float((dt_recon[sp] <= 8).mean())         # scan within 8cm of recon
    precision = float((dt_scan[rp] <= 8).mean())        # recon within 8cm of scan
    log(f"scan-wall px {int(sp.sum())}, recon-wall px {int(rp.sum())}")
    log(f"boundary deviation scan->recon: median {med:.1f}cm  p90 {p90:.1f}cm")
    log(f"recall (scan wall captured within 8cm): {recall*100:.1f}%")
    log(f"precision (recon wall backed by scan within 8cm): {precision*100:.1f}%")

    vis = np.full((H, W, 3), 255, np.uint8)
    vis[sp & ~rp] = (0, 0, 255)     # red  = scan wall, missed by recon
    vis[rp & ~sp] = (0, 160, 0)     # green= recon wall, no scan support (invented/thickness)
    vis[sp & rp] = (60, 60, 60)     # grey = agree
    cv2.imwrite(str(out_dir / "accuracy_overlay.png"), vis)
    log(f"overlay -> {out_dir}/accuracy_overlay.png  (grey=agree, red=missed, green=recon-only)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
