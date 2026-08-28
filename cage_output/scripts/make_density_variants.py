"""Density-map variants for the CAGE domain-transfer test.

Structured3D density maps come from panorama depth (near-uniform surface
sampling). A handheld LiDAR scan is wildly non-uniform: dwell time and range
make some pixels 25x denser than others, so normalising by the raw max leaves
the floorplan at ~4% brightness. Each variant is one hypothesis about how to
close that gap; all are fed to the same pretrained checkpoint.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
from recon.io_las import load_scan
from recon.isolate import select_z_band, isolate_unit

OUT = Path(__file__).resolve().parent
YAW_DEG = 5.229


def density_from(ps_xy, min_coords, max_coords, width=256, height=256, clip_pct=None):
    image_res = np.array((width, height))
    coordinates = np.round(
        (ps_xy - min_coords[None, :]) / (max_coords[None, :] - min_coords[None, :])
        * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                             image_res - 1)
    density = np.zeros((height, width), dtype=np.float32)
    uniq, counts = np.unique(coordinates, return_counts=True, axis=0)
    uniq = uniq.astype(np.int32)
    counts = counts.astype(np.float32)
    if clip_pct is not None:
        counts = np.minimum(counts, np.percentile(counts, clip_pct))
    density[uniq[:, 1], uniq[:, 0]] = counts
    density = density / density.max()
    return density


def main():
    scan = load_scan(str(ROOT / "koushikexport.las"), max_points=8_000_000)
    z_band = select_z_band(scan.xyz[:, 2])
    unit, stats = isolate_unit(scan, np.zeros((0, 3)), z_band)
    print(f"isolated {stats['kept']:,} pts, z_band={z_band}", flush=True)

    xyz = unit.xyz.copy()
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T
    xy = xyz[:, :2]

    # Robust footprint: the raw min/max is dragged out by a thin drift streak
    # off the balcony, which shrinks the flat to a corner of the image.
    lo_r = np.percentile(xy, 0.5, axis=0)
    hi_r = np.percentile(xy, 99.5, axis=0)
    keep = np.all((xy >= lo_r) & (xy <= hi_r), axis=1)
    xy_r = xy[keep]
    print(f"raw bbox   : {(xy.max(0)-xy.min(0))}", flush=True)
    print(f"robust bbox: {(xy_r.max(0)-xy_r.min(0))}  kept {keep.sum():,}", flush=True)

    variants = {}
    for name, pts, clip in [
        ("raw",        xy,   None),
        ("robust",     xy_r, None),
        ("robust_c99", xy_r, 99.0),
        ("robust_c95", xy_r, 95.0),
        ("robust_c90", xy_r, 90.0),
    ]:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        pad = 0.1 * (mx - mn)
        mn, mx = mn - pad, mx + pad
        d = density_from(pts, mn, mx, clip_pct=clip)
        np.save(OUT / f"density_{name}.npy", d)
        cv2.imwrite(str(OUT / f"density_{name}.png"), (d * 255).astype(np.uint8))
        nz = d[d > 0]
        variants[name] = {
            "min_coords": mn.tolist(), "max_coords": mx.tolist(),
            "mm_per_px": [float((mx[0]-mn[0])/256*1000), float((mx[1]-mn[1])/256*1000)],
            "nonzero_px": int(nz.size), "median_nonzero": float(np.median(nz)),
            "clip_pct": clip, "n_points": int(len(pts)),
        }
        print(f"{name:12s} mm/px={variants[name]['mm_per_px'][0]:5.1f},"
              f"{variants[name]['mm_per_px'][1]:5.1f}  nz={nz.size:5d}"
              f"  median={np.median(nz):.3f}", flush=True)

    (OUT / "variants.json").write_text(json.dumps(
        {"centre_xy": centre.tolist(), "yaw_deg": YAW_DEG,
         "z_band": [float(z_band[0]), float(z_band[1])],
         "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
