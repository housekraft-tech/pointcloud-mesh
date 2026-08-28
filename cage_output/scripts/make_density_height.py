"""Height-banded density maps.

CAGE fails in the open-plan centre. The density map is the suspect: it counts
every point in the storey, so floor return and furniture clutter light up the
middle of a room as brightly as a wall. Structured3D's panorama clouds are
clean, empty rooms -- clutter is the domain gap, not resolution.

A LiDAR scan knows the height of every return, and that is information the
density map throws away. Counting only points in a band above furniture and
below the ceiling should leave walls and nothing else.
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

# (name, low above floor, high above floor, clip percentile)
BANDS = [
    ("wall_10_22", 1.0, 2.2, 95.0),
    ("wall_12_20", 1.2, 2.0, 95.0),
    ("wall_15_24", 1.5, 2.4, 95.0),
    ("wall_10_22_c90", 1.0, 2.2, 90.0),
    ("wall_10_22_noclip", 1.0, 2.2, None),
]


def density_from(xy, mn, mx, clip_pct, width=256, height=256):
    res = np.array((width, height))
    c = np.round((xy - mn[None, :]) / (mx[None, :] - mn[None, :]) * res[None])
    c = np.minimum(np.maximum(c, np.zeros_like(res)), res - 1)
    d = np.zeros((height, width), dtype=np.float32)
    u, cnt = np.unique(c, return_counts=True, axis=0)
    u = u.astype(np.int32); cnt = cnt.astype(np.float32)
    if clip_pct is not None:
        cnt = np.minimum(cnt, np.percentile(cnt, clip_pct))
    d[u[:, 1], u[:, 0]] = cnt
    return d / d.max()


def main():
    scan = load_scan(str(ROOT / "koushikexport.las"), max_points=8_000_000)
    z_band = select_z_band(scan.xyz[:, 2])
    unit, st = isolate_unit(scan, np.zeros((0, 3)), z_band)
    print(f"isolated {st['kept']:,}  z_floor={z_band[0]:.3f} z_ceil={z_band[1]:.3f}",
          flush=True)

    xyz = unit.xyz.copy()
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T

    # Use the SAME footprint as the robust variants so results are comparable
    var = json.loads((OUT / "variants.json").read_text())
    mn = np.array(var["variants"]["robust"]["min_coords"])
    mx = np.array(var["variants"]["robust"]["max_coords"])

    z_floor = z_band[0]
    out = {}
    for name, lo, hi, clip in BANDS:
        m = (xyz[:, 2] >= z_floor + lo) & (xyz[:, 2] <= z_floor + hi)
        pts = xyz[m][:, :2]
        d = density_from(pts, mn, mx, clip)
        np.save(OUT / f"density_{name}.npy", d)
        cv2.imwrite(str(OUT / f"density_{name}.png"), (d * 255).astype(np.uint8))
        nz = d[d > 0]
        out[name] = {"min_coords": mn.tolist(), "max_coords": mx.tolist(),
                     "mm_per_px": [float((mx[0]-mn[0])/256*1000),
                                   float((mx[1]-mn[1])/256*1000)],
                     "band_m": [lo, hi], "clip_pct": clip,
                     "n_points": int(m.sum()), "nonzero_px": int(nz.size)}
        print(f"{name:18s} band={lo}-{hi}m pts={m.sum():>9,} "
              f"nz={nz.size:5d} ({100*nz.size/65536:.1f}% of image)", flush=True)

    var["variants"].update(out)
    (OUT / "variants.json").write_text(json.dumps(var, indent=2))


if __name__ == "__main__":
    main()
