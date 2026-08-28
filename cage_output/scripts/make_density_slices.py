"""Density maps per height band, on ONE shared footprint.

A top-down density map collapses the whole storey, so it cannot tell a
full-height wall from a downstand beam with open air beneath it -- both project
to the same bright line. Slicing first and projecting each band separately
keeps that distinction: a wall is a boundary present in every band, a beam is
one present only near the ceiling, and a door is a wall that goes missing in
the low bands.

All bands share the same min/max coords, so pixel (i, j) means the same place
in every slice and the stack can be differenced directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from snap_to_planes import load_frame  # noqa: E402

OUT = BASE / "slices"

# (name, low, high) above the measured floor
BANDS = [
    ("s00_kneed",  0.15, 0.55),   # below any sill, above the skirting
    ("s01_sill",   0.55, 0.95),   # window sills live here
    ("s02_mid",    0.95, 1.45),   # clear of most furniture
    ("s03_high",   1.45, 1.95),   # above worktops, below door heads
    ("s04_head",   1.95, 2.35),   # door heads and transoms
    ("s05_beam",   2.35, 2.70),   # downstands, beams, bulkheads
]
CLIP_PCT = 95.0


def density(xy, mn, mx, clip=CLIP_PCT, w=256, h=256):
    res = np.array((w, h))
    c = np.round((xy - mn[None, :]) / (mx[None, :] - mn[None, :]) * res[None])
    c = np.minimum(np.maximum(c, np.zeros_like(res)), res - 1)
    d = np.zeros((h, w), np.float32)
    u, cnt = np.unique(c, return_counts=True, axis=0)
    u = u.astype(np.int32); cnt = cnt.astype(np.float32)
    if clip is not None and cnt.size:
        cnt = np.minimum(cnt, np.percentile(cnt, clip))
    d[u[:, 1], u[:, 0]] = cnt
    return d / d.max() if d.max() > 0 else d


def main():
    OUT.mkdir(exist_ok=True)
    xyz, z_band, centre, _ = load_frame()
    zf = z_band[0]

    # Shared footprint: the same robust bbox the single-slice runs used, so
    # results are comparable with everything already measured.
    var = json.loads((BASE / "scores" / "variants.json").read_text())
    mn = np.array(var["variants"]["robust"]["min_coords"])
    mx = np.array(var["variants"]["robust"]["max_coords"])
    print(f"shared footprint {mx-mn} m, "
          f"{1000*(mx[0]-mn[0])/256:.1f} mm/px", flush=True)

    meta = {}
    for name, lo, hi in BANDS:
        m = (xyz[:, 2] >= zf + lo) & (xyz[:, 2] <= zf + hi)
        d = density(xyz[m][:, :2], mn, mx)
        np.save(OUT / f"density_{name}.npy", d)
        cv2.imwrite(str(OUT / f"density_{name}.png"), (d * 255).astype(np.uint8))
        nz = int((d > 0).sum())
        meta[name] = {"min_coords": mn.tolist(), "max_coords": mx.tolist(),
                      "band_m": [lo, hi], "clip_pct": CLIP_PCT,
                      "n_points": int(m.sum()), "nonzero_px": nz,
                      "mm_per_px": [float(1000*(mx[0]-mn[0])/256),
                                    float(1000*(mx[1]-mn[1])/256)]}
        print(f"{name:12s} {lo:.2f}-{hi:.2f} m  pts={m.sum():>9,}  "
              f"nz={nz:5d} ({100*nz/65536:4.1f}%)", flush=True)

    # merge into the shared variants file so infer.py/eval can find them
    var["variants"].update(meta)
    (BASE / "scores" / "variants.json").write_text(json.dumps(var, indent=2))
    (OUT / "bands.json").write_text(json.dumps(
        {"floor_z": float(zf), "bands": BANDS, "meta": meta}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
