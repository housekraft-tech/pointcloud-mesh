"""Prototype of the plan's Task 6 (furniture removal) for the Poisson path:
keep only points explained by STRUCTURAL planes -- floor, ceiling, and
tall (wall-like) vertical planes -- and drop everything else (sofas, tables,
clutter blobs that Poisson melts into lumps). Writes
output/<scan>_iso/isolated_structural.las for re-meshing.

Usage: python strip_furniture.py [koushik|mujammel]
"""
import sys
import time

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from scripts.recon.io_las import load_scan, save_scan_las
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes

WALL_MIN_ZSPAN_FRAC = 0.55   # vertical plane counts as structural if it spans
                             # >= this fraction of the storey height
FLOOR_BAND_M = 0.12          # keep skirting/floor points near the floor plane
CEIL_BAND_M = 0.20           # ceiling fixtures band


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(name):
    out_dir = WT + rf"\output\{name}_iso"
    scan = load_scan(out_dir + r"\isolated.las")
    log(f"{name}: loaded {scan.n:,} points")

    try:
        import open3d as o3d
        o3d.utility.random.seed(0)
    except Exception:
        pass

    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    R = dominant_axes(estimate_normals(scan.xyz[sub]))
    aligned = axis_align(scan, R)  # same point ORDER as scan -> masks transfer
    xyz = aligned.xyz

    z = xyz[:, 2]
    z_floor = float(np.percentile(z, 1.0))
    z_ceiling = float(np.percentile(z, 99.0))
    storey = z_ceiling - z_floor

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    log(f"detect_planes: {len(planes)} planes in {time.time()-t0:.0f}s")

    keep = np.zeros(scan.n, dtype=bool)
    keep |= np.abs(z - z_floor) <= FLOOR_BAND_M
    keep |= np.abs(z - z_ceiling) <= CEIL_BAND_M

    n_struct_planes = 0
    for p in planes:
        if p.label in ("floor", "ceiling"):
            keep[p.inlier_idx] = True
            n_struct_planes += 1
            continue
        pz = z[p.inlier_idx]
        if pz.size and (pz.max() - pz.min()) >= WALL_MIN_ZSPAN_FRAC * storey:
            keep[p.inlier_idx] = True
            n_struct_planes += 1

    kept = scan.subset(keep)  # ORIGINAL (un-rotated) coordinates
    log(f"structural planes kept: {n_struct_planes}/{len(planes)}; "
        f"points kept {kept.n:,}/{scan.n:,} (dropped {scan.n - kept.n:,} furniture/clutter)")

    out_las = out_dir + r"\isolated_structural.las"
    save_scan_las(kept, out_las)
    log(f"wrote {out_las}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "koushik")
