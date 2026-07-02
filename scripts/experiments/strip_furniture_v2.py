"""Furniture strip v2 -- user-specified rule (geometric, not plane-membership):

  KEEP everything within a corridor around wall LINES (nothing is ever placed
       "inside" a wall -- doors, frames, skirting, windows, curtains all live
       there and must survive), full height;
  KEEP a thin constant floor band everywhere (so the floor stays one clean
       plane -- furniture feet sit ABOVE it and get dropped);
  KEEP the ceiling band (ceiling + beams + cornices);
  KEEP column footprints, full height;
  DROP everything else = free-standing clutter in room interiors.

v1's plane-membership filter tore holes in walls (unclaimed wall points were
stripped) and left furniture-foot lumps on the floor. This version cannot
touch walls by construction.

Usage: python strip_furniture_v2.py [koushik|mujammel]
Writes output/<scan>_iso/isolated_structural_v2.las
"""
import sys
import time

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.recon.io_las import load_scan, save_scan_las
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import extract_columns_beams
from scripts.recon.regularize import snap_walls, pair_thickness
from diag_floorplan2d_v3 import group_wall_runs_v3

FLOOR_BAND_M = 0.08     # constant floor: keep only this thin band at floor level
CEIL_BAND_M = 0.45      # ceiling + beams + cornices
WALL_EXTRA_M = 0.22     # corridor beyond each wall's half-thickness
COL_MARGIN_M = 0.15


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
    aligned = axis_align(scan, R)  # same point ORDER -> mask maps back to scan
    xyz = aligned.xyz
    z = xyz[:, 2]
    z_floor = float(np.percentile(z, 1.0))
    z_ceiling = float(np.percentile(z, 99.0))

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, _ = group_wall_runs_v3(verticals, xyz)
    cols, _beams = extract_columns_beams(verticals, xyz, z_floor, z_ceiling)
    walls = pair_thickness(snap_walls(runs, np.eye(3)), xyz)
    log(f"{len(walls)} walls, {len(cols)} columns for the keep-corridor")

    keep = np.zeros(scan.n, dtype=bool)
    keep |= z <= z_floor + FLOOR_BAND_M          # constant floor, everywhere
    keep |= z >= z_ceiling - CEIL_BAND_M         # ceiling + beams

    xy = xyz[:, :2]
    for w in walls:
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        d = p1 - p0
        L2 = float(d @ d)
        if L2 == 0:
            continue
        # distance from every point to this wall SEGMENT (vectorized),
        # corridor width = half thickness + margin, full height
        t_par = np.clip((xy - p0) @ d / L2, 0.0, 1.0)
        closest = p0 + t_par[:, None] * d
        dist = np.linalg.norm(xy - closest, axis=1)
        # steps can protrude (relief/pillar fronts) -- widen by max step offset
        step_reach = max((abs(s.offset_m) for s in w.get("steps", [])), default=0.0)
        corridor = w.get("thickness_m", 0.1) / 2.0 + WALL_EXTRA_M + step_reach
        keep |= dist <= corridor

    for c in cols:
        fp = np.asarray(c.footprint, float)
        lo = fp.min(axis=0) - COL_MARGIN_M
        hi = fp.max(axis=0) + COL_MARGIN_M
        keep |= (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & \
                (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])

    kept = scan.subset(keep)
    log(f"kept {kept.n:,}/{scan.n:,} (dropped {scan.n - kept.n:,} interior clutter)")
    out_las = out_dir + r"\isolated_structural_v2.las"
    save_scan_las(kept, out_las)
    log(f"wrote {out_las}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "koushik")
