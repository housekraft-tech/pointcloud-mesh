"""Find open door leaves, so they stop being mistaken for walls.

A door standing open is a planar vertical panel about 2 m tall and 0.6-1.0 m
long, hinged at one end to a wall and sticking out roughly perpendicular to it.
To a top-down projection it looks exactly like a short wall; to the void finder
it looks like the doorway is half blocked. Both are wrong, and both distort the
measured opening width in the same direction -- narrower than the truth.

The panel's own plane sits at a coordinate where no wall was measured, which is
what separates it from structure: mask out everything near a measured wall
face, raster what is left, and look for thin, tall, door-length components with
one end touching a wall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))

from snap_to_planes import load_frame, measure_faces, structural  # noqa: E402

CELL = 0.04
WALL_CLEAR = 0.13     # a point this close to a measured face is wall, not leaf
BAND = (0.35, 1.90)   # above floor: the leaf body, clear of floor and ceiling
LEN_RANGE = (0.50, 1.20)
MAX_THICK = 0.14
MIN_ZSPAN = 1.10
MIN_CELLS = 12
HINGE_MAX = 0.35      # one end must be this close to a wall face


def leaf_candidates(xyz, faces, z_floor):
    m = ((xyz[:, 2] >= z_floor + BAND[0]) & (xyz[:, 2] <= z_floor + BAND[1]))
    pts = xyz[m]
    # Only STRUCTURAL faces define "this is a wall". detect_wall_faces returns
    # every measurable surface -- 361 of them here, about 70 mm apart on
    # average -- and masking around all of them erases the whole cloud. The
    # weak ones are furniture and, importantly, the door leaves themselves.
    sx, sy = structural(faces["x"]), structural(faces["y"])
    fx = np.array([f.value for f in sx]) if sx else np.array([])
    fy = np.array([f.value for f in sy]) if sy else np.array([])
    print(f"  structural faces: {len(fx)} in x, {len(fy)} in y "
          f"(of {len(faces['x'])} / {len(faces['y'])} measured)", flush=True)

    keep = np.ones(len(pts), bool)
    if fx.size:
        keep &= np.min(np.abs(pts[:, 0][:, None] - fx[None, :]), axis=1) > WALL_CLEAR
    if fy.size:
        keep &= np.min(np.abs(pts[:, 1][:, None] - fy[None, :]), axis=1) > WALL_CLEAR
    free = pts[keep]
    print(f"  {len(pts):,} points in band, {len(free):,} clear of every "
          f"measured wall face", flush=True)
    return free, fx, fy


def components(free, cell=CELL):
    mn = free[:, :2].min(axis=0)
    ij = np.floor((free[:, :2] - mn) / cell).astype(int)
    nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    occ = np.zeros((nx, ny), bool)
    occ[ij[:, 0], ij[:, 1]] = True
    zmin = np.full((nx, ny), np.inf, np.float32)
    zmax = np.full((nx, ny), -np.inf, np.float32)
    np.minimum.at(zmin, (ij[:, 0], ij[:, 1]), free[:, 2])
    np.maximum.at(zmax, (ij[:, 0], ij[:, 1]), free[:, 2])
    lab, n = ndimage.label(occ, structure=np.ones((3, 3), bool))
    return lab, n, mn, cell, zmin, zmax


def analyse(lab, n, mn, cell, zmin, zmax, fx, fy):
    out = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) < MIN_CELLS:
            continue
        xy = mn + (idx + 0.5) * cell
        c = xy.mean(axis=0)
        d = xy - c
        cov = d.T @ d / len(d)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        proj = d @ evecs
        length = float(proj[:, 0].max() - proj[:, 0].min())
        thick = float(proj[:, 1].max() - proj[:, 1].min())
        if not (LEN_RANGE[0] <= length <= LEN_RANGE[1]) or thick > MAX_THICK:
            continue
        zs0 = zmin[idx[:, 0], idx[:, 1]]
        zs1 = zmax[idx[:, 0], idx[:, 1]]
        zspan = float(np.nanmax(zs1) - np.nanmin(zs0))
        if zspan < MIN_ZSPAN:
            continue
        # hinge: one end of the long axis must sit against a measured wall
        e0 = c + evecs[:, 0] * proj[:, 0].min()
        e1 = c + evecs[:, 0] * proj[:, 0].max()
        def near_wall(p):
            dx = np.min(np.abs(p[0] - fx)) if fx.size else 9e9
            dy = np.min(np.abs(p[1] - fy)) if fy.size else 9e9
            return min(dx, dy)
        d0, d1 = near_wall(e0), near_wall(e1)
        if min(d0, d1) > HINGE_MAX:
            continue
        out.append({
            "centre": [round(float(c[0]), 4), round(float(c[1]), 4)],
            "end_a": [round(float(e0[0]), 4), round(float(e0[1]), 4)],
            "end_b": [round(float(e1[0]), 4), round(float(e1[1]), 4)],
            "length_m": round(length, 4), "thickness_m": round(thick, 4),
            "z_span_m": round(zspan, 4),
            "hinge_dist_m": round(float(min(d0, d1)), 4),
            "n_cells": int(len(idx)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    a = ap.parse_args()

    xyz, z_band, centre, _ = load_frame()
    zf = z_band[0]
    faces = measure_faces(xyz, z_band)
    free, fx, fy = leaf_candidates(xyz, faces, zf)
    lab, n, mn, cell, zmin, zmax = components(free)
    print(f"  {n} connected components clear of the walls", flush=True)
    leaves = analyse(lab, n, mn, cell, zmin, zmax, fx, fy)

    print(f"\n{len(leaves)} open-door-leaf candidates")
    print(f"{'len mm':>7s} {'thick mm':>9s} {'z span mm':>10s} "
          f"{'hinge mm':>9s} {'cells':>6s}")
    for l in sorted(leaves, key=lambda l: -l["length_m"]):
        print(f"{1000*l['length_m']:7.0f} {1000*l['thickness_m']:9.0f} "
              f"{1000*l['z_span_m']:10.0f} {1000*l['hinge_dist_m']:9.0f} "
              f"{l['n_cells']:6d}")

    dest = BASE / "openings" / f"leaves_{a.run}.json"
    dest.write_text(json.dumps({"z_floor": zf, "params": {
        "cell": CELL, "wall_clear": WALL_CLEAR, "band": BAND,
        "len_range": LEN_RANGE, "max_thick": MAX_THICK,
        "min_zspan": MIN_ZSPAN, "hinge_max": HINGE_MAX},
        "leaves": leaves}, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
