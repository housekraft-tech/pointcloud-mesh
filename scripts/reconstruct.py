"""Plane-first modular reconstruction pipeline (CLI).

Phase A (current): ingest -> clean -> isolate the scanned unit, writing the
isolated cloud + a report. Later phases add planes, walls/relief, openings,
solids, floor plan, and modular GLB export.

Usage:
    python scripts/reconstruct.py <scan.las|.laz> <out_dir> [options]
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as a script (python scripts/reconstruct.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.recon import io_las, clean, isolate  # noqa: E402
from scripts.recon.trajectory import approx_trajectory, load_trajectory  # noqa: E402


DEFAULT_CONFIG = {
    # ingest
    "max_points": None,            # uniform subsample cap (None = full resolution)
    # clean
    "crop_lo_pct": 1.0,            # percentile bbox crop to cull gross drift
    "crop_hi_pct": 99.0,
    "crop_margin_m": 0.5,
    "remove_outliers": True,       # Open3D statistical outlier removal
    "outlier_nb": 20,
    "outlier_std_ratio": 2.0,
    # trajectory
    "traj_dt_s": 0.25,             # time-slice width for approx walk-path
    # isolate
    "z_bin_m": 0.05,
    "z_min_height_m": 1.8,
    "z_max_height_m": 4.5,
    "iso_cell_m": 0.25,            # XY occupancy cell for connectivity
    "iso_max_gap_cells": 1,        # bridge gaps up to this many cells
    "iso_max_dist_m": None,        # optional distance-from-path cap (off by default)
    # scope
    "balcony": "include",          # "include" | "stop-at-door"
}


def run_isolation(in_path, out_dir, config, trajectory_path=None):
    """Ingest -> clean -> isolate the unit. Returns (ScanData, stats)."""
    os.makedirs(out_dir, exist_ok=True)

    scan = io_las.load_scan(in_path, max_points=config["max_points"])
    n_loaded = scan.n

    scan = clean.percentile_crop(
        scan, lo=config["crop_lo_pct"], hi=config["crop_hi_pct"], margin_m=config["crop_margin_m"]
    )
    n_cropped = scan.n

    if config["remove_outliers"]:
        scan = clean.remove_outliers(
            scan, nb_neighbors=config["outlier_nb"], std_ratio=config["outlier_std_ratio"]
        )
    n_denoised = scan.n

    if trajectory_path:
        traj = load_trajectory(trajectory_path)
    elif scan.gps_time is not None:
        traj = approx_trajectory(scan.gps_time, scan.xyz, dt_s=config["traj_dt_s"])
    else:
        traj = __import__("numpy").zeros((0, 3))

    z_band = isolate.select_z_band(
        scan.xyz[:, 2],
        bin_m=config["z_bin_m"],
        min_height_m=config["z_min_height_m"],
        max_height_m=config["z_max_height_m"],
    )
    unit, stats = isolate.isolate_unit(
        scan,
        traj,
        z_band,
        cell_m=config["iso_cell_m"],
        max_gap_cells=config["iso_max_gap_cells"],
        max_dist_m=config["iso_max_dist_m"],
    )
    stats.update(
        loaded=n_loaded,
        after_crop=n_cropped,
        after_denoise=n_denoised,
        trajectory_vertices=int(traj.shape[0]),
    )
    return unit, stats


def _write_report(out_dir, in_path, stats, unit):
    xy = unit.xyz[:, :2]
    fx = (xy[:, 0].max() - xy[:, 0].min()) if unit.n else 0.0
    fy = (xy[:, 1].max() - xy[:, 1].min()) if unit.n else 0.0
    lines = [
        f"input: {in_path}",
        f"loaded points: {stats['loaded']}",
        f"after percentile crop: {stats['after_crop']}",
        f"after outlier removal: {stats['after_denoise']}",
        f"trajectory vertices: {stats['trajectory_vertices']}",
        f"z-band (floor, ceiling): ({stats['z_floor']:.3f}, {stats['z_ceiling']:.3f}) m"
        f"  => height {stats['z_ceiling'] - stats['z_floor']:.3f} m",
        f"isolated unit points: {stats['kept']}",
        f"dropped (drift + neighbour): {stats['dropped']}",
        f"unit XY footprint: {fx:.1f} x {fy:.1f} m",
    ]
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plane-first modular reconstruction (Phase A).")
    ap.add_argument("in_path")
    ap.add_argument("out_dir")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--no-outliers", action="store_true", help="skip statistical outlier removal")
    ap.add_argument("--balcony", choices=["include", "stop-at-door"], default="include")
    args = ap.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    if args.max_points is not None:
        config["max_points"] = args.max_points
    if args.no_outliers:
        config["remove_outliers"] = False
    config["balcony"] = args.balcony

    unit, stats = run_isolation(args.in_path, args.out_dir, config, trajectory_path=args.trajectory)
    io_las.save_scan_las(unit, os.path.join(args.out_dir, "isolated.las"))
    for line in _write_report(args.out_dir, args.in_path, stats, unit):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
