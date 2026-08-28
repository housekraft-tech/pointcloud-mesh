"""Build a Structured3D-convention top-down density map from the Koushik scan.

Mirrors CAGE/RoomFormer's data_preprocess/stru3d/stru3d_utils.generate_density
exactly (256x256, 10% bbox padding, count histogram normalised by max), so the
pretrained checkpoint sees the input distribution it was trained on.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))

from recon.io_las import load_scan
from recon.isolate import select_z_band, isolate_unit

OUT = Path(__file__).resolve().parent
LAS = ROOT / "koushikexport.las"
MAX_POINTS = 8_000_000
YAW_DEG = 5.229  # from koushik_output/modular/manifest.json


def generate_density(point_cloud, width=256, height=256):
    """Verbatim port of stru3d_utils.generate_density."""
    ps = point_cloud * -1
    ps[:, 0] *= -1
    ps[:, 1] *= -1

    image_res = np.array((width, height))
    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords
    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {"min_coords": min_coords, "max_coords": max_coords,
                          "image_res": image_res}

    coordinates = np.round(
        (ps[:, :2] - min_coords[None, :2])
        / (max_coords[None, :2] - min_coords[None, :2]) * image_res[None])
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                             image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)
    return density, normalization_dict


def main():
    print(f"loading {LAS} (subsample to {MAX_POINTS:,})", flush=True)
    scan = load_scan(str(LAS), max_points=MAX_POINTS)
    print(f"  loaded {scan.n:,} points", flush=True)

    z = scan.xyz[:, 2]
    z_band = select_z_band(z)
    print(f"  z_band = {z_band}", flush=True)

    unit, stats = isolate_unit(scan, np.zeros((0, 3)), z_band)
    print(f"  isolated: kept {stats['kept']:,} dropped {stats['dropped']:,}", flush=True)

    xyz = unit.xyz.copy()
    # centre, then de-rotate by the pipeline's yaw so the flat is axis-aligned
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T

    ext = xyz.max(axis=0) - xyz.min(axis=0)
    print(f"  extent after isolation: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m", flush=True)

    density, nd = generate_density(xyz.copy())

    np.save(OUT / "density.npy", density)
    meta = {
        "min_coords": nd["min_coords"].tolist(),
        "max_coords": nd["max_coords"].tolist(),
        "image_res": nd["image_res"].tolist(),
        "centre_xy": centre.tolist(),
        "yaw_deg": YAW_DEG,
        "z_band": [float(z_band[0]), float(z_band[1])],
        "n_points_used": int(unit.n),
        "extent_m": ext.tolist(),
    }
    (OUT / "density_meta.json").write_text(json.dumps(meta, indent=2))

    import cv2
    cv2.imwrite(str(OUT / "density.png"), (density * 255).astype(np.uint8))
    # metres per density pixel -- needed to convert predicted polygons back to metres
    mx = (nd["max_coords"][0] - nd["min_coords"][0]) / 256.0
    my = (nd["max_coords"][1] - nd["min_coords"][1]) / 256.0
    print(f"  density scale: {mx*1000:.1f} mm/px (x), {my*1000:.1f} mm/px (y)", flush=True)
    print(f"  nonzero pixels: {(density>0).sum()} / 65536", flush=True)
    print("wrote density.npy / density.png / density_meta.json", flush=True)


if __name__ == "__main__":
    main()
