"""Analyze a LAS file: header info, available attributes, and point density.

Usage:
    python scripts/analyze_las.py [path/to/file.las]

Default path points at data/koushikexport.las if no argument is given.
"""

import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "koushikexport.las"
CHUNK_SIZE = 2_000_000
TARGET_SAMPLE_SIZE = 500_000


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main(path):
    path = Path(path)
    print(f"File: {path}")
    print(f"Size: {human_bytes(path.stat().st_size)}\n")

    with laspy.open(path) as f:
        header = f.header
        point_count = header.point_count
        pf = header.point_format
        mins, maxs = np.array(header.mins), np.array(header.maxs)
        extent = maxs - mins

        print("--- Header ---")
        print(f"LAS version: {header.version}")
        print(f"Point format: {pf.id}")
        print(f"Point count: {point_count:,}")
        print(f"Available dimensions: {sorted(pf.dimension_names)}")
        print(f"Bounding box min: {mins}")
        print(f"Bounding box max: {maxs}")
        print(f"Extent (X,Y,Z) meters: {extent}")
        crs = header.parse_crs()
        print(f"CRS: {crs if crs else 'none / local coordinate system'}")
        print()

        has_rgb = "red" in pf.dimension_names
        has_intensity = "intensity" in pf.dimension_names
        has_classification = "classification" in pf.dimension_names
        has_gps_time = "gps_time" in pf.dimension_names

        # Stream through in chunks to keep memory bounded on a 789MB file.
        stride = max(1, point_count // TARGET_SAMPLE_SIZE)
        sample_xyz = []
        class_hist = {}
        intensity_min, intensity_max, intensity_sum, intensity_n = None, None, 0.0, 0
        rgb_min, rgb_max = None, None

        f2 = laspy.open(path)
        with f2 as reader:
            for chunk in reader.chunk_iterator(CHUNK_SIZE):
                sample_xyz.append(
                    np.column_stack([chunk.x[::stride], chunk.y[::stride], chunk.z[::stride]])
                )

                if has_classification:
                    vals, counts = np.unique(chunk.classification, return_counts=True)
                    for v, c in zip(vals, counts):
                        class_hist[int(v)] = class_hist.get(int(v), 0) + int(c)

                if has_intensity:
                    inten = np.asarray(chunk.intensity)
                    cmin, cmax = inten.min(), inten.max()
                    intensity_min = cmin if intensity_min is None else min(intensity_min, cmin)
                    intensity_max = cmax if intensity_max is None else max(intensity_max, cmax)
                    intensity_sum += inten.sum()
                    intensity_n += inten.size

                if has_rgb:
                    rgb = np.column_stack([chunk.red, chunk.green, chunk.blue])
                    cmin, cmax = rgb.min(axis=0), rgb.max(axis=0)
                    rgb_min = cmin if rgb_min is None else np.minimum(rgb_min, cmin)
                    rgb_max = cmax if rgb_max is None else np.maximum(rgb_max, cmax)

        sample = np.concatenate(sample_xyz, axis=0)
        print(f"--- Sampled {sample.shape[0]:,} / {point_count:,} points for spacing analysis ---\n")

        print("--- Available data / attributes ---")
        print(f"RGB color: {'yes' if has_rgb else 'no'}"
              + (f" (range R:{rgb_min[0]}-{rgb_max[0]} G:{rgb_min[1]}-{rgb_max[1]} B:{rgb_min[2]}-{rgb_max[2]})" if has_rgb else ""))
        print(f"Intensity: {'yes' if has_intensity else 'no'}"
              + (f" (range {intensity_min}-{intensity_max}, mean {intensity_sum / intensity_n:.1f})" if has_intensity else ""))
        print(f"GPS time (per-point timestamps): {'yes' if has_gps_time else 'no'}")
        if has_classification:
            print(f"Classification codes present: {class_hist}")
        else:
            print("Classification: no per-point classification field")
        print()

        # Nearest-neighbor spacing on a subsample of the sample (KD-tree on 500k pts is fine,
        # but querying itself against itself for NN is O(n log n) - still do it on full sample).
        print("--- Point spacing (resolution) analysis ---")
        tree = cKDTree(sample)
        # query 2 nearest (1st is the point itself)
        dists, _ = tree.query(sample, k=2, workers=-1)
        nn_dist = dists[:, 1]
        nn_dist = nn_dist[np.isfinite(nn_dist)]

        median_mm = np.median(nn_dist) * 1000
        p90_mm = np.percentile(nn_dist, 90) * 1000
        mean_mm = np.mean(nn_dist) * 1000

        print(f"Median nearest-neighbor spacing: {median_mm:.2f} mm")
        print(f"Mean nearest-neighbor spacing:   {mean_mm:.2f} mm")
        print(f"90th percentile spacing:         {p90_mm:.2f} mm")
        print()
        print("--- Interpretation ---")
        print(f"A feature needs roughly 3x the median spacing across its width to be reliably")
        print(f"resolved in the raw cloud (Nyquist-ish rule of thumb for surface reconstruction).")
        print(f"=> Reliably resolvable groove/ledge width at this density: ~{median_mm * 3:.1f} mm and wider.")
        print(f"Narrower relief than that will be under-sampled and risks being smoothed away")
        print(f"during meshing regardless of reconstruction method.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    main(target)
