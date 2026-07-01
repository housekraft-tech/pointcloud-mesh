"""LAS -> cleaned, reconstructed mesh (OBJ) - first pass.

Pipeline: load (chunked) -> voxel downsample -> statistical outlier removal
-> normal estimation -> Poisson surface reconstruction -> low-density trim -> OBJ export.

Usage:
    python scripts/reconstruct_mesh.py [input.las] [output.obj]
"""

import sys
import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT = ROOT / "output" / "mesh_v1.obj"

VOXEL_SIZE = 0.015  # 15mm - close to native median spacing, caps density unevenness
POISSON_DEPTH = 10  # ~8cm cells at this extent; matches what the point density can support
DENSITY_TRIM_PERCENTILE = 5  # drop the lowest-confidence 5% of Poisson vertices
CHUNK_SIZE = 2_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def validate_las_file(path):
    path = Path(path)
    size = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
        point_size = header.point_format.size
        expected = header.offset_to_point_data + header.point_count * point_size
    if size < expected:
        raise ValueError(
            f"LAS file looks truncated: {size:,} bytes on disk, "
            f"header expects at least {expected:,} bytes "
            f"({header.point_count:,} points x {point_size} bytes). "
            f"Re-upload the full file (~753 MB)."
        )
    return header.point_count, point_size, size


def load_las_as_o3d(path):
    path = Path(path)
    point_count, point_size, size = validate_las_file(path)
    log(
        f"Loading {path} ({size / 1e6:.1f} MB, {point_count:,} points, "
        f"{point_size} bytes/point) ..."
    )

    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z])
    log(f"Loaded {xyz.shape[0]:,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    dims = las.point_format.dimension_names
    if "intensity" in dims:
        gray = (np.asarray(las.intensity, dtype=np.float64) / 255.0).clip(0, 1)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([gray, gray, gray]))
    return pcd


def main(input_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = load_las_as_o3d(Path(input_path))

    log(f"Voxel downsampling at {VOXEL_SIZE * 1000:.0f}mm ...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    log(f"Points after downsample: {len(pcd.points):,}")

    log("Removing statistical outliers ...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log(f"Points after outlier removal: {len(pcd.points):,}")

    log("Estimating normals ...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 4, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    log(f"Running Poisson surface reconstruction (depth={POISSON_DEPTH}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    log(f"Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    log(f"Trimming lowest {DENSITY_TRIM_PERCENTILE}% confidence vertices ...")
    densities = np.asarray(densities)
    threshold = np.percentile(densities, DENSITY_TRIM_PERCENTILE)
    mesh.remove_vertices_by_mask(densities < threshold)
    log(f"Trimmed mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    mesh.compute_vertex_normals()

    log(f"Writing {output_path} ...")
    o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
    log("Done.")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    main(inp, outp)
