"""LAS -> cleaned, reconstructed mesh (OBJ) - first pass.

Pipeline: load -> voxel downsample -> statistical outlier removal
-> normal estimation -> Poisson surface reconstruction -> low-density trim -> OBJ export.

Usage:
    python scripts/reconstruct_mesh.py [input.las] [output.obj]
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
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
HEARTBEAT_SEC = 15  # progress ping interval during long Open3D calls


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_elapsed(seconds):
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


@contextmanager
def log_stage(name, heartbeat=True):
    """Log stage start/end and periodic 'still running' heartbeats."""
    log(f"{name} — started")
    t0 = time.time()
    stop = threading.Event()

    def ping():
        while not stop.wait(HEARTBEAT_SEC):
            log(f"{name} — still running ({fmt_elapsed(time.time() - t0)} elapsed)")

    thread = None
    if heartbeat:
        thread = threading.Thread(target=ping, daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1)
        log(f"{name} — done ({fmt_elapsed(time.time() - t0)})")


def validate_las_file(path):
    path = Path(path)
    size = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
        point_size = header.point_format.size
        expected = header.offset_to_point_data + header.point_count * point_size
    # LAZ is compressed on disk; only check raw byte size for uncompressed LAS.
    if path.suffix.lower() == ".las" and size < expected:
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
    with log_stage(
        f"Loading point cloud ({size / 1e6:.1f} MB, {point_count:,} points)"
    ):
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
    pipeline_t0 = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_threads = os.cpu_count() or 1
    log(f"Pipeline started — {n_threads} CPU threads available (Poisson uses all)")
    log(f"Settings: voxel={VOXEL_SIZE * 1000:.0f}mm, poisson_depth={POISSON_DEPTH}")

    pcd = load_las_as_o3d(Path(input_path))

    with log_stage(f"Voxel downsampling at {VOXEL_SIZE * 1000:.0f}mm"):
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    log(f"Points after downsample: {len(pcd.points):,}")

    with log_stage("Statistical outlier removal"):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log(f"Points after outlier removal: {len(pcd.points):,}")

    with log_stage(
        f"Normal estimation (hybrid search, radius={VOXEL_SIZE * 4 * 1000:.0f}mm, "
        f"max_nn=30, {len(pcd.points):,} points)"
    ):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 4, max_nn=30
            )
        )

    with log_stage(
        f"Normal orientation (consistent tangent plane, k=15, {len(pcd.points):,} points)"
    ):
        pcd.orient_normals_consistent_tangent_plane(k=15)

    with log_stage(f"Poisson reconstruction (depth={POISSON_DEPTH}, n_threads={n_threads})"):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=POISSON_DEPTH, n_threads=n_threads
            )
    log(f"Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    with log_stage(f"Trimming lowest {DENSITY_TRIM_PERCENTILE}% confidence vertices"):
        densities = np.asarray(densities)
        threshold = np.percentile(densities, DENSITY_TRIM_PERCENTILE)
        mesh.remove_vertices_by_mask(densities < threshold)
    log(f"Trimmed mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    with log_stage("Computing vertex normals"):
        mesh.compute_vertex_normals()

    with log_stage(f"Writing {output_path}"):
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)

    log(f"Done — total wall time {fmt_elapsed(time.time() - pipeline_t0)}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    main(inp, outp)
