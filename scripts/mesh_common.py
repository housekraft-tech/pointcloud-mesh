"""Shared mesh reconstruction pipeline."""

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_SEC = 15


@dataclass
class MeshConfig:
    name: str
    voxel_size: float
    poisson_depth: int
    density_trim_percentile: int = 5
    outlier_std_ratio: float = 2.0
    crop_radius_m: float | None = None
    max_points: int | None = None
    skip_outlier_removal: bool = False


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
    if path.suffix.lower() == ".las" and size < expected:
        raise ValueError(
            f"LAS file looks truncated: {size:,} bytes on disk, "
            f"header expects at least {expected:,} bytes. Re-upload the full file."
        )
    return header


def _cloud_center(las):
    mins = np.array(las.header.mins)
    maxs = np.array(las.header.maxs)
    return (mins + maxs) / 2.0


def _header_center(header):
    mins = np.array(header.mins)
    maxs = np.array(header.maxs)
    return (mins + maxs) / 2.0


def _chunk_colors_from_las(chunk):
    dims = chunk.point_format.dimension_names
    if "intensity" not in dims:
        return None
    gray = (np.asarray(chunk.intensity, dtype=np.float64) / 255.0).clip(0, 1)
    return np.column_stack([gray, gray, gray])


def _stream_las_points(path, header, crop_radius_m, max_points):
    center = _header_center(header)
    radius_sq = crop_radius_m**2 if crop_radius_m is not None else None
    if crop_radius_m is not None:
        log(
            f"Spatial crop (streaming): {crop_radius_m:.1f}m radius around "
            f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        )
    if max_points is not None:
        log(f"Point cap: stop after {max_points:,} points kept")

    chunk_size = 500_000
    if max_points is not None:
        chunk_size = min(chunk_size, max(max_points, 1))

    xyz_parts = []
    color_parts = []
    has_colors = False
    total_kept = 0
    total_scanned = 0

    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                [np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)]
            )
            total_scanned += len(xyz)
            colors = _chunk_colors_from_las(chunk)

            if radius_sq is not None:
                dist_sq = np.sum((xyz - center) ** 2, axis=1)
                mask = dist_sq <= radius_sq
                xyz = xyz[mask]
                if colors is not None:
                    colors = colors[mask]

            if len(xyz) == 0:
                continue

            if max_points is not None:
                remaining = max_points - total_kept
                if remaining <= 0:
                    break
                if len(xyz) > remaining:
                    xyz = xyz[:remaining]
                    if colors is not None:
                        colors = colors[:remaining]

            xyz_parts.append(xyz)
            if colors is not None:
                color_parts.append(colors)
                has_colors = True
            total_kept += len(xyz)

            if max_points is not None and total_kept >= max_points:
                log(f"Reached point cap ({max_points:,}); stopping stream")
                break

    if crop_radius_m is not None or max_points is not None:
        log(
            f"Streamed {total_scanned:,} points from file, kept {total_kept:,} "
            f"for meshing"
        )

    if total_kept == 0 and crop_radius_m is not None:
        log(
            f"WARNING: spatial crop ({crop_radius_m:.1f}m) kept 0 points after scanning "
            f"{total_scanned:,}; retrying without crop"
        )
        return _stream_las_points(path, header, None, max_points)

    if not xyz_parts:
        xyz = np.empty((0, 3), dtype=np.float64)
        colors = None
    else:
        xyz = np.vstack(xyz_parts)
        colors = np.vstack(color_parts) if has_colors and color_parts else None

    return xyz, colors


def load_las_as_o3d(path, crop_radius_m=None, max_points=None):
    path = Path(path)
    header = validate_las_file(path)
    size = path.stat().st_size
    use_streaming = crop_radius_m is not None or max_points is not None

    with log_stage(
        f"Loading point cloud ({size / 1e6:.1f} MB, {header.point_count:,} points in file)"
    ):
        if use_streaming:
            xyz, colors = _stream_las_points(path, header, crop_radius_m, max_points)
        else:
            las = laspy.read(path)
            xyz = np.column_stack([las.x, las.y, las.z])
            colors = _chunk_colors_from_las(las)

            if crop_radius_m is not None:
                center = _cloud_center(las)
                dist = np.linalg.norm(xyz - center, axis=1)
                mask = dist <= crop_radius_m
                kept = int(mask.sum())
                log(
                    f"Spatial crop: {crop_radius_m:.1f}m radius around "
                    f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
                    f"-> {kept:,} / {len(xyz):,} points"
                )
                xyz = xyz[mask]
                if colors is not None:
                    colors = colors[mask]

    if xyz.shape[0] == 0:
        raise ValueError(
            "No points loaded for meshing. Check LAS file contents, crop_radius_m, "
            "and max_points settings."
        )

    log(f"Using {xyz.shape[0]:,} points for meshing")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def recenter_pcd(pcd):
    """Subtract point mean so KD-tree normal search works on large survey coordinates."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    center = pts.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector(pts - center)
    log(
        f"Recentered cloud by ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
        f"for stable normal estimation"
    )
    return center


def crop_pcd_to_percentile_bounds(pcd, low_pct=1.0, high_pct=99.0, margin_m=0.5):
    """Phase 0: drop the sparse SLAM-drift/ghost-point tail that inflates a
    raw bounding box (confirmed on real scans: 99% of points sit in a tight
    room volume while raw bbox balloons 6-7x from stray points), which was
    the dominant cause of ~30 minute Poisson reconstruction times."""
    from floorplan_geometry import crop_to_percentile_bounds  # local import: keeps the pure module import-order independent of mesh_common
    xyz = np.asarray(pcd.points)
    _lo, _hi, keep_mask, stats = crop_to_percentile_bounds(xyz, low_pct, high_pct, margin_m)
    keep_idx = np.nonzero(keep_mask)[0]
    cropped = pcd.select_by_index(keep_idx)
    log(
        f"Phase 0 crop: kept {stats['kept_points']:,}/{stats['input_points']:,} points "
        f"({stats['dropped_fraction']*100:.2f}% dropped); "
        f"bounds {stats['raw_bounds_min']} -> {stats['raw_bounds_max']} "
        f"became {stats['cropped_bounds_min']} -> {stats['cropped_bounds_max']}"
    )
    return cropped, stats


def _finite_normal_fraction(pcd):
    if not pcd.has_normals():
        return 0.0
    normals = np.asarray(pcd.normals)
    if normals.size == 0:
        return 0.0
    return float(np.isfinite(normals).all(axis=1).mean())


def _normals_acceptable(pcd, min_fraction=0.9):
    return pcd.has_normals() and _finite_normal_fraction(pcd) >= min_fraction


def _try_estimate_normals(pcd, search_param, label, fast_normal_computation=False):
    pcd.estimate_normals(
        search_param=search_param,
        fast_normal_computation=fast_normal_computation,
    )
    frac = _finite_normal_fraction(pcd)
    ok = _normals_acceptable(pcd)
    log(
        f"Normal estimate ({label}): has_normals={pcd.has_normals()}, "
        f"finite={frac * 100:.1f}%"
    )
    return ok


def _orient_normals(pcd):
    try:
        pcd.orient_normals_consistent_tangent_plane(k=15)
        log("Normals oriented with consistent tangent plane (k=15)")
    except RuntimeError as exc:
        log(
            f"Tangent-plane orientation failed ({exc}); "
            "falling back to orient_normals_towards_camera_location"
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))


def estimate_and_orient_normals(pcd, voxel_size, fast_normals=False):
    n_points = len(pcd.points)
    if n_points < 100:
        raise ValueError(
            f"Need at least 100 points for normal estimation, got {n_points:,}"
        )

    radius_primary = max(voxel_size * 4.0, 0.05)
    radius_wide = max(voxel_size * 8.0, 0.15)

    strategies = [
        (
            "hybrid",
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_primary, max_nn=30),
            f"radius={radius_primary:.4f}m, max_nn=30",
        ),
        (
            "hybrid-wide",
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_wide, max_nn=50),
            f"radius={radius_wide:.4f}m, max_nn=50",
        ),
        (
            "knn",
            o3d.geometry.KDTreeSearchParamKNN(knn=30),
            "knn=30",
        ),
    ]

    last_label = None
    for _name, param, label in strategies:
        last_label = label
        if _try_estimate_normals(
            pcd, param, label, fast_normal_computation=fast_normals
        ):
            break
    else:
        raise RuntimeError(
            f"Normal estimation failed after all strategies (last try: {last_label})"
        )

    _orient_normals(pcd)


def run_pipeline(input_path, output_path, config: MeshConfig):
    pipeline_t0 = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_threads = os.cpu_count() or 1
    log(f"=== {config.name} ===")
    log(
        f"CPU threads: {n_threads} | voxel: {config.voxel_size * 1000:.1f}mm | "
        f"poisson depth: {config.poisson_depth}"
    )
    if config.crop_radius_m:
        log(f"Test patch crop: {config.crop_radius_m:.1f}m radius")
    if config.max_points is not None:
        log(f"Load cap: {config.max_points:,} points max")

    pcd = load_las_as_o3d(
        input_path,
        crop_radius_m=config.crop_radius_m,
        max_points=config.max_points,
    )
    n_after_load = len(pcd.points)
    recenter_pcd(pcd)

    with log_stage(f"Voxel downsampling at {config.voxel_size * 1000:.1f}mm"):
        pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)
    n_after_downsample = len(pcd.points)
    log(f"Points after downsample: {n_after_downsample:,}")

    if config.skip_outlier_removal:
        log("Skipping statistical outlier removal")
        n_after_outlier = n_after_downsample
    else:
        with log_stage("Statistical outlier removal"):
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=config.outlier_std_ratio
            )
        n_after_outlier = len(pcd.points)
        log(f"Points after outlier removal: {n_after_outlier:,}")

    if n_after_outlier < 100:
        raise ValueError(
            "Need at least 100 points after preprocessing for normal estimation; "
            f"got {n_after_outlier:,}. Stage counts: "
            f"after_load={n_after_load:,}, after_downsample={n_after_downsample:,}, "
            f"after_outlier_removal={n_after_outlier:,}. "
            "Try a smaller voxel_size, disable crop, or increase max_points."
        )

    n_after = n_after_outlier
    fast_normals = n_after < 500_000
    if fast_normals:
        log("Using fast normal computation (small point count)")

    with log_stage(f"Normal estimation + orientation ({n_after:,} points)"):
        estimate_and_orient_normals(pcd, config.voxel_size, fast_normals=fast_normals)

    with log_stage(
        f"Poisson reconstruction (depth={config.poisson_depth}, n_threads={n_threads})"
    ):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=config.poisson_depth, n_threads=n_threads
            )
    log(f"Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    with log_stage(f"Trimming lowest {config.density_trim_percentile}% confidence vertices"):
        densities = np.asarray(densities)
        threshold = np.percentile(densities, config.density_trim_percentile)
        mesh.remove_vertices_by_mask(densities < threshold)
    log(f"Trimmed mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    with log_stage("Computing vertex normals"):
        mesh.compute_vertex_normals()

    with log_stage(f"Writing {output_path}"):
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)

    log(f"Done — total wall time {fmt_elapsed(time.time() - pipeline_t0)}")
