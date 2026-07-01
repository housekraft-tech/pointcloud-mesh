"""Walls + rectilinear grooves + opening frames for Blender.

Grooves are rectangular, square, or L-shaped (no curves).
  - Walls: RANSAC vertical planes -> flat Delaunay mesh per wall (flat band only)
  - Openings: voids in full wall inlier UV grid (doors / windows / balcony)
  - Grooves: points off each wall plane -> wall-local axis-aligned box cells
  - Primary output: reconstructed.obj (walls + grooves + opening frames)

Usage:
    python segment_walls_and_grooves.py [input.laz] [output_dir] [--parts]

With --parts: also writes walls/wall_*.obj and grooves/*.
Always writes output_dir/reconstructed.obj and manifest.json.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from mesh_common import (
    ROOT,
    load_las_as_o3d,
    log,
    log_stage,
    recenter_pcd,
)

DEFAULT_INPUT = ROOT / "data" / "koushikexport.laz"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "walls_and_grooves"

VOXEL_SIZE = 0.015
RANSAC_DISTANCE = 0.015
WALL_FLAT_BAND = 0.006
GROOVE_DEVIATION_MAX = 0.015
GROOVE_CELL_UV = 0.010
GROOVE_CELL_DEPTH = 0.008
OPENING_CELL = 0.05
OPENING_MIN_POINTS_PER_CELL = 3
OPENING_MIN_W = 0.45
OPENING_MIN_H = 0.45
OPENING_MAX_W = 4.0
OPENING_MAX_H = 3.5
OPENING_MAX_AREA_FRAC = 0.30
OPENING_FRAME_DEPTH = 0.075
OPENING_FRAME_WIDTH = 0.05
MIN_PLANE_POINTS = 8_000
MAX_PLANES = 40
RANSAC_ITERATIONS = 5_000
MIN_GROOVE_POINTS = 80
MIN_GROOVE_CELL_POINTS = 4
MIN_OPENING_CELLS = 2
MAX_POINTS = None

FLOOR_NZ_MIN = 0.85
WALL_NZ_MAX = 0.35


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def wall_uv_basis(normal):
    """U along wall, V aligned with world up projected onto the wall plane."""
    normal = normal / np.linalg.norm(normal)
    world_up = np.array([0.0, 0.0, 1.0])
    v = world_up - normal * np.dot(world_up, normal)
    if np.linalg.norm(v) < 1e-6:
        u, v = plane_basis(normal)
        return u, v
    v = v / np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    return u, v


def plane_normal(plane_model):
    a, b, c, _d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    return n / np.linalg.norm(n)


def signed_plane_distance(points, plane_model):
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ n + float(d)


def project_to_plane(points, plane_model):
    signed = signed_plane_distance(points, plane_model)
    n = plane_normal(plane_model)
    pts = np.asarray(points, dtype=np.float64)
    return pts - signed[:, None] * n


def refine_plane_model(points, plane_model):
    """Refine RANSAC plane with SVD least-squares fit on inliers."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return list(plane_model)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    n0 = plane_normal(plane_model)
    if float(np.dot(normal, n0)) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return [float(normal[0]), float(normal[1]), float(normal[2]), d]


def wall_local_frame(projected_points, plane_model):
    """Origin, u, v, and per-point UV from wall_uv_basis on projected points."""
    n = plane_normal(plane_model)
    u, v = wall_uv_basis(n)
    projected = np.asarray(projected_points, dtype=np.float64)
    origin = projected.min(axis=0)
    rel = projected - origin
    uv = np.column_stack([rel @ u, rel @ v])
    return origin, u, v, uv


def split_plane_inliers(plane_pcd, plane_model):
    """Wall: |distance| <= WALL_FLAT_BAND; groove: (WALL_FLAT_BAND, GROOVE_DEVIATION_MAX]."""
    signed = signed_plane_distance(np.asarray(plane_pcd.points), plane_model)
    dist = np.abs(signed)
    wall_idx = np.where(dist <= WALL_FLAT_BAND)[0]
    groove_idx = np.where(
        (dist > WALL_FLAT_BAND) & (dist <= GROOVE_DEVIATION_MAX)
    )[0]
    far_idx = np.where(dist > GROOVE_DEVIATION_MAX)[0]
    wall_pcd = plane_pcd.select_by_index(wall_idx)
    groove_pcd = plane_pcd.select_by_index(groove_idx)
    far_pcd = plane_pcd.select_by_index(far_idx)
    return wall_pcd, groove_pcd, far_pcd


def _centroid_inside_opening_rects(cu, cv, opening_uv_rects):
    for u_min, u_max, v_min, v_max in opening_uv_rects:
        if u_min <= cu <= u_max and v_min <= cv <= v_max:
            return True
    return False


def plane_cloud_to_mesh(
    plane_pcd,
    plane_model,
    origin=None,
    u=None,
    v=None,
    opening_uv_rects=None,
):
    pts = np.asarray(plane_pcd.points)
    if len(pts) < 3:
        return None

    projected = project_to_plane(pts, plane_model)
    if origin is None or u is None or v is None:
        n = plane_normal(plane_model)
        u, v = wall_uv_basis(n)
        origin = projected.min(axis=0)

    rel = projected - origin
    uv = np.column_stack([rel @ u, rel @ v])

    try:
        tri = Delaunay(uv)
    except Exception:
        return None

    simplices = tri.simplices
    if opening_uv_rects:
        keep = []
        for simplex in simplices:
            cu = float(uv[simplex, 0].mean())
            cv = float(uv[simplex, 1].mean())
            if not _centroid_inside_opening_rects(cu, cv, opening_uv_rects):
                keep.append(simplex)
        if not keep:
            return None
        simplices = np.asarray(keep, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(projected)
    mesh.triangles = o3d.utility.Vector3iVector(simplices)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _concat_point_clouds(clouds):
    clouds = [c for c in clouds if c is not None and len(c.points) > 0]
    if not clouds:
        return o3d.geometry.PointCloud()
    if len(clouds) == 1:
        return clouds[0]
    out = o3d.geometry.PointCloud()
    pts = np.vstack([np.asarray(c.points) for c in clouds])
    out.points = o3d.utility.Vector3dVector(pts)
    return out


def extract_planes(pcd):
    planes = []
    groove_by_wall = []
    residual_parts = []
    floor_candidates = []
    remaining = pcd
    total = len(pcd.points)
    wall_id = 0

    for idx in range(MAX_PLANES):
        if len(remaining.points) < MIN_PLANE_POINTS:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=RANSAC_DISTANCE,
            ransac_n=3,
            num_iterations=RANSAC_ITERATIONS,
        )
        if len(inliers) < MIN_PLANE_POINTS:
            break

        fraction = len(inliers) / max(total, 1)
        if fraction < 0.005:
            break

        plane_pcd = remaining.select_by_index(inliers)
        inlier_pts = np.asarray(plane_pcd.points)
        plane_model = refine_plane_model(inlier_pts, plane_model)
        n = plane_normal(plane_model)
        nz_abs = abs(float(n[2]))
        pts_arr = inlier_pts
        mean_z = float(pts_arr[:, 2].mean()) if len(pts_arr) else 0.0

        remaining = remaining.select_by_index(inliers, invert=True)

        if nz_abs > FLOOR_NZ_MIN:
            floor_candidates.append((mean_z, plane_model, len(pts_arr)))
            log(
                f"Plane {idx + 1}: floor candidate mean_z={mean_z:.3f} "
                f"({len(inliers):,} inliers, {fraction * 100:.1f}% of cloud)"
            )
            continue

        if nz_abs >= WALL_NZ_MAX:
            log(
                f"Plane {idx + 1}: skipped non-vertical (|nz|={nz_abs:.2f}); "
                f"{len(inliers):,} inliers -> residual"
            )
            residual_parts.append(plane_pcd)
            continue

        wall_pcd, groove_pcd, far_pcd = split_plane_inliers(plane_pcd, plane_model)

        if len(far_pcd.points) > 0:
            residual_parts.append(far_pcd)

        if len(wall_pcd.points) < MIN_PLANE_POINTS:
            log(
                f"Plane {idx + 1}: skipped wall ({len(wall_pcd.points):,} flat points); "
                f"groove pts {len(groove_pcd.points):,}"
            )
            if len(groove_pcd.points) > 0:
                residual_parts.append(groove_pcd)
            continue

        wall_id += 1
        planes.append((wall_id, plane_model, wall_pcd, plane_pcd))
        if len(groove_pcd.points) >= MIN_GROOVE_POINTS:
            groove_by_wall.append((wall_id, plane_model, groove_pcd))
        elif len(groove_pcd.points) > 0:
            residual_parts.append(groove_pcd)

        log(
            f"Plane {idx + 1} wall_{wall_id:03d}: {len(wall_pcd.points):,} wall, "
            f"{len(groove_pcd.points):,} groove ({fraction * 100:.1f}% of cloud in RANSAC)"
        )

    floor_z = None
    floor_plane_model = None
    if floor_candidates:
        _mean_z, floor_plane_model, _npts = min(floor_candidates, key=lambda x: x[0])
        floor_z = _mean_z
        log(f"Floor Z (lowest horizontal plane mean): {floor_z:.3f} m")

    residual = _concat_point_clouds([remaining] + residual_parts)
    log(
        f"Extracted {len(planes)} walls, {len(groove_by_wall)} groove sets; "
        f"{len(residual.points):,} residual points"
    )
    return planes, groove_by_wall, residual, floor_z, floor_plane_model



def _merge_uv_cells(occupied_uv):
    """Merge occupied (iu, iv) cells into axis-aligned rectangles via U then V runs."""
    if not occupied_uv:
        return []

    bars = []
    for iv in sorted({iv for _iu, iv in occupied_uv}):
        row = sorted(iu for iu, jv in occupied_uv if jv == iv)
        start = prev = row[0]
        for iu in row[1:]:
            if iu == prev + 1:
                prev = iu
            else:
                bars.append((start, prev, iv))
                start = prev = iu
        bars.append((start, prev, iv))

    by_span = defaultdict(list)
    for iu0, iu1, iv in bars:
        by_span[(iu0, iu1)].append(iv)

    rects = []
    for (iu0, iu1), ivs in by_span.items():
        ivs = sorted(set(ivs))
        start = prev = ivs[0]
        for iv in ivs[1:]:
            if iv == prev + 1:
                prev = iv
            else:
                rects.append((iu0, iu1, start, prev))
                start = prev = iv
        rects.append((iu0, iu1, start, prev))
    return rects


def _opening_interior_void_cells(occupied, iu0, iu1, iv0, iv1):
    """Empty cells not reachable from the grid border (door/window holes)."""
    from collections import deque

    exterior = set()
    queue = deque()

    def seed(iu, iv):
        if (iu, iv) in occupied or (iu, iv) in exterior:
            return
        exterior.add((iu, iv))
        queue.append((iu, iv))

    for iu in range(iu0, iu1 + 1):
        seed(iu, iv0)
        seed(iu, iv1)
    for iv in range(iv0, iv1 + 1):
        seed(iu0, iv)
        seed(iu1, iv)

    while queue:
        iu, iv = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = iu + di, iv + dj
            if not (iu0 <= ni <= iu1 and iv0 <= nj <= iv1):
                continue
            if (ni, nj) in occupied or (ni, nj) in exterior:
                continue
            exterior.add((ni, nj))
            queue.append((ni, nj))

    voids = set()
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            if (iu, iv) not in occupied and (iu, iv) not in exterior:
                voids.add((iu, iv))
    return voids


def _classify_opening(width, height, sill):
    if width >= 1.0 and height >= 2.0 and sill < 0.25:
        return "balcony_door"
    if 0.65 <= width <= 1.4 and 1.75 <= height <= 2.5 and sill < 0.25:
        return "door"
    if sill >= 0.4 and width >= OPENING_MIN_W and height >= OPENING_MIN_H:
        return "window"
    return "unknown_opening"


def _world_z_at_uv(origin, u, v, uu, vv):
    p = origin + u * uu + v * vv
    return float(p[2])


def _floor_z_at_xy(floor_plane_model, x, y):
    a, b, c, d = floor_plane_model
    if abs(c) < 1e-9:
        return None
    return float(-(a * x + b * y + d) / c)


def _opening_sill_m(origin, u, v, u_min, u_max, v_min, floor_plane_model, floor_z_fallback):
    bottom_corners = (
        origin + u * u_min + v * v_min,
        origin + u * u_max + v * v_min,
    )
    if floor_plane_model is not None:
        sills = []
        for p in bottom_corners:
            fz = _floor_z_at_xy(floor_plane_model, float(p[0]), float(p[1]))
            if fz is not None:
                sills.append(float(p[2]) - fz)
        if sills:
            return min(sills)
    return min(float(p[2]) for p in bottom_corners) - floor_z_fallback


def _append_oriented_box(mesh, p000, p100, p010, p110, p001, p101, p011, p111):
    """Add 8 corners and 12 triangles (two per face)."""
    base = len(mesh.vertices)
    corners = [p000, p100, p010, p110, p001, p101, p011, p111]
    mesh.vertices.extend(corners)
    tris = [
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ]
    mesh.triangles.extend(
        o3d.utility.Vector3iVector([[base + a, base + b, base + c] for a, b, c in tris])
    )


def _box_from_uv_rect(origin, u_vec, v_vec, inward, u_min, u_max, v_min, v_max, depth):
    def corner(uu, vv, dd):
        return origin + u_vec * uu + v_vec * vv + inward * dd

    p000 = corner(u_min, v_min, 0.0)
    p100 = corner(u_max, v_min, 0.0)
    p010 = corner(u_min, v_max, 0.0)
    p110 = corner(u_max, v_max, 0.0)
    p001 = corner(u_min, v_min, depth)
    p101 = corner(u_max, v_min, depth)
    p011 = corner(u_min, v_max, depth)
    p111 = corner(u_max, v_max, depth)
    mesh = o3d.geometry.TriangleMesh()
    _append_oriented_box(mesh, p000, p100, p010, p110, p001, p101, p011, p111)
    return mesh


def _opening_frame_mesh(origin, u, v, inward, u_min, u_max, v_min, v_max):
    """Rectilinear frame (four jambs + lintel) extruded into the wall."""
    fw = OPENING_FRAME_WIDTH
    depth = OPENING_FRAME_DEPTH
    combined = o3d.geometry.TriangleMesh()
    combined += _box_from_uv_rect(origin, u, v, inward, u_min, u_max, v_min, v_min + fw, depth)
    combined += _box_from_uv_rect(origin, u, v, inward, u_min, u_min + fw, v_min, v_max, depth)
    combined += _box_from_uv_rect(origin, u, v, inward, u_max - fw, u_max, v_min, v_max, depth)
    combined += _box_from_uv_rect(origin, u, v, inward, u_min, u_max, v_max - fw, v_max, depth)
    if len(combined.triangles) == 0:
        return None
    combined.remove_duplicated_vertices()
    combined.remove_degenerate_triangles()
    combined.compute_vertex_normals()
    return combined


def detect_openings_on_wall(
    wall_id,
    full_inlier_pcd,
    plane_model,
    floor_z,
    floor_plane_model=None,
):
    if full_inlier_pcd is None or len(full_inlier_pcd.points) < MIN_PLANE_POINTS // 4:
        return [], [], []

    pts = np.asarray(full_inlier_pcd.points, dtype=np.float64)
    n = plane_normal(plane_model)
    inward = n.copy()

    signed = signed_plane_distance(pts, plane_model)
    flat_mask = np.abs(signed) <= WALL_FLAT_BAND * 2.0
    flat_pts = pts[flat_mask]
    if len(flat_pts) < 100:
        flat_pts = pts

    projected_full = project_to_plane(pts, plane_model)
    origin, u, v, _uv_full = wall_local_frame(projected_full, plane_model)

    projected_flat = project_to_plane(flat_pts, plane_model)
    rel = projected_flat - origin
    uv = np.column_stack([rel @ u, rel @ v])

    cell_counts = defaultdict(int)
    iu = np.floor(uv[:, 0] / OPENING_CELL).astype(np.int64)
    iv = np.floor(uv[:, 1] / OPENING_CELL).astype(np.int64)
    for k in range(len(flat_pts)):
        cell_counts[(int(iu[k]), int(iv[k]))] += 1

    occupied = {
        key for key, cnt in cell_counts.items() if cnt >= OPENING_MIN_POINTS_PER_CELL
    }
    if not occupied:
        return [], [], []

    iu0 = min(c[0] for c in occupied)
    iu1 = max(c[0] for c in occupied)
    iv0 = min(c[1] for c in occupied)
    iv1 = max(c[1] for c in occupied)

    wall_bbox_area = (iu1 - iu0 + 1) * OPENING_CELL * (iv1 - iv0 + 1) * OPENING_CELL

    void_cells = _opening_interior_void_cells(occupied, iu0, iu1, iv0, iv1)
    if not void_cells:
        return [], [], []

    rects = _merge_uv_cells(void_cells)
    openings = []
    frame_meshes = []
    opening_uv_rects = []

    if floor_z is None:
        floor_z = float(pts[:, 2].min())

    for iu_a, iu_b, iv_a, iv_b in rects:
        n_iu = iu_b - iu_a + 1
        n_iv = iv_b - iv_a + 1
        if n_iu < MIN_OPENING_CELLS or n_iv < MIN_OPENING_CELLS:
            continue

        u_min = iu_a * OPENING_CELL
        u_max = (iu_b + 1) * OPENING_CELL
        v_min = iv_a * OPENING_CELL
        v_max = (iv_b + 1) * OPENING_CELL
        width = u_max - u_min
        height = v_max - v_min

        if width < OPENING_MIN_W or height < OPENING_MIN_H:
            continue
        if width > OPENING_MAX_W or height > OPENING_MAX_H:
            continue
        if width * height > OPENING_MAX_AREA_FRAC * wall_bbox_area:
            continue

        sill = _opening_sill_m(
            origin, u, v, u_min, u_max, v_min, floor_plane_model, floor_z
        )

        otype = _classify_opening(width, height, sill)
        uv_rect = [u_min, u_max, v_min, v_max]
        opening_uv_rects.append(uv_rect)

        frame = _opening_frame_mesh(origin, u, v, inward, u_min, u_max, v_min, v_max)
        if frame is not None:
            frame_meshes.append(frame)

        openings.append({
            "wall_id": wall_id,
            "type": otype,
            "width_m": round(width, 3),
            "height_m": round(height, 3),
            "sill_m": round(sill, 3),
            "uv_rect": uv_rect,
        })

    return openings, frame_meshes, opening_uv_rects


def _max_depth_in_rect(cell_depth_max, iu0, iu1, iv0, iv1):
    d = 0.0
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            key = (iu, iv)
            if key in cell_depth_max:
                d = max(d, cell_depth_max[key])
    return max(d, GROOVE_CELL_DEPTH * 0.5)



def rectilinear_groove_mesh_on_wall(groove_pcd, plane_model, frame_origin=None, frame_u=None, frame_v=None):
    if groove_pcd is None or len(groove_pcd.points) < MIN_GROOVE_POINTS:
        return None

    pts = np.asarray(groove_pcd.points, dtype=np.float64)
    n = plane_normal(plane_model)
    u, v = wall_uv_basis(n)
    signed = signed_plane_distance(pts, plane_model)
    recess_sign = float(np.sign(np.median(signed)))
    if recess_sign == 0.0:
        recess_sign = 1.0
    inward = -n * recess_sign

    projected = project_to_plane(pts, plane_model)
    if frame_origin is not None and frame_u is not None and frame_v is not None:
        origin = np.asarray(frame_origin, dtype=np.float64)
        u = np.asarray(frame_u, dtype=np.float64)
        v = np.asarray(frame_v, dtype=np.float64)
    else:
        origin, u, v, _ = wall_local_frame(projected, plane_model)

    rel = projected - origin
    uv = np.column_stack([rel @ u, rel @ v])
    depth = np.abs(signed)

    iu = np.floor(uv[:, 0] / GROOVE_CELL_UV).astype(np.int64)
    iv = np.floor(uv[:, 1] / GROOVE_CELL_UV).astype(np.int64)

    cell_counts = defaultdict(int)
    cell_depth_max = defaultdict(float)
    for k in range(len(pts)):
        key = (int(iu[k]), int(iv[k]))
        cell_counts[key] += 1
        cell_depth_max[key] = max(cell_depth_max[key], depth[k])

    occupied_uv = {
        key for key, cnt in cell_counts.items() if cnt >= MIN_GROOVE_CELL_POINTS
    }
    if not occupied_uv:
        return None

    rects = _merge_uv_cells(occupied_uv)
    combined = o3d.geometry.TriangleMesh()
    for iu0, iu1, iv0, iv1 in rects:
        u_min = iu0 * GROOVE_CELL_UV
        u_max = (iu1 + 1) * GROOVE_CELL_UV
        v_min = iv0 * GROOVE_CELL_UV
        v_max = (iv1 + 1) * GROOVE_CELL_UV
        depth_val = _max_depth_in_rect(cell_depth_max, iu0, iu1, iv0, iv1)
        box = _box_from_uv_rect(origin, u, v, inward, u_min, u_max, v_min, v_max, depth_val)
        combined += box

    if len(combined.triangles) == 0:
        return None
    combined.remove_duplicated_vertices()
    combined.remove_degenerate_triangles()
    combined.compute_vertex_normals()
    return combined


def _world_axis_box(x0, x1, y0, y1, z0, z1):
    mesh = o3d.geometry.TriangleMesh()
    p000 = np.array([x0, y0, z0])
    p100 = np.array([x1, y0, z0])
    p010 = np.array([x0, y1, z0])
    p110 = np.array([x1, y1, z0])
    p001 = np.array([x0, y0, z1])
    p101 = np.array([x1, y0, z1])
    p011 = np.array([x0, y1, z1])
    p111 = np.array([x1, y1, z1])
    _append_oriented_box(mesh, p000, p100, p010, p110, p001, p101, p011, p111)
    return mesh


def world_voxel_groove_mesh(pcd):
    if pcd is None or len(pcd.points) < MIN_GROOVE_CELL_POINTS:
        return None

    pts = np.asarray(pcd.points, dtype=np.float64)
    origin = pts.min(axis=0)
    rel = pts - origin
    ix = np.floor(rel[:, 0] / GROOVE_CELL_UV).astype(np.int64)
    iy = np.floor(rel[:, 1] / GROOVE_CELL_UV).astype(np.int64)
    iz = np.floor(rel[:, 2] / GROOVE_CELL_DEPTH).astype(np.int64)

    cell_counts = defaultdict(int)
    for k in range(len(pts)):
        cell_counts[(int(ix[k]), int(iy[k]), int(iz[k]))] += 1

    by_layer = defaultdict(set)
    for (i, j, k), cnt in cell_counts.items():
        if cnt >= MIN_GROOVE_CELL_POINTS:
            by_layer[k].add((i, j))

    if not by_layer:
        return None

    combined = o3d.geometry.TriangleMesh()
    for iz_val, occupied_ij in by_layer.items():
        rects = _merge_uv_cells(occupied_ij)
        z0 = origin[2] + iz_val * GROOVE_CELL_DEPTH
        z1 = z0 + GROOVE_CELL_DEPTH
        for i0, i1, j0, j1 in rects:
            x0 = origin[0] + i0 * GROOVE_CELL_UV
            x1 = origin[0] + (i1 + 1) * GROOVE_CELL_UV
            y0 = origin[1] + j0 * GROOVE_CELL_UV
            y1 = origin[1] + (j1 + 1) * GROOVE_CELL_UV
            combined += _world_axis_box(x0, x1, y0, y1, z0, z1)

    if len(combined.triangles) == 0:
        return None
    combined.remove_duplicated_vertices()
    combined.remove_degenerate_triangles()
    combined.compute_vertex_normals()
    return combined


def combine_meshes(meshes):
    meshes = [m for m in meshes if m is not None and len(m.triangles) > 0]
    if not meshes:
        return None
    out = o3d.geometry.TriangleMesh()
    for m in meshes:
        out += m
    out.remove_duplicated_vertices()
    out.remove_degenerate_triangles()
    out.compute_vertex_normals()
    return out if len(out.triangles) > 0 else None



def main(input_path, output_dir, write_parts=False):
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    walls_dir = output_dir / "walls"
    grooves_dir = output_dir / "grooves"
    if write_parts:
        walls_dir.mkdir(parents=True, exist_ok=True)
        grooves_dir.mkdir(parents=True, exist_ok=True)

    log("=== WALLS + RECTILINEAR GROOVES + OPENINGS PIPELINE ===")
    log(
        f"Voxel: {VOXEL_SIZE * 1000:.0f}mm | RANSAC: {RANSAC_DISTANCE * 1000:.0f}mm | "
        f"groove cell: {GROOVE_CELL_UV * 1000:.0f}mm UV | opening cell: {OPENING_CELL * 1000:.0f}mm"
    )

    pcd = load_las_as_o3d(input_path, max_points=MAX_POINTS)
    recenter_pcd(pcd)

    with log_stage(f"Voxel downsampling at {VOXEL_SIZE * 1000:.0f}mm"):
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    log(f"Points for segmentation: {len(pcd.points):,}")

    with log_stage("RANSAC plane extraction"):
        planes, groove_by_wall, residual_pcd, floor_z, floor_plane_model = extract_planes(pcd)

    if floor_z is None:
        floor_z = float(np.asarray(pcd.points)[:, 2].min())
        log(f"No horizontal floor plane; using cloud min Z as floor: {floor_z:.3f} m")

    manifest = {
        "reconstructed": "reconstructed.obj",
        "floor_z_m": round(floor_z, 4),
        "floor_plane": (
            [float(x) for x in floor_plane_model]
            if floor_plane_model is not None
            else None
        ),
        "walls": [],
        "openings": [],
        "grooves_by_wall": [],
        "misc_grooves": None,
        "grooves_all": None,
        "settings": {
            "voxel_mm": VOXEL_SIZE * 1000,
            "wall_flat_mm": WALL_FLAT_BAND * 1000,
            "groove_deviation_mm": GROOVE_DEVIATION_MAX * 1000,
            "groove_cell_uv_mm": GROOVE_CELL_UV * 1000,
            "groove_cell_depth_mm": GROOVE_CELL_DEPTH * 1000,
            "opening_cell_mm": OPENING_CELL * 1000,
        },
    }

    all_groove_meshes = []
    all_reconstructed_parts = []

    wall_frames = {}

    with log_stage(f"Building {len(planes)} wall meshes + opening detection"):
        for wall_id, plane_model, wall_pcd, full_inlier_pcd in planes:
            full_pts = np.asarray(full_inlier_pcd.points, dtype=np.float64)
            plane_model = refine_plane_model(full_pts, plane_model)

            projected_full = project_to_plane(full_pts, plane_model)
            origin, u, v, _ = wall_local_frame(projected_full, plane_model)
            wall_frames[wall_id] = (origin, u, v)

            wall_openings, opening_frames, opening_uv_rects = detect_openings_on_wall(
                wall_id,
                full_inlier_pcd,
                plane_model,
                floor_z,
                floor_plane_model=floor_plane_model,
            )

            mesh = plane_cloud_to_mesh(
                wall_pcd,
                plane_model,
                origin=origin,
                u=u,
                v=v,
                opening_uv_rects=opening_uv_rects or None,
            )
            if mesh is None or len(mesh.triangles) == 0:
                log(f"  wall_{wall_id:03d}: skipped (mesh build failed)")
            else:
                all_reconstructed_parts.append(mesh)
                wall_entry = {
                    "wall_id": wall_id,
                    "points": len(wall_pcd.points),
                    "triangles": len(mesh.triangles),
                    "plane": [float(x) for x in plane_model],
                }
                if write_parts:
                    out = walls_dir / f"wall_{wall_id:03d}.obj"
                    o3d.io.write_triangle_mesh(str(out), mesh)
                    wall_entry["file"] = str(out.relative_to(output_dir))
                manifest["walls"].append(wall_entry)
                log(
                    f"  wall_{wall_id:03d}: {len(wall_pcd.points):,} pts, "
                    f"{len(mesh.triangles):,} tris"
                )

            for op in wall_openings:
                manifest["openings"].append({
                    "wall_id": op["wall_id"],
                    "type": op["type"],
                    "width_m": op["width_m"],
                    "height_m": op["height_m"],
                    "sill_m": op["sill_m"],
                })
            all_reconstructed_parts.extend(opening_frames)
            if wall_openings:
                log(
                    f"  wall_{wall_id:03d} openings: "
                    + ", ".join(
                        f"{o['type']}({o['width_m']}x{o['height_m']}m)" for o in wall_openings
                    )
                )

    with log_stage(f"Rectilinear grooves on {len(groove_by_wall)} walls"):
        for wall_id, plane_model, groove_pcd in groove_by_wall:
            frame = wall_frames.get(wall_id)
            if frame is not None:
                origin, u, v = frame
                groove_mesh = rectilinear_groove_mesh_on_wall(
                    groove_pcd, plane_model, frame_origin=origin, frame_u=u, frame_v=v
                )
            else:
                groove_mesh = rectilinear_groove_mesh_on_wall(groove_pcd, plane_model)
            rel_name = f"wall_{wall_id:03d}_grooves.obj"
            entry = {
                "wall_id": wall_id,
                "points": len(groove_pcd.points),
                "triangles": 0,
            }
            if write_parts:
                entry["file"] = str((grooves_dir / rel_name).relative_to(output_dir))
            if groove_mesh is not None:
                if write_parts:
                    out = grooves_dir / rel_name
                    o3d.io.write_triangle_mesh(str(out), groove_mesh)
                entry["triangles"] = len(groove_mesh.triangles)
                all_groove_meshes.append(groove_mesh)
                all_reconstructed_parts.append(groove_mesh)
                log(
                    f"  {rel_name}: {len(groove_pcd.points):,} pts, "
                    f"{len(groove_mesh.triangles):,} tris"
                )
            else:
                log(f"  {rel_name}: skipped (rectilinear mesh failed)")
            manifest["grooves_by_wall"].append(entry)

    residual_count = len(residual_pcd.points) if residual_pcd else 0
    with log_stage(f"World-voxel misc grooves ({residual_count:,} pts)"):
        misc_mesh = world_voxel_groove_mesh(residual_pcd) if residual_count else None
        if misc_mesh is not None:
            all_groove_meshes.append(misc_mesh)
            all_reconstructed_parts.append(misc_mesh)
            if write_parts:
                misc_path = grooves_dir / "misc_grooves.obj"
                o3d.io.write_triangle_mesh(str(misc_path), misc_mesh)
                manifest["misc_grooves"] = {
                    "file": str(misc_path.relative_to(output_dir)),
                    "points": residual_count,
                    "triangles": len(misc_mesh.triangles),
                }
            else:
                manifest["misc_grooves"] = {
                    "points": residual_count,
                    "triangles": len(misc_mesh.triangles),
                }
            log(
                f"misc_grooves: {residual_count:,} pts, "
                f"{len(misc_mesh.triangles):,} tris"
            )
        else:
            log("No misc_grooves mesh (too few residual points or no dense cells)")

    if write_parts:
        grooves_all = combine_meshes(all_groove_meshes)
        if grooves_all is not None:
            all_path = grooves_dir / "grooves_all.obj"
            o3d.io.write_triangle_mesh(str(all_path), grooves_all)
            manifest["grooves_all"] = {
                "file": str(all_path.relative_to(output_dir)),
                "triangles": len(grooves_all.triangles),
            }
            log(f"grooves_all.obj: {len(grooves_all.triangles):,} tris")

    reconstructed = combine_meshes(all_reconstructed_parts)
    recon_path = output_dir / "reconstructed.obj"
    if reconstructed is not None:
        o3d.io.write_triangle_mesh(str(recon_path), reconstructed)
        manifest["reconstructed_triangles"] = len(reconstructed.triangles)
        log(f"reconstructed.obj: {len(reconstructed.triangles):,} triangles")
    else:
        manifest["reconstructed_triangles"] = 0
        log("reconstructed.obj: empty (no meshes produced)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = int(time.time() - t0)
    log(f"Done in {elapsed // 60}m {elapsed % 60}s")
    log(f"Output: {output_dir}")
    if write_parts:
        log("Also wrote walls/ and grooves/ (--parts)")
    log("Primary: reconstructed.obj + manifest.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment walls, grooves, and openings.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Input LAZ/LAS point cloud",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--parts",
        action="store_true",
        help="Also write walls/wall_*.obj and grooves/* part files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.input), Path(args.output_dir), write_parts=args.parts)
