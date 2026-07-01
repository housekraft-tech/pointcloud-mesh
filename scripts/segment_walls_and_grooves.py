"""Walls + rectilinear grooves for Blender.

Grooves are rectangular, square, or L-shaped (no curves).
  - Walls: RANSAC planes -> flat Delaunay mesh per wall (flat band only)
  - Grooves: points off each wall plane -> wall-local axis-aligned box cells,
    merged along U/V into rects / L-shapes
  - Residual non-wall points -> world-axis voxel boxes (misc grooves)

Usage:
    python segment_walls_and_grooves.py [input.laz] [output_dir]

Blender: walls/wall_*.obj, grooves/wall_*_grooves.obj, grooves/misc_grooves.obj,
         grooves/grooves_all.obj
"""

import json
import sys
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
RANSAC_DISTANCE = 0.018
WALL_FLAT_BAND = 0.006
GROOVE_DEVIATION_MAX = 0.015
GROOVE_CELL_UV = 0.012
GROOVE_CELL_DEPTH = 0.008
MIN_PLANE_POINTS = 8_000
MAX_PLANES = 40
RANSAC_ITERATIONS = 2_000
MIN_GROOVE_POINTS = 80
MIN_GROOVE_CELL_POINTS = 4
MAX_POINTS = None


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
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


def plane_cloud_to_mesh(plane_pcd, plane_model):
    pts = np.asarray(plane_pcd.points)
    if len(pts) < 3:
        return None

    n = plane_normal(plane_model)
    projected = project_to_plane(pts, plane_model)

    u, v = plane_basis(n)
    origin = projected.mean(axis=0)
    rel = projected - origin
    uv = np.column_stack([rel @ u, rel @ v])

    try:
        tri = Delaunay(uv)
    except Exception:
        return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(projected)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
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
        wall_pcd, groove_pcd, far_pcd = split_plane_inliers(plane_pcd, plane_model)

        remaining = remaining.select_by_index(inliers, invert=True)
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
        planes.append((plane_model, wall_pcd))
        if len(groove_pcd.points) >= MIN_GROOVE_POINTS:
            groove_by_wall.append((wall_id, plane_model, groove_pcd))
        elif len(groove_pcd.points) > 0:
            residual_parts.append(groove_pcd)

        log(
            f"Plane {idx + 1} wall_{wall_id:03d}: {len(wall_pcd.points):,} wall, "
            f"{len(groove_pcd.points):,} groove ({fraction * 100:.1f}% of cloud in RANSAC)"
        )

    residual = _concat_point_clouds([remaining] + residual_parts)
    log(
        f"Extracted {len(planes)} walls, {len(groove_by_wall)} groove sets; "
        f"{len(residual.points):,} residual points"
    )
    return planes, groove_by_wall, residual



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


def _max_depth_in_rect(cell_depth_max, iu0, iu1, iv0, iv1):
    d = 0.0
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            key = (iu, iv)
            if key in cell_depth_max:
                d = max(d, cell_depth_max[key])
    return max(d, GROOVE_CELL_DEPTH * 0.5)



def rectilinear_groove_mesh_on_wall(groove_pcd, plane_model):
    if groove_pcd is None or len(groove_pcd.points) < MIN_GROOVE_POINTS:
        return None

    pts = np.asarray(groove_pcd.points, dtype=np.float64)
    n = plane_normal(plane_model)
    u, v = plane_basis(n)
    signed = signed_plane_distance(pts, plane_model)
    recess_sign = float(np.sign(np.median(signed)))
    if recess_sign == 0.0:
        recess_sign = 1.0
    inward = -n * recess_sign

    projected = project_to_plane(pts, plane_model)
    origin = projected.min(axis=0)
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



def main(input_path, output_dir):
    t0 = time.time()
    output_dir = Path(output_dir)
    walls_dir = output_dir / "walls"
    grooves_dir = output_dir / "grooves"
    walls_dir.mkdir(parents=True, exist_ok=True)
    grooves_dir.mkdir(parents=True, exist_ok=True)

    log("=== WALLS + RECTILINEAR GROOVES PIPELINE ===")
    log(
        f"Voxel: {VOXEL_SIZE * 1000:.0f}mm | RANSAC: {RANSAC_DISTANCE * 1000:.0f}mm | "
        f"groove cell: {GROOVE_CELL_UV * 1000:.0f}mm UV, {GROOVE_CELL_DEPTH * 1000:.0f}mm depth"
    )

    pcd = load_las_as_o3d(input_path, max_points=MAX_POINTS)
    recenter_pcd(pcd)

    with log_stage(f"Voxel downsampling at {VOXEL_SIZE * 1000:.0f}mm"):
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    log(f"Points for segmentation: {len(pcd.points):,}")

    with log_stage("RANSAC plane extraction"):
        planes, groove_by_wall, residual_pcd = extract_planes(pcd)

    manifest = {
        "walls": [],
        "grooves_by_wall": [],
        "misc_grooves": None,
        "grooves_all": None,
        "settings": {
            "voxel_mm": VOXEL_SIZE * 1000,
            "wall_flat_mm": WALL_FLAT_BAND * 1000,
            "groove_deviation_mm": GROOVE_DEVIATION_MAX * 1000,
            "groove_cell_uv_mm": GROOVE_CELL_UV * 1000,
            "groove_cell_depth_mm": GROOVE_CELL_DEPTH * 1000,
        },
    }

    all_groove_meshes = []

    with log_stage(f"Building {len(planes)} wall meshes"):
        for i, (plane_model, plane_pcd) in enumerate(planes, start=1):
            mesh = plane_cloud_to_mesh(plane_pcd, plane_model)
            if mesh is None or len(mesh.triangles) == 0:
                log(f"  wall_{i:03d}: skipped (mesh build failed)")
                continue
            out = walls_dir / f"wall_{i:03d}.obj"
            o3d.io.write_triangle_mesh(str(out), mesh)
            manifest["walls"].append({
                "file": str(out.relative_to(output_dir)),
                "points": len(plane_pcd.points),
                "triangles": len(mesh.triangles),
                "plane": [float(x) for x in plane_model],
            })
            log(
                f"  wall_{i:03d}.obj: {len(plane_pcd.points):,} pts, "
                f"{len(mesh.triangles):,} tris"
            )

    with log_stage(f"Rectilinear grooves on {len(groove_by_wall)} walls"):
        for wall_id, plane_model, groove_pcd in groove_by_wall:
            groove_mesh = rectilinear_groove_mesh_on_wall(groove_pcd, plane_model)
            rel_name = f"wall_{wall_id:03d}_grooves.obj"
            entry = {
                "wall_id": wall_id,
                "file": str((grooves_dir / rel_name).relative_to(output_dir)),
                "points": len(groove_pcd.points),
                "triangles": 0,
            }
            if groove_mesh is not None:
                out = grooves_dir / rel_name
                o3d.io.write_triangle_mesh(str(out), groove_mesh)
                entry["triangles"] = len(groove_mesh.triangles)
                all_groove_meshes.append(groove_mesh)
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
            misc_path = grooves_dir / "misc_grooves.obj"
            o3d.io.write_triangle_mesh(str(misc_path), misc_mesh)
            all_groove_meshes.append(misc_mesh)
            manifest["misc_grooves"] = {
                "file": str(misc_path.relative_to(output_dir)),
                "points": residual_count,
                "triangles": len(misc_mesh.triangles),
            }
            log(
                f"misc_grooves.obj: {residual_count:,} pts, "
                f"{len(misc_mesh.triangles):,} tris"
            )
        else:
            log("No misc_grooves mesh (too few residual points or no dense cells)")

    grooves_all = combine_meshes(all_groove_meshes)
    if grooves_all is not None:
        all_path = grooves_dir / "grooves_all.obj"
        o3d.io.write_triangle_mesh(str(all_path), grooves_all)
        manifest["grooves_all"] = {
            "file": str(all_path.relative_to(output_dir)),
            "triangles": len(grooves_all.triangles),
        }
        log(f"grooves_all.obj: {len(grooves_all.triangles):,} tris")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = int(time.time() - t0)
    log(f"Done in {elapsed // 60}m {elapsed % 60}s")
    log(f"Output: {output_dir}")
    log(
        "Blender: walls/wall_*.obj + grooves/wall_*_grooves.obj + "
        "grooves/misc_grooves.obj + grooves/grooves_all.obj"
    )


if __name__ == "__main__":
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    main(inp, out)
