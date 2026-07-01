"""Pure numpy+cv2 floor plan reconstruction primitives.

No open3d import anywhere in this file -- this is what makes it fully
unit-testable on a 32-bit Python install that cannot install open3d at all.
Any function here that needs open3d-loaded data takes plain numpy arrays;
the open3d-facing wrapper lives in mesh_common.py.
"""
import numpy as np
import cv2
from collections import defaultdict, deque


# ---------- plane / frame math ----------

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


def refine_plane_model(points, plane_model):
    """SVD least-squares plane refit on the given points."""
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


def wall_uv_basis(normal):
    """U along the wall (perpendicular to normal, in the horizontal plane),
    V = world-up projected onto the wall plane."""
    normal = normal / np.linalg.norm(normal)
    world_up = np.array([0.0, 0.0, 1.0])
    v = world_up - normal * np.dot(world_up, normal)
    if np.linalg.norm(v) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v
    v = v / np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    return u, v


def project_to_plane(points, plane_model):
    signed = signed_plane_distance(points, plane_model)
    n = plane_normal(plane_model)
    pts = np.asarray(points, dtype=np.float64)
    return pts - signed[:, None] * n


def points_to_wall_uv(points, plane_model, origin_xyz, u_axis, v_axis=None):
    """Project 3D points onto a wall plane, express as (u=along wall, v=height)."""
    if v_axis is None:
        v_axis = np.array([0.0, 0.0, 1.0])
    projected = project_to_plane(points, plane_model)
    rel = projected - np.asarray(origin_xyz, dtype=np.float64)
    u = rel @ np.asarray(u_axis, dtype=np.float64)
    v = rel @ np.asarray(v_axis, dtype=np.float64)
    return np.column_stack([u, v])


# ---------- Phase 0: bounding-box auto-crop ----------

def crop_to_percentile_bounds(xyz, low_pct=1.0, high_pct=99.0, margin_m=0.5):
    """Robust bounding box from per-axis percentiles + margin, dropping the
    sparse SLAM-drift/ghost-point tail that inflates a raw min/max bbox.

    Confirmed on the real koushikexport.las/mujammelexport.las scans: 99% of
    points sit in a ~11x12x3.3m room while raw bbox balloons to 30-85m due to
    a sparse stray tail; this recovers the tight room bounds.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("crop_to_percentile_bounds: empty point array")
    lo = np.percentile(xyz, low_pct, axis=0) - margin_m
    hi = np.percentile(xyz, high_pct, axis=0) + margin_m
    keep_mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    stats = {
        "input_points": int(xyz.shape[0]),
        "kept_points": int(keep_mask.sum()),
        "dropped_points": int((~keep_mask).sum()),
        "dropped_fraction": float((~keep_mask).sum() / xyz.shape[0]),
        "raw_bounds_min": xyz.min(axis=0).tolist(),
        "raw_bounds_max": xyz.max(axis=0).tolist(),
        "cropped_bounds_min": lo.tolist(),
        "cropped_bounds_max": hi.tolist(),
    }
    return lo, hi, keep_mask, stats
