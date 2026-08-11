"""Plane fitting with a canonical normal orientation.

Every plane in the pipeline is fitted to its own points and to nothing else.
Canonical orientation exists so that two parallel patches -- a wall face and
the 75 mm step in front of it -- yield comparable plane offsets regardless of
which way PCA happened to point their normals.
"""
from __future__ import annotations

import numpy as np


def canonical_normal(n: np.ndarray) -> np.ndarray:
    """Flip a normal so its largest-magnitude component is positive.

    Ties are broken toward the lowest axis index, which keeps the choice
    deterministic for normals like (0.5, -0.5, 0).
    """
    n = np.asarray(n, dtype=np.float64)
    dominant = int(np.argmax(np.abs(n)))
    return -n if n[dominant] < 0 else n.copy()


def fit_plane(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through the points.

    Returns (normal, d) satisfying normal @ x + d == 0, with a unit,
    canonically-oriented normal.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {xyz.shape[0]}")

    centroid = xyz.mean(axis=0)
    centred = xyz - centroid
    # smallest singular vector of the centred cloud is the plane normal
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = canonical_normal(vt[-1])
    normal /= np.linalg.norm(normal)
    d = float(-normal @ centroid)
    return normal, d


def plane_distance(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Signed perpendicular distance of each point to the plane, in metres."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz @ np.asarray(normal, dtype=np.float64) + d


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic orthonormal basis (u, v) spanning the plane.

    u is built from the world axis least aligned with the normal, so the basis
    is stable for a given normal and never degenerate.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(normal)))] = 1.0

    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v
