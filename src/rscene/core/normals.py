"""kNN PCA normal and curvature estimation.

Normals are canonically oriented (see fitting.canonical_normal) rather than
consistently oriented toward a viewpoint: patch growing compares normals with
abs(dot), so sign carries no information and a canonical choice keeps results
reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(xyz: np.ndarray, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-point normals and surface variation from k nearest neighbours.

    Returns (normals, curvature). curvature is lambda_0 / sum(lambda), which is
    ~0 on a flat surface and rises toward 1/3 at a corner.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < k:
        raise ValueError(f"need at least k={k} points, got {xyz.shape[0]}")

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)

    nbr = xyz[idx]                                       # (N, k, 3)
    centred = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k

    # eigh returns ascending eigenvalues; columns of v are the eigenvectors
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    curvature = w[:, 0] / np.clip(w.sum(axis=1), 1e-18, None)

    # canonical orientation, vectorised: flip rows whose dominant component < 0
    dominant = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(normals)), dominant])
    sign[sign == 0] = 1.0
    normals = normals * sign[:, None]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals, curvature
