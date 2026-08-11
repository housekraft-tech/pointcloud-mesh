"""Every threshold in the pipeline, in one place, one commented line each.

Values marked "calibration pending" are starting points to be tuned against
the golden synthetic scenes and the reference scans. They are not claims of
achieved accuracy.
"""
from __future__ import annotations

from copy import deepcopy

DEFAULT_CONFIG: dict = {
    # --- reproducibility ---
    "seed": 0,                     # seeds every RNG in the pipeline
    # --- normals ---
    "normal_k": 24,                # neighbours used for PCA normal estimation
    # --- patch extraction (spec tolerances) ---
    "tau_fit_m": 0.003,            # plane inlier distance
    "tau_feature_m": 0.008,        # min depth to count as a feature, not roughness
    "patch_angle_tol_deg": 8.0,    # max normal deviation when growing a patch
    "patch_connect_radius_m": 0.05,  # CALIBRATION PENDING -- neighbour radius enforcing patch connectivity; must stay below the narrowest feature width
    "min_patch_points": 200,       # tuned against the golden room (Task 10): below 184 lets a curvature-gate splinter off the extrusion face and a groove fragment survive as spurious patches with wrong plane offsets, corrupting the extrusion-depth and groove-depth measurements; stays below ~441, the switch-box recess sample count, so the box patch is never excluded by size alone (it fails to form for a different reason -- see task-10-report.md)
    "patch_max_curvature": 0.01,   # CALIBRATION PENDING -- points blended across a geometric edge run hotter than this; gated out of seeding and growth so edges can't chain into spurious slivers
    "patch_neighbor_k": 256,       # CALIBRATION PENDING -- vectorised k-NN precompute size for growth; a point denser than this falls back to an exact radius query, so correctness never depends on this value
    "refit_interval": 200,         # points added between plane refits while growing
    # --- coplanarity (recorded, never applied) ---
    "coplanar_dist_tol_m": 0.005,  # max plane-offset difference within a class
    "coplanar_angle_tol_deg": 2.0,  # max normal deviation within a class
    # --- intersection lines ---
    "min_intersection_angle_deg": 0.5,  # planes closer than this angle return None; guards against numerical instability in near-parallel cases
    # --- adjacency ---
    "adjacency_radius_m": 0.05,    # max gap between patches counted as adjacent
    # --- frame (measured, never enforced) ---
    "floor_normal_tol_deg": 15.0,  # max tilt from world Z for a floor/ceiling patch
}


def merged_config(overrides: dict | None = None) -> dict:
    """Return DEFAULT_CONFIG updated with overrides, rejecting unknown keys."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"unknown config key {key!r}")
        cfg[key] = value
    return cfg
