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
    "min_patch_points": 100,       # smallest patch kept; floor set by switch-box sample count (~441 raw points) -- a bare count, no density/extent requirement: on the real crop 130/200 patches are under 200 points, median in-plane bbox 0.23 x 0.49 m at only 4.4% fill of a 6.1 mm-spaced surface; a fill/density gate is the intended Plan 2 fix
    "patch_max_curvature": 0.01,   # CALIBRATION PENDING -- points blended across a geometric edge run hotter than this; gated out of seeding only (Task 10), so a contaminated point can't found a spurious sliver, but an already-founded patch may still recruit it under the normal-agreement and tau_fit gates
    "patch_neighbor_k": 256,       # correctly sized, not a free memory win: at 6.1 mm real spacing a 50 mm ball holds ~211 points, so dropping to ~48 (previously logged as "5x memory for free") would force the exact-radius fallback for nearly every point -- it's a memory-for-time trade, not free
    "refit_interval": 200,         # points added between plane refits while growing
    # --- coplanarity (recorded, never applied) ---
    "coplanar_dist_tol_m": 0.005,  # max plane-offset difference within a class
    "coplanar_angle_tol_deg": 2.0,  # max normal deviation within a class
    # --- intersection lines ---
    "min_intersection_angle_deg": 0.5,  # planes closer than this angle return None; guards against numerical instability in near-parallel cases -- OPEN DECISION: this is below coplanar_angle_tol_deg (2.0), so pairs 0.5-2 deg apart are classed as one coplanarity group yet still yield an intersection line; unresolved for whoever wires edge construction
    # --- adjacency ---
    "adjacency_radius_m": 0.05,    # max gap between patches counted as adjacent
    # --- frame (measured, never enforced) ---
    "floor_normal_tol_deg": 15.0,  # max tilt from world Z for a floor/ceiling patch
    "vertical_normal_max_z": 0.2,  # patch counted as vertical when abs(normal.z) is below this; patches between roughly 11.5 and 78.5 deg off vertical fall into neither the horizontal nor vertical set (four such patches exist in the real crop)
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
