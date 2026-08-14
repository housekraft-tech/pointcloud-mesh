"""Face roles.

Only the roles that can be decided from a face's own geometry plus the measured
frame live here: floor, ceiling, wall, oblique, unknown. Roles that depend on
relationships between parts -- a step face, an opening reveal, a beam soffit --
are decided where those relationships are built, not guessed here.

Nothing in this module moves geometry. It reads orientation and height and
writes a label.
"""
from __future__ import annotations

import numpy as np


def classify_faces(faces, frame, config: dict) -> None:
    """Assign `Face.role` in place. Every face gets a role, never None."""
    tol_cos = float(np.cos(np.radians(config["floor_normal_tol_deg"])))
    vert_max_z = float(config["vertical_normal_max_z"])
    band = float(config["classify_slab_band_m"])
    floor_z = frame.floor_z
    ceiling_z = frame.ceiling_z

    for f in sorted(faces, key=lambda g: g.face_id):
        nz = abs(float(f.normal[2]))
        if nz >= tol_cos:
            z = float(f.centroid[2])
            near_floor = floor_z is not None and abs(z - floor_z) <= band
            near_ceiling = ceiling_z is not None and abs(z - ceiling_z) <= band
            if near_floor and near_ceiling:
                # degenerate storey; pick the nearer level
                f.role = "floor" if abs(z - floor_z) <= abs(z - ceiling_z) else "ceiling"
            elif near_floor:
                f.role = "floor"
            elif near_ceiling:
                f.role = "ceiling"
            else:
                f.role = "unknown"
        elif nz < vert_max_z:
            f.role = "wall"
        else:
            f.role = "oblique"
