"""Rectangular wall features, and the gate that keeps concrete out.

A bare-shell concrete building is rectilinear by construction: formwork makes
flat faces meeting at sharp edges. A real feature -- an extrusion, a recess, a
groove, an electrical box -- is a rectangular prism. An irregular lump of
leftover concrete is not, and cannot be made to fit one.

That is the whole discriminator, and it needs no heuristics about size or
position. `rect_fit` is the fraction of the feature's own in-plane bounding box
that its points actually fill; a rectangular face fills its box, a scattered
mass does not. Anything below the gate is quarantined -- returned to the caller
for the scene's `unmodeled` set, never silently deleted and never promoted.

Measured defect this module is designed around: a wall pair's own partner
looks exactly like a feature of the other face -- the far face of a 200 mm
wall is geometrically indistinguishable from a 200 mm extrusion on the near
face. `extract_features` therefore runs AFTER `assemble_walls` and takes the
resulting `walls` list so it can exclude every face already claimed as a wall
pair's partner from candidacy. A face can still serve as some OTHER
candidate's parent even while it is itself a wall-pair member (a genuine
extrusion standing off a paired wall's face is real); only being promoted TO
a feature is what pairing forecloses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faces import Face, median_spacing
from .graph import perpendicular_offset
from .scene import Measurement


@dataclass
class Feature:
    """A rectangular departure from a parent face."""

    feature_id: str
    kind: str                       # "extrusion" | "intrusion"
    parent_face: int                # the larger face this feature stands proud of / recesses into
    source_face: int                # the candidate face this feature was built from
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    depth: Measurement
    rect_fit: float


def _rect_fit(face: Face, xyz: np.ndarray, spacing: float) -> float:
    """Fraction of the face's in-plane bbox its own points fill."""
    area = face.area_bound_m2()
    if area <= 0 or spacing <= 0:
        return 0.0
    expected = area / (spacing ** 2)
    return float(min(1.0, face.n_points / expected)) if expected > 0 else 0.0


def extract_features(
    faces: list[Face], xyz: np.ndarray, config: dict, walls: list
) -> tuple[list[Feature], list[int]]:
    """Find rectangular features standing proud of or recessed into wall faces.

    Must be called AFTER `assemble_walls`, with its `walls` result passed in.
    A candidate is a wall face parallel to a larger wall face, offset by less
    than `feature_max_depth_m`, and smaller than it -- EXCLUDING any face that
    is `face_a` or `face_b` of a paired (`face_b is not None`) `Wall`: that
    face is already claimed as a wall's own partner, and re-emerges looking
    exactly like a feature of it (the measured defect this function is built
    to avoid). Faces that remain candidates can still be measured against a
    wall-pair member acting as PARENT -- only being consumed as a feature's
    own source face is excluded.

    Candidates that fail the rectangularity gate are quarantined by face id
    rather than deleted.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    spacing = median_spacing(xyz, seed=int(config["seed"]))
    max_depth = float(config["feature_max_depth_m"])
    min_fit = float(config["feature_min_rect_fit"])
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))

    pair_members: set[int] = set()
    for w in walls:
        if w.face_b is not None:
            pair_members.add(w.face_a)
            pair_members.add(w.face_b)

    wall_faces = sorted([f for f in faces if f.role == "wall"], key=lambda f: f.face_id)
    candidates = [f for f in wall_faces if f.face_id not in pair_members]

    features: list[Feature] = []
    quarantined: list[int] = []
    counter = 0

    for cand in candidates:
        parent = None
        best = np.inf
        for other in wall_faces:
            if other.face_id == cand.face_id:
                continue
            if other.area_bound_m2() <= cand.area_bound_m2():
                continue
            if abs(float(cand.normal @ other.normal)) < cos_tol:
                continue
            offset = perpendicular_offset(cand, other)
            if offset <= 0 or offset > max_depth:
                continue
            if offset < best:
                best, parent = offset, other
        if parent is None:
            continue

        fit = _rect_fit(cand, xyz, spacing)
        if fit < min_fit:
            quarantined.append(cand.face_id)
            continue

        # sign: does the candidate stand toward the parent's interior side?
        along = float((cand.centroid - parent.centroid) @ parent.normal)
        kind = "extrusion" if along > 0 else "intrusion"

        counter += 1
        features.append(Feature(
            feature_id=f"F{counter:02d}",
            kind=kind,
            parent_face=parent.face_id,
            source_face=cand.face_id,
            u_range=cand.u_range,
            v_range=cand.v_range,
            depth=Measurement(
                value=float(best),
                method="perpendicular offset to parent face",
                n_points=int(cand.n_points),
                p95_residual=float(max(cand.p95_residual_m, parent.p95_residual_m)),
            ),
            rect_fit=float(fit),
        ))

    return features, sorted(quarantined)
