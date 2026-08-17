"""The scene document -- the single source of truth.

JSON rather than executable code: the LLM never edits geometry here, so
code-as-truth would cost determinism and diffability without buying anything.
A readable room.py is emitted FROM this document later, not into it.

Every dimension is a Measurement carrying how it was measured, how many points
backed it and how well they fitted. A number a designer cannot audit is a
number they will eventually stop trusting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .patches import Patch


@dataclass
class Measurement:
    """A dimension plus the evidence behind it.

    `n_bins` is an optional sibling to `n_points`/`p95_residual`: for a
    measurement assembled from several local sub-measurements (e.g. wall
    thickness binned across an overlap region -- see `core/parts.py`), it
    states how many of those sub-measurements actually qualified and back
    this specific value (for wall thickness: the DOMINANT segment's bin
    count, not the whole field's -- see `ThicknessField`), so a caller can
    tell a genuine multi-bin median from a single-cell or whole-overlap
    fallback without parsing `method`. `None` for measurements that were
    never binned (length, height, ...).
    """

    value: float
    method: str
    n_points: int
    p95_residual: float
    n_bins: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "n_points": self.n_points,
            "p95_residual": self.p95_residual,
            "n_bins": self.n_bins,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Measurement":
        return Measurement(
            value=payload["value"], method=payload["method"],
            n_points=payload["n_points"], p95_residual=payload["p95_residual"],
            n_bins=payload.get("n_bins"),
        )


@dataclass
class Provenance:
    """Where the scene came from, so any result can be reproduced."""

    scan_path: str
    scan_sha256: str
    pipeline_version: str
    timestamp: str
    config: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_path": self.scan_path, "scan_sha256": self.scan_sha256,
            "pipeline_version": self.pipeline_version, "timestamp": self.timestamp,
            "config": self.config,
        }

    @staticmethod
    def from_dict(p: dict[str, Any]) -> "Provenance":
        return Provenance(
            scan_path=p["scan_path"], scan_sha256=p["scan_sha256"],
            pipeline_version=p["pipeline_version"], timestamp=p["timestamp"],
            config=p["config"],
        )


@dataclass
class Frame:
    """The measured frame. Reported as data; no geometry is rotated to match."""

    z_axis: list[float]
    xy_rotation_deg: float
    floor_z: Optional[float] = None
    ceiling_z: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_axis": list(self.z_axis), "xy_rotation_deg": self.xy_rotation_deg,
            "floor_z": self.floor_z, "ceiling_z": self.ceiling_z,
        }

    @staticmethod
    def from_dict(p: dict[str, Any]) -> "Frame":
        return Frame(
            z_axis=p["z_axis"], xy_rotation_deg=p["xy_rotation_deg"],
            floor_z=p["floor_z"], ceiling_z=p["ceiling_z"],
        )


def _patch_to_dict(p: Patch) -> dict[str, Any]:
    return {
        "patch_id": p.patch_id,
        "normal": [float(x) for x in p.normal],
        "d": float(p.d),
        "point_idx": [int(i) for i in p.point_idx],
        "n_points": p.n_points,
        "p95_residual_m": p.p95_residual_m,
        "centroid": [float(x) for x in p.centroid],
        "u_range": list(p.u_range),
        "v_range": list(p.v_range),
    }


def _patch_from_dict(p: dict[str, Any]) -> Patch:
    return Patch(
        patch_id=p["patch_id"],
        normal=np.array(p["normal"], dtype=np.float64),
        d=p["d"],
        point_idx=np.array(p["point_idx"], dtype=np.int64),
        n_points=p["n_points"],
        p95_residual_m=p["p95_residual_m"],
        centroid=np.array(p["centroid"], dtype=np.float64),
        u_range=tuple(p["u_range"]),
        v_range=tuple(p["v_range"]),
    )


def _measure_to_dict(m: Optional["Measurement"]):
    return None if m is None else m.to_dict()


def _measure_from_dict(p):
    return None if p is None else Measurement.from_dict(p)


def _face_to_dict(f) -> dict[str, Any]:
    return {
        "face_id": f.face_id,
        "normal": [float(x) for x in f.normal],
        "d": float(f.d),
        "patch_ids": list(f.patch_ids),
        "point_idx": [int(i) for i in f.point_idx],
        "loose_idx": [int(i) for i in f.loose_idx],
        "n_points": f.n_points,
        "p95_residual_m": f.p95_residual_m,
        "centroid": [float(x) for x in f.centroid],
        "u_range": list(f.u_range),
        "v_range": list(f.v_range),
        "role": f.role,
        "interior_sign": f.interior_sign,
    }


def _face_from_dict(p: dict[str, Any]):
    from .faces import Face
    return Face(
        face_id=p["face_id"],
        normal=np.array(p["normal"], dtype=np.float64),
        d=p["d"],
        patch_ids=list(p["patch_ids"]),
        point_idx=np.array(p["point_idx"], dtype=np.int64),
        loose_idx=np.array(p["loose_idx"], dtype=np.int64),
        n_points=p["n_points"],
        p95_residual_m=p["p95_residual_m"],
        centroid=np.array(p["centroid"], dtype=np.float64),
        u_range=tuple(p["u_range"]),
        v_range=tuple(p["v_range"]),
        role=p["role"],
        interior_sign=p["interior_sign"],
    )


def _wall_to_dict(w) -> dict[str, Any]:
    return {
        "wall_id": w.wall_id, "face_a": w.face_a, "face_b": w.face_b,
        "thickness": _measure_to_dict(w.thickness),
        "thickness_field": None if w.thickness_field is None else w.thickness_field.to_dict(),
        "length": _measure_to_dict(w.length),
        "height": _measure_to_dict(w.height),
        "centroid": [float(x) for x in w.centroid],
        "normal": [float(x) for x in w.normal],
    }


def _wall_from_dict(p: dict[str, Any]):
    from .parts import ThicknessField, Wall
    return Wall(
        wall_id=p["wall_id"], face_a=p["face_a"], face_b=p["face_b"],
        thickness=_measure_from_dict(p["thickness"]),
        thickness_field=(
            None if p["thickness_field"] is None
            else ThicknessField.from_dict(p["thickness_field"])
        ),
        length=_measure_from_dict(p["length"]),
        height=_measure_from_dict(p["height"]),
        centroid=np.array(p["centroid"], dtype=np.float64),
        normal=np.array(p["normal"], dtype=np.float64),
    )


def _feature_to_dict(f) -> dict[str, Any]:
    return {
        "feature_id": f.feature_id, "kind": f.kind, "parent_face": f.parent_face,
        "source_face": f.source_face,
        "u_range": list(f.u_range), "v_range": list(f.v_range),
        "depth": _measure_to_dict(f.depth), "rect_fit": f.rect_fit,
    }


def _feature_from_dict(p: dict[str, Any]):
    from .features import Feature
    return Feature(
        feature_id=p["feature_id"], kind=p["kind"], parent_face=p["parent_face"],
        source_face=p["source_face"],
        u_range=tuple(p["u_range"]), v_range=tuple(p["v_range"]),
        depth=_measure_from_dict(p["depth"]), rect_fit=p["rect_fit"],
    )


@dataclass
class Scene:
    """The whole scene document. Plan 1 populates patches; parts arrive in Plan 2."""

    provenance: Provenance
    frame: Frame
    patches: list[Patch] = field(default_factory=list)
    coplanarity_classes: list[list[int]] = field(default_factory=list)
    adjacency: list[tuple[int, int]] = field(default_factory=list)
    unassigned_points: int = 0
    diagnostics: dict = field(default_factory=dict)
    faces: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    features: list = field(default_factory=list)
    unmodeled: list[int] = field(default_factory=list)


def scene_to_json(scene: Scene) -> str:
    """Serialise deterministically: sorted keys, fixed indent."""
    payload = {
        "adjacency": [list(pair) for pair in scene.adjacency],
        "coplanarity_classes": scene.coplanarity_classes,
        "diagnostics": scene.diagnostics,
        "faces": [_face_to_dict(f) for f in scene.faces],
        "features": [_feature_to_dict(f) for f in scene.features],
        "frame": scene.frame.to_dict(),
        "patches": [_patch_to_dict(p) for p in scene.patches],
        "provenance": scene.provenance.to_dict(),
        "unassigned_points": scene.unassigned_points,
        "unmodeled": list(scene.unmodeled),
        "walls": [_wall_to_dict(w) for w in scene.walls],
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def scene_from_json(text: str) -> Scene:
    payload = json.loads(text)
    return Scene(
        provenance=Provenance.from_dict(payload["provenance"]),
        frame=Frame.from_dict(payload["frame"]),
        patches=[_patch_from_dict(p) for p in payload["patches"]],
        coplanarity_classes=payload["coplanarity_classes"],
        adjacency=[tuple(pair) for pair in payload["adjacency"]],
        unassigned_points=payload["unassigned_points"],
        diagnostics=payload["diagnostics"],
        faces=[_face_from_dict(p) for p in payload["faces"]],
        walls=[_wall_from_dict(p) for p in payload["walls"]],
        features=[_feature_from_dict(p) for p in payload["features"]],
        unmodeled=list(payload["unmodeled"]),
    )
