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
    """A dimension plus the evidence behind it."""

    value: float
    method: str
    n_points: int
    p95_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "n_points": self.n_points,
            "p95_residual": self.p95_residual,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Measurement":
        return Measurement(
            value=payload["value"], method=payload["method"],
            n_points=payload["n_points"], p95_residual=payload["p95_residual"],
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


def scene_to_json(scene: Scene) -> str:
    """Serialise deterministically: sorted keys, fixed indent."""
    payload = {
        "adjacency": [list(pair) for pair in scene.adjacency],
        "coplanarity_classes": scene.coplanarity_classes,
        "diagnostics": scene.diagnostics,
        "frame": scene.frame.to_dict(),
        "patches": [_patch_to_dict(p) for p in scene.patches],
        "provenance": scene.provenance.to_dict(),
        "unassigned_points": scene.unassigned_points,
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
    )
