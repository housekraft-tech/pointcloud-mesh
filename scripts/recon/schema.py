"""Data model for the plane-first modular reconstruction pipeline.

Extends scripts/floorplan_schema.py (Wall, Opening) with the point-cloud
container (ScanData), detected planes (Plane), and the relief/element types
(WallStep, Column, Beam). Pure numpy -- imports without Open3D/trimesh so the
downstream geometry modules stay unit-testable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class ScanData:
    """A point cloud plus its optional per-point attributes, all row-aligned."""
    xyz: np.ndarray  # (N, 3) float64, metres, world frame
    gps_time: Optional[np.ndarray] = None  # (N,) float64 or None
    rgb: Optional[np.ndarray] = None  # (N, 3) uint8 or None
    intensity: Optional[np.ndarray] = None  # (N,) or None

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask) -> "ScanData":
        """Return a new ScanData keeping rows selected by a boolean/index mask.

        Every non-None attribute is indexed identically so rows stay aligned.
        """
        mask = np.asarray(mask)
        return ScanData(
            xyz=self.xyz[mask],
            gps_time=None if self.gps_time is None else self.gps_time[mask],
            rgb=None if self.rgb is None else self.rgb[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
        )


@dataclass
class Plane:
    """An infinite plane normal . x + d = 0 with its supporting points."""
    normal: tuple  # unit (a, b, c)
    d: float
    label: str  # "floor" | "ceiling" | "vertical"
    inlier_idx: np.ndarray  # indices into the source cloud

    def signed_distance(self, pts) -> np.ndarray:
        """Signed perpendicular distance of each point to the plane (metres)."""
        pts = np.asarray(pts, dtype=float)
        return pts @ np.asarray(self.normal, dtype=float) + self.d


@dataclass
class WallStep:
    """A planar wall face at a given perpendicular offset within a wall run.

    Steps capture in/out relief (pilasters, niches, pillar faces) that a single
    flat plane per wall would erase. offset_m is signed along the wall normal
    relative to the wall's reference (main) face.
    """
    offset_m: float
    u_min_m: float
    u_max_m: float
    z_min_m: float
    z_max_m: float


@dataclass
class Column:
    """A vertical prismatic pillar, described by its horizontal footprint."""
    column_id: str
    footprint: list  # list[(x, y)] polygon, world frame
    z_min_m: float
    z_max_m: float


@dataclass
class Beam:
    """A horizontal prismatic beam under the ceiling."""
    beam_id: str
    p0: tuple  # (x, y) centreline endpoint
    p1: tuple
    width_m: float
    depth_m: float
    z_min_m: float
    z_max_m: float


def new_column_id(index: int) -> str:
    return f"column_{index:03d}"


def new_beam_id(index: int) -> str:
    return f"beam_{index:03d}"


def column_to_dict(column: Column) -> dict:
    return asdict(column)


def beam_to_dict(beam: Beam) -> dict:
    return asdict(beam)
