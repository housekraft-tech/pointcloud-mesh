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


@dataclass
class Feature:
    """One detected plane / relief element belonging to a wall, in the
    wall's own 1D u-coordinate (not XY-clipped) -- lets a wall be split
    into room-owned WallSegments later without re-deriving relief geometry.
    """
    kind: str  # "face" | "groove" | "l_extrusion" | "beam_soffit"
               #   | "column_attach" | "opening"
    u_min_m: float
    u_max_m: float
    z_min_m: float
    z_max_m: float
    offset_m: float  # signed depth from the wall's reference face
    meta: dict = field(default_factory=dict)  # e.g. {"type": "door", "walked": True}


@dataclass
class WallSegment:
    """The unit that becomes exactly one mesh: a wall clipped to one room's
    (or unassigned) ownership range, carrying only the features that fall
    in that u-range.
    """
    wall_id: str
    room_id: Optional[str]  # None -> unassigned (never silently dropped)
    u_min_m: float
    u_max_m: float
    features: list  # list[Feature], clipped to [u_min_m, u_max_m]


@dataclass
class ElementDims:
    """Measured/refined/assumed dimensions of one element (door, window,
    column, beam, ...), carried alongside human-readable review flags.
    """
    width_m: float
    height_m: float
    depth_m: float
    source: str  # "measured" | "refined" | "assumed"
    flags: list = field(default_factory=list)


def new_column_id(index: int) -> str:
    return f"column_{index:03d}"


def new_beam_id(index: int) -> str:
    return f"beam_{index:03d}"


def new_wall_id(index: int) -> str:
    return f"wall_{index:03d}"


def new_opening_id(index: int) -> str:
    return f"opening_{index:03d}"


def new_room_id(index: int) -> str:
    return f"room_{index:03d}"


def column_to_dict(column: Column) -> dict:
    return asdict(column)


def beam_to_dict(beam: Beam) -> dict:
    return asdict(beam)


def feature_to_dict(feature: Feature) -> dict:
    return asdict(feature)


def wall_segment_to_dict(segment: WallSegment) -> dict:
    d = asdict(segment)
    d["features"] = [asdict(f) if not isinstance(f, dict) else f for f in segment.features]
    return d


# ---------------------------------------------------------------------------
# Dimension priors + manifest assembly (Task 8)
# ---------------------------------------------------------------------------

DEFAULT_PRIORS = {
    "door_h_m": 2.13,
    "door_h_tol_m": 0.25,
    "ceiling_m": 2.75,
    "ceiling_tol_m": 0.35,
}


def check_priors(kind: str, dims: ElementDims, priors: dict) -> list:
    """Compare a measured element's dims against expected-value priors.

    Pure and non-destructive: flags never delete/drop an element, they only
    surface human-readable strings for review (same philosophy as
    structure.extract_unclassified -- nothing found is silently dropped).
    """
    flags = []
    if kind == "door":
        prior = priors.get("door_h_m")
        tol = priors.get("door_h_tol_m", 0.0)
        if prior is not None and abs(dims.height_m - prior) > tol:
            flags.append(f"door_height_unusual: {dims.height_m:.2f}m (prior {prior:.2f}m)")
    elif kind == "ceiling":
        prior = priors.get("ceiling_m")
        tol = priors.get("ceiling_tol_m", 0.0)
        if prior is not None and abs(dims.height_m - prior) > tol:
            flags.append(f"ceiling_height_unusual: {dims.height_m:.2f}m (prior {prior:.2f}m)")
    return flags


def _polygon_area_m2(footprint) -> float:
    """Shoelace-formula area (m^2) of a closed 2D polygon [(x, y), ...]."""
    if not footprint or len(footprint) < 3:
        return 0.0
    xs = [float(p[0]) for p in footprint]
    ys = [float(p[1]) for p in footprint]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def build_manifest(walls, openings, columns, beams, rooms, z_floor, z_ceiling, config) -> dict:
    """Assemble the final JSON-ready manifest for one storey.

    walls: list of wall dicts, the shape produced by
        structure.group_wall_runs / regularize.pair_thickness / recenter_walls
        (keys used here: p0, p1, thickness_m, thickness_source, steps;
        length_m is derived from p0/p1 when not already present).
    openings: dict[int, list[dict]] keyed by wall index -- the shape
        produced by openings.detect_openings (u0/u1 or u_min_m/u_max_m,
        z0/z1, width_m/height_m/sill_m, type, walked, confidence).
    columns / beams: list[Column] / list[Beam].
    rooms: list of polygons (list[(x, y)]), e.g. from
        floorplan2d.build_room_polygons.
    config: echoed verbatim under "config"; config.get("priors") overrides
        DEFAULT_PRIORS for the check_priors sanity flags below.
    """
    priors = {**DEFAULT_PRIORS, **((config or {}).get("priors") or {})}

    wall_dicts = []
    for i, w in enumerate(walls):
        w = w if isinstance(w, dict) else asdict(w)
        p0 = tuple(w["p0"])
        p1 = tuple(w["p1"])
        length_m = w.get("length_m")
        if length_m is None:
            length_m = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        steps = w.get("steps", [])
        steps_out = [asdict(s) if not isinstance(s, dict) else s for s in steps]
        wall_dicts.append({
            "wall_id": new_wall_id(i),
            "p0": p0,
            "p1": p1,
            "length_m": float(length_m),
            "thickness_m": float(w.get("thickness_m", 0.0)),
            "thickness_source": w.get("thickness_source", "assumed"),
            "steps": steps_out,
        })

    opening_dicts = []
    opening_index = 0
    for wall_idx, wall_openings in (openings or {}).items():
        wall_id = new_wall_id(wall_idx) if isinstance(wall_idx, int) else wall_idx
        for o in wall_openings:
            u_min = o.get("u_min_m", o.get("u0"))
            u_max = o.get("u_max_m", o.get("u1"))
            z_min = o.get("z_min_m", o.get("z0"))
            z_max = o.get("z_max_m", o.get("z1"))
            width_m = o.get("width_m")
            if width_m is None and u_min is not None and u_max is not None:
                width_m = u_max - u_min
            height_m = o.get("height_m")
            if height_m is None and z_min is not None and z_max is not None:
                height_m = z_max - z_min
            sill_m = o.get("sill_m", z_min)
            kind = o.get("type", "unknown_opening")

            dims = ElementDims(
                width_m=float(width_m) if width_m is not None else 0.0,
                height_m=float(height_m) if height_m is not None else 0.0,
                depth_m=float(o.get("depth_m", 0.0)),
                source=o.get("source", "measured"),
            )
            flags = check_priors(kind, dims, priors) if kind == "door" else list(o.get("flags", []))

            opening_dicts.append({
                "opening_id": new_opening_id(opening_index),
                "wall_id": wall_id,
                "type": kind,
                "u_min_m": u_min,
                "u_max_m": u_max,
                "sill_m": sill_m,
                "width_m": width_m,
                "height_m": height_m,
                "walked": o.get("walked", False),
                "confidence": o.get("confidence"),
                "oversized": o.get("oversized", False),
                "flags": flags,
            })
            opening_index += 1

    column_dicts = []
    for i, c in enumerate(columns):
        column_dicts.append({
            "column_id": new_column_id(i),
            "footprint": [tuple(p) for p in c.footprint],
            "z_min_m": float(c.z_min_m),
            "z_max_m": float(c.z_max_m),
            "height_m": float(c.z_max_m - c.z_min_m),
        })

    beam_dicts = []
    for i, b in enumerate(beams):
        beam_dicts.append({
            "beam_id": new_beam_id(i),
            "p0": tuple(b.p0),
            "p1": tuple(b.p1),
            "span_m": float(np.hypot(b.p1[0] - b.p0[0], b.p1[1] - b.p0[1])),
            "width_m": float(b.width_m),
            "depth_m": float(b.depth_m),
            "z_min_m": float(b.z_min_m),
            "z_max_m": float(b.z_max_m),
        })

    room_dicts = []
    for i, r in enumerate(rooms):
        room_dicts.append({
            "room_id": new_room_id(i),
            "footprint": [tuple(p) for p in r],
            "area_m2": _polygon_area_m2(r),
        })

    storey_height_m = float(z_ceiling - z_floor)
    ceiling_dims = ElementDims(width_m=0.0, height_m=storey_height_m, depth_m=0.0, source="measured")
    storey_flags = check_priors("ceiling", ceiling_dims, priors)

    return {
        "walls": wall_dicts,
        "openings": opening_dicts,
        "columns": column_dicts,
        "beams": beam_dicts,
        "rooms": room_dicts,
        "storey": {
            "z_floor_m": float(z_floor),
            "z_ceiling_m": float(z_ceiling),
            "height_m": storey_height_m,
            "flags": storey_flags,
        },
        "config": config,
    }
