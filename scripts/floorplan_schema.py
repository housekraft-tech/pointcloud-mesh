"""manifest.json data model shared by floorplan_reconstruct.py and validate_measurements.py."""
from dataclasses import dataclass, field, asdict


@dataclass
class Opening:
    opening_id: str
    wall_id: str
    type: str  # "door" | "window" | "balcony_door" | "unknown_opening"
    u_min_m: float
    u_max_m: float
    sill_m: float
    height_m: float
    width_m: float
    edge_method: str  # "density_half_max" (v1; "reveal_plane" reserved for later)
    both_faces_confirmed: bool


@dataclass
class Wall:
    wall_id: str
    p0: tuple  # (x, y) centerline endpoint, meters, world frame
    p1: tuple
    length_m: float
    thickness_m: float
    thickness_source: str  # "measured" | "assumed"
    plane_front: list  # [a, b, c, d] of the front face
    plane_back: list  # [a, b, c, d] of the back face, or None if thickness_source == "assumed"
    origin_xyz: tuple  # wall-local frame origin used for u/v projection
    u_axis: tuple
    v_axis: tuple
    floor_z_m: float
    ceiling_z_m: float
    region_band_m: float  # perpendicular band width used for point selection during refit
    region_corner_margin_m: float  # corner exclusion margin used during refit
    openings: list = field(default_factory=list)  # list[Opening]
    grooves: list = field(default_factory=list)  # reserved for Phase 2, empty here


def new_wall_id(index):
    return f"wall_{index:03d}"


def wall_to_dict(wall: Wall) -> dict:
    d = asdict(wall)
    d["openings"] = [asdict(o) if not isinstance(o, dict) else o for o in wall.openings]
    return d


def opening_to_dict(opening: Opening) -> dict:
    return asdict(opening)
