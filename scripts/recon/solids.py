"""Solids: extrude + boolean cut, turning the plane-first manifold into named
watertight trimesh.Trimesh solids (walls, slabs, columns, beams) ready for
`assemble.py`.

Not required to be pure (see plan Global Constraints: solids.py is one of the
modules allowed to depend on trimesh/manifold3d).

Coordinate conventions (matching this repo's existing precedent in
`floorplan_reconstruct.py::_walls_to_obj_mesh` and `floorplan_schema.Wall`'s
own docstring):
  - `wall.p0`/`wall.p1` is the wall's horizontal CENTERLINE, not a face line.
  - `u` (used by both `WallStep.u_min_m/u_max_m` and `Opening.u_min_m/u_max_m`)
    is metric distance from `p0` along the unit direction `u_hat = (p1-p0)/|p1-p0|`.
  - `n_hat = (-u_hat.y, u_hat.x)` is the horizontal perpendicular ("wall
    normal") direction, rotated 90 degrees counter-clockwise from `u_hat`.
    `WallStep.offset_m` is a signed distance along `n_hat`, centered on the
    wall centerline (offset_m == 0.0 reproduces the plain rectangular wall
    box `_walls_to_obj_mesh` already builds with Open3D). This is a
    documented interpretation choice: the schema docstring only says
    offset is "relative to the wall's reference (main) face", but there is
    no separate "reference face" concept carried on floorplan_schema.Wall,
    so centering on the centerline reuses the one geometric reference this
    repo's Wall already has and keeps offset_m==0 backward compatible with
    the existing non-stepped wall mesh builder.
  - `Opening.sill_m`/`height_m` are relative to `wall.floor_z_m` (confirmed
    by `segment_walls_and_grooves.py::_opening_sill_m`, which computes
    sill = corner_z - floor_z). `WallStep.z_min_m`/`z_max_m` are absolute
    world Z (same frame as `wall.floor_z_m`/`ceiling_z_m`), since nothing in
    the schema documents them as floor-relative the way Opening's are.

Integration note for Task 17: the eventual `structure.py`/`regularize.py`
wall type isn't finalized yet. These functions only touch `p0`, `p1`,
`thickness_m` (optional, falls back to `DEFAULT_WALL_THICKNESS_M`), and
`floor_z_m` (optional, falls back to 0.0) on the `wall` argument -- so any
future wall object exposing at least those attributes will work unmodified.
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon

# Fallback wall thickness (metres) used only when the `wall` object passed in
# doesn't carry an explicit thickness_m (floorplan_schema.Wall always does,
# so this is a defensive default for future/duck-typed wall objects). Picked
# to sit in the middle of this repo's established plausible-thickness band --
# see floorplan_reconstruct.py's DEFAULT_CONFIG["pair_min_thickness_m"]=0.06 /
# ["pair_max_thickness_m"]=0.35 -- and matches typical interior partition
# thickness (~0.1-0.2 m).
DEFAULT_WALL_THICKNESS_M = 0.15

# How far an opening cutter box overshoots the wall's own thickness on each
# perpendicular side, so the boolean difference fully punches through even if
# the wall solid's faces aren't perfectly flat/aligned. Matches the task
# spec's suggested "wall thickness + 0.1 m on each side".
OPENING_OVERSHOOT_M = 0.1


def _horizontal_axes(wall) -> tuple:
    """Return (u_hat, n_hat): unit along-wall and unit perpendicular-in-plane
    horizontal directions, derived from wall.p0/p1 (see module docstring).
    """
    p0 = np.asarray(wall.p0, dtype=float)
    p1 = np.asarray(wall.p1, dtype=float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        raise ValueError("wall.p0 and wall.p1 must be distinct points")
    u_hat = d / length
    n_hat = np.array([-u_hat[1], u_hat[0]])
    return u_hat, n_hat


def _wall_thickness(wall) -> float:
    thickness = getattr(wall, "thickness_m", None)
    if thickness is None or thickness <= 0:
        return DEFAULT_WALL_THICKNESS_M
    return float(thickness)


def _axes_transform(u_hat, n_hat, center) -> np.ndarray:
    """4x4 transform placing a unit box's local +X/+Y/+Z axes onto
    u_hat/n_hat/world-Z, centered at `center`. Used so `trimesh.creation.box`
    (which builds an axis-aligned box centered at the origin) can be dropped
    directly into the wall/beam's local frame without a separate rotate+
    translate step.
    """
    transform = np.eye(4)
    transform[:3, 0] = [u_hat[0], u_hat[1], 0.0]
    transform[:3, 1] = [n_hat[0], n_hat[1], 0.0]
    transform[:3, 2] = [0.0, 0.0, 1.0]
    transform[:3, 3] = center
    return transform


def _ensure_watertight(mesh: trimesh.Trimesh, context: str) -> trimesh.Trimesh:
    """Assert watertightness, attempting a hole-fill + normal-fix repair pass
    first. Raises a clear RuntimeError if repair doesn't recover it.

    FUTURE fallback (not implemented here, per task spec): if manifold3d
    itself is unavailable or its boolean result can't be repaired, the plan
    calls for retrying the boolean op with Blender's `bpy` EXACT solver. That
    would hook in right here, before the final raise below -- `bpy` is not
    expected to be installed in this environment so it isn't implemented.
    """
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise RuntimeError(
            f"{context}: result is not watertight even after fill_holes/fix_normals repair "
            "(bpy EXACT-solver fallback not implemented in this environment)"
        )
    return mesh


def _step_box(wall, step, thickness_m: float) -> trimesh.Trimesh:
    """Box for one WallStep: u_min_m..u_max_m along the wall, z_min_m..z_max_m
    vertically, thickness_m wide and CENTERED on offset_m perpendicular to the
    wall centerline (see module docstring for the offset convention).
    """
    u_hat, n_hat = _horizontal_axes(wall)
    p0 = np.asarray(wall.p0, dtype=float)
    u_mid = (step.u_min_m + step.u_max_m) / 2.0
    z_mid = (step.z_min_m + step.z_max_m) / 2.0
    center_xy = p0 + u_mid * u_hat + step.offset_m * n_hat
    center = np.array([center_xy[0], center_xy[1], z_mid])
    extents = (
        step.u_max_m - step.u_min_m,
        thickness_m,
        step.z_max_m - step.z_min_m,
    )
    transform = _axes_transform(u_hat, n_hat, center)
    return trimesh.creation.box(extents=extents, transform=transform)


def wall_to_solid(wall, steps: list) -> trimesh.Trimesh:
    """Stepped extrusion of a wall into one watertight solid.

    `steps` is a list[WallStep] (scripts/recon/schema.py). Each step becomes
    its own box (see `_step_box`); a single step reproduces a plain
    rectangular wall. Multiple steps (e.g. a base wall run plus a pilaster/
    pillar relief at a different offset) are combined with a real boolean
    union (`trimesh.boolean.union`, backed by manifold3d) rather than naive
    mesh concatenation: steps commonly touch or overlap (a relief step's
    u-range sits inside the base step's u-range), and naive concatenation
    would leave doubled/overlapping interior faces at those shared
    boundaries -- still technically "watertight" by the edge-adjacency
    definition, but not a single clean manifold and not a trustworthy volume.
    manifold3d is already a hard dependency for `cut_openings`, so reusing it
    here avoids that whole class of bug for a small extra cost.
    """
    if not steps:
        raise ValueError("wall_to_solid requires at least one WallStep")
    thickness = _wall_thickness(wall)
    boxes = [_step_box(wall, step, thickness) for step in steps]
    if len(boxes) == 1:
        solid = boxes[0]
    else:
        solid = trimesh.boolean.union(boxes)
    return _ensure_watertight(solid, "wall_to_solid")


def slab_to_solid(plane, polygon, thickness: float) -> trimesh.Trimesh:
    """Extrude a 2D polygon (list[(x, y)] or shapely.Polygon) along a
    (near-horizontal) plane's normal by `thickness`, positioned at the
    plane's height.

    Position convention: for `plane.label == "floor"`, the slab sits BELOW
    the plane (the floor plane is the walkable top surface, the slab is the
    structural depth beneath it). For `"ceiling"`, the slab sits ABOVE the
    plane analogously. For any other/missing label (e.g. a duck-typed plane
    object in a test), the slab is centered on the plane -- an arbitrary but
    harmless default since only the extruded volume is guaranteed by the
    contract, not which side of the plane it's biased toward.
    """
    poly = polygon if isinstance(polygon, ShapelyPolygon) else ShapelyPolygon(polygon)
    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)

    normal = np.asarray(plane.normal, dtype=float)
    if abs(normal[2]) < 1e-6:
        raise ValueError("slab_to_solid requires a near-horizontal plane (floor/ceiling)")
    plane_z = -float(plane.d) / normal[2]

    label = getattr(plane, "label", None)
    if label == "ceiling":
        z_offset = plane_z
    elif label == "floor":
        z_offset = plane_z - thickness
    else:
        z_offset = plane_z - thickness / 2.0
    mesh.apply_translation([0.0, 0.0, z_offset])

    return _ensure_watertight(mesh, "slab_to_solid")


def column_to_solid(column) -> trimesh.Trimesh:
    """Extrude a Column's horizontal footprint polygon from z_min_m to
    z_max_m.
    """
    poly = ShapelyPolygon(column.footprint)
    height = column.z_max_m - column.z_min_m
    mesh = trimesh.creation.extrude_polygon(poly, height=height)
    mesh.apply_translation([0.0, 0.0, column.z_min_m])
    return _ensure_watertight(mesh, "column_to_solid")


def beam_to_solid(beam) -> trimesh.Trimesh:
    """Box along the beam's p0-p1 centerline.

    Field interpretation (scripts/recon/schema.py's Beam): `width_m` is the
    horizontal cross-section dimension perpendicular to the centerline (how
    wide the beam looks from below/the side). The vertical extent is taken
    directly from `z_min_m`/`z_max_m` -- both are absolute world Z and fully
    determine the beam's height, so `depth_m` (documented as "how tall") is
    not separately consumed here: using both would either be redundant (if
    depth_m == z_max_m - z_min_m, the common case) or ambiguous (if they
    disagree, nothing in the schema says whether the beam should be
    top-aligned, bottom-aligned, or centered within the mismatch). Trusting
    the explicit absolute bounds avoids inventing an alignment rule.
    """
    u_hat, n_hat = _horizontal_axes(beam)
    p0 = np.asarray(beam.p0, dtype=float)
    p1 = np.asarray(beam.p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    z_mid = (beam.z_min_m + beam.z_max_m) / 2.0
    center_xy = (p0 + p1) / 2.0
    center = np.array([center_xy[0], center_xy[1], z_mid])
    extents = (length, beam.width_m, beam.z_max_m - beam.z_min_m)
    transform = _axes_transform(u_hat, n_hat, center)
    mesh = trimesh.creation.box(extents=extents, transform=transform)
    return _ensure_watertight(mesh, "beam_to_solid")


def _opening_cutter_box(opening, wall, thickness_m: float) -> trimesh.Trimesh:
    u_hat, n_hat = _horizontal_axes(wall)
    p0 = np.asarray(wall.p0, dtype=float)
    floor_z = float(getattr(wall, "floor_z_m", 0.0))

    z_min = floor_z + opening.sill_m
    z_max = z_min + opening.height_m
    v_half = thickness_m / 2.0 + OPENING_OVERSHOOT_M

    u_mid = (opening.u_min_m + opening.u_max_m) / 2.0
    z_mid = (z_min + z_max) / 2.0
    center_xy = p0 + u_mid * u_hat  # v centered on the wall centerline (v_mid=0)
    center = np.array([center_xy[0], center_xy[1], z_mid])

    extents = (
        opening.u_max_m - opening.u_min_m,
        2.0 * v_half,
        z_max - z_min,
    )
    transform = _axes_transform(u_hat, n_hat, center)
    return trimesh.creation.box(extents=extents, transform=transform)


def cut_openings(wall_solid: trimesh.Trimesh, openings: list, wall) -> trimesh.Trimesh:
    """Boolean-subtract each opening's cutter box from wall_solid.

    `openings` is a list of floorplan_schema.Opening (this pipeline's shared
    Opening representation, per the plan's Global Constraints). Each cutter
    spans the opening's own u_min_m..u_max_m and sill_m..sill_m+height_m,
    overshooting the wall's perpendicular thickness by OPENING_OVERSHOOT_M on
    each side so the cut fully punches through regardless of small alignment
    error between wall_solid's actual faces and the nominal thickness.

    Uses `trimesh.boolean.difference` (manifold3d backend). Watertightness is
    asserted after every single cut (not just at the end) so a bad opening
    is caught at the element that caused it, not blamed on a later one.
    """
    solid = wall_solid
    thickness = _wall_thickness(wall)
    for opening in openings:
        cutter = _opening_cutter_box(opening, wall, thickness)
        solid = trimesh.boolean.difference([solid, cutter])
        _ensure_watertight(
            solid, f"cut_openings (opening_id={getattr(opening, 'opening_id', '?')})"
        )
    return solid
