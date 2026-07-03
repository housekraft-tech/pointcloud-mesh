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
  - `WallStep.offset_m` is a signed distance perpendicular to the wall,
    centered on the wall centerline (offset_m == 0.0 reproduces the plain
    rectangular wall box `_walls_to_obj_mesh` already builds with Open3D).
    The perpendicular axis is NOT a locally-rotated `n_hat = (-u_hat.y,
    u_hat.x)` derived from this wall's own, arbitrary p0->p1 ordering --
    it is whichever of the two WORLD axes (X or Y) is nearest-perpendicular
    to the wall's own direction, always taken with its POSITIVE sign, e.g.
    +X (never -X) for a wall that runs along Y. This is the exact same
    convention `recon.structure.group_wall_runs` uses to compute
    WallStep.offset_m in the first place (see `_plane_axis_stats`, which
    always measures a plane's offset along the positive `ex`/`ey` it is
    closer to, never a per-wall-local sign) -- `_wall_world_axes` below
    reproduces it here from `wall.p0`/`wall.p1` alone so a WallStep
    produced by group_wall_runs and consumed by `wall_to_solid` means the
    same physical offset in both places. Using the wall's own locally
    rotated `n_hat` instead (the previous behavior of this module) is a
    convention MISMATCH: for a wall whose own p0->p1 happens to run in the
    world-negative direction along its snapped axis, `n_hat` and the
    world-positive axis point opposite ways, so a relief step (e.g. a
    pillar 0.3 m proud of its wall) would be built on the wrong side of the
    wall entirely. This is a documented interpretation choice: the schema
    docstring only says offset is "relative to the wall's reference (main)
    face", but there is no separate "reference face" concept carried on
    floorplan_schema.Wall, so centering on the centerline reuses the one
    geometric reference this repo's Wall already has and keeps offset_m==0
    backward compatible with the existing non-stepped wall mesh builder.
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

from types import SimpleNamespace

import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon

from .schema import WallStep

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
    horizontal directions, derived from wall.p0/p1.

    n_hat here is a LOCAL perpendicular (rotated 90 degrees CCW from
    u_hat) -- it is NOT the world-axis-snapped perpendicular WallStep.offset_m
    is measured along (see `_wall_world_axes` and the module docstring for
    why those differ). Still safe to use for `beam_to_solid` and
    `_opening_cutter_box`/`cut_openings`: both only ever build a box that is
    SYMMETRIC about n_hat == 0 (a beam's width, an opening cutter's
    thickness-plus-overshoot), so n_hat's sign is irrelevant there -- unlike
    `_step_box`, which places a step at a specific SIGNED offset and so
    needs the world-axis-consistent convention instead.
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


def _wall_world_axes(wall) -> tuple:
    """Return (u_is_x, u_sign): which WORLD axis the wall's own p0->p1
    direction is closest to, and that direction's sign along that axis.

    This mirrors `recon.structure.group_wall_runs`'s own dominant-axis snap
    (`_plane_axis_stats`, which classifies a plane's normal against the
    nearest of the two dominant horizontal axes ex/ey) applied instead to a
    wall's along-wall direction: by this pipeline's Manhattan-alignment
    convention (see frame.axis_align -- the cloud is always axis-aligned
    before structure.py runs), a wall's own p0->p1 direction is close to
    either world X or world Y, and its perpendicular ("normal") axis is
    simply the OTHER one.

    u_is_x is True when the wall runs along world X (so its perpendicular
    axis is world Y -- a group_wall_runs run with direction="y"); False
    when it runs along world Y (perpendicular axis world X, direction="x").

    u_sign is +1.0 if wall.p1 sits at a larger coordinate than wall.p0 along
    the snapped u axis, else -1.0. This only affects how u_min_m/u_max_m
    (measured from p0 along the wall's OWN direction, per the module
    docstring) map onto world coordinates -- it must NOT be used to flip
    the perpendicular axis's sign (see module docstring): WallStep.offset_m
    is always signed along the snapped axis's POSITIVE direction, exactly
    like group_wall_runs's own `_plane_axis_stats` always measures a
    plane's offset along +ex/+ey and never negates it based on which side
    of the wall the plane happens to be.
    """
    p0 = np.asarray(wall.p0, dtype=float)
    p1 = np.asarray(wall.p1, dtype=float)
    d = p1 - p0
    if float(np.linalg.norm(d)) < 1e-9:
        raise ValueError("wall.p0 and wall.p1 must be distinct points")
    u_is_x = abs(d[0]) >= abs(d[1])
    u_component = d[0] if u_is_x else d[1]
    u_sign = 1.0 if u_component >= 0 else -1.0
    return u_is_x, u_sign


def _step_box(wall, step, thickness_m: float) -> trimesh.Trimesh:
    """Box for one WallStep.

    Along the wall (u): spans u_min_m..u_max_m, measured from wall.p0 along
    the wall's own p0->p1 direction (unchanged from before).

    Vertically (z): spans z_min_m..z_max_m (unchanged from before).

    Perpendicular to the wall (v, the offset_m axis): spans from
    `min(0.0, step.offset_m) - thickness_m/2` to `max(0.0, step.offset_m) +
    thickness_m/2`, measured along the WORLD axis nearest the wall's own
    direction (see `_wall_world_axes`) -- the same axis/sign
    `recon.structure.group_wall_runs` uses for WallStep.offset_m itself,
    not a locally rotated axis derived from this wall's own arbitrary
    p0/p1 ordering (see module docstring).

    This v-range is deliberately NOT a fixed-width slab centered on
    offset_m (the previous behavior): for the main step (offset_m == 0.0)
    it reduces to exactly the old symmetric [-thickness/2, +thickness/2]
    box (backward compatible with the plain non-stepped wall). For any
    other step it ALWAYS fully contains that same [-thickness/2,
    +thickness/2] reference span too -- e.g. a pillar-front step at
    offset_m=+0.3 spans continuously from the main wall's own back face
    (-thickness/2) out to the pillar's own front face plus half-thickness
    (+0.3 + thickness/2), rather than floating as an independent slab only
    thickness_m wide centered at +0.3. That guarantees every step's box
    touches/overlaps the main wall's box (they share the whole reference
    span), so `trimesh.boolean.union` in `wall_to_solid` always comes back
    as ONE connected solid instead of disconnected floating bodies.

    No rotation matrix is used: since the wall is treated as exactly
    axis-aligned (u/v snapped to world X/Y per `_wall_world_axes`), the box
    is a plain world-axis-aligned box positioned by translation only --
    this also sidesteps any risk of a mirrored (negative-determinant)
    transform silently inverting face winding/normals when the wall's own
    p0->p1 direction is the negative of its snapped world axis.
    """
    u_is_x, u_sign = _wall_world_axes(wall)
    p0 = np.asarray(wall.p0, dtype=float)

    v_lo = min(0.0, step.offset_m) - thickness_m / 2.0
    v_hi = max(0.0, step.offset_m) + thickness_m / 2.0
    v_mid = (v_lo + v_hi) / 2.0
    v_extent = v_hi - v_lo

    u_mid = (step.u_min_m + step.u_max_m) / 2.0
    z_mid = (step.z_min_m + step.z_max_m) / 2.0
    u_extent = step.u_max_m - step.u_min_m
    z_extent = step.z_max_m - step.z_min_m

    if u_is_x:
        # Wall runs along world X; perpendicular (offset) axis is world Y,
        # always POSITIVE (never flipped by u_sign) to match
        # group_wall_runs's own convention.
        center = np.array([p0[0] + u_sign * u_mid, p0[1] + v_mid, z_mid])
        extents = (u_extent, v_extent, z_extent)
    else:
        # Wall runs along world Y; perpendicular axis is world X, positive.
        center = np.array([p0[0] + v_mid, p0[1] + u_sign * u_mid, z_mid])
        extents = (v_extent, u_extent, z_extent)

    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


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


# ---------------------------------------------------------------------------
# Task 9 (Architecture Revision R1): per-segment meshing
# ---------------------------------------------------------------------------

def _wattr(wall, key, default=None):
    """Read `key` off a wall that may be a group_wall_runs dict OR an object
    (SimpleNamespace / future Wall type) -- both shapes flow through this
    module's callers, so accept either uniformly."""
    if isinstance(wall, dict):
        return wall.get(key, default)
    return getattr(wall, key, default)


# Feature kinds that ADD material as a stepped extrusion (each becomes a
# WallStep). "groove" is subtractive relief, "opening" is a through-cut --
# both handled separately below.
_ADDITIVE_KINDS = ("face", "l_extrusion", "column_attach", "beam_soffit")


def _perp_box(seg_wall, u_lo, u_hi, z_lo, z_hi, v_lo, v_hi) -> trimesh.Trimesh:
    """World-axis-aligned box in the (segment) wall's frame: u measured from
    seg_wall.p0 along its snapped world axis (same convention as `_step_box`),
    perpendicular ("v") spanning [v_lo, v_hi] along the world axis nearest the
    wall's normal (always its POSITIVE sign, per `_wall_world_axes`), z from
    z_lo to z_hi. Used to build groove-recess cutters."""
    u_is_x, u_sign = _wall_world_axes(seg_wall)
    p0 = np.asarray(seg_wall.p0, dtype=float)
    u_mid = (u_lo + u_hi) / 2.0
    v_mid = (v_lo + v_hi) / 2.0
    z_mid = (z_lo + z_hi) / 2.0
    if u_is_x:
        center = np.array([p0[0] + u_sign * u_mid, p0[1] + v_mid, z_mid])
        extents = (u_hi - u_lo, v_hi - v_lo, z_hi - z_lo)
    else:
        center = np.array([p0[0] + v_mid, p0[1] + u_sign * u_mid, z_mid])
        extents = (v_hi - v_lo, u_hi - u_lo, z_hi - z_lo)
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def segment_to_mesh(segment, wall) -> trimesh.Trimesh:
    """Build ONE watertight mesh for a room-owned WallSegment (Task 9 / R1).

    Reuses the SAME CSG primitives as the whole-wall builders: a stepped
    additive extrusion (`wall_to_solid`) for the segment's own u-range built
    from its clipped ADDITIVE features (face / l_extrusion / column_attach /
    beam_soffit -- each a WallStep at its own offset & z-range), then boolean
    subtraction of its "opening" features (through-cuts, via `cut_openings`)
    and "groove" features (front-face recesses, via `_perp_box`). Relief is
    thereby preserved per segment (the user's top priority): l_extrusions /
    column_attach protrude, beam_soffits leave a hollow gap below (they start
    at their elevated z_min, not the floor), grooves recess, openings punch
    through.

    `segment.features` are in WALL-LOCAL u (metres from wall.p0). A per-segment
    wall is built whose own p0/p1 are the segment's world endpoints, so the
    existing `_step_box` / `_opening_cutter_box` (both measuring u from p0)
    place everything correctly even for a segment that doesn't start at the
    parent wall's p0.
    """
    p0 = np.asarray(_wattr(wall, "p0"), dtype=float)
    p1 = np.asarray(_wattr(wall, "p1"), dtype=float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        raise ValueError("segment_to_mesh: wall.p0 and wall.p1 must be distinct")
    u_hat = d / length
    thickness = _wattr(wall, "thickness_m", None) or DEFAULT_WALL_THICKNESS_M
    floor_z = float(_wattr(wall, "floor_z_m", 0.0) or 0.0)

    seg_p0 = p0 + u_hat * segment.u_min_m
    seg_p1 = p0 + u_hat * segment.u_max_m
    seg_len = float(segment.u_max_m - segment.u_min_m)
    seg_wall = SimpleNamespace(
        p0=(float(seg_p0[0]), float(seg_p0[1])),
        p1=(float(seg_p1[0]), float(seg_p1[1])),
        thickness_m=float(thickness),
        floor_z_m=floor_z,
    )

    def _to_local(u):
        return float(u - segment.u_min_m)

    additive, grooves, openings = [], [], []
    for f in segment.features:
        lo = max(0.0, _to_local(f.u_min_m))
        hi = min(seg_len, _to_local(f.u_max_m))
        if hi - lo <= 1e-6:
            continue
        if f.kind in _ADDITIVE_KINDS:
            additive.append(WallStep(offset_m=f.offset_m, u_min_m=lo, u_max_m=hi,
                                     z_min_m=f.z_min_m, z_max_m=f.z_max_m))
        elif f.kind == "groove":
            grooves.append((lo, hi, f.z_min_m, f.z_max_m, abs(f.offset_m)))
        elif f.kind == "opening":
            openings.append(SimpleNamespace(
                opening_id=str(f.meta.get("opening_id", "opening")),
                u_min_m=lo, u_max_m=hi,
                sill_m=float(f.z_min_m - floor_z),
                height_m=float(f.z_max_m - f.z_min_m),
            ))

    if not additive:
        # Only subtractive features (or none) fell in this segment -- still give
        # it a plain full-height face so it stays a real watertight wall piece
        # (never silently dropped). Ceiling from the wall if it carries one,
        # else the tallest feature's top.
        z_top = _wattr(wall, "ceiling_z_m", None)
        if z_top is None:
            tops = [f.z_max_m for f in segment.features]
            z_top = max(tops) if tops else floor_z + DEFAULT_WALL_THICKNESS_M
        additive = [WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=seg_len,
                             z_min_m=floor_z, z_max_m=float(z_top))]

    solid = wall_to_solid(seg_wall, additive)

    if openings:
        solid = cut_openings(solid, openings, seg_wall)

    for (lo, hi, z_lo, z_hi, depth) in grooves:
        depth = min(depth, seg_wall.thickness_m * 0.95)
        # Recess the front (world-positive perp) face inward by `depth` over the
        # groove's u/z range. The front face sits at +thickness/2 (see
        # `_step_box`); overshoot outward so the cut fully clears that face.
        v_lo = seg_wall.thickness_m / 2.0 - depth
        v_hi = seg_wall.thickness_m / 2.0 + OPENING_OVERSHOOT_M
        cutter = _perp_box(seg_wall, lo, hi, z_lo, z_hi, v_lo, v_hi)
        solid = trimesh.boolean.difference([solid, cutter])
        _ensure_watertight(solid, "segment_to_mesh (groove)")

    return _ensure_watertight(solid, "segment_to_mesh")
