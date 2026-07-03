"""Room-owned wall model assembly (Task 9, under Architecture Revision R1).

R1 restructured the final meshing so that **walls own a feature graph and
rooms own wall segments** (superseding the earlier flat per-wall element dict,
which produced duplicate/fragmented panels where two rooms share a wall and
silently dropped walls no closed room claimed). This module implements that:

  1. `classify_wall_features(wall, z_floor, z_ceiling, openings=None)`
     turns a wall's relief (its `WallStep`s from structure.group_wall_runs)
     plus its detected openings into a list of typed `Feature`s
     ("face"/"groove"/"l_extrusion"/"column_attach"/"beam_soffit"/"opening").
     This classifier is the gap the plan left implicit -- no earlier task
     assigns `Feature.kind`; R1 says it belongs here, where features are first
     consumed.
  2. `split_wall_into_segments(wall, features, rooms, wall_id)` splits a wall
     ONCE at the union of the room-boundary u-coordinates that fall in its
     u-range, producing one room-owned `WallSegment` per stretch. A stretch no
     room's edge overlaps becomes ONE segment with `room_id=None` (never
     dropped -- the exact v4/v5 "missing wall" bug R1 fixes).
  3. `build_room_model(...)` assembles the final `{collection_name: [meshes]}`
     dict: one `Room_NN_<area>m2` collection per room (its wall-segment meshes
     + a floor panel), a `Walls_unassigned` collection, plus `Columns`/`Beams`.

Coordinate convention (SHARED with openings.py and floorplan2d.py, and asserted
in tests/recon/test_model.py): every `Feature`/`WallSegment` u-coordinate is
WALL-LOCAL -- metres from `wall.p0` along the p0->p1 direction, in [0, length].
`structure.group_wall_runs` emits `WallStep.u_min_m` in ABSOLUTE world-u (the
raw `pts @ u_vec` projection); `classify_wall_features` converts that to
wall-local u by subtracting p0's own projection, so a wall whose p0 sits away
from the world origin is still described in a clean [0, length] frame.
z_min_m/z_max_m stay absolute world Z (same frame as z_floor/z_ceiling).

Pure-Python relief classification (no trimesh) lives here so it stays
unit-testable anywhere; the actual meshing is delegated to
`solids.segment_to_mesh`.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from .schema import Feature, WallSegment, new_wall_id


# --- classification thresholds (validated in sharp_preview_v3.build_stretches
# and sharp_preview_v5_rooms; see those modules' docstrings) -----------------

# A face whose bottom sits no more than this above the floor is treated as
# floor-anchored and filled DOWN to the floor (furniture/scan occlusion, not a
# real elevated feature). Above this it is a candidate beam soffit.
OCCLUSION_FILL_M = 1.2
# A face whose top reaches within this of the ceiling is filled UP to the
# ceiling (a wall reaches the ceiling; scan noise leaves the top a little short).
CEIL_SNAP_M = 0.5
# |offset| at/below this counts as the reference (main) face; the main step
# comes back from group_wall_runs at exactly 0.0, so a tiny tolerance suffices.
FACE_OFFSET_TOL_M = 0.02
# A PROUD step narrower than this is a pilaster/pillar attach ("column_attach");
# wider is a longer "l_extrusion". Matches structure.extract_columns_beams'
# max_size_m column-scale threshold.
COLUMN_ATTACH_MAX_W_M = 0.6
# Discard a classified feature shorter than this in Z (scan noise sliver).
MIN_FEATURE_H_M = 0.15

# --- room-edge-to-wall matching (validated in sharp_preview_v5_rooms) -------

# Max perpendicular distance between a room-polygon edge and a wall for the
# edge to be considered "on" that wall.
EDGE_MATCH_M = 0.45
# Ignore room-polygon edges shorter than this (numeric slivers).
MIN_EDGE_M = 0.15
# A room edge must overlap the wall's u-range by at least this fraction of its
# own length to claim it.
MIN_OVERLAP_FRAC = 0.3
# Two split boundaries closer than this collapse into one (avoid slivers).
BOUNDARY_EPS_M = 0.02

# Floor-panel slab depth (m) extruded down under each room.
FLOOR_SLAB_M = 0.12


def _wget(wall, key, default=None):
    """Read `key` off a wall that may be a group_wall_runs dict OR an object."""
    if isinstance(wall, dict):
        return wall.get(key, default)
    return getattr(wall, key, default)


def _wall_frame(wall):
    """(p0, p1, u_hat, length, u_is_x): u_hat is the unit p0->p1 direction,
    u_is_x True when the wall RUNS along world X (so its perpendicular/normal
    axis is world Y). Derived from geometry, independent of any stored
    `direction` label (which structure.group_wall_runs defines as the NORMAL's
    axis -- the opposite one)."""
    p0 = np.asarray(_wget(wall, "p0"), dtype=float)
    p1 = np.asarray(_wget(wall, "p1"), dtype=float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        raise ValueError("wall.p0 and wall.p1 must be distinct points")
    u_hat = d / length
    u_is_x = abs(u_hat[0]) >= abs(u_hat[1])
    return p0, p1, u_hat, length, u_is_x


# ---------------------------------------------------------------------------
# 1. feature classification (the gap this task fills)
# ---------------------------------------------------------------------------

def _classify_step_kind(offset_m, width_m, floor_anchored, reaches_ceiling):
    """The core rule (kept tiny & documented, per the task): a step's kind from
    its offset relative to the reference face and its z placement."""
    if not floor_anchored and reaches_ceiling:
        return "beam_soffit"
    if not floor_anchored and not reaches_ceiling:
        return None  # elevated but not ceiling-hung -> noise (v3 drops these)
    # floor-anchored: distinguish by perpendicular offset from the main face
    if abs(offset_m) <= FACE_OFFSET_TOL_M:
        return "face"
    if offset_m > 0:  # proud of the face
        return "column_attach" if width_m <= COLUMN_ATTACH_MAX_W_M else "l_extrusion"
    return "groove"  # recessed behind the face


def _opening_bounds(op):
    """Normalize an opening (dict from openings.detect_openings, with u0/u1/
    z0/z1, OR a build_manifest-style u_min_m/z_min_m dict, OR an object) to
    (u_min, u_max, z_min, z_max)."""
    def g(*keys):
        for k in keys:
            if isinstance(op, dict):
                if k in op and op[k] is not None:
                    return op[k]
            elif getattr(op, k, None) is not None:
                return getattr(op, k)
        return None
    return (g("u0", "u_min_m"), g("u1", "u_max_m"),
            g("z0", "z_min_m", "sill_m"), g("z1", "z_max_m"))


def classify_wall_features(wall, z_floor: float, z_ceiling: float, openings=None):
    """Classify a wall's `WallStep`s (+ its openings) into typed `Feature`s.

    `wall` is a group_wall_runs run dict (or any object) carrying `p0`, `p1`
    and `steps` (list[WallStep], with u in ABSOLUTE world-u as group_wall_runs
    emits them). `openings` is an optional list of opening dicts/objects for
    THIS wall (openings.detect_openings' per-wall list), each becoming an
    "opening"-kind Feature at its own u/z range.

    Returns list[Feature] in WALL-LOCAL u (see module docstring). z-ranges:
    floor-anchored faces/relief are filled down to z_floor (and up to
    z_ceiling when they reach near it); beam soffits keep their elevated
    z_min (so a hollow gap below is preserved).
    """
    p0, _p1, u_hat, length, _u_is_x = _wall_frame(wall)
    p0_u = float(np.dot(p0, u_hat))  # p0's own absolute-u -> local u = abs - p0_u

    feats = []
    for step in _wget(wall, "steps", []) or []:
        u0 = float(step.u_min_m) - p0_u
        u1 = float(step.u_max_m) - p0_u
        # clamp into the wall's own [0, length] (guards tiny numeric spill)
        u0 = max(0.0, min(u0, length))
        u1 = max(0.0, min(u1, length))
        if u1 - u0 <= 1e-6:
            continue

        z_min = float(step.z_min_m)
        z_max = float(step.z_max_m)
        floor_anchored = (z_min - z_floor) <= OCCLUSION_FILL_M
        reaches_ceiling = (z_ceiling - z_max) <= CEIL_SNAP_M

        kind = _classify_step_kind(step.offset_m, u1 - u0, floor_anchored, reaches_ceiling)
        if kind is None:
            continue

        if kind == "beam_soffit":
            fz0, fz1 = z_min, z_ceiling
        else:
            fz0 = z_floor  # occlusion-fill down
            fz1 = z_ceiling if reaches_ceiling else z_max
        if fz1 - fz0 < MIN_FEATURE_H_M:
            continue

        feats.append(Feature(kind=kind, u_min_m=u0, u_max_m=u1,
                             z_min_m=fz0, z_max_m=fz1, offset_m=float(step.offset_m)))

    for op in (openings or []):
        u0, u1, z0, z1 = _opening_bounds(op)
        if None in (u0, u1, z0, z1):
            continue
        meta = {}
        for k in ("type", "walked", "confidence", "opening_id"):
            v = op.get(k) if isinstance(op, dict) else getattr(op, k, None)
            if v is not None:
                meta[k] = v
        feats.append(Feature(kind="opening", u_min_m=float(u0), u_max_m=float(u1),
                             z_min_m=float(z0), z_max_m=float(z1), offset_m=0.0, meta=meta))

    return feats


# ---------------------------------------------------------------------------
# 2. room ownership split
# ---------------------------------------------------------------------------

def _normalize_rooms(rooms):
    """Accept rooms as list of shapely Polygons, coord-lists, or (id, poly)
    tuples. Return list of (room_id, ShapelyPolygon)."""
    out = []
    for i, r in enumerate(rooms or []):
        if (isinstance(r, tuple) and len(r) == 2
                and (hasattr(r[1], "exterior") or isinstance(r[0], str))):
            rid, poly = r
        else:
            rid, poly = None, r
        if not hasattr(poly, "exterior"):
            poly = ShapelyPolygon(poly)
        if rid is None:
            rid = f"room_{i:02d}"
        out.append((rid, poly))
    return out


def _matching_edge_intervals(wall, rooms):
    """For each room edge that lies ON this wall (parallel, within EDGE_MATCH_M
    perpendicular, overlapping the wall's u-range), return
    (room_id, local_lo, local_hi, perp_dist) with u in wall-local coords.

    Mirrors sharp_preview_v5_rooms' room-edge-to-wall matching (direction /
    perpendicular-offset / u-overlap), inverted to run per-wall."""
    p0, p1, u_hat, length, u_is_x = _wall_frame(wall)
    p0_u = float(np.dot(p0, u_hat))
    wall_perp = float(p0[1]) if u_is_x else float(p0[0])
    wall_u_lo, wall_u_hi = p0_u, p0_u + length

    intervals = []
    for rid, poly in rooms:
        coords = list(poly.exterior.coords)
        for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
            ex, ey = xb - xa, yb - ya
            elen = float(np.hypot(ex, ey))
            if elen < MIN_EDGE_M:
                continue
            edge_is_x = abs(ex) >= abs(ey)  # edge runs along world X
            if edge_is_x != u_is_x:
                continue  # not parallel to this wall
            e_perp = (ya + yb) / 2.0 if edge_is_x else (xa + xb) / 2.0
            if abs(wall_perp - e_perp) > EDGE_MATCH_M:
                continue
            e_lo = min(xa, xb) if edge_is_x else min(ya, yb)
            e_hi = max(xa, xb) if edge_is_x else max(ya, yb)
            overlap = min(e_hi, wall_u_hi) - max(e_lo, wall_u_lo)
            if overlap < MIN_OVERLAP_FRAC * elen:
                continue
            lo = max(e_lo, wall_u_lo) - p0_u
            hi = min(e_hi, wall_u_hi) - p0_u
            lo = max(0.0, min(lo, length))
            hi = max(0.0, min(hi, length))
            if hi - lo <= BOUNDARY_EPS_M:
                continue
            intervals.append((rid, lo, hi, abs(wall_perp - e_perp)))
    return intervals


def _clip_features(features, u0, u1):
    clipped = []
    for f in features:
        lo = max(f.u_min_m, u0)
        hi = min(f.u_max_m, u1)
        if hi - lo <= 1e-6:
            continue
        clipped.append(Feature(kind=f.kind, u_min_m=lo, u_max_m=hi,
                              z_min_m=f.z_min_m, z_max_m=f.z_max_m,
                              offset_m=f.offset_m, meta=dict(f.meta)))
    return clipped


def split_wall_into_segments(wall, features, rooms, wall_id=None):
    """Split a wall into room-owned `WallSegment`s at the union of the
    room-boundary u-coordinates falling inside its own u-range.

    Every stretch is owned by the room whose matching edge is nearest in
    perpendicular distance (so a wall SHARED by two rooms on opposite faces is
    emitted ONCE, not duplicated); a stretch no room edge covers becomes a
    single `room_id=None` segment. Features are clipped to each segment's
    [u_min_m, u_max_m]. Returns list[WallSegment].
    """
    _p0, _p1, _u_hat, length, _u_is_x = _wall_frame(wall)
    wall_id = wall_id if wall_id is not None else str(_wget(wall, "wall_id", "wall"))
    rooms = _normalize_rooms(rooms)
    intervals = _matching_edge_intervals(wall, rooms)

    # boundaries: wall ends + every matching interval endpoint
    bounds = {0.0, length}
    for (_rid, lo, hi, _d) in intervals:
        bounds.add(lo)
        bounds.add(hi)
    bounds = sorted(bounds)
    # dedupe near-coincident boundaries
    merged_bounds = [bounds[0]]
    for b in bounds[1:]:
        if b - merged_bounds[-1] > BOUNDARY_EPS_M:
            merged_bounds.append(b)
    bounds = merged_bounds

    def _owner(mid):
        best = None
        for (rid, lo, hi, dist) in intervals:
            if lo <= mid <= hi and (best is None or dist < best[1]):
                best = (rid, dist)
        return best[0] if best else None

    # owner per sub-interval, then merge adjacent same-owner runs
    raw = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= BOUNDARY_EPS_M:
            continue
        raw.append((a, b, _owner((a + b) / 2.0)))

    segments = []
    for a, b, owner in raw:
        if segments and segments[-1][2] == owner:
            segments[-1] = (segments[-1][0], b, owner)
        else:
            segments.append((a, b, owner))

    out = []
    for a, b, owner in segments:
        out.append(WallSegment(wall_id=wall_id, room_id=owner, u_min_m=a, u_max_m=b,
                              features=_clip_features(features, a, b)))
    return out


# ---------------------------------------------------------------------------
# 3. final assembly
# ---------------------------------------------------------------------------

def _room_collection_name(index: int, poly) -> str:
    """Room_NN_<area>m2, matching sharp_preview_v5_rooms' naming."""
    return f"Room_{index:02d}_{poly.area:.0f}m2"


class RoomModel(dict):
    """dict[str, list[trimesh.Trimesh]] subclass returned by `build_room_model`.

    Behaves exactly like the plain dict callers already expect (iteration,
    `in`, `.values()`, item access -- see tests/recon/test_model.py), but also
    carries `.drops`: a dict of failure-category -> list[dict] records for
    every piece of geometry that failed to mesh and was excluded from the
    model. This exists so a per-segment/panel/column/beam meshing failure is
    COUNTED and identifiable (never silently swallowed), matching the
    pipeline's "never silently drop a wall" guarantee -- see
    scripts/isolidarflow.py's `degenerate_walls_dropped` for the sibling
    pattern this mirrors at the wall-regularization stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drops = {"segments": [], "floor_panels": [], "columns": [], "beams": []}


def build_room_model(walls, openings_by_wall, columns, beams, rooms,
                     z_floor: float, z_ceiling: float, slab_m: float = FLOOR_SLAB_M):
    """Assemble the final room-owned model.

    Returns a `RoomModel` (dict[str, list[trimesh.Trimesh]] subclass) with one
    `Room_NN_<area>m2` collection per room (its wall-segment meshes + a floor
    panel), a `Walls_unassigned` collection (segments no room claimed), and
    `Columns` / `Beams` collections. Every mesh in the model is watertight.
    The returned object's `.drops` attribute records anything that failed to
    mesh and was therefore excluded (see `RoomModel` docstring) -- callers
    that only need the plain dict behaviour need not change anything.

    Design decisions (flagged; the plan left these to this task):
      - Floor is a per-room panel (matches R1's "floor panel per room" and
        sharp_preview_v5_rooms), NOT a single whole-footprint slab -- keeps
        everything room-owned and avoids the shared-geometry duplication R1
        set out to remove. Ceiling is omitted at this layer (add analogously if
        wanted); the user's priority is wall relief, and a per-room ceiling
        panel would just mirror the floor.
      - `openings_by_wall` is keyed by wall index (openings.detect_openings'
        shape).
    """
    import trimesh
    from .solids import segment_to_mesh, column_to_solid, beam_to_solid

    # Assign display collection names (Room_NN_<area>m2) as room ids so each
    # segment carries the collection it belongs to.
    named_rooms = [(_room_collection_name(i, poly), poly)
                   for i, (_rid, poly) in enumerate(_normalize_rooms(rooms))]

    model = RoomModel()

    def _add(collection, mesh):
        model.setdefault(collection, []).append(mesh)

    openings_by_wall = openings_by_wall or {}
    for wi, wall in enumerate(walls):
        wall_id = str(_wget(wall, "wall_id", None) or new_wall_id(wi))
        feats = classify_wall_features(wall, z_floor, z_ceiling,
                                       openings=openings_by_wall.get(wi))
        segments = split_wall_into_segments(wall, feats, named_rooms, wall_id=wall_id)
        for seg in segments:
            try:
                mesh = segment_to_mesh(seg, wall)
            except Exception as exc:
                # A single bad segment must not sink the whole model -- but it
                # must be COUNTED, not silently absorbed.
                model.drops["segments"].append({
                    "wall_id": wall_id, "room_id": seg.room_id,
                    "u_min_m": seg.u_min_m, "u_max_m": seg.u_max_m,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            collection = seg.room_id if seg.room_id is not None else "Walls_unassigned"
            _add(collection, mesh)

    # per-room floor panel
    for name, poly in named_rooms:
        if name not in model:
            model[name] = []  # keep the room collection even if it got no walls
        try:
            panel = trimesh.creation.extrude_polygon(poly, slab_m)
            panel.apply_translation([0.0, 0.0, z_floor - slab_m])
            model[name].append(panel)
        except Exception as exc:
            model.drops["floor_panels"].append({
                "room": name, "error": f"{type(exc).__name__}: {exc}",
            })

    for i, c in enumerate(columns or []):
        try:
            _add("Columns", column_to_solid(c))
        except Exception as exc:
            cid = str(_wget(c, "column_id", None) or f"column_{i:03d}")
            model.drops["columns"].append({
                "column_id": cid, "error": f"{type(exc).__name__}: {exc}",
            })
    for i, b in enumerate(beams or []):
        try:
            _add("Beams", beam_to_solid(b))
        except Exception as exc:
            bid = str(_wget(b, "beam_id", None) or f"beam_{i:03d}")
            model.drops["beams"].append({
                "beam_id": bid, "error": f"{type(exc).__name__}: {exc}",
            })

    return model
