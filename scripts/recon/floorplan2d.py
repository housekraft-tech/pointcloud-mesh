"""2D arrangement: wall centerlines -> room polygons -> DXF/SVG floor plans.

Consumes scripts/floorplan_schema.py's Wall/Opening dataclasses (this
pipeline's established shared type for wall geometry) and produces:

- `build_room_polygons`: closed room footprints from the wall centerline
  arrangement, via shapely's polygonize.
- `write_dxf` / `write_svg`: CAD/vector floor plan exports with walls,
  openings, and rooms on separate layers/groups, following this repo's
  existing floorplan rendering convention (see
  scripts/floorplan_geometry.py's render_floorplan_image: dark wall lines,
  openings highlighted, "X.XXm/XXXmm" length labels).
"""
from __future__ import annotations

import numpy as np
import ezdxf
import svgwrite
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize, unary_union

LAYER_NAMES = ("WALLS", "OPENINGS", "ROOMS", "DIMS")


# ---------- shared wall/opening geometry helpers ----------

def _wall_frame(wall):
    """Return (p0, p1, unit_dir, unit_normal) as (2,) float arrays for a wall's
    centerline. unit_normal is the left-hand perpendicular of unit_dir, used
    to inflate the centerline out to the wall's two long edges."""
    p0 = np.asarray(wall.p0, dtype=float)
    p1 = np.asarray(wall.p1, dtype=float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length == 0:
        return p0, p1, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    u = d / length
    n = np.array([-u[1], u[0]])
    return p0, p1, u, n


def _wall_corners(wall):
    """Four corner points (world XY, wound consistently) of the wall's
    rectangular footprint: its centerline inflated by +-half thickness."""
    p0, p1, u, n = _wall_frame(wall)
    half_t = wall.thickness_m / 2.0
    return [
        p0 + n * half_t,
        p1 + n * half_t,
        p1 - n * half_t,
        p0 - n * half_t,
    ]


def _opening_corners(wall, opening):
    """Four corner points (world XY) of an opening marker rectangle, placed
    along its wall's u-axis from p0 (per floorplan_schema.py's Opening:
    u_min_m/u_max_m are relative to the wall's own u-axis from p0), inflated
    by +-half the wall's thickness for visual clarity."""
    p0, p1, u, n = _wall_frame(wall)
    op_p0 = p0 + u * opening.u_min_m
    op_p1 = p0 + u * opening.u_max_m
    half_t = wall.thickness_m / 2.0
    return [
        op_p0 + n * half_t,
        op_p1 + n * half_t,
        op_p1 - n * half_t,
        op_p0 - n * half_t,
    ]


def _wall_length_label(wall) -> str:
    # Matches scripts/floorplan_geometry.py's render_floorplan_image label
    # convention: "X.XXm/XXXmm" (length in metres / thickness in mm).
    return f"{wall.length_m:.2f}m/{wall.thickness_m * 1000:.0f}mm"


# ---------- room polygons ----------

def _extended_centerline(wall, epsilon_m: float) -> LineString:
    """The wall's centerline, extrapolated by epsilon_m past each endpoint.

    Two walls meeting at a corner should share an exact endpoint, but small
    upstream numerical slop can leave them just short of touching -- which
    would leave shapely.ops.polygonize unable to find a closed ring at all.
    A small linear extension past each endpoint closes that kind of gap
    without materially changing wall geometry.
    """
    p0, p1, u, _n = _wall_frame(wall)
    return LineString([tuple(p0 - u * epsilon_m), tuple(p1 + u * epsilon_m)])


def build_room_polygons(walls, epsilon_m: float = 0.05) -> list:
    """Polygonize wall centerlines into closed room polygons.

    Uses each wall's CENTERLINE (not interior face) as the polygonization
    boundary: the returned polygons' areas equal the footprint enclosed by
    the centerline loop. For a rectangular exterior wall loop with
    centerlines forming an AxB metre rectangle, the returned polygon's area
    is ~A*B -- NOT the smaller interior clear-floor area (centerline area
    minus each wall's own half-thickness eaten into the room). A caller
    wanting interior/clear floor area would need to buffer/offset the
    polygon inward by the surrounding walls' half-thickness itself; that is
    out of scope here.

    Returns an empty list if the walls don't form a closed loop (no
    hallucinated rooms from an open boundary), and filters out any
    degenerate (near-zero-area) polygons.
    """
    if not walls:
        return []
    lines = [_extended_centerline(w, epsilon_m) for w in walls]
    # polygonize requires fully-noded linework: unary_union splits every
    # line at each crossing/touching point first, otherwise crossing (but
    # un-noded) LineStrings don't produce any closed ring at all.
    merged = unary_union(MultiLineString(lines))
    polygons = list(polygonize(merged))
    return [p for p in polygons if p.area > 1e-6]


# ---------- DXF export ----------

def write_dxf(walls, openings, rooms, path) -> str:
    """Write walls/openings/rooms to a DXF file with WALLS/OPENINGS/ROOMS/DIMS
    layers. Walls are drawn as closed rectangles (centerline +- half
    thickness) on WALLS, with a length/thickness text label on DIMS per
    wall. Openings are drawn as closed rectangles at their wall-relative u
    position on OPENINGS. Rooms are drawn as closed polylines on ROOMS.
    Returns `path`.
    """
    doc = ezdxf.new()
    for name in LAYER_NAMES:
        doc.layers.new(name=name)
    msp = doc.modelspace()

    walls_by_id = {w.wall_id: w for w in walls}

    for wall in walls:
        corners = [tuple(c) for c in _wall_corners(wall)]
        msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": "WALLS"})

        mid = tuple((np.asarray(wall.p0, dtype=float) + np.asarray(wall.p1, dtype=float)) / 2.0)
        text = msp.add_text(
            _wall_length_label(wall),
            dxfattribs={"layer": "DIMS", "height": 0.1},
        )
        text.set_placement(mid)

    for opening in openings:
        wall = walls_by_id.get(opening.wall_id)
        if wall is None:
            # An opening referencing a wall not passed in `walls` can't be
            # placed -- skip rather than crash on a lookup miss.
            continue
        corners = [tuple(c) for c in _opening_corners(wall, opening)]
        msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": "OPENINGS"})

    for room in rooms:
        coords = [(float(x), float(y)) for x, y in room.exterior.coords]
        msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": "ROOMS"})

    doc.saveas(str(path))
    return str(path)


# ---------- SVG export ----------

def write_svg(walls, openings, rooms, path, px_per_meter: float = 100.0) -> str:
    """Write walls/openings/rooms to an SVG floor plan, roughly matching
    scripts/floorplan_geometry.py's render_floorplan_image visual style:
    walls as thick dark shapes, openings highlighted in green, rooms as
    filled/outlined polygons. Returns `path`.
    """
    if not walls:
        dwg = svgwrite.Drawing(str(path), size=("400px", "200px"), profile="full")
        dwg.save()
        return str(path)

    all_pts = np.vstack([np.vstack([w.p0, w.p1]) for w in walls])
    margin_m = 0.5
    min_xy = all_pts.min(axis=0) - margin_m
    max_xy = all_pts.max(axis=0) + margin_m
    width_px = float((max_xy[0] - min_xy[0]) * px_per_meter)
    height_px = float((max_xy[1] - min_xy[1]) * px_per_meter)

    def to_px(pt):
        x = (pt[0] - min_xy[0]) * px_per_meter
        y = (max_xy[1] - pt[1]) * px_per_meter  # flip Y for image/SVG coords
        return (float(x), float(y))

    dwg = svgwrite.Drawing(
        str(path), size=(f"{width_px:.0f}px", f"{height_px:.0f}px"), profile="full"
    )

    room_group = dwg.add(dwg.g(id="rooms"))
    for room in rooms:
        pts = [to_px(c) for c in room.exterior.coords]
        room_group.add(
            dwg.polygon(points=pts, fill="#dce6f2", stroke="#8fa8c4", stroke_width=1, fill_opacity=0.6)
        )

    wall_group = dwg.add(dwg.g(id="walls"))
    label_group = dwg.add(dwg.g(id="labels"))
    for wall in walls:
        corners = [to_px(c) for c in _wall_corners(wall)]
        wall_group.add(dwg.polygon(points=corners, fill="#282828", stroke="#282828"))
        mid = to_px((np.asarray(wall.p0, dtype=float) + np.asarray(wall.p1, dtype=float)) / 2.0)
        label_group.add(
            dwg.text(_wall_length_label(wall), insert=mid, fill="#c80000", font_size="10px")
        )

    opening_group = dwg.add(dwg.g(id="openings"))
    walls_by_id = {w.wall_id: w for w in walls}
    for opening in openings:
        wall = walls_by_id.get(opening.wall_id)
        if wall is None:
            continue
        corners = [to_px(c) for c in _opening_corners(wall, opening)]
        opening_group.add(dwg.polygon(points=corners, fill="#00aa00", stroke="#006600"))

    dwg.save()
    return str(path)
