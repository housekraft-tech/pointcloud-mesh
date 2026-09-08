"""Make planar polygons safe for the native SketchUp face builder.

SketchUp merges vertices closer than about 0.025 mm and rejects loops that
repeat a vertex or touch another loop at a point. Grid-cell unions and strip
extensions produce exactly those: pinch points and hairline slivers. Cleaning
is a morphological opening by 0.15 mm followed by a 0.03 mm simplification,
so no boundary moves by more than 0.15 mm (at convex corners only), far below
every evidence bound used in this flow.

Sanitized parts also carry ``planar_loops``: the exact cleaned face rings in
3D. The native exporter uses them directly instead of re-deriving loops from
a triangulation, which reintroduces degenerate slivers.
"""
import numpy as np
import shapely
import trimesh
from shapely.geometry.polygon import orient

from filter_soulace_overlap import polygon_parts, mesh_from_polygon

EPS = 1.5e-4
INSET = 4e-5          # dilation is this much smaller than the erosion
SIMPLIFY = 1e-5
SEPARATION = 3e-5     # minimum distance between vertices of different loops
MIN_PART_M2 = 1e-7


def loops_separated(geometry):
    """No two loop vertices (other than ring neighbours) closer than SEPARATION."""
    from scipy.spatial import cKDTree
    for polygon in polygon_parts(geometry):
        rings = [np.asarray(r.coords)[:-1] for r in [polygon.exterior, *polygon.interiors]]
        if any(len(r) < 3 for r in rings):
            return False
        xy = np.vstack(rings)
        ring_id = np.concatenate([np.full(len(r), i) for i, r in enumerate(rings)])
        position = np.concatenate([np.arange(len(r)) for r in rings])
        sizes = np.array([len(r) for r in rings])
        tree = cKDTree(xy)
        for i, j in tree.query_pairs(SEPARATION):
            if ring_id[i] != ring_id[j]:
                return False
            n = sizes[ring_id[i]]
            if (position[i] - position[j]) % n not in (1, n - 1):
                return False
    return True


def clean_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    base = geometry.buffer(0).simplify(SIMPLIFY, preserve_topology=True)
    cleaned = base
    for scale in (1, 2, 4):
        eps = EPS * scale
        # Erode by eps and dilate by slightly less: pieces that separated stay
        # apart by at least 2*INSET, so no loop touches or repeats a vertex.
        cleaned = base.buffer(-eps, join_style='mitre').buffer(eps - INSET, quad_segs=1).buffer(0)
        cleaned = cleaned.simplify(SIMPLIFY, preserve_topology=True).buffer(0)
        parts = [p for p in polygon_parts(cleaned) if p.area >= MIN_PART_M2]
        cleaned = shapely.union_all(parts) if parts else shapely.Polygon()
        if cleaned.is_empty or loops_separated(cleaned):
            return cleaned
    return cleaned


def loops_3d(region, origin, u, v):
    """Face loops (exterior first, counter-clockwise in the u,v basis) in 3D."""
    faces = []
    for polygon in polygon_parts(region):
        polygon = orient(polygon, sign=1.0)
        rings = []
        for ring in [polygon.exterior, *polygon.interiors]:
            xy = np.asarray(ring.coords)[:-1]
            if len(xy) < 3:
                continue
            rings.append((origin + xy[:, :1] * u + xy[:, 1:] * v).tolist())
        if rings:
            faces.append(rings)
    return faces


def _finish(pieces):
    out = trimesh.util.concatenate(pieces)
    out.merge_vertices(digits_vertex=8)
    out.update_faces(out.nondegenerate_faces(height=1e-8) & (out.area_faces >= 1e-12))
    out.remove_unreferenced_vertices()
    return out


def sanitize_surface_part(part):
    """Re-mesh an open planar surface part from cleaned per-plane polygons."""
    from architectural_surface_refinement import _all_planar_patches
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    pieces, loops = [], []
    for poly, origin, u, v in _all_planar_patches(mesh):
        region = clean_polygon(poly)
        if region.is_empty:
            continue
        piece = mesh_from_polygon(region, origin, u, v)
        if piece is not None:
            pieces.append(piece); loops.extend(loops_3d(region, origin, u, v))
    if not pieces:
        return None
    out = _finish(pieces)
    return {**part, 'v': out.vertices.tolist(), 'f': out.faces.tolist(), 'planar_loops': loops}


def sanitize_solid_floor(part):
    """Re-extrude a floor solid from its cleaned top polygon; datums unchanged."""
    from architectural_surface_refinement import floor_top, _all_planar_patches
    result = floor_top(part)
    if result is None:
        return None
    top, z, bottom = result
    region = clean_polygon(top)
    if region.is_empty or z - bottom <= 0:
        return None
    pieces = []
    for poly in polygon_parts(region):
        piece = trimesh.creation.extrude_polygon(poly, height=z - bottom, engine='earcut')
        piece.vertices[:, 2] += bottom
        pieces.append(piece)
    out = _finish(pieces); out.fix_normals()
    ex, ey, ez = np.eye(3)
    loops = loops_3d(region, np.array([0, 0, z]), ex, ey)
    loops += [[ring[::-1] for ring in face] for face in loops_3d(region, np.array([0, 0, bottom]), ex, ey)]
    for poly, origin, u, v in _all_planar_patches(out):
        if abs(float(np.cross(u, v) @ ez)) > .5:
            continue  # top and bottom already listed from the exact polygon
        loops.extend(loops_3d(poly.buffer(0), origin, u, v))
    return {**part, 'v': out.vertices.tolist(), 'f': out.faces.tolist(), 'planar_loops': loops}
