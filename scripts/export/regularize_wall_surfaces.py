"""Regularize existing wall planes, preserving rectangular openings.

Islands of one plane are bridged across gaps up to 0.6 m only where the raw
returns carry the surface within 50 mm (measured surface the model dropped).

Completion is local to an existing plane. The outline moves at most 50 mm
from the existing surface. A hole that is completely surrounded by one
measured plane, is not rectangular and is no larger than a scan shadow
(``max_gap_m2``) is a scan gap and is filled as inferred continuity; a
rectangular hole is a door, window, vent or service opening and stays open.
Filled area is inferred surface continuity, not measured area. No new wall
run, opposite face or thickness is generated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from wall_surface_cleanup import planes_of


def is_opening(hole, min_side=.04, rectangularity=.85):
    """Rectangular holes wider than a vent slot are real openings (a circle scores 0.785)."""
    width, height = np.subtract(hole.bounds[2:], hole.bounds[:2])
    return hole.area / max(width * height, 1e-12) >= rectangularity and min(width, height) > min_side


def grid_samples(region, pitch):
    u0, v0, u1, v1 = region.bounds
    us = np.arange(u0 + pitch / 2, u1, pitch); vs = np.arange(v0 + pitch / 2, v1, pitch)
    if not len(us) or not len(vs):
        return np.zeros((0, 2))
    uv = np.stack(np.meshgrid(us, vs, indexing='ij'), -1).reshape(-1, 2)
    return uv[shapely.contains_xy(region, uv[:, 0], uv[:, 1])]


def bridge_islands(region, support, close_m=.3, pitch=.025):
    """Join islands of one plane across gaps where the raw returns carry the surface.

    A plane fragmented by cleanup thresholds is closed at ``close_m`` (gaps up
    to twice that), but every bridged cell must have a raw return within 50 mm
    of the plane: the bridge is measured surface the model had dropped, never
    an invented one. Without a support function nothing wider than the 50 mm
    crack repair is bridged.
    """
    if support is None:
        return region, 0.
    closed = region.buffer(close_m, join_style=2).buffer(-close_m, join_style=2)
    bridged = closed.difference(region)
    if bridged.is_empty or bridged.area < 1e-6:
        return region, 0.
    uv = grid_samples(bridged, pitch)
    if not len(uv):
        return region, 0.
    ok = support(uv)
    if not ok.any():
        return region, 0.
    half = pitch * .55
    cells = shapely.union_all(shapely.box(uv[ok, 0] - half, uv[ok, 1] - half, uv[ok, 0] + half, uv[ok, 1] + half))
    supported = cells.intersection(bridged)
    return shapely.union_all([region, supported]), float(supported.area)


def regularize(region, reach=.05, max_gap_m2=1., max_gap_m=1.5, support=None):
    region, bridged_m2 = bridge_islands(region, support)
    holes = [shapely.Polygon(r) for p in polygon_parts(region) for r in p.interiors]
    protected, gaps = [], []
    for hole in holes:
        width, height = np.subtract(hole.bounds[2:], hole.bounds[:2])
        if hole.area <= .04 and max(width, height) <= .3 and not is_opening(hole):
            gaps.append(hole)          # speckle, always inferred continuity
        elif is_opening(hole) or hole.area > max_gap_m2 or max(width, height) > max_gap_m:
            protected.append(hole)
        else:
            gaps.append(hole)          # blobby scan shadow inside one plane
    # Repair narrow cracks, then remove fine boundary teeth. Plane coordinates
    # do not move; only the outline changes within the local repair envelope.
    cleaned = region.buffer(reach / 2, join_style=2).buffer(-reach / 2, join_style=2)
    cleaned = cleaned.buffer(-.0075, join_style=2).buffer(.0075, join_style=2)
    cleaned = cleaned.simplify(.015, preserve_topology=True)
    filled = []
    for poly in polygon_parts(cleaned):
        inner = [r for r in poly.interiors if shapely.Polygon(r).area > .04]
        filled.append(shapely.Polygon(poly.exterior, inner))
    cleaned = shapely.union_all(filled).intersection(region.buffer(reach, join_style=2))
    if gaps:
        cleaned = shapely.union_all([cleaned, *gaps])
    if protected:
        cleaned = cleaned.difference(shapely.union_all(protected))
    # Do not turn a narrow actual wall strip into an empty result.
    if cleaned.is_empty or cleaned.area < .9 * region.area:
        return region, {'accepted': False}
    return cleaned, {'accepted': True, 'added_area_m2': cleaned.difference(region).area + bridged_m2,
                     'removed_area_m2': region.difference(cleaned).area,
                     'bridged_measured_area_m2': bridged_m2,
                     'filled_gap_area_m2': float(sum(g.area for g in gaps)),
                     'protected_openings': len(protected)}


def regularize_within(region, mask, reach=.05, margin=.3, support=None):
    """Regularize only the part of the plane inside ``mask`` (partly exterior faces)."""
    zone = mask.buffer(margin, join_style=2)
    inside = region.intersection(zone)
    proposed, row = regularize(inside, reach=reach, support=support)
    if not row['accepted']:
        return region, row
    return shapely.union_all([proposed, region.difference(zone)]), row


def load_selection(path):
    """Selected planes per wall.

    Accepts a plain list of wall names (every vertical plane of each wall,
    value None) or the plane-level form written by select_exterior_walls.py,
    ``{"walls": {name: [{"key", "mask_uv"}, ...]}}`` (value: plane key ->
    in-plane mask polygon or None for the whole plane), so the interior face
    of an exterior wall is left exactly as measured.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {name: None for name in data}
    selection = {}
    for name, planes in data['walls'].items():
        chosen = {}
        for entry in planes:
            key = tuple(np.round(entry['key'] if isinstance(entry, dict) else entry, 5))
            mask = entry.get('mask_uv') if isinstance(entry, dict) else None
            chosen[key] = shapely.union_all([shapely.Polygon(ring) for ring in mask]) if mask else None
        selection[name] = chosen
    return selection


def plane_selection(selected_planes, plane):
    """(selected, mask) for one plane; the mask is None for a whole-plane selection."""
    if selected_planes is None:
        return True, None
    key = tuple(np.round(plane['key'], 5))
    return key in selected_planes, selected_planes.get(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--selection', help='JSON list of wall names, or {"walls": {name: [plane keys]}} from select_exterior_walls.py')
    parser.add_argument('--evidence-cache', help='raw_evidence.py npz; enables bridging of island gaps where the returns support it')
    parser.add_argument('--support-m', type=float, default=.05)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    selected = load_selection(args.selection) if args.selection else None
    tree = None
    if args.evidence_cache:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.load(args.evidence_cache, allow_pickle=True)['p'], leafsize=64, compact_nodes=False, balanced_tree=False)

    def supporter(plane):
        if tree is None:
            return None

        def support(uv):
            xyz = plane['origin'] + uv[:, :1] * plane['u'] + uv[:, 1:] * plane['v']
            return tree.query(xyz, distance_upper_bound=args.support_m, workers=-1)[0] <= args.support_m
        return support

    parts, report = [], []
    for part in json.loads(Path(args.model).read_text())['parts']:
        if selected is not None and part['name'] not in selected:
            continue
        if not part['kind'].startswith(('wall', 'parapet')):
            continue
        planes = list(planes_of(part))
        if any(p['snap'] > .0005 for p in planes):
            continue
        selected_planes = selected[part['name']] if selected is not None else None
        meshes, rows = [], []
        for plane in planes:
            poly = plane['poly']
            chosen, mask = plane_selection(selected_planes, plane)
            if chosen and abs(plane['normal'][2]) < .01 and poly.area > .1:
                support = supporter(plane)
                proposed, row = regularize(poly, support=support) if mask is None else regularize_within(poly, mask, support=support)
            else:
                proposed, row = poly, {'accepted': False}
            row['selected'] = bool(chosen)
            row['partial_mask'] = mask is not None
            row['plane_key'] = [float(x) for x in plane['key']]
            if not proposed.is_empty:
                meshes.append(mesh_from_polygon(proposed, plane['origin'], plane['u'], plane['v']))
            rows.append(row)
        if not any(r['accepted'] for r in rows) or not meshes:
            continue
        mesh = trimesh.util.concatenate(meshes)
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
        replacement = {**part, 'name': part['name'] + ' - smooth wall plane',
                       'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                       'source_group_names': [part['name']], 'merge_coplanar_faces': True,
                       'evidence_status': 'existing_wall_plane_with_local_inferred_gap_completion_50mm',
                       'completion_is_measured': False}
        replacement.pop('planar_loops', None)
        parts.append(replacement)
        report.append({'wall': part['name'], 'planes': rows})
        print(f'Regularized {part["name"]}', flush=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'walls': len(parts), 'added_area_m2': sum(r.get('added_area_m2', 0) for p in report for r in p['planes']),
                      'removed_area_m2': sum(r.get('removed_area_m2', 0) for p in report for r in p['planes'])}))


if __name__ == '__main__':
    main()
