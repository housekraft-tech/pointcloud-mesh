"""Force the exterior to rectangular blocks: only axis-aligned edges, coplanar planes merged.

Declared inference for a drawing-clean exterior. For every exterior plane
(and every recessed face):

1. pieces under ``min_piece_m2`` are dropped;
2. each outer ring is snapped to axis-aligned edges with a rising tolerance
   (50, 100, 200, 350 mm); a ring that still cannot be snapped becomes the
   bounding rectangle of its piece;
3. holes are kept only as rectangles: rectangular openings keep their
   straightened rectangle, larger irregular holes (recess voids) become their
   bounding rectangle, small holes are filled;
4. planes of different parts that are parallel within 1 degree, within
   ``merge_gap_m`` of each other and on the same level are merged into one
   block on the mean plane, so one wall run is one face.

Non-exterior planes of a merged part (its interior face, top, reveals) travel
unchanged with the block, because replacing a native group hides all of it.
Every block reports the outline displacement it needed; the evidence status
ends in INFERRED.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from regularize_wall_surfaces import is_opening, load_selection, orthogonalize_ring, plane_selection
from wall_surface_cleanup import planes_of

TOLERANCES = (.05, .1, .2, .35)


def is_axis_aligned(poly, tol=1e-6):
    for ring in [poly.exterior, *poly.interiors]:
        d = np.diff(np.asarray(ring.coords), axis=0)
        if not np.all((np.abs(d[:, 0]) < tol) | (np.abs(d[:, 1]) < tol)):
            return False
    return True


def square_ring(coords, fallback_box):
    """Axis-aligned ring or its bounding box; the snapper may keep diagonal
    corners where edges are steep, so alignment is checked, never assumed."""
    for tolerance in TOLERANCES:
        snapped = orthogonalize_ring(coords, tolerance)
        if snapped is not None and is_axis_aligned(snapped):
            return snapped, tolerance
    return fallback_box, None


def rectilinear(region, min_piece_m2=.1, min_hole_m2=.3):
    """Axis-aligned version of a plane polygon; returns (polygon, report)."""
    pieces, report = [], {'pieces': 0, 'dropped_m2': 0., 'boxed_rings': 0, 'snapped_rings': 0, 'max_tolerance_m': 0.}
    for piece in polygon_parts(region):
        if piece.area < min_piece_m2:
            report['dropped_m2'] += piece.area
            continue
        outer, tolerance = square_ring(piece.exterior.coords, shapely.box(*piece.bounds))
        if tolerance is None:
            report['boxed_rings'] += 1
        else:
            report['snapped_rings'] += 1; report['max_tolerance_m'] = max(report['max_tolerance_m'], tolerance)
        holes = []
        for ring in piece.interiors:
            hole = shapely.Polygon(ring)
            if is_opening(hole):
                box, tolerance = square_ring(ring.coords, shapely.box(*hole.bounds))
                holes.append(shapely.box(*box.bounds))
            elif hole.area >= min_hole_m2:
                holes.append(shapely.box(*hole.bounds))
        result = outer.difference(shapely.union_all(holes)) if holes else outer
        pieces.append(result)
        report['pieces'] += 1
    if not pieces:
        return None, report
    merged = shapely.union_all(pieces).buffer(0)
    report['displacement_m2'] = float(merged.symmetric_difference(region).area)
    return merged, report


def plane_bucket(plane, gap):
    n = np.asarray(plane['normal'])
    axis = int(np.argmax(np.abs(n)))
    sign = 1. if n[axis] >= 0 else -1.
    return (axis, tuple(np.round(n * sign, 2)), int(np.floor(plane['offset'] * sign / gap)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fragment', action='append', required=True, help='patch files, later ones supersede earlier chains')
    parser.add_argument('--selection', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--merge-gap-m', type=float, default=.015)
    args = parser.parse_args()
    selection = load_selection(args.selection)
    # Resolve chains: keep only final parts, remember what each replaces.
    chain = {}
    for fragment in args.fragment:
        for part in json.loads(Path(fragment).read_text())['parts']:
            chain[part['name']] = part
    replaced = {s for p in chain.values() for s in p.get('source_group_names', [])}
    final = [p for name, p in chain.items() if name not in replaced]

    def root_source(part):
        sources = part.get('source_group_names', [])
        while sources and sources[0] in chain:
            sources = chain[sources[0]].get('source_group_names', [])
        return sources[0] if sources else None

    # Collect exterior planes per part, squared.
    squared = []          # (part, plane, polygon, report)
    others = defaultdict(list)   # part name -> non-exterior planes (kept as-is)
    for part in final:
        if not part.get('kind', '').startswith(('wall', 'parapet')):
            continue
        source = root_source(part) or part['name']
        listed = source in selection
        chosen_planes = selection.get(source)     # None with listed=True means every plane
        is_recess = 'recessed face' in part['name']
        for plane in planes_of(part):
            chosen, mask = (True, None) if is_recess else plane_selection(chosen_planes, plane) if listed else (False, None)
            if chosen and abs(plane['normal'][2]) < .01:
                poly, report = rectilinear(plane['poly'])
                if poly is not None:
                    squared.append((part, plane, poly, report))
                    continue
            others[part['name']].append(plane)
    # Merge coplanar exterior planes across parts on the same level.
    buckets = defaultdict(list)
    for item in squared:
        part, plane = item[0], item[1]
        buckets[(part.get('level', 0), plane_bucket(plane, args.merge_gap_m))].append(item)
    # Buckets are floor(offset/gap); neighbouring buckets within one gap are joined.
    keys = sorted(buckets)
    joined, used = [], set()
    for key in keys:
        if key in used:
            continue
        group = list(buckets[key]); used.add(key)
        neighbour = (key[0], (key[1][0], key[1][1], key[1][2] + 1))
        if neighbour in buckets and neighbour not in used:
            group.extend(buckets[neighbour]); used.add(neighbour)
        joined.append(group)
    out_parts, report, carried = [], [], set()
    block_index = 0
    for group in joined:
        parts_in = {}
        for part, plane, poly, rep in group:
            parts_in[part['name']] = part
        # Mean plane of the group, area weighted.
        normals = np.array([np.asarray(pl['normal']) * np.sign(np.asarray(pl['normal']) @ np.asarray(group[0][1]['normal'])) for _, pl, _, _ in group])
        areas = np.array([poly.area for _, _, poly, _ in group])
        n = np.average(normals, axis=0, weights=areas); n /= np.linalg.norm(n)
        offsets = np.array([float(np.asarray(pl['origin']) @ n) for _, pl, _, _ in group])
        offset = float(np.average(offsets, weights=areas))
        origin = n * offset
        u, v = group[0][1]['u'], group[0][1]['v']
        # Re-express every squared polygon in the common frame and union.
        polys = []
        for _, pl, poly, _ in group:
            mesh = mesh_from_polygon(poly, pl['origin'], pl['u'], pl['v'])
            if mesh is None:
                continue
            uv = np.stack(((mesh.vertices - origin) @ u, (mesh.vertices - origin) @ v), 1)
            polys.append(shapely.union_all(shapely.polygons(uv[mesh.faces])).buffer(0))
        merged = shapely.union_all(polys).buffer(0)
        merged, rep_merge = rectilinear(merged)      # square the union again (seams between pieces)
        if merged is None:
            continue
        meshes = [mesh_from_polygon(merged, origin, u, v)]
        for name in parts_in:
            if name in carried:
                continue                 # its non-exterior planes already travel with an earlier block
            carried.add(name)
            for plane in others[name]:
                meshes.append(mesh_from_polygon(plane['poly'], plane['origin'], plane['u'], plane['v']))
        meshes = [m for m in meshes if m is not None]
        mesh = trimesh.util.concatenate(meshes)
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
        level = group[0][0].get('level', 0)
        name = f'L{level} exterior block {block_index:02d} - rectilinear - INFERRED outline'
        block_index += 1
        out_parts.append({'name': name, 'kind': 'wall_measured', 'level': level, 'colour': [221, 190, 110],
                          'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                          'source_group_names': sorted(parts_in), 'merge_coplanar_faces': True,
                          'evidence_status': 'exterior_planes_squared_and_merged_INFERRED_outline',
                          'completion_is_measured': False})
        report.append({'block': name, 'merged_parts': sorted(parts_in), 'offset_spread_mm': float((offsets.max() - offsets.min()) * 1000),
                       'area_m2': float(merged.area), 'planes': [{'part': p['name'], **r} for p, _, _, r in group], 'union': rep_merge})
        print(f'{name}: {len(parts_in)} parts, {merged.area:.1f} m2, offsets spread {(offsets.max() - offsets.min()) * 1000:.0f} mm', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'blocks': len(out_parts), 'merged_parts': sum(len(r['merged_parts']) for r in report),
                      'boxed_rings': sum(p['boxed_rings'] for r in report for p in r['planes']),
                      'dropped_m2': round(sum(p['dropped_m2'] for r in report for p in r['planes']), 2)}))


if __name__ == '__main__':
    main()
