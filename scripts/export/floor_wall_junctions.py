"""Measure and, within a 15 mm bound, close floor-to-wall junction gaps.

For every level the visible floor tops are grouped by exact datum. Every wall
face's base edge (its lowest vertical-face edges) is sampled and compared with
the floor region at the matching datum:

- horizontal gap: XY distance from the base sample to the floor region
- vertical gap: base z minus floor datum

With --fix, two bounded edits are made:

- a wall base floating above its floor datum is seated onto it, only when the
  offset is at most 15 mm (the local-cleanup displacement bound);
- the floor is extended by the strip between its boundary and a wall base
  edge up to 150 mm away, never past the wall face, and only where the strip
  lies within 50 mm of raw returns: the strip is a hypothesis clipped to the
  registered evidence exactly like fresh wall faces are, so the unsupported
  part (for instance under a wall) is not built.

Larger gaps stay open and are listed for review. No geometry is invented
elsewhere; wall thickness and site accuracy remain uncertified.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon, LineString
from scipy.spatial import cKDTree

from filter_soulace_overlap import polygon_parts, supported_polygon, mesh_from_polygon
from polygon_hygiene import clean_polygon

MAX_FIX_M = .015        # vertical seating bound
MAX_STRIP_M = .15       # floor strip reach towards a wall base
MIN_STRIP_M2 = .0005    # supported strip pieces smaller than this are noise
SUPPORT_M = .05
FLOOR_KINDS = ('floor', 'ground', 'landing')


def is_floor(part):
    kind = part.get('kind', '')
    return any(k in kind for k in FLOOR_KINDS) and 'stair' not in kind


def is_wall(part):
    return part.get('kind', '').startswith(('wall', 'parapet'))


def floor_regions(parts):
    """Union of horizontal top triangles per (level, datum)."""
    table = defaultdict(list)
    for part in parts:
        if not is_floor(part):
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        up = mesh.face_normals[:, 2] > .999999
        for z in np.unique(np.round(mesh.triangles_center[up, 2], 4)):
            ids = np.flatnonzero(up & (abs(mesh.triangles_center[:, 2] - z) < 5e-4))
            table[(part.get('level', 0), float(z))].append(mesh.triangles[ids][:, :, :2])
    regions = {}
    for key, tris in table.items():
        regions[key] = shapely.union_all(shapely.polygons(np.concatenate(tris))).buffer(0)
    return regions


def wall_base_samples(part, step=.02):
    """Sample the lowest edges of the vertical faces of one wall part."""
    v = np.asarray(part['v']); f = np.asarray(part['f'])
    mesh = trimesh.Trimesh(v, f, process=False)
    vertical = abs(mesh.face_normals[:, 2]) < .01
    if not vertical.any():
        return None
    z0 = float(v[f[vertical]][:, :, 2].min())
    edges = mesh.edges_unique
    low = (abs(v[edges[:, 0], 2] - z0) < 1e-6) & (abs(v[edges[:, 1], 2] - z0) < 1e-6)
    face_edges = mesh.edges_unique[np.unique(mesh.faces_unique_edges[vertical])]
    low_set = {tuple(sorted(e)) for e in edges[low]}
    segments = [e for e in face_edges if tuple(sorted(e)) in low_set]
    if not segments:
        return None
    samples, seg_list = [], []
    for a, b in segments:
        pa, pb = v[a, :2], v[b, :2]
        length = np.linalg.norm(pb - pa)
        count = max(2, int(np.ceil(length / step)) + 1)
        samples.append(pa + (pb - pa) * np.linspace(0, 1, count)[:, None])
        seg_list.append((pa, pb))
    return {'z': z0, 'xy': np.vstack(samples), 'segments': seg_list}


def measure(parts):
    regions = floor_regions(parts)
    rows = []
    for part in parts:
        if not is_wall(part):
            continue
        base = wall_base_samples(part)
        if base is None:
            continue
        level = part.get('level', 0)
        candidates = [(abs(base['z'] - z), z) for (lvl, z) in regions if lvl == level]
        if not candidates:
            continue
        dz, datum = min(candidates)
        if dz > .12:
            rows.append({'name': part['name'], 'level': level, 'base_z_m': base['z'], 'nearest_floor_datum_m': datum,
                         'vertical_gap_mm': (base['z'] - datum) * 1000, 'status': 'base_not_at_any_floor_datum'})
            continue
        region = regions[(level, datum)]
        gap = shapely.distance(shapely.points(base['xy']), region)
        near = gap <= .30
        if not near.any():
            rows.append({'name': part['name'], 'level': level, 'base_z_m': base['z'], 'nearest_floor_datum_m': datum,
                         'vertical_gap_mm': (base['z'] - datum) * 1000, 'status': 'no_floor_within_300mm_of_base'})
            continue
        g = gap[near]
        rows.append({'name': part['name'], 'level': level, 'base_z_m': base['z'], 'nearest_floor_datum_m': datum,
                     'vertical_gap_mm': float((base['z'] - datum) * 1000), 'base_samples': int(len(base['xy'])),
                     'samples_near_floor': int(near.sum()),
                     'horizontal_gap_max_mm': float(g.max() * 1000), 'horizontal_gap_p95_mm': float(np.percentile(g, 95) * 1000),
                     'samples_gap_over_1mm_percent': float(np.mean(g > .001) * 100),
                     'samples_gap_over_15mm_percent': float(np.mean(g > MAX_FIX_M) * 100),
                     'status': 'touching' if g.max() <= .001 and abs(base['z'] - datum) <= .001 else 'gap'})
    return regions, rows


def summarise(rows):
    gaps = [r for r in rows if r['status'] == 'gap']
    return {'walls_checked': len(rows), 'walls_touching': sum(r['status'] == 'touching' for r in rows),
            'walls_with_gap': len(gaps),
            'walls_with_horizontal_gap_over_1mm': sum(r.get('horizontal_gap_max_mm', 0) > 1 for r in gaps),
            'walls_with_vertical_gap_over_1mm': sum(abs(r['vertical_gap_mm']) > 1 for r in gaps),
            'worst_horizontal_gap_mm': max([r.get('horizontal_gap_max_mm', 0) for r in gaps], default=0),
            'worst_vertical_gap_mm': max([abs(r['vertical_gap_mm']) for r in gaps], default=0),
            'walls_not_on_a_floor_datum': sum(r['status'] not in ('gap', 'touching') for r in rows)}


def extension_strips(part, region):
    """Strip between each base segment and its projection on the floor boundary."""
    base = wall_base_samples(part)
    pieces = []
    xy = np.vstack([np.vstack(s) for s in base['segments']])
    # Work against the local floor only; complex floors are slow to intersect.
    local = region.intersection(shapely.box(*(xy.min(0) - .3), *(xy.max(0) + .3)))
    if local.is_empty:
        return None
    region = local
    merged = shapely.line_merge(shapely.multilinestrings([np.vstack(s) for s in base['segments']]))
    lines = list(getattr(merged, 'geoms', [merged]))
    for line in lines:
        line = line.simplify(1e-6)
        coords = np.asarray(line.coords)
        pa, pb = coords[0], coords[-1]
        if line.length < 1e-6:
            continue
        pts = shapely.line_interpolate_point(line, np.linspace(0, line.length, max(3, int(line.length / .01) + 2)))
        d = shapely.distance(pts, region)
        if d.max() <= 1e-6 or d.min() > MAX_STRIP_M:
            continue
        reach = min(d.max(), MAX_STRIP_M)
        # Strip between the base edge and the present floor boundary only:
        # the sweep of the edge towards the floor, cut back to the floor's
        # neighbourhood so nothing is added on the far side of the wall face.
        strip = line.buffer(reach + 5e-4, cap_style='flat')
        piece = strip.intersection(region.buffer(reach + 1e-3)).difference(region)
        if not piece.is_empty and piece.area > 0:
            pieces.append(piece)
    return shapely.union_all(pieces).buffer(0) if pieces else None


def rebuild_floor(part, new_region_by_datum):
    """Rebuild one floor part with extended top regions, preserving datums."""
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    up = mesh.face_normals[:, 2] > .999999
    datums = np.unique(np.round(mesh.triangles_center[up, 2], 4))
    pieces = []
    solid = mesh.is_watertight and mesh.volume > 1e-9
    for z in datums:
        region = new_region_by_datum.get(float(z))
        if region is None:
            continue
        region = clean_polygon(region)
        if region.is_empty:
            continue
        if solid:
            bottom = float(mesh.vertices[:, 2].min())
            for poly in polygon_parts(region):
                piece = trimesh.creation.extrude_polygon(poly, height=float(z) - bottom, engine='earcut')
                piece.vertices[:, 2] += bottom
                pieces.append(piece)
        else:
            piece = mesh_from_polygon(region, np.array([0, 0, float(z)]), np.array([1., 0, 0]), np.array([0, 1., 0]))
            if piece is not None:
                pieces.append(piece)
    if not pieces:
        return None
    out = trimesh.util.concatenate(pieces)
    out.merge_vertices(digits_vertex=8)
    out.update_faces(out.nondegenerate_faces(height=1e-8) & (out.area_faces >= 1e-12))
    out.remove_unreferenced_vertices()
    if solid:
        out.fix_normals()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fix', action='store_true')
    parser.add_argument('--evidence-spec', help='flow manifest or support scan list for raw support')
    parser.add_argument('--evidence-cache', help='npz cache written by raw_evidence.py')
    parser.add_argument('--working-out', help='write the full model with the fixes substituted')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.model).read_text()); parts = payload['parts']
    regions, rows = measure(parts)
    report = {'source': str(Path(args.model).resolve()), 'fix_bound_mm': MAX_FIX_M * 1000,
              'support_bound_mm': SUPPORT_M * 1000, 'before': summarise(rows), 'walls_before': rows}
    print(json.dumps(report['before'], indent=2), flush=True)
    if not args.fix:
        (out / 'floor_wall_junction_audit.json').write_text(json.dumps(report, indent=2))
        return
    if not (args.evidence_spec and args.evidence_cache):
        raise SystemExit('--fix needs --evidence-spec and --evidence-cache for raw support')
    from raw_evidence import load_evidence, model_bounds
    raw, provenance = load_evidence(args.evidence_spec, model_bounds(parts), args.evidence_cache)
    report['raw_evidence'] = provenance
    tree = cKDTree(raw)
    by_name = {p['name']: p for p in parts}
    # 1. Horizontal: floor strips towards wall base edges, per (level, datum).
    additions = defaultdict(list); refused = []
    for row in rows:
        if row['status'] != 'gap' or row.get('horizontal_gap_max_mm', 0) <= 1:
            continue
        level, datum = row['level'], row['nearest_floor_datum_m']
        region = regions[(level, datum)]
        strip = extension_strips(by_name[row['name']], region)
        if strip is None or strip.is_empty:
            refused.append({'wall': row['name'], 'reason': 'base further than 150 mm from the floor everywhere; left open'})
            continue
        supported, _ = supported_polygon(strip, np.array([0, 0, datum]), np.array([1., 0, 0]), np.array([0, 1., 0]),
                                         tree, cutoff=SUPPORT_M, min_area=MIN_STRIP_M2)
        supported = supported.buffer(0)
        if supported.is_empty or supported.area < MIN_STRIP_M2:
            refused.append({'wall': row['name'], 'reason': 'strip has no raw floor returns within 50 mm; left open',
                            'strip_area_m2': strip.area})
            continue
        additions[(level, datum)].append(supported)
        if strip.area - supported.area > 1e-6:
            refused.append({'wall': row['name'], 'reason': 'part of the strip is unsupported and stays open',
                            'strip_area_m2': strip.area, 'supported_area_m2': supported.area})
    new_regions = {}
    for key, strips in additions.items():
        new_regions[key] = shapely.union_all([regions[key], *strips]).buffer(0)
    replaced = {}
    floor_changes = []
    for part in parts:
        if not is_floor(part):
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        up = mesh.face_normals[:, 2] > .999999
        level = part.get('level', 0)
        own = {}
        for z in np.unique(np.round(mesh.triangles_center[up, 2], 4)):
            key = (level, float(z))
            if key not in new_regions:
                continue
            ids = np.flatnonzero(up & (abs(mesh.triangles_center[:, 2] - z) < 5e-4))
            mine = shapely.union_all(shapely.polygons(mesh.triangles[ids][:, :, :2])).buffer(0)
            # Give each floor only the strips that touch its own footprint.
            gained = new_regions[key].difference(regions[key]).intersection(mine.buffer(MAX_STRIP_M + 2e-3))
            if gained.is_empty or gained.area < 1e-9:
                continue
            own[float(z)] = shapely.union_all([mine, gained]).buffer(0)
            regions[key] = regions[key].difference(gained)  # not assigned twice
            new_regions[key] = new_regions[key].difference(gained)
        if not own:
            continue
        rebuilt = rebuild_floor(part, own)
        if rebuilt is None:
            continue
        replaced[part['name']] = {**part, 'name': part['name'] + ' - junction closed',
                                  'v': rebuilt.vertices.tolist(), 'f': rebuilt.faces.tolist(),
                                  'source_group_names': [part['name']],
                                  'evidence_status': 'floor_extended_to_wall_base_up_to_150mm_clipped_to_50mm_raw_support'}
        before_area = sum(shapely.union_all(shapely.polygons(
            mesh.triangles[np.flatnonzero(up & (abs(mesh.triangles_center[:, 2] - z) < 5e-4))][:, :, :2])).area for z in own)
        floor_changes.append({'floor': part['name'], 'added_area_m2': float(sum(o.area for o in own.values()) - before_area),
                              'datums_m': sorted(own)})
    # 2. Vertical: seat floating wall bases onto the datum within the bound.
    wall_changes = []
    for row in rows:
        dz = row['vertical_gap_mm'] / 1000
        if row['status'] != 'gap' or not (.001 < dz <= MAX_FIX_M):
            continue
        part = by_name[row['name']]
        v = np.asarray(part['v'], float); f = np.asarray(part['f'])
        mesh = trimesh.Trimesh(v, f, process=False)
        vertical = abs(mesh.face_normals[:, 2]) < .01
        base_vertices = np.unique(f[vertical][abs(v[f[vertical]][:, :, 2] - row['base_z_m']) < 1e-6])
        v[base_vertices, 2] = row['nearest_floor_datum_m']
        replaced[part['name']] = {**part, 'name': part['name'] + ' - base seated', 'v': v.tolist(), 'f': f.tolist(),
                                  'source_group_names': [part['name']],
                                  'evidence_status': 'wall_base_lowered_to_floor_datum_within_15mm'}
        wall_changes.append({'wall': part['name'], 'lowered_mm': float(dz * 1000), 'vertices': int(len(base_vertices))})
    refined = [replaced.get(p['name'], p) for p in parts]
    _, rows_after = measure(refined)
    report.update(after=summarise(rows_after), walls_after=rows_after, floor_changes=floor_changes,
                  wall_changes=wall_changes, refused=refused)
    (out / 'floor_wall_junction_audit.json').write_text(json.dumps(report, indent=2))
    (out / 'junction_fix.build.json').write_text(json.dumps({'parts': list(replaced.values())}, separators=(',', ':')))
    if args.working_out:
        Path(args.working_out).write_text(json.dumps({**payload, 'parts': refined}, separators=(',', ':')))
    print(json.dumps({'after': report['after'], 'floors_rebuilt': len(floor_changes), 'walls_seated': len(wall_changes),
                      'refused': len(refused)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
