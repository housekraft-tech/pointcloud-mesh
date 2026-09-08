"""Clean wall-face artifacts without losing any scan coverage.

Per visible wall group and per exact plane the triangles are united, which
removes duplicated and overlapping triangles inside the group. Then:

- cross-group coplanar overlap (two groups drawing the same patch of the same
  plane, the flicker seen on exterior faces) is assigned to the group with the
  larger plane and removed from the other;
- islands below 0.02 m2 are removed only if every raw return they represented
  (within 50 mm) is still within 50 mm of the cleaned model, so no previously
  represented scan sample is lost; islands that would lose support stay;
- enclosed pores below 0.005 m2 are filled only when the whole pore lies within
  50 mm of raw returns, as the floor cleanup already does.

Vertices are snapped onto their bucketed plane; the largest displacement is
recorded and must stay below 0.5 mm, otherwise the group is left untouched.
Nothing here adds wall runs, backs or openings.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import open3d as o3d

from filter_soulace_overlap import polygon_parts, supported_polygon, mesh_from_polygon

ISLAND_M2 = .02
PORE_M2 = .005
SUPPORT_M = .05
MAX_SNAP_M = .0005


def is_wall(part):
    return part.get('kind', '').startswith(('wall', 'parapet'))


def planes_of(part):
    """Yield (key, polygon, origin, u, v, snap) for each near-exact plane."""
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    buckets = defaultdict(list)
    for i, (normal, centre, area) in enumerate(zip(mesh.face_normals, mesh.triangles_center, mesh.area_faces)):
        if area < 1e-10:
            continue
        n = normal.copy()
        if n[np.argmax(np.abs(n))] < 0:
            n = -n
        buckets[(*np.round(n, 5), round(float(n @ centre), 5))].append(i)
    for key, ids in buckets.items():
        tri = mesh.triangles[ids]
        # Median plane of the bucket; snapping displacement is checked later.
        normals = mesh.face_normals[ids] * np.sign(mesh.face_normals[ids] @ np.asarray(key[:3]))[:, None]
        n = np.average(normals, axis=0, weights=mesh.area_faces[ids]); n /= np.linalg.norm(n)
        offset = float(np.median((tri.reshape(-1, 3) @ n)))
        snap = float(np.max(np.abs(tri.reshape(-1, 3) @ n - offset)))
        origin = n * offset
        seed = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, seed); u /= np.linalg.norm(u); v = np.cross(n, u)
        projected = np.stack(((tri - origin) @ u, (tri - origin) @ v), axis=-1)
        poly = shapely.union_all(shapely.polygons(projected)).buffer(0)
        yield {'key': key, 'normal': n, 'offset': offset, 'poly': poly, 'origin': origin, 'u': u, 'v': v,
               'snap': snap, 'triangle_area': float(mesh.area_faces[ids].sum())}


def scene_of(parts):
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    for part in parts:
        scene.add_triangles(o3d.core.Tensor(np.asarray(part['v'], np.float32)),
                            o3d.core.Tensor(np.asarray(part['f'], np.uint32)))
    return scene


def distances(scene, points):
    out = []
    for start in range(0, len(points), 200000):
        q = o3d.core.Tensor(points[start:start + 200000].astype(np.float32))
        out.append(scene.compute_distance(q).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def polygon_samples(poly, origin, u, v, step=.02):
    xmin, ymin, xmax, ymax = poly.bounds
    xs = np.arange(xmin, xmax + step, step); ys = np.arange(ymin, ymax + step, step)
    xx, yy = np.meshgrid(xs, ys); xy = np.column_stack((xx.ravel(), yy.ravel()))
    inside = shapely.contains_xy(poly, xy[:, 0], xy[:, 1])
    coords = np.asarray(poly.exterior.coords)
    xy = np.vstack([xy[inside], coords])
    return origin + xy[:, 0, None] * u + xy[:, 1, None] * v


def clean(parts, raw):
    tree = cKDTree(raw)
    walls = [p for p in parts if is_wall(p)]
    others = [p for p in parts if not is_wall(p)]
    groups = {}
    skipped = []
    for part in walls:
        planes = list(planes_of(part))
        worst = max((p['snap'] for p in planes), default=0.)
        if worst > MAX_SNAP_M:
            skipped.append({'name': part['name'], 'reason': f'plane snap {worst * 1000:.2f} mm exceeds bound; untouched'})
            continue
        groups[part['name']] = {'part': part, 'planes': planes, 'snap_mm': worst * 1000,
                                'source_area_m2': float(sum(p['triangle_area'] for p in planes)),
                                'union_area_m2': float(sum(p['poly'].area for p in planes))}
    # Cross-group coplanar overlap: the group with the larger plane keeps it.
    overlaps = []
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a]['part'].get('level') != groups[b]['part'].get('level'):
                continue
            for pa in groups[a]['planes']:
                for pb in groups[b]['planes']:
                    if np.linalg.norm(pa['normal'] - pb['normal']) > 1e-5 or abs(pa['offset'] - pb['offset']) > 1e-3:
                        continue
                    # Express pb in pa's basis: same plane, so a rigid 2D map.
                    shift = (pb['origin'] - pa['origin'])
                    coords = np.asarray(pb['poly'].exterior.coords) if pb['poly'].geom_type == 'Polygon' else None
                    xy_b = []
                    for poly in polygon_parts(pb['poly']):
                        ext = np.asarray(poly.exterior.coords)
                        pts = pb['origin'] + ext[:, 0, None] * pb['u'] + ext[:, 1, None] * pb['v']
                        ring = np.column_stack(((pts - pa['origin']) @ pa['u'], (pts - pa['origin']) @ pa['v']))
                        holes = []
                        for interior in poly.interiors:
                            h = np.asarray(interior.coords)
                            hp = pb['origin'] + h[:, 0, None] * pb['u'] + h[:, 1, None] * pb['v']
                            holes.append(np.column_stack(((hp - pa['origin']) @ pa['u'], (hp - pa['origin']) @ pa['v'])))
                        xy_b.append(Polygon(ring, holes))
                    b_in_a = shapely.union_all(xy_b).buffer(0)
                    inter = pa['poly'].intersection(b_in_a)
                    if inter.is_empty or inter.area <= 1e-6:
                        continue
                    loser_is_b = pb['poly'].area <= pa['poly'].area
                    if loser_is_b:
                        # Map the intersection back into pb's basis and subtract.
                        pieces = []
                        for poly in polygon_parts(inter):
                            ext = np.asarray(poly.exterior.coords)
                            pts = pa['origin'] + ext[:, 0, None] * pa['u'] + ext[:, 1, None] * pa['v']
                            pieces.append(Polygon(np.column_stack(((pts - pb['origin']) @ pb['u'], (pts - pb['origin']) @ pb['v']))))
                        pb['poly'] = pb['poly'].difference(shapely.union_all(pieces).buffer(0)).buffer(0)
                    else:
                        pa['poly'] = pa['poly'].difference(inter).buffer(0)
                    overlaps.append({'kept': a if loser_is_b else b, 'trimmed': b if loser_is_b else a,
                                     'overlap_area_m2': float(inter.area)})
    # Islands: drop small ones provisionally, then restore any whose raw
    # support would otherwise be lost.
    island_records = []
    for name, group in groups.items():
        for pi, plane in enumerate(group['planes']):
            for poly in polygon_parts(plane['poly']):
                if poly.area < ISLAND_M2:
                    island_records.append({'group': name, 'plane': pi, 'poly': poly, 'area': poly.area})
    for group in groups.values():
        for plane in group['planes']:
            plane['kept'] = shapely.union_all([p for p in polygon_parts(plane['poly']) if p.area >= ISLAND_M2]).buffer(0) \
                if not plane['poly'].is_empty else plane['poly']
    # Raw returns represented by the removed islands.
    island_raw = []
    for rec in island_records:
        plane = groups[rec['group']]['planes'][rec['plane']]
        samples = polygon_samples(rec['poly'], plane['origin'], plane['u'], plane['v'])
        ids = sorted({i for lst in tree.query_ball_point(samples, SUPPORT_M, workers=8) for i in lst})
        rec['raw_ids'] = np.asarray(ids, dtype=np.int64)
        island_raw.append(rec['raw_ids'])
    restored = 0
    if island_records:
        ids_all = np.unique(np.concatenate(island_raw)) if island_raw else np.zeros(0, np.int64)
        probe = raw[ids_all]
        before_d = distances(scene_of(walls), probe) if len(probe) else np.zeros(0)
        provisional = build_parts(groups, others=[])
        after_d = distances(scene_of(provisional), probe) if len(probe) else np.zeros(0)
        lost = set(ids_all[(before_d <= SUPPORT_M) & (after_d > SUPPORT_M)].tolist())
        for rec in island_records:
            if lost.intersection(rec['raw_ids'].tolist()):
                plane = groups[rec['group']]['planes'][rec['plane']]
                plane['kept'] = shapely.union_all([plane['kept'], rec['poly']]).buffer(0)
                rec['restored'] = True; restored += 1
            else:
                rec['restored'] = False
    # Pores: fill only wholly supported small holes.
    for name, group in groups.items():
        filled = 0; filled_area = 0.; unresolved = 0
        for plane in group['planes']:
            new_holes = []
            for poly in polygon_parts(plane['kept']):
                for ring in poly.interiors:
                    hole = Polygon(ring)
                    if hole.area > PORE_M2:
                        continue
                    supported, _ = supported_polygon(hole, plane['origin'], plane['u'], plane['v'], tree,
                                                     cutoff=SUPPORT_M, min_area=1e-8)
                    if hole.difference(supported).area < 1e-10:
                        new_holes.append(hole)
                    else:
                        unresolved += 1
            if new_holes:
                plane['kept'] = shapely.union_all([plane['kept'], *new_holes]).buffer(0)
                filled += len(new_holes); filled_area += sum(h.area for h in new_holes)
        group.update(pores_filled=filled, pores_filled_area_m2=float(filled_area), pores_unresolved=unresolved)
    cleaned = build_parts(groups, others=[])
    audit = {'groups': [], 'cross_group_overlaps_resolved': overlaps,
             'overlap_area_removed_m2': float(sum(o['overlap_area_m2'] for o in overlaps)),
             'islands_below_threshold': len(island_records),
             'islands_removed': int(sum(not r.get('restored', False) for r in island_records)),
             'islands_restored_for_scan_support': restored,
             'islands_removed_area_m2': float(sum(r['area'] for r in island_records if not r.get('restored', False))),
             'skipped_groups': skipped}
    by_name = {p['name']: p for p in cleaned}
    for name, group in groups.items():
        new = by_name.get(name)
        audit['groups'].append({'name': name, 'source_triangle_area_m2': group['source_area_m2'],
                                'union_area_m2': group['union_area_m2'],
                                'duplicate_area_removed_m2': max(0., group['source_area_m2'] - group['union_area_m2']),
                                'final_area_m2': float(sum(p['kept'].area for p in group['planes'])),
                                'plane_snap_max_mm': group['snap_mm'], 'pores_filled': group['pores_filled'],
                                'pores_filled_area_m2': group['pores_filled_area_m2'],
                                'pores_unresolved': group['pores_unresolved'],
                                'islands_removed': int(sum(1 for r in island_records if r['group'] == name and not r.get('restored', False))),
                                'triangles_before': len(group['part']['f']), 'triangles_after': len(new['f']) if new else 0})
    return cleaned, audit


def build_parts(groups, others):
    parts = list(others)
    for name, group in groups.items():
        pieces = []
        for plane in group['planes']:
            region = plane.get('kept', plane['poly'])
            if region.is_empty:
                continue
            piece = mesh_from_polygon(region, plane['origin'], plane['u'], plane['v'])
            if piece is not None:
                pieces.append(piece)
        if not pieces:
            continue
        mesh = trimesh.util.concatenate(pieces)
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8) & (mesh.area_faces >= 1e-12))
        mesh.remove_unreferenced_vertices()
        parts.append({**group['part'], 'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist()})
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--evidence-spec', required=True); parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--working-out', help='write the full model with cleaned walls substituted')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from raw_evidence import load_evidence, model_bounds
    from run_architectural_flow import compare_coverage
    payload = json.loads(Path(args.model).read_text()); parts = payload['parts']
    raw, provenance = load_evidence(args.evidence_spec, model_bounds(parts), args.evidence_cache)
    cleaned, audit = clean(parts, raw)
    by_name = {p['name']: p for p in cleaned}
    walls_before = [p for p in parts if is_wall(p)]
    walls_after = [by_name.get(p['name'], p) for p in walls_before]
    coverage, _, _ = compare_coverage(walls_before, walls_after, raw)
    changed = []
    for part in walls_before:
        new = by_name.get(part['name'])
        if new is None:
            continue
        if len(new['f']) == len(part['f']) and np.allclose(np.asarray(new['v']).sum(), np.asarray(part['v']).sum()):
            continue
        changed.append({**new, 'name': part['name'] + ' - cleaned', 'source_group_names': [part['name']],
                        'evidence_status': 'exact_plane_union; islands removed only with scan support retained; pores filled within 50 mm raw envelope'})
    audit.update(source=str(Path(args.model).resolve()), raw_evidence=provenance, wall_coverage=coverage,
                 changed_groups=len(changed), site_accuracy_certified=False)
    (out / 'wall_cleanup_audit.json').write_text(json.dumps(audit, indent=2))
    (out / 'wall_cleanup.build.json').write_text(json.dumps({'parts': changed}, separators=(',', ':')))
    if args.working_out:
        replaced = {c['source_group_names'][0]: c for c in changed}
        working = [replaced.get(p['name'], p) for p in parts]
        Path(args.working_out).write_text(json.dumps({**payload, 'parts': working}, separators=(',', ':')))
    print(json.dumps({k: v for k, v in audit.items() if k not in ('groups', 'cross_group_overlaps_resolved', 'raw_evidence')}, indent=2), flush=True)
    print(json.dumps({'lost_over_50mm': coverage['previously_represented_scan_samples_lost_over_50mm'],
                      'gained_within_50mm': coverage['newly_represented_scan_samples_within_50mm'],
                      'area_before': coverage['before']['surface_area_m2'], 'area_after': coverage['after']['surface_area_m2']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
