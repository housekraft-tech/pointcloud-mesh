"""Build observed vertical faces for scan planes that no visible wall covers.

Input is the read-only wall completeness audit. A region qualifies only when:

- it is classified as a probable opposite face of a visible wall or has no
  parallel visible plane at all (surfaces within 60 mm of an existing face
  are finishes, furniture or door leaves until reviewed, and stay out);
- at least 60 percent of its samples are more than 50 mm from every visible
  wall, the unmatched occupancy is at least 1 m2, and its plane residual is
  within 15 mm;
- it lies within 0.45 m of the property's own floor footprint, so surfaces of
  neighbouring buildings or corridors outside the unit are not modelled.

The face is a hypothesis plane bounded to the full-density 50 mm raw-return
envelope, exactly like the existing bounded wall faces, so only observed area
is built. Faces are named as observed and their identity stays unverified:
no back face, thickness or opening is inferred from them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import box
from scipy.spatial import cKDTree

from filter_soulace_overlap import supported_polygon, mesh_from_polygon
from polygon_hygiene import clean_polygon

ACCEPT = ('probable_opposite_face_of_visible_wall_thickness_unverified', 'no_parallel_visible_wall_plane_nearby')
MIN_UNMATCHED_PERCENT = 60
MIN_AREA_M2 = 1.0
MAX_RESIDUAL_MM = 15
FOOTPRINT_MARGIN_M = .45
SUPPORT_M = .05
COLOUR = [214, 176, 120]


def footprint(parts):
    tris = []
    for part in parts:
        kind = part.get('kind', '')
        if 'floor' not in kind and 'ground' not in kind:
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        up = mesh.face_normals[:, 2] > .999
        tris.append(mesh.triangles[up][:, :, :2])
    return shapely.union_all(shapely.polygons(np.concatenate(tris))).buffer(FOOTPRINT_MARGIN_M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--evidence-spec', required=True); parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--level-rotations')
    parser.add_argument('--working-out')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from raw_evidence import load_evidence, model_bounds
    payload = json.loads(Path(args.model).read_text()); parts = payload['parts']
    audit = json.loads(Path(args.audit).read_text())
    rotations = {}
    if args.level_rotations:
        rotations = {int(k): np.asarray(v)[:3, :3] for k, v in json.loads(Path(args.level_rotations).read_text()).items()}
    raw, provenance = load_evidence(args.evidence_spec, model_bounds(parts), args.evidence_cache)
    tree = cKDTree(raw)
    unit = footprint(parts)
    built, rows = [], []
    for region in audit['regions']:
        row = {'id': region['id'], 'level': region['level'], 'classification': region['classification'],
               'unmatched_area_m2': region['unmatched_occupied_grid_area_m2'],
               'unmatched_percent': region['unmatched_points_percent'],
               'plane_residual_p95_mm': region['plane_residual_p95_mm'],
               'nearest_parallel_plane': region.get('nearest_parallel_plane')}
        if region['classification'] not in ACCEPT:
            row['decision'] = 'left_for_review: close to an existing face'
        elif region['unmatched_points_percent'] < MIN_UNMATCHED_PERCENT:
            row['decision'] = 'left_for_review: mostly already represented'
        elif region['unmatched_occupied_grid_area_m2'] < MIN_AREA_M2:
            row['decision'] = 'left_for_review: below 1 m2'
        elif region['plane_residual_p95_mm'] > MAX_RESIDUAL_MM:
            row['decision'] = 'left_for_review: not planar within 15 mm'
        else:
            rotation = rotations.get(region['level'])
            axis = region['normal_axis']; tangent = 1 - axis
            lo, hi = np.asarray(region['bounds_m'][0]), np.asarray(region['bounds_m'][1])
            # Level frame: plane normal along the axis at the median coordinate.
            n_level = np.eye(3)[axis]; u_level = np.eye(3)[tangent]; v_level = np.array([0, 0, 1.])
            origin_level = n_level * region['plane_coordinate_m']
            to_model = (lambda q: q @ rotation.T) if rotation is not None else (lambda q: q)
            n = to_model(n_level); u = to_model(u_level); v = to_model(v_level); origin = to_model(origin_level)
            centre_xy = to_model(np.array([(lo + hi) / 2])[0] * 1.0)[:2]
            if not unit.intersects(shapely.LineString([to_model(np.where(np.arange(3) == tangent, lo, (lo + hi) / 2))[:2],
                                                       to_model(np.where(np.arange(3) == tangent, hi, (lo + hi) / 2))[:2]])):
                row['decision'] = 'left_for_review: outside the unit footprint'
                rows.append(row); continue
            rect = box(lo[tangent] - .05, lo[2] - .05, hi[tangent] + .05, hi[2] + .05)
            bounded, info = supported_polygon(rect, origin, u, v, tree, cutoff=SUPPORT_M, min_area=.02)
            bounded = clean_polygon(bounded)
            row['bounded_area_m2'] = float(bounded.area)
            if bounded.is_empty or bounded.area < .5:
                row['decision'] = 'left_for_review: under 0.5 m2 survives the 50 mm envelope'
                rows.append(row); continue
            mesh = mesh_from_polygon(bounded, origin, u, v)
            mesh.merge_vertices(digits_vertex=8)
            near = row['nearest_parallel_plane'] or {}
            name = f"L{region['level']} observed vertical face {region['id']} - identity unverified"
            built.append({'name': name, 'kind': 'wall_face_observed', 'level': region['level'],
                          'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(), 'colour': COLOUR,
                          'merge_coplanar_faces': True, 'construction_solid': False, 'open_surface': True,
                          'thickness_verified': False, 'modeled_thickness_m': None,
                          'evidence_status': 'full_density_50mm_bounded_surface_from_independent_scan_plane_detection',
                          'semantic_identity': 'unverified: planar vertical scan surface, could be wall face, glazing, door or fixed furniture',
                          'nearest_parallel_visible_wall': near.get('wall'), 'offset_from_it_m': near.get('offset_m')})
            row['decision'] = 'built_as_observed_face'; row['name'] = name
        rows.append(row)
    report = {'source_audit': str(Path(args.audit).resolve()), 'raw_evidence': provenance,
              'rules': {'accepted_classifications': ACCEPT, 'min_unmatched_percent': MIN_UNMATCHED_PERCENT,
                        'min_unmatched_area_m2': MIN_AREA_M2, 'max_plane_residual_mm': MAX_RESIDUAL_MM,
                        'footprint_margin_m': FOOTPRINT_MARGIN_M, 'support_bound_mm': SUPPORT_M * 1000},
              'built': len(built), 'built_area_m2': float(sum(trimesh.Trimesh(p['v'], p['f'], process=False).area for p in built)),
              'regions': rows, 'site_accuracy_certified': False,
              'limitations': 'Observed faces are measured surfaces with unverified identity; they add no back faces, thickness or openings.'}
    (out / 'observed_faces_audit.json').write_text(json.dumps(report, indent=2))
    (out / 'observed_faces.build.json').write_text(json.dumps({'parts': built}, separators=(',', ':')))
    if args.working_out:
        Path(args.working_out).write_text(json.dumps({**payload, 'parts': parts + built}, separators=(',', ':')))
    print(json.dumps({'built': len(built), 'built_area_m2': report['built_area_m2'],
                      'decisions': {d: sum(r['decision'] == d for r in rows) for d in {r['decision'] for r in rows}}}, indent=2), flush=True)
    for r in rows:
        if r['decision'] == 'built_as_observed_face':
            print(r['id'], round(r['bounded_area_m2'], 2), 'm2', r['nearest_parallel_plane'], flush=True)


if __name__ == '__main__':
    main()
