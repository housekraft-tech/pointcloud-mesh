"""Audit modeled wall faces, including possible opposite faces in other groups.

Parallel overlap is geometric evidence only: adjacent finishes or shaft walls
can also overlap. This check does not certify physical wall thickness.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from wall_surface_cleanup import planes_of
from filter_soulace_overlap import polygon_parts
from render_native_geometry import render


def projected_polygon(plane, target):
    def ring(r):
        xy = np.asarray(r.coords)
        xyz = plane['origin'] + xy[:, :1] * plane['u'] + xy[:, 1:] * plane['v']
        delta = xyz - target['origin']
        return np.column_stack((delta @ target['u'], delta @ target['v']))
    return shapely.union_all([shapely.Polygon(ring(p.exterior), [ring(h) for h in p.interiors])
                              for p in polygon_parts(plane['poly'])]).buffer(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='Wall-face audit')
    args = parser.parse_args()
    parts = json.loads(Path(args.model).read_text())['parts']
    walls = [p for p in parts if p['kind'].startswith(('wall', 'parapet'))]
    planes = []
    for part in walls:
        for plane in planes_of(part):
            if abs(plane['normal'][2]) < .015 and plane['poly'].area >= .02:
                planes.append({**plane, 'group': part['name'], 'level': part.get('level', 0)})
    rows = []
    for part in walls:
        own = [p for p in planes if p['group'] == part['name']]
        if not own:
            continue
        main = max(own, key=lambda p: p['poly'].area)
        overlaps, local, candidates = [], [], []
        for other in planes:
            if other is main or other['level'] != main['level']:
                continue
            if abs(main['normal'] @ other['normal']) < np.cos(np.deg2rad(1)):
                continue
            gap = abs(float((other['origin'] - main['origin']) @ main['normal']))
            if not .05 <= gap <= .65:
                continue
            overlap = main['poly'].intersection(projected_polygon(other, main))
            if overlap.area < .02:
                continue
            overlaps.append(overlap)
            if other['group'] == main['group']:
                local.append(overlap)
            candidates.append({'group': other['group'], 'separation_mm': gap * 1000,
                               'overlap_m2': overlap.area})
        coverage = shapely.union_all(overlaps).area / main['poly'].area if overlaps else 0
        local_coverage = shapely.union_all(local).area / main['poly'].area if local else 0
        state = 'no_substantial_opposite_face' if coverage < .1 else ('partial_opposite_face' if coverage < .8 else 'substantial_opposite_face_candidate')
        rows.append({'wall': part['name'], 'status': state, 'main_face_area_m2': main['poly'].area,
                     'opposite_coverage_fraction': coverage, 'same_group_opposite_coverage_fraction': local_coverage,
                     'watertight': bool(trimesh.Trimesh(part['v'], part['f'], process=True).is_watertight),
                     'candidates': sorted(candidates, key=lambda r: -r['overlap_m2'])[:5]})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    counts = {state: sum(r['status'] == state for r in rows) for state in sorted({r['status'] for r in rows})}
    (out / 'single_sided_audit.json').write_text(json.dumps({'counts': counts, 'walls': rows,
        'interpretation': 'Geometric face overlap only; matching physical wall identity and thickness require scan verification.'}, indent=2))
    colours = {'no_substantial_opposite_face': [215, 70, 65], 'partial_opposite_face': [235, 165, 55],
               'substantial_opposite_face_candidate': [90, 165, 130]}
    by_name = {r['wall']: r for r in rows}
    preview = [{**p, 'colour': colours[by_name[p['name']]['status']]} for p in walls if p['name'] in by_name]
    render(preview, out / 'wall_sides.png', args.label + ' | red: one-sided / amber: partial / green: possible pair')
    print(json.dumps({'counts': counts, 'walls': [{k: r[k] for k in ('wall', 'status', 'opposite_coverage_fraction')} for r in rows]}, indent=2))


if __name__ == '__main__':
    main()


