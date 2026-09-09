"""Separate tiny detached wall-plane speckles from retained architectural faces.

Only components below 25 cm2, below 100 mm in both plane directions, and
more than 20 mm from a larger component on the same plane are hidden. Larger
and linear components are retained. This is a display classification, not
proof that a fragment has no physical counterpart; source groups remain as
hidden references in the native revision.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from wall_surface_cleanup import planes_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    changed, records = [], []
    for part in json.loads(Path(args.model).read_text())['parts']:
        if not part['kind'].startswith(('wall', 'parapet')):
            continue
        pieces, removed = [], []
        planes = list(planes_of(part))
        if any(p['snap'] > .0005 for p in planes):
            continue
        for plane in planes:
            components = polygon_parts(plane['poly'])
            anchors = [p for p in components if p.area >= .0025 or max(np.subtract(p.bounds[2:], p.bounds[:2])) >= .1]
            anchor = shapely.union_all(anchors)
            keep = []
            for poly in components:
                extent = np.subtract(poly.bounds[2:], poly.bounds[:2])
                if (not anchor.is_empty and poly.area < .0025 and max(extent) < .1
                        and poly.distance(anchor) > .02):
                    removed.append({'area_m2': poly.area, 'extent_mm': (extent * 1000).tolist(),
                                    'distance_to_larger_face_mm': poly.distance(anchor) * 1000})
                else:
                    keep.append(poly)
            region = shapely.union_all(keep)
            if not region.is_empty:
                pieces.append(mesh_from_polygon(region, plane['origin'], plane['u'], plane['v']))
        if not removed or not pieces:
            continue
        mesh = trimesh.util.concatenate(pieces)
        result = {**part, 'name': part['name'] + ' - detached speckles hidden',
                  'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                  'source_group_names': [part['name']],
                  'evidence_status': 'detached_speckles_display_filtered_original_retained'}
        result.pop('planar_loops', None)
        changed.append(result)
        records.append({'wall': part['name'], 'fragments': len(removed),
                        'area_m2': sum(r['area_m2'] for r in removed), 'removed': removed})
    (out / 'patches.build.json').write_text(json.dumps({'parts': changed}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({'wall_groups': len(changed), 'detached_fragments': sum(r['fragments'] for r in records),
                      'area_m2': sum(r['area_m2'] for r in records)}), flush=True)


if __name__ == '__main__':
    main()
