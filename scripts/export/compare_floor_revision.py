"""Compare native floor footprints on a 0.00001 mm numerical grid."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts


def footprints(path):
    result = {}
    for part in json.loads(Path(path).read_text())['parts']:
        if 'floor' not in part['kind']:
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        up = mesh.face_normals[:, 2] > .999999
        for z in np.unique(np.round(mesh.triangles_center[up, 2], 4)):
            ids = up & (abs(mesh.triangles_center[:, 2] - z) < 1e-6)
            triangles = np.round(mesh.triangles[ids, :, :2], 8)
            poly = shapely.union_all(shapely.polygons(triangles), grid_size=1e-8)
            result.setdefault((part['level'], float(z)), []).append(poly)
    return {key: shapely.union_all(value, grid_size=1e-8) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    before, after = footprints(args.before), footprints(args.after)
    assert before.keys() == after.keys(), 'Floor datums changed'
    rows = []
    for key, region in after.items():
        old = before[key]
        holes = [shapely.Polygon(r) for p in polygon_parts(region) for r in p.interiors]
        new_holes = [h for h in holes if h.representative_point().within(old)]
        rows.append({'level': key[0], 'datum_m': key[1],
                     'added_area_m2': region.difference(old).area,
                     'removed_area_m2': old.difference(region).area,
                     'new_hole_area_m2': sum(h.area for h in new_holes),
                     'new_holes': len(new_holes),
                     'islands_before': len(polygon_parts(old)),
                     'islands_after': len(polygon_parts(region))})
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
