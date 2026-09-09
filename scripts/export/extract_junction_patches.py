"""Keep existing native floors intact and export only supported added strips."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import mesh_from_polygon


def top(part, z):
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    ids = (mesh.face_normals[:, 2] > .999999) & (abs(mesh.triangles_center[:, 2] - z) < 1e-6)
    return shapely.union_all(shapely.polygons(np.round(mesh.triangles[ids, :, :2], 8)), grid_size=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--repairs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = {p['name']: p for p in json.loads(Path(args.source).read_text())['parts']}
    parts = []
    for repair in json.loads(Path(args.repairs).read_text())['parts']:
        if 'floor' not in repair['kind']:
            raise ValueError('This adapter only handles additive floor repairs')
        original = source[repair['source_group_names'][0]]
        mesh = trimesh.Trimesh(repair['v'], repair['f'], process=False)
        datums = np.unique(np.round(mesh.triangles_center[mesh.face_normals[:, 2] > .999999, 2], 4))
        for z in datums:
            added = top(repair, z).difference(top(original, z))
            if added.is_empty or added.area < 1e-9:
                continue
            patch = mesh_from_polygon(added, np.array([0., 0., z]), np.array([1., 0., 0.]), np.array([0., 1., 0.]))
            parts.append({'name': f'L{repair["level"]} supported floor junction strips {len(parts) + 1}',
                          'kind': 'floor_junction_patch', 'level': repair['level'],
                          'v': patch.vertices.tolist(), 'f': patch.faces.tolist(),
                          'source_group_names': [], 'merge_coplanar_faces': True,
                          'evidence_status': '15mm_junction_strips_clipped_to_50mm_raw_support',
                          'parent_floor': original['name']})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'parts': parts}, separators=(',', ':')))
    print(f'{len(parts)} additive floor patches; original geometry preserved')


if __name__ == '__main__':
    main()
