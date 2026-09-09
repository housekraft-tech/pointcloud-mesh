"""Turn unclassified scan-detail surfaces into rectilinear planes (declared inference).

The flat's checked model carries one large "Observed scan detail -
UNCLASSIFIED" mesh: wall regions, ceilings and clutter that earlier stages
did not classify. Its triangles are grouped by axis-aligned normal and by
offset (20 mm bins); every group with enough area becomes one plane whose
outline is squared with the wall rule (vertical) or the slab rule
(horizontal). Triangles that are not axis-aligned (furniture, curtains,
people) are dropped. The planes replace the scan-detail group, so the
rectilinear merge can join them with the wall blocks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from rectilinear_exterior import rectilinear

WALL_COLOUR = [221, 190, 110]
CEILING_COLOUR = [200, 200, 205]


def frames():
    return {0: (np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, -1., 0])),
            1: (np.array([0, 1., 0]), np.array([0, 0, 1.]), np.array([1., 0, 0])),
            2: (np.array([0, 0, 1.]), np.array([1., 0, 0]), np.array([0, 1., 0]))}


def planes_from_mesh(mesh, bin_m=.02, min_area_m2=.5, assign_m=.03):
    result = []
    normals, centres, areas = mesh.face_normals, mesh.triangles_center, mesh.area_faces
    for axis, (n, u, v) in frames().items():
        aligned = np.abs(normals[:, axis]) > .97
        if not aligned.any():
            continue
        offsets = centres[aligned] @ n
        idx = np.flatnonzero(aligned)
        bins = np.floor(offsets / bin_m).astype(np.int64)
        weight = {}
        for b, a in zip(bins, areas[idx]):
            weight[b] = weight.get(b, 0.) + a
        used = np.zeros(len(idx), bool)
        for b in sorted(weight, key=lambda k: -(weight.get(k - 1, 0) + weight[k] + weight.get(k + 1, 0))):
            total = weight.get(b - 1, 0) + weight[b] + weight.get(b + 1, 0)
            if total < min_area_m2:
                break
            centre = (b + .5) * bin_m
            member = (~used) & (np.abs(offsets - centre) <= assign_m)
            if areas[idx[member]].sum() < min_area_m2:
                continue
            used |= member
            offset = float(np.average(offsets[member], weights=areas[idx[member]]))
            tri = mesh.triangles[idx[member]]
            origin = n * offset
            uv = np.stack(((tri - origin) @ u, (tri - origin) @ v), -1)
            poly = shapely.union_all(shapely.polygons(uv)).buffer(0)
            squared, report = rectilinear(poly, min_piece_m2=.3, box_holes=axis != 2)
            if squared is None:
                continue
            result.append({'axis': axis, 'offset': offset, 'origin': origin, 'u': u, 'v': v, 'poly': squared,
                           'measured_m2': float(areas[idx[member]].sum()), 'report': report})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kinds', default='scan_detail')
    args = parser.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(','))
    parts = json.loads(Path(args.model).read_text())['parts']
    out_parts, report = [], []
    for part in parts:
        if not part.get('kind', '').startswith(kinds):
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        planes = planes_from_mesh(mesh)
        for k, plane in enumerate(planes):
            m = mesh_from_polygon(plane['poly'], plane['origin'], plane['u'], plane['v'])
            if m is None:
                continue
            vertical = plane['axis'] != 2
            name = f"Scan detail plane {'xyz'[plane['axis']]}{k:02d} at {plane['offset']:.3f} m - rectilinear - INFERRED outline"
            out_parts.append({'name': name, 'kind': 'wall_measured' if vertical else 'ceiling_scan_plane',
                              'level': part.get('level', 0), 'colour': WALL_COLOUR if vertical else CEILING_COLOUR,
                              'v': m.vertices.tolist(), 'f': m.faces.tolist(), 'source_group_names': [part['name']],
                              'merge_coplanar_faces': True, 'evidence_status': 'scan_detail_dominant_plane_squared_INFERRED_outline',
                              'completion_is_measured': False})
            report.append({'name': name, 'measured_m2': plane['measured_m2'], 'plane_m2': float(plane['poly'].area), **plane['report']})
        print(f"{part['name']}: {len(planes)} planes, {sum(p['measured_m2'] for p in planes):.1f} m2 of {mesh.area:.1f} m2 axis-aligned", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'planes': len(out_parts), 'plane_area_m2': round(sum(r['plane_m2'] for r in report), 1)}))


if __name__ == '__main__':
    main()
