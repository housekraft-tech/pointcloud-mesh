"""Square floor and ceiling outlines (declared inference) so every slab is rectilinear.

For each floor or ceiling part (stairs excluded): a watertight slab is
rebuilt as an extrusion between its own bottom and top heights of its
squared plan outline; a surface slab has each horizontal plane squared in
place. Squaring uses the same rule as the walls: snap to axis-aligned edges
with a rising tolerance, else the bounding rectangle; holes survive only as
rectangles (stairwells, voids over 0.3 m2); pieces under 0.1 m2 are dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from rectilinear_exterior import rectilinear
from wall_surface_cleanup import planes_of

SLAB_KINDS = ('floor', 'ceiling', 'roof')
SKIP = ('stair', 'junction_patch')


def plan_outline(mesh):
    up = mesh.face_normals[:, 2] > .99
    if not up.any():
        return None
    return shapely.union_all(shapely.polygons(np.round(mesh.triangles[up][:, :, :2], 6))).buffer(0)


def square_part(part):
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    if mesh.is_watertight:
        outline = plan_outline(mesh)
        if outline is None:
            return None, None
        squared, report = rectilinear(outline, box_holes=False)
        if squared is None:
            return None, report
        z0, z1 = float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max())
        pieces = []
        for piece in polygon_parts(squared):
            solid = trimesh.creation.extrude_polygon(piece, z1 - z0)
            solid.apply_translation([0, 0, z0])
            pieces.append(solid)
        new = trimesh.util.concatenate(pieces)
        return new, {**report, 'solid': True, 'thickness_m': z1 - z0}
    meshes, reports = [], []
    for plane in planes_of(part):
        if abs(plane['normal'][2]) < .99:
            m = mesh_from_polygon(plane['poly'], plane['origin'], plane['u'], plane['v'])
        else:
            squared, report = rectilinear(plane['poly'], box_holes=False)
            reports.append(report)
            if squared is None:
                continue
            m = mesh_from_polygon(squared, plane['origin'], plane['u'], plane['v'])
        if m is not None:
            meshes.append(m)
    if not meshes:
        return None, None
    new = trimesh.util.concatenate(meshes)
    return new, {'solid': False, 'planes': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    parts = json.loads(Path(args.model).read_text())['parts']
    out_parts, report = [], []
    for part in parts:
        kind = part.get('kind', '')
        if not any(k in kind for k in SLAB_KINDS) or any(k in kind for k in SKIP):
            continue
        if kind in ('floor_single_plane', 'ground_single_plane'):
            continue
        new, rep = square_part(part)
        if new is None:
            report.append({'part': part['name'], 'decision': 'kept', 'report': rep}); continue
        new.merge_vertices(digits_vertex=8)
        new.update_faces(new.nondegenerate_faces(height=1e-8))
        out_parts.append({**{k: v for k, v in part.items() if k not in ('planar_loops',)},
                          'name': part['name'] + ' - rectilinear slab - INFERRED outline',
                          'v': new.vertices.tolist(), 'f': new.faces.tolist(), 'source_group_names': [part['name']],
                          'merge_coplanar_faces': True, 'evidence_status': 'slab_outline_squared_INFERRED',
                          'completion_is_measured': False})
        report.append({'part': part['name'], 'decision': 'squared', 'report': rep})
        print(f"Squared {part['name']}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'slabs_squared': len(out_parts), 'kept': sum(r['decision'] == 'kept' for r in report)}))


if __name__ == '__main__':
    main()
