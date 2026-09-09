"""Find wall faces the model lacks, directly in the raw returns (declared inference of identity).

A wall face is a vertical surface: in plan, a 25 mm cell whose returns stack
through most of the storey height. Cells like that which lie more than
``clearance`` from every existing wall solid or face block are grouped into
connected runs; a run that is long (>= ``min_length``) and thin (<= 0.2 m)
is a straight face. Each becomes a rectangular face from the lowest to the
highest stacked return, named as recovered from returns, so the solid stage
can pair it or give it an inferred thickness.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy import ndimage

from complete_exterior_rectangles import level_datums
from filter_soulace_overlap import mesh_from_polygon

COLOUR = [221, 190, 110]


def existing_footprint(fragments, model):
    polys = []
    for f in fragments:
        for p in json.loads(Path(f).read_text())['parts']:
            if not p.get('kind', '').startswith(('wall', 'parapet')):
                continue
            m = trimesh.Trimesh(p['v'], p['f'], process=False)
            vert = np.abs(m.face_normals[:, 2]) < .05
            if vert.any():
                t = m.triangles[vert][:, :, :2]
                polys.append(shapely.union_all([shapely.LineString(tri[:2]).buffer(.02) for tri in t[::2]]))
            up = m.face_normals[:, 2] > .99
            if up.any() and m.is_watertight:
                polys.append(shapely.union_all(shapely.polygons(np.round(m.triangles[up][:, :, :2], 6))).buffer(0))
    for p in model:
        if p.get('kind', '').startswith(('wall', 'parapet')):
            m = trimesh.Trimesh(p['v'], p['f'], process=False)
            vert = np.abs(m.face_normals[:, 2]) < .05
            if vert.any():
                t = m.triangles[vert][:, :, :2]
                polys.append(shapely.union_all([shapely.LineString(tri[:2]).buffer(.02) for tri in t[::4]]))
    return shapely.union_all(polys) if polys else shapely.Polygon()


def find_faces(returns, floor_z, ceiling_z, existing, pitch=.025, zbin=.05, min_fraction=.5, clearance=.15,
               min_length=.5, max_width=.2, level=0):
    band = (returns[:, 2] > floor_z + .25) & (returns[:, 2] < ceiling_z - .25)
    pts = returns[band]
    origin = pts[:, :2].min(0) - pitch
    ij = np.floor((pts[:, :2] - origin) / pitch).astype(np.int64)
    zb = np.floor((pts[:, 2] - floor_z) / zbin).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    combined = np.unique(np.stack((key, zb), 1), axis=0)
    cell_keys, stacks = np.unique(combined[:, 0], return_counts=True)
    height_bins = (ceiling_z - floor_z - .5) / zbin
    solid_cells = cell_keys[stacks >= min_fraction * height_bins]
    if not len(solid_cells):
        return []
    ii, jj = solid_cells // 1_000_003, solid_cells % 1_000_003
    shape = (ii.max() + 2, jj.max() + 2)
    grid = np.zeros(shape, bool); grid[ii, jj] = True
    centres = origin + (np.stack((ii, jj), 1) + .5) * pitch
    keep = ~shapely.contains_xy(existing.buffer(clearance), centres[:, 0], centres[:, 1]) if not existing.is_empty else np.ones(len(ii), bool)
    grid[:] = False; grid[ii[keep], jj[keep]] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3), bool))
    labels, count = ndimage.label(grid, np.ones((3, 3), bool))
    faces = []
    for k in range(1, count + 1):
        cells = np.argwhere(labels == k)
        if len(cells) < 20:
            continue
        xy = origin + (cells + .5) * pitch
        centre = xy.mean(0)
        u, s, vt = np.linalg.svd(xy - centre, full_matrices=False)
        along, across = vt[0], vt[1]
        a = (xy - centre) @ along; c = (xy - centre) @ across
        length = float(a.max() - a.min()); width = float(np.percentile(c, 95) - np.percentile(c, 5))
        if length < min_length or width > max_width:
            continue
        # Returns of this run give its height range.
        inside = (np.abs((pts[:, :2] - centre) @ across) <= max_width) & ((pts[:, :2] - centre) @ along >= a.min()) & ((pts[:, :2] - centre) @ along <= a.max())
        if inside.sum() < 50:
            continue
        z0, z1 = float(np.percentile(pts[inside, 2], 2)), float(np.percentile(pts[inside, 2], 98))
        z0 = floor_z if z0 - floor_z < .5 else z0
        z1 = ceiling_z if ceiling_z - z1 < .5 else z1
        normal = np.array([across[0], across[1], 0.])
        corners = [centre + along * a.min(), centre + along * a.max()]
        v = np.array([[corners[0][0], corners[0][1], z0], [corners[1][0], corners[1][1], z0],
                      [corners[1][0], corners[1][1], z1], [corners[0][0], corners[0][1], z1]])
        faces.append({'name': f'L{level} recovered wall face {len(faces):02d} from returns - identity INFERRED',
                      'kind': 'wall_measured', 'level': level, 'colour': COLOUR,
                      'v': v.tolist(), 'f': [[0, 1, 2], [0, 2, 3]], 'merge_coplanar_faces': True,
                      'evidence_status': 'vertical_return_stack_without_model_face_identity_INFERRED',
                      'completion_is_measured': True, 'length_m': length, 'width_m': width, 'returns': int(inside.sum())})
    return faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--fragment', action='append', default=[], help='existing wall fragments (blocks, solids) to keep clear of')
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())['parts']
    datums = level_datums(model)
    returns = np.load(args.evidence_cache, allow_pickle=True)['p']
    existing = existing_footprint(args.fragment, model)
    levels = sorted(datums)
    out_parts, report = [], []
    for k, level in enumerate(levels):
        floor_z = datums[level]
        ceiling_z = datums[levels[k + 1]] - .1 if k + 1 < len(levels) else None
        if ceiling_z is None:
            z = returns[:, 2]; h, e = np.histogram(z[(z > floor_z + 2.) & (z < floor_z + 4.5)], bins=np.arange(floor_z + 2., floor_z + 4.5, .02))
            ceiling_z = float(e[np.argmax(h)]) if h.size and h.max() > 0 else floor_z + 2.8
        faces = find_faces(returns, floor_z, ceiling_z, existing, level=level)
        for f in faces:
            report.append({k: f[k] for k in ('name', 'length_m', 'width_m', 'returns')})
            out_parts.append({k: v for k, v in f.items() if k not in ('length_m', 'width_m', 'returns')})
        print(f'level {level}: floor {floor_z:.3f} ceiling {ceiling_z:.3f}: {len(faces)} recovered faces, {sum(f["length_m"] for f in faces):.1f} m', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'faces': len(out_parts)}))


if __name__ == '__main__':
    main()
