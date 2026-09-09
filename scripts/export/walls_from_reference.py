"""Add walls from a reference model (drawing-based) where the scan model has none.

The reference build (for example a manually drawn SketchUp export) is
registered to the scan frame by raster cross-correlation of the wall
footprints (rotation in 1-degree steps, optional mirror, translation by
FFT). Every reference wall whose plan is covered less than ``min_cover`` by
the existing wall solids is a candidate; it is added as a solid box between
the level datums only when the raw returns show at least ``min_returns``
within 0.25 m of it at wall height, so nothing is added that the scan
contradicts or never approached. Added walls are tagged INFERRED from the
reference and coloured apart.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy.signal import fftconvolve

from complete_exterior_rectangles import level_datums

REFERENCE_COLOUR = [205, 175, 140]


def wall_lines(part, step=2):
    m = trimesh.Trimesh(part['v'], part['f'], process=False)
    vert = np.abs(m.face_normals[:, 2]) < .05
    if not vert.any():
        return None, m
    t = m.triangles[vert][:, :, :2]
    return shapely.union_all([shapely.LineString(tri[:2]).buffer(.02) for tri in t[::step]]), m


def solid_footprint(fragments):
    polys = []
    for f in fragments:
        for p in json.loads(Path(f).read_text())['parts']:
            m = trimesh.Trimesh(p['v'], p['f'], process=False)
            up = m.face_normals[:, 2] > .99
            if up.any():
                polys.append(shapely.union_all(shapely.polygons(np.round(m.triangles[up][:, :, :2], 6))).buffer(0))
    return shapely.union_all(polys) if polys else shapely.Polygon()


def raster(geom, origin, shape, pitch):
    xs = origin[0] + (np.arange(shape[0]) + .5) * pitch; ys = origin[1] + (np.arange(shape[1]) + .5) * pitch
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return shapely.contains_xy(geom, X.ravel(), Y.ravel()).reshape(shape).astype(float)


def register(reference_walls, target, pitch=.05):
    ref = shapely.union_all([g for g in reference_walls if g is not None])
    b = target.bounds; origin = (b[0] - 3, b[1] - 3)
    shape = (int((b[2] - b[0] + 6) / pitch), int((b[3] - b[1] + 6) / pitch))
    R = raster(target, origin, shape, pitch)
    best = None
    for ang in [*np.arange(-5, 5.1, 1.), 90, 180, 270]:
        for mirror in (False, True):
            g = shapely.affinity.rotate(ref, ang, origin=(0, 0))
            if mirror:
                g = shapely.affinity.scale(g, xfact=-1, yfact=1, origin=(0, 0))
            gb = g.bounds; go = (gb[0], gb[1]); gs = (int((gb[2] - gb[0]) / pitch) + 1, int((gb[3] - gb[1]) / pitch) + 1)
            if gs[0] > R.shape[0] or gs[1] > R.shape[1]:
                continue
            M = raster(g, go, gs, pitch)
            c = fftconvolve(R, M[::-1, ::-1], mode='valid')
            k = np.unravel_index(np.argmax(c), c.shape); score = float(c[k] / max(M.sum(), 1))
            if best is None or score > best['score']:
                best = {'score': score, 'angle': float(ang), 'mirror': bool(mirror),
                        'dx': float(origin[0] + k[0] * pitch - go[0]), 'dy': float(origin[1] + k[1] * pitch - go[1])}
    return best


def transform(geom, reg):
    g = shapely.affinity.rotate(geom, reg['angle'], origin=(0, 0))
    if reg['mirror']:
        g = shapely.affinity.scale(g, xfact=-1, yfact=1, origin=(0, 0))
    return shapely.affinity.translate(g, reg['dx'], reg['dy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='reopened_visible.build.json of the scan model (datums)')
    parser.add_argument('--reference', required=True, help='reference build.json (walls with kind "wall")')
    parser.add_argument('--fragment', action='append', required=True, help='wall solids already built')
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-cover', type=float, default=.6)
    parser.add_argument('--min-returns', type=int, default=3000)
    parser.add_argument('--level', type=int, default=0)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())['parts']
    reference = json.loads(Path(args.reference).read_text())['parts']
    datums = level_datums(model)
    returns = np.load(args.evidence_cache, allow_pickle=True)['p']
    footprint = solid_footprint(args.fragment)
    ref_walls = [(p, *wall_lines(p)) for p in reference if p.get('kind') == 'wall']
    reg = register([g for _, g, _ in ref_walls], footprint)
    print('registration', reg, flush=True)
    floor_z = datums.get(args.level, float(np.min(returns[:, 2])))
    levels = sorted(datums)
    if args.level + 1 in datums:
        ceiling_z = datums[args.level + 1] - .1
    else:
        z = returns[:, 2]; h, e = np.histogram(z[(z > floor_z + 2.) & (z < floor_z + 4.5)], bins=np.arange(floor_z + 2., floor_z + 4.5, .02))
        ceiling_z = float(e[np.argmax(h)]) if h.size and h.max() > 0 else floor_z + 2.8
    cover = footprint.buffer(.15)
    out_parts, report = [], []
    for part, lines, mesh in ref_walls:
        if lines is None:
            continue
        g = transform(lines, reg)
        frac = g.intersection(cover).area / max(g.area, 1e-9)
        # Plan rectangle of the reference wall: its rotated bounding box.
        hull = transform(shapely.MultiPoint(mesh.vertices[:, :2]).convex_hull, reg)
        rect = hull.minimum_rotated_rectangle
        band = (returns[:, 2] > floor_z + .3) & (returns[:, 2] < ceiling_z - .3)
        near = shapely.contains_xy(rect.buffer(.25), returns[band, 0], returns[band, 1]).sum()
        row = {'reference_wall': part['name'], 'covered_fraction': float(frac), 'returns_near': int(near)}
        if frac >= args.min_cover:
            row['decision'] = 'already_in_model'; report.append(row); continue
        if near < args.min_returns:
            row['decision'] = 'no_scan_support'; report.append(row); continue
        z0 = float(mesh.vertices[:, 2].min()); z1 = float(mesh.vertices[:, 2].max())
        height = z1 - z0
        top = ceiling_z if abs(height - (ceiling_z - floor_z)) < .6 else floor_z + height
        solid = trimesh.creation.extrude_polygon(rect, top - floor_z)
        solid.apply_translation([0, 0, floor_z])
        name = f"L{args.level} wall from reference {part['name']} - INFERRED from drawing, scan-supported"
        out_parts.append({'name': name, 'kind': 'wall_solid', 'level': args.level, 'colour': REFERENCE_COLOUR,
                          'v': solid.vertices.tolist(), 'f': solid.faces.tolist(), 'merge_coplanar_faces': True,
                          'evidence_status': 'reference_model_wall_not_in_scan_model_sparse_returns_INFERRED',
                          'completion_is_measured': False})
        row.update({'decision': 'added', 'name': name, 'plan_m2': float(rect.area)})
        report.append(row)
        print(f"added {part['name']}: covered {frac:.2f}, {near} returns near", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps({'registration': reg, 'walls': report}, indent=2))
    print(json.dumps({'added': len(out_parts), 'already_in_model': sum(r['decision'] == 'already_in_model' for r in report),
                      'no_scan_support': sum(r['decision'] == 'no_scan_support' for r in report)}))


if __name__ == '__main__':
    main()
