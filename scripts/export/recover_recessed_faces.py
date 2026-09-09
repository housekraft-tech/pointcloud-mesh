"""Recover recessed and stepped faces the model lacks behind exterior voids.

Inside the rectangle of every exterior wall plane, a void where the raw
returns show a surface 50-300 mm off the plane is a recess, a step or another
face. When the model already carries a surface within 30 mm of those returns
nothing is done. Otherwise the returns are fitted with a plane parallel to
the wall (the dominant offset), the cells they cover are meshed and the face
is added as an observed vertical face with unverified identity, in the same
amber convention as observed_wall_faces.py: measured surface, no thickness,
no opening, no back face.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from regularize_wall_surfaces import load_selection, plane_selection, regularize
from wall_surface_cleanup import planes_of, scene_of, distances

COLOUR = [221, 190, 110]   # wall colour: identity is still unverified in the name and evidence status


def dominant_offset(offsets, bin_m=.01):
    edges = np.arange(offsets.min(), offsets.max() + bin_m, bin_m)
    if len(edges) < 2:
        return float(np.median(offsets))
    counts, edges = np.histogram(offsets, edges)
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2)


def recover(parts, selection, returns, scene, min_void_m2=.3, band_m=.35, fit_band_m=.015,
            min_returns=200, represented_m=.03, pitch=.025, min_face_m2=.1):
    faces, report = [], []
    for part in parts:
        if part['name'] not in selection:
            continue
        for plane_index, plane in enumerate(planes_of(part)):
            chosen, mask = plane_selection(selection[part['name']], plane)
            if not chosen or mask is not None or abs(plane['normal'][2]) > .01 or plane['poly'].area < .5:
                continue
            n = np.asarray(plane['normal']); offset = plane['offset']
            signed = returns @ n - offset
            near = np.abs(signed) < band_m
            pts, d = returns[near], signed[near]
            uv = np.stack(((pts - plane['origin']) @ plane['u'], (pts - plane['origin']) @ plane['v']), 1)
            main = [p for p in polygon_parts(plane['poly']) if p.area >= .1] or list(polygon_parts(plane['poly']))
            frame = shapely.box(*shapely.union_all(main).bounds)
            for index, void in enumerate(polygon_parts(frame.difference(plane['poly']))):
                if void.area < min_void_m2:
                    continue
                inside = shapely.contains_xy(void, uv[:, 0], uv[:, 1])
                if inside.sum() < min_returns:
                    continue
                dv = d[inside]
                if (np.abs(dv) < .05).mean() > .5:
                    continue                      # in-plane surface: the regularizer recovers it
                peak = dominant_offset(dv)
                if abs(peak) < .05:
                    continue
                on_face = inside.copy(); on_face[inside] = np.abs(dv - peak) < fit_band_m
                count = int(on_face.sum())
                if count < min_returns:
                    continue
                sample = pts[on_face]
                probe = sample[np.random.default_rng(0).choice(len(sample), min(2000, len(sample)), replace=False)]
                represented = float((distances(scene, probe) < represented_m).mean())
                row = {'wall': part['name'], 'void_m2': float(void.area), 'offset_mm': round(peak * 1000),
                       'returns_on_face': count, 'represented_fraction': represented}
                if represented > .7:
                    row['decision'] = 'already_in_model'; report.append(row); continue
                cell_uv = uv[on_face]
                keys, counts = np.unique(np.floor(cell_uv / pitch).astype(np.int64), axis=0, return_counts=True)
                keys = keys[counts >= 2]
                if not len(keys):
                    row['decision'] = 'too_sparse'; report.append(row); continue
                cells = shapely.union_all(shapely.box(keys[:, 0] * pitch, keys[:, 1] * pitch,
                                                      (keys[:, 0] + 1) * pitch, (keys[:, 1] + 1) * pitch))
                region = cells.buffer(.025, join_style=2).buffer(-.025, join_style=2).intersection(void.buffer(.05, join_style=2))
                region = shapely.union_all([p for p in polygon_parts(region) if p.area >= min_face_m2])
                if region.is_empty:
                    row['decision'] = 'too_fragmented'; report.append(row); continue
                # Same straightening as the walls: notch fill, gap fill, axis snap.
                straight, straightening = regularize(region)
                if straightening['accepted']:
                    region = straight
                    row['straightened'] = {k: round(v, 3) for k, v in straightening.items() if isinstance(v, float)}
                origin = plane['origin'] + n * peak
                mesh = mesh_from_polygon(region, origin, plane['u'], plane['v'])
                if mesh is None:
                    row['decision'] = 'no_mesh'; report.append(row); continue
                name = f"L{part.get('level', 0)} recessed face behind {part['name'].split(' - ')[0]} p{plane_index}v{index} - identity unverified"
                faces.append({'name': name, 'kind': 'wall_face_observed', 'level': part.get('level', 0),
                              'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(), 'colour': COLOUR,
                              'merge_coplanar_faces': True,
                              'evidence_status': 'returns_within_15mm_of_parallel_plane_behind_exterior_void_not_in_model',
                              'completion_is_measured': True})
                row.update({'decision': 'built_as_observed_face', 'name': name, 'area_m2': float(region.area)})
                report.append(row)
                print(f'Recovered {name}: {region.area:.2f} m2 at {peak * 1000:.0f} mm', flush=True)
    return faces, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    parts = json.loads(Path(args.model).read_text())['parts']
    selection = load_selection(args.selection)
    returns = np.load(args.evidence_cache, allow_pickle=True)['p']
    scene = scene_of(parts)
    faces, report = recover(parts, selection, returns, scene)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'recessed_faces.build.json').write_text(json.dumps({'parts': faces}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'faces': len(faces), 'area_m2': round(sum(r.get('area_m2', 0) for r in report), 2),
                      'already_in_model': sum(r['decision'] == 'already_in_model' for r in report)}))


if __name__ == '__main__':
    main()
