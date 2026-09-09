"""Measure what the exterior refinement added and removed against the raw returns.

For every replacement part, each regularized plane is compared with the same
plane of the source part (same plane key): the added region is sampled on a
grid and the distance from every sample to the nearest raw return is
measured; the removed region likewise. Added surface within 50 mm of a return
was measured surface the model had dropped; added surface farther away is
inferred continuity and is reported as such. Nothing here certifies site
accuracy; it bounds how much of the revision is not measured.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

from select_exterior_walls import load_returns
from wall_surface_cleanup import planes_of


def grid_samples(region, pitch=.05):
    if region.is_empty or region.area < 1e-6:
        return np.zeros((0, 2))
    u0, v0, u1, v1 = region.bounds
    us = np.arange(u0 + pitch / 2, u1, pitch); vs = np.arange(v0 + pitch / 2, v1, pitch)
    if not len(us) or not len(vs):
        return np.array([[region.representative_point().x, region.representative_point().y]])
    uv = np.stack(np.meshgrid(us, vs, indexing='ij'), -1).reshape(-1, 2)
    keep = shapely.contains_xy(region, uv[:, 0], uv[:, 1])
    return uv[keep] if keep.any() else np.array([[region.representative_point().x, region.representative_point().y]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-model', required=True, help='reopened_visible.build.json the patches replace')
    parser.add_argument('--patches', required=True)
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--support-m', type=float, default=.05)
    args = parser.parse_args()
    source = {p['name']: p for p in json.loads(Path(args.previous_model).read_text())['parts']}
    patches = json.loads(Path(args.patches).read_text())['parts']
    tree = cKDTree(load_returns(args.evidence_cache), leafsize=64, compact_nodes=False, balanced_tree=False)
    report, totals = [], {'added_m2': 0., 'added_supported_m2': 0., 'removed_m2': 0., 'removed_supported_m2': 0.}
    for patch in patches:
        before = {tuple(np.round(p['key'], 5)): p for p in planes_of(source[patch['source_group_names'][0]])}
        for plane in planes_of(patch):
            old = before.get(tuple(np.round(plane['key'], 5)))
            if old is None:
                continue
            row = {'wall': patch['source_group_names'][0], 'plane_key': [float(x) for x in plane['key']]}
            for label, region in (('added', plane['poly'].difference(old['poly'])), ('removed', old['poly'].difference(plane['poly']))):
                area = float(region.area)
                if area < 1e-6:
                    row[f'{label}_m2'] = 0.
                    continue
                uv = grid_samples(region)
                xyz = plane['origin'] + uv[:, :1] * plane['u'] + uv[:, 1:] * plane['v']
                distance, _ = tree.query(xyz, workers=-1)
                supported = float(np.mean(distance <= args.support_m))
                row[f'{label}_m2'] = area
                row[f'{label}_scan_supported_fraction'] = supported
                row[f'{label}_median_distance_mm'] = float(np.median(distance) * 1000)
                totals[f'{label}_m2'] += area
                totals[f'{label}_supported_m2'] += area * supported
            if row['added_m2'] or row['removed_m2']:
                report.append(row)
    totals = {k: round(v, 3) for k, v in totals.items()}
    totals['added_inferred_m2'] = round(totals['added_m2'] - totals['added_supported_m2'], 3)
    totals['support_distance_m'] = args.support_m
    totals['site_accuracy_certified'] = False
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'totals': totals, 'planes': report}, indent=2))
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
