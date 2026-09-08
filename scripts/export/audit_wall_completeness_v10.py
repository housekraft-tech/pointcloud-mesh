"""Read-only scan-to-native wall audit. Unmatched planes are review candidates, not walls."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output_final/soulace_exterior_v10/coverage'


def wall(part):
    return part.get('kind', '').startswith(('wall', 'parapet'))


def normals(points):
    tree = cKDTree(points)
    normal = np.empty_like(points)
    vertical = np.zeros(len(points), bool)
    for start in range(0, len(points), 30000):
        dist, ids = tree.query(points[start:start + 30000], k=20, workers=8)
        p = points[ids]
        p = p - p.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', p, p) / 20
        values, vectors = np.linalg.eigh(cov)
        n = vectors[:, :, 0]
        normal[start:start + len(n)] = n
        vertical[start:start + len(n)] = ((abs(n[:, 2]) < .15) &
            (values[:, 0] < .00025) & (values[:, 1] > .0002) &
            (values[:, 0] < .12 * values[:, 1]) & (dist[:, -1] < .35))
    return normal, vertical


def model_distances(points, parts):
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    names = []
    for part in parts:
        scene.add_triangles(o3d.core.Tensor(np.asarray(part['v'], np.float32)),
                            o3d.core.Tensor(np.asarray(part['f'], np.uint32)))
        names.append(part['name'])
    distances = []
    nearest = []
    for start in range(0, len(points), 100000):
        p = points[start:start + 100000]
        result = scene.compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))
        distances.append(np.linalg.norm(result['points'].numpy() - p, axis=1))
        nearest.append(result['geometry_ids'].numpy())
    return np.concatenate(distances), np.concatenate(nearest), names


def detect_regions(p, n, vertical, d, nearest, names):
    regions = []
    membership = []
    levels = [(0, .15, 3.1041), (1, 3.3541, 6.4544), (2, 6.7044, 9.8)]
    for level, zmin, zmax in levels:
        for axis in (0, 1):
            tangent = 1 - axis
            eligible = np.flatnonzero(vertical & (abs(n[:, axis]) > .98) &
                (p[:, 2] >= zmin) & (p[:, 2] <= zmax))
            # Detect visible plane locations from the scan, not candidate boxes.
            bins = np.arange(-15, 15.0201, .02)
            hist, _ = np.histogram(p[eligible, axis], bins)
            peaks, _ = find_peaks(ndimage.gaussian_filter1d(hist.astype(float), 1),
                                 distance=4, prominence=35, height=60)
            for peak in peaks:
                c0 = float((bins[peak] + bins[peak + 1]) * .5)
                ids = eligible[abs(p[eligible, axis] - c0) <= .035]
                if len(ids) < 100:
                    continue
                center = float(np.median(p[ids, axis]))
                ids = eligible[abs(p[eligible, axis] - center) <= .035]
                grid = np.floor(p[ids][:, [tangent, 2]] / .1).astype(int)
                origin = grid.min(axis=0)
                local = grid - origin
                occupied = np.zeros(local.max(axis=0) + 1, bool)
                occupied[tuple(local.T)] = True
                # Eight-neighbour cells; no dilation across openings or occlusions.
                labels, _ = ndimage.label(occupied, np.ones((3, 3)))
                point_labels = labels[tuple(local.T)]
                for label in np.unique(point_labels):
                    qids = ids[point_labels == label]
                    if len(qids) < 100:
                        continue
                    q = p[qids]
                    lo, hi = np.min(q, axis=0), np.max(q, axis=0)
                    if hi[tangent] - lo[tangent] < .5 or hi[2] - lo[2] < .7:
                        continue
                    missing = d[qids] > .05
                    missing_grid = np.unique(np.floor(q[missing][:, [tangent, 2]] / .1).astype(int), axis=0)
                    occupancy = int(np.sum(labels == label))
                    missing_area = len(missing_grid) * .01
                    if missing_area < .15:
                        continue
                    unique, counts = np.unique(nearest[qids], return_counts=True)
                    order = np.argsort(counts)[::-1][:3]
                    region = {
                        'id': f'L{level}_{"X" if axis == 0 else "Y"}_{len(regions):03d}',
                        'level': level, 'normal_axis': axis, 'plane_coordinate_m': float(np.median(q[:, axis])),
                        'bounds_m': [lo.tolist(), hi.tolist()], 'sample_points': len(qids),
                        'occupied_10cm_cells': occupancy, 'occupied_grid_area_m2': occupancy * .01,
                        'unmatched_occupied_grid_area_m2': missing_area,
                        'unmatched_points_percent': float(np.mean(missing) * 100),
                        'within_20mm_percent': float(np.mean(d[qids] <= .02) * 100),
                        'within_50mm_percent': float(np.mean(d[qids] <= .05) * 100),
                        'distance_p95_mm': float(np.percentile(d[qids], 95) * 1000),
                        'plane_residual_p95_mm': float(np.percentile(abs(q[:, axis] - np.median(q[:, axis])), 95) * 1000),
                        'nearest_visible_wall_groups': [{'name': names[int(unique[i])], 'sample_count': int(counts[i])} for i in order],
                        'classification': 'coherent_vertical_scan_surface_requires_semantic_review',
                        'warning': 'Grid area is diagnostic occupancy, not measured wall area. Could be a door, glazing, furniture, railing, an offset face, or a wall.'
                    }
                    regions.append(region)
                    membership.append(qids)
    return regions, membership


def render(p, vertical, d, regions, memberships):
    fig, axes = plt.subplots(3, 3, figsize=(20, 17), constrained_layout=True)
    levels = [(0, .15, 3.1041), (1, 3.3541, 6.4544), (2, 6.7044, 9.8)]
    for row, (level, lo, hi) in enumerate(levels):
        ids = np.flatnonzero(vertical & (p[:, 2] >= lo) & (p[:, 2] <= hi))
        ids = ids[::2]
        supported = d[ids] <= .05
        for col, dims in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row, col]
            ax.scatter(p[ids[supported], dims[0]], p[ids[supported], dims[1]], s=.35, c='#a2b3b9', rasterized=True)
            ax.scatter(p[ids[~supported], dims[0]], p[ids[~supported], dims[1]], s=.7, c='#cf4036', rasterized=True)
            for region, membership in zip(regions, memberships):
                if region['level'] != level or region['unmatched_occupied_grid_area_m2'] < .8:
                    continue
                centroid = p[membership].mean(axis=0)
                ax.text(centroid[dims[0]], centroid[dims[1]], region['id'], fontsize=7, color='#26334e',
                        bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=.5))
            ax.set_aspect('equal'); ax.grid(alpha=.15)
            ax.set_title(f'L{level}: {("Plan", "Front elevation", "Side elevation")[col]}')
            ax.set_xlabel('XYZ'[dims[0]] + ' (m)'); ax.set_ylabel('XYZ'[dims[1]] + ' (m)')
    fig.suptitle('Soulace checked native model | independently detected vertical scan surfaces\nRed: >50 mm from a visible wall. Red does not automatically mean a missing wall.', fontsize=17)
    fig.savefig(OUT / 'wall_coverage_multiview.png', dpi=145)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'vertical_evidence.npz'
    model = json.loads((ROOT / 'output_final/soulace_architectural_v9/reopened_visible.build.json').read_text())
    parts = [part for part in model['parts'] if wall(part)]
    if cache.exists():
        data = np.load(cache)
        p, n, vertical, d, nearest = (data[k] for k in ('p', 'n', 'vertical', 'd', 'nearest'))
        names = [part['name'] for part in parts]
    else:
        source = np.load(ROOT / 'output_final/scan_first_diagnostics/soulace/sample.npz')
        p = source['p']
        _, ids = np.unique(np.floor(p / .06).astype(np.int32), axis=0, return_index=True)
        p = p[ids]
        print('normal diagnostic points', len(p), flush=True)
        n, vertical = normals(p)
        print('vertical points', int(np.sum(vertical)), flush=True)
        d, nearest, names = model_distances(p, parts)
        np.savez_compressed(cache, p=p, n=n, vertical=vertical, d=d, nearest=nearest)
    regions, memberships = detect_regions(p, n, vertical, d, nearest, names)
    order = np.argsort([-r['unmatched_occupied_grid_area_m2'] for r in regions])
    regions = [regions[i] for i in order]
    memberships = [memberships[i] for i in order]
    summaries = []
    for level, lo, hi in [(0, .15, 3.1041), (1, 3.3541, 6.4544), (2, 6.7044, 9.8)]:
        m = vertical & (p[:, 2] >= lo) & (p[:, 2] <= hi)
        summaries.append({'level': level, 'visible_wall_groups': sum(x.get('level') == level for x in parts),
            'vertical_diagnostic_sample_count': int(sum(m)),
            'vertical_sample_within_50mm_of_wall_percent': float(np.mean(d[m] <= .05) * 100),
            'note': 'Denominator includes doors, glazing, furnishings, railings and out-of-scope exterior returns; not wall completeness.'})
    report = {'source': 'actual native-visible reopened_visible.build.json',
        'scan': 'sample.npz: every eighth raw return then 25 mm occupied voxels, further 60 mm diagnostic subsample; original points retained',
        'coordinate_frame': 'existing raw L0-to-common scan transform, no fitted realignment',
        'policy': 'Read-only review; no region automatically classified as wall or promoted into model.',
        'level_summary': summaries, 'regions': regions,
        'limitations': ['Cannot certify every wall is present from incomplete scan returns.',
            'Planar surfaces can be furniture, doors, glass, rails, neighbouring buildings or wall faces.',
            'Main-axis candidate extraction does not fully audit diagonal/curved or highly occluded walls.',
            'Occupancy grid area is not an exact measured surface area.']}
    (OUT / 'wall_completeness_audit.json').write_text(json.dumps(report, indent=2))
    render(p, vertical, d, regions, memberships)
    np.savez_compressed(OUT / 'region_memberships.npz', **{r['id']: ids for r, ids in zip(regions, memberships)})
    print(json.dumps(summaries, indent=2), flush=True)
    for r in regions[:30]:
        print(r['id'], 'area', round(r['unmatched_occupied_grid_area_m2'], 2), 'missing%', round(r['unmatched_points_percent'], 1),
              'plane', round(r['plane_coordinate_m'], 3), 'bounds', np.round(r['bounds_m'], 2).tolist(), flush=True)


if __name__ == '__main__':
    main()
