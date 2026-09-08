"""Read-only wall review for any checked native model: artifacts and coverage.

Two questions, answered separately and never used to edit geometry here:

1. Topology: per wall group and per exact plane, how many tiny islands,
   pores and duplicated triangles exist, and which groups share coplanar
   overlapping faces (the flicker seen on exterior faces).
2. Completeness: which coherent vertical scan surfaces lie more than 50 mm
   from every visible wall. Each region is compared with the nearest wall
   group's own planes; a parallel plane offset by a plausible wall thickness
   is reported as a probable opposite face, still requiring semantic review.

Scan points come either from a level-registered sample (.npz with 'p') or
from the raw LAS files of a flow manifest, streamed with one return per
occupied voxel. Nothing here certifies that a region is a wall.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
import open3d as o3d

from filter_soulace_overlap import polygon_parts

WALL_KINDS = ('wall', 'parapet')
ISLAND_M2 = .02
PORE_M2 = .005


def is_wall(part):
    return part.get('kind', '').startswith(WALL_KINDS)


def plane_buckets(mesh, rotation=None):
    """Exact coplanar buckets in a canonical frame; geometry is not altered."""
    xyz = mesh.vertices if rotation is None else np.round(mesh.vertices @ rotation, 7)
    canon = trimesh.Trimesh(xyz, mesh.faces, process=False)
    buckets = defaultdict(list)
    for i, (normal, centre, area) in enumerate(zip(canon.face_normals, canon.triangles_center, canon.area_faces)):
        if area < 1e-8:
            continue
        normal = normal.copy()
        if normal[np.argmax(np.abs(normal))] < 0:
            normal *= -1
        axis = np.argmax(np.abs(normal))
        if normal[axis] > .99999:
            normal = np.eye(3)[axis]
        buckets[(*np.round(normal, 5), round(float(normal @ centre), 5))].append(i)
    return canon, buckets


def topology(parts, rotations):
    records, patches = [], []
    for part in parts:
        if not is_wall(part):
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        canon, buckets = plane_buckets(mesh, rotations.get(part.get('level', 0)))
        record = {'name': part['name'], 'level': part.get('level', 0), 'kind': part['kind'],
                  'mesh_area_m2': float(mesh.area), 'triangles': len(mesh.faces), 'planes': []}
        for index, (key, ids) in enumerate(buckets.items()):
            normal = np.asarray(key[:3], float); normal /= np.linalg.norm(normal)
            seed = np.eye(3)[np.argmin(np.abs(normal))]
            u = np.cross(normal, seed); u /= np.linalg.norm(u); v = np.cross(normal, u)
            triangles = canon.triangles[ids]
            projected = np.round(np.stack((triangles @ u, triangles @ v), axis=-1), 7)
            shape = shapely.union_all(shapely.polygons(projected)).buffer(0)
            polygons = sorted(polygon_parts(shape), key=lambda p: p.area, reverse=True)
            holes = sorted((Polygon(r).area for p in polygons for r in p.interiors), reverse=True)
            small = [p.area for p in polygons if p.area < ISLAND_M2]
            area_sum = float(canon.area_faces[ids].sum())
            record['planes'].append({
                'index': index, 'normal': normal.tolist(), 'offset': key[3], 'area_m2': float(shape.area),
                'triangles': len(ids), 'duplicate_triangle_area_m2': max(0., area_sum - shape.area),
                'components': len(polygons), 'small_islands': len(small), 'small_islands_area_m2': float(sum(small)),
                'holes': len(holes), 'pores': int(sum(a < PORE_M2 for a in holes)),
                'pores_area_m2': float(sum(a for a in holes if a < PORE_M2)),
                'largest_component_m2': polygons[0].area if polygons else 0.})
            patches.append({'name': part['name'], 'level': part.get('level', 0), 'key': key,
                            'normal': normal, 'shape': shape, 'index': index})
        for field in ('small_islands', 'small_islands_area_m2', 'pores', 'pores_area_m2', 'duplicate_triangle_area_m2'):
            record[field] = float(sum(p[field] for p in record['planes']))
        record['small_islands'] = int(record['small_islands']); record['pores'] = int(record['pores'])
        records.append(record)
    overlaps = []
    for i, a in enumerate(patches):
        for b in patches[i + 1:]:
            if a['name'] == b['name'] or a['level'] != b['level']:
                continue
            if np.linalg.norm(a['normal'] - b['normal']) > 1e-5 or abs(a['key'][3] - b['key'][3]) > .001:
                continue
            area = a['shape'].intersection(b['shape']).area
            if area > PORE_M2:
                overlaps.append({'a': a['name'], 'a_plane': a['index'], 'b': b['name'], 'b_plane': b['index'],
                                 'overlap_area_m2': float(area)})
    return {'wall_groups': len(records), 'small_island_threshold_m2': ISLAND_M2, 'pore_threshold_m2': PORE_M2,
            'total_small_islands': int(sum(r['small_islands'] for r in records)),
            'total_small_islands_area_m2': float(sum(r['small_islands_area_m2'] for r in records)),
            'total_pores': int(sum(r['pores'] for r in records)),
            'total_pores_area_m2': float(sum(r['pores_area_m2'] for r in records)),
            'total_duplicate_triangle_area_m2': float(sum(r['duplicate_triangle_area_m2'] for r in records)),
            'cross_group_coplanar_overlaps': overlaps,
            'cross_group_overlap_area_m2': float(sum(o['overlap_area_m2'] for o in overlaps)),
            'groups': records,
            'notes': ['Topology alone is not evidence that a component is erroneous.',
                      'No gap is classified as a real opening from geometry alone.',
                      'Removal or filling requires the separate raw-scan support audit.']}


def normals(points):
    tree = cKDTree(points)
    normal = np.empty_like(points); vertical = np.zeros(len(points), bool)
    for start in range(0, len(points), 30000):
        dist, ids = tree.query(points[start:start + 30000], k=20, workers=8)
        p = points[ids]; p = p - p.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', p, p) / 20)
        n = vectors[:, :, 0]
        normal[start:start + len(n)] = n
        vertical[start:start + len(n)] = ((abs(n[:, 2]) < .15) & (values[:, 0] < .00025) & (values[:, 1] > .0002)
                                          & (values[:, 0] < .12 * values[:, 1]) & (dist[:, -1] < .35))
    return normal, vertical


def model_distances(points, parts):
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    for part in parts:
        scene.add_triangles(o3d.core.Tensor(np.asarray(part['v'], np.float32)),
                            o3d.core.Tensor(np.asarray(part['f'], np.uint32)))
    distances, nearest = [], []
    for start in range(0, len(points), 100000):
        p = points[start:start + 100000]
        result = scene.compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))
        distances.append(np.linalg.norm(result['points'].numpy() - p, axis=1))
        nearest.append(result['geometry_ids'].numpy())
    return np.concatenate(distances), np.concatenate(nearest)


def level_bands(parts):
    """Wall z-extent per level, trimmed so floor and ceiling returns are excluded."""
    bands = {}
    for level in sorted({p.get('level', 0) for p in parts}):
        z = np.concatenate([np.asarray(p['v'])[:, 2] for p in parts if p.get('level', 0) == level])
        bands[level] = (float(z.min() + .15), float(z.max() - .10))
    return bands


def wall_planes(parts, rotations):
    """Axis-aligned plane coordinates per wall group, in the audit frame."""
    table = {}
    for part in parts:
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        _, buckets = plane_buckets(mesh, rotations.get(part.get('level', 0)))
        rows = []
        for key, ids in buckets.items():
            normal = np.asarray(key[:3]); axis = int(np.argmax(np.abs(normal)))
            if abs(normal[axis]) < .98 or axis == 2:
                continue
            area = float(mesh.area_faces[ids].sum())
            if area < .05:
                continue
            tri = mesh.triangles[ids].reshape(-1, 3)
            rows.append({'axis': axis, 'coordinate': float(key[3] / normal[axis]), 'area_m2': area,
                         'lo': tri.min(0).tolist(), 'hi': tri.max(0).tolist()})
        table[part['name']] = rows
    return table


def classify(region, planes_by_name):
    """Compare a region with the planes of its nearest visible wall groups."""
    axis = region['normal_axis']; tangent = 1 - axis
    lo, hi = np.asarray(region['bounds_m'][0]), np.asarray(region['bounds_m'][1])
    best = None
    for near in region['nearest_visible_wall_groups']:
        for plane in planes_by_name.get(near['name'], []):
            if plane['axis'] != axis:
                continue
            overlap = min(hi[tangent], plane['hi'][tangent]) - max(lo[tangent], plane['lo'][tangent])
            z_overlap = min(hi[2], plane['hi'][2]) - max(lo[2], plane['lo'][2])
            if overlap < .3 or z_overlap < .3:
                continue
            offset = abs(plane['coordinate'] - region['plane_coordinate_m'])
            if best is None or offset < best['offset_m']:
                best = {'wall': near['name'], 'offset_m': float(offset), 'tangent_overlap_m': float(overlap)}
    if best is None:
        return 'no_parallel_visible_wall_plane_nearby', None
    if .06 <= best['offset_m'] <= .40:
        return 'probable_opposite_face_of_visible_wall_thickness_unverified', best
    if best['offset_m'] < .06:
        return 'near_parallel_offset_below_wall_thickness_possible_finish_or_furniture', best
    return 'parallel_surface_beyond_wall_thickness_requires_review', best


def detect_regions(p, n, vertical, d, nearest, names, bands, rotations):
    regions, membership = [], []
    for level, (zmin, zmax) in bands.items():
        rotation = rotations.get(level)
        q_all = p if rotation is None else p @ rotation
        for axis in (0, 1):
            tangent = 1 - axis
            eligible = np.flatnonzero(vertical & (abs(n[:, axis]) > .98) & (p[:, 2] >= zmin) & (p[:, 2] <= zmax))
            if not len(eligible):
                continue
            coords = q_all[eligible, axis]
            bins = np.arange(coords.min() - .05, coords.max() + .07, .02)
            hist, _ = np.histogram(coords, bins)
            peaks, _ = find_peaks(ndimage.gaussian_filter1d(hist.astype(float), 1), distance=4, prominence=35, height=60)
            for peak in peaks:
                c0 = float((bins[peak] + bins[peak + 1]) * .5)
                ids = eligible[abs(q_all[eligible, axis] - c0) <= .035]
                if len(ids) < 100:
                    continue
                center = float(np.median(q_all[ids, axis]))
                ids = eligible[abs(q_all[eligible, axis] - center) <= .035]
                grid = np.floor(q_all[ids][:, [tangent, 2]] / .1).astype(int)
                local = grid - grid.min(axis=0)
                occupied = np.zeros(local.max(axis=0) + 1, bool); occupied[tuple(local.T)] = True
                labels, _ = ndimage.label(occupied, np.ones((3, 3)))
                point_labels = labels[tuple(local.T)]
                for label in np.unique(point_labels):
                    qids = ids[point_labels == label]
                    if len(qids) < 100:
                        continue
                    q = q_all[qids]; lo, hi = q.min(0), q.max(0)
                    if hi[tangent] - lo[tangent] < .5 or hi[2] - lo[2] < .7:
                        continue
                    missing = d[qids] > .05
                    missing_grid = np.unique(np.floor(q[missing][:, [tangent, 2]] / .1).astype(int), axis=0)
                    missing_area = len(missing_grid) * .01
                    if missing_area < .15:
                        continue
                    unique, counts = np.unique(nearest[qids], return_counts=True)
                    order = np.argsort(counts)[::-1][:3]
                    regions.append({
                        'id': f'L{level}_{"X" if axis == 0 else "Y"}_{len(regions):03d}', 'level': level,
                        'normal_axis': axis, 'plane_coordinate_m': float(np.median(q[:, axis])),
                        'bounds_m': [lo.tolist(), hi.tolist()], 'sample_points': int(len(qids)),
                        'occupied_grid_area_m2': float(int(np.sum(labels == label)) * .01),
                        'unmatched_occupied_grid_area_m2': float(missing_area),
                        'unmatched_points_percent': float(np.mean(missing) * 100),
                        'within_50mm_percent': float(np.mean(d[qids] <= .05) * 100),
                        'distance_p95_mm': float(np.percentile(d[qids], 95) * 1000),
                        'plane_residual_p95_mm': float(np.percentile(abs(q[:, axis] - np.median(q[:, axis])), 95) * 1000),
                        'nearest_visible_wall_groups': [{'name': names[int(unique[i])], 'sample_count': int(counts[i])} for i in order]})
                    membership.append(qids)
    return regions, membership


def render(p, vertical, d, regions, memberships, bands, out, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    levels = sorted(bands)
    fig, axes = plt.subplots(len(levels), 3, figsize=(20, 6 * len(levels)), constrained_layout=True, squeeze=False)
    for row, level in enumerate(levels):
        lo, hi = bands[level]
        ids = np.flatnonzero(vertical & (p[:, 2] >= lo) & (p[:, 2] <= hi))[::2]
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
    fig.suptitle(f'{label} | independently detected vertical scan surfaces\n'
                 'Red: >50 mm from a visible wall. Red does not automatically mean a missing wall.', fontsize=17)
    fig.savefig(out / 'wall_coverage_multiview.png', dpi=145)
    plt.close(fig)


def load_scan(args, parts):
    if args.sample:
        data = np.load(args.sample)
        p = data['p']
        provenance = {'sample': str(Path(args.sample).resolve()), 'points': int(len(p))}
    else:
        from run_architectural_flow import read_manifest, read_registered_points
        manifest = read_manifest(args.manifest)
        matrix = np.asarray(manifest['model_to_building'])
        scans = [{**s, 'scan_to_model': (matrix @ np.asarray(s['scan_to_model'])).tolist()} for s in manifest['scans']]
        xyz = np.vstack([p['v'] for p in parts])
        p, provenance = read_registered_points(scans, np.array([xyz.min(0) - .3, xyz.max(0) + .3]), voxel_m=.02)
    _, ids = np.unique(np.floor(p / .06).astype(np.int32), axis=0, return_index=True)
    return p[ids], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='reopened_visible.build.json of a checked native model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--sample', help='npz with p in the model frame')
    parser.add_argument('--manifest', help='flow manifest whose raw LAS scans are streamed instead')
    parser.add_argument('--level-rotations', help='JSON {level: 4x4} common-to-level; rotation used for axis bucketing')
    args = parser.parse_args()
    if bool(args.sample) == bool(args.manifest):
        raise SystemExit('Give exactly one of --sample or --manifest')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    parts = json.loads(Path(args.model).read_text())['parts']
    walls = [p for p in parts if is_wall(p)]
    rotations = {}
    if args.level_rotations:
        rotations = {int(k): np.asarray(v)[:3, :3] for k, v in json.loads(Path(args.level_rotations).read_text()).items()}
    topo = topology(parts, rotations)
    (out / 'wall_topology.json').write_text(json.dumps({'source': str(Path(args.model).resolve()), **topo}, indent=2))
    print(json.dumps({k: v for k, v in topo.items() if k not in ('groups', 'cross_group_coplanar_overlaps', 'notes')}, indent=2), flush=True)

    p, provenance = load_scan(args, parts)
    print('diagnostic points', len(p), flush=True)
    n, vertical = normals(p)
    d, nearest = model_distances(p, walls)
    names = [w['name'] for w in walls]
    bands = level_bands(walls)
    regions, memberships = detect_regions(p, n, vertical, d, nearest, names, bands, rotations)
    order = np.argsort([-r['unmatched_occupied_grid_area_m2'] for r in regions])
    regions = [regions[i] for i in order]; memberships = [memberships[i] for i in order]
    planes = wall_planes(walls, rotations)
    tally = defaultdict(float)
    for region in regions:
        region['classification'], region['nearest_parallel_plane'] = classify(region, planes)
        tally[region['classification']] += region['unmatched_occupied_grid_area_m2']
    summaries = []
    for level, (lo, hi) in bands.items():
        m = vertical & (p[:, 2] >= lo) & (p[:, 2] <= hi)
        summaries.append({'level': level, 'visible_wall_groups': sum(w.get('level', 0) == level for w in walls),
                          'z_band_m': [lo, hi], 'vertical_diagnostic_sample_count': int(m.sum()),
                          'vertical_sample_within_50mm_of_wall_percent': float(np.mean(d[m] <= .05) * 100),
                          'note': 'Denominator includes doors, glazing, furnishings, railings and out-of-scope returns; not wall completeness.'})
    report = {'source': str(Path(args.model).resolve()), 'scan': provenance,
              'policy': 'Read-only review; no region is classified as a wall or promoted into the model here.',
              'level_summary': summaries,
              'unmatched_area_by_classification_m2': dict(tally),
              'regions': regions,
              'limitations': ['Cannot certify every wall is present from incomplete scan returns.',
                              'Planar surfaces can be furniture, doors, glass, rails, neighbouring buildings or wall faces.',
                              'Main-axis extraction does not audit diagonal, curved or heavily occluded walls.',
                              'Occupancy grid area is not a measured surface area.']}
    (out / 'wall_completeness_audit.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / 'region_memberships.npz', p=p, **{r['id']: ids for r, ids in zip(regions, memberships)})
    render(p, vertical, d, regions, memberships, bands, out, args.label)
    print(json.dumps(summaries, indent=2), flush=True)
    print(json.dumps(dict(tally), indent=2), flush=True)
    for r in regions[:25]:
        print(r['id'], 'area', round(r['unmatched_occupied_grid_area_m2'], 2), 'missing%', round(r['unmatched_points_percent'], 1),
              'plane', round(r['plane_coordinate_m'], 3), r['classification'], r['nearest_parallel_plane'], flush=True)


if __name__ == '__main__':
    main()
