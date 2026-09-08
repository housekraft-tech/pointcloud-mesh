"""Project-independent, LiDAR-measured rectilinear CAD candidate rebuild.

This is NOT a raw-cloud room detector. Input CAD supplies hypotheses and opening
boundaries. LAS supplies plane positions and support. Missing returns are not
declared free space; small continuity fills are audited, never called measured.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import laspy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import contains_xy, union_all
from shapely.geometry import Polygon, box
from shapely.ops import transform as transform_polygon
import trimesh


@dataclass(frozen=True)
class Settings:
    allow_continuity: bool = True
    target_m: float = .010
    support_m: float = .050
    grid_m: float = .050
    association_m: float = .050
    fit_grid_m: float = .100
    max_fit_shift_m: float = .050
    min_patch_area_m2: float = .020
    min_rectangle_area_m2: float = .080
    min_rectangle_extent_m: float = .040
    min_rectangle_support: float = .80
    max_rectangles_per_patch: int = 32
    close_cells: int = 1
    normal_angle_deg: float = 20.0
    max_normal_variation: float = .15
    fit_min_votes: int = 20
    max_axis_plane_spread_m: float = .020
    seed: int = 9210

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Settings must be finite')
        if not 0 < self.target_m <= self.support_m <= .050:
            raise ValueError('Target <= support <= 50 mm is required')
        if not 0 < self.association_m <= self.support_m:
            raise ValueError('Association must be within the support envelope')
        if not 0 < self.grid_m <= .050 or not 0 < self.max_fit_shift_m <= .050:
            raise ValueError('Grid and permitted shift must be <= 50 mm')
        if not 0 <= self.close_cells <= 1:
            raise ValueError('Continuity closing is limited to one support cell')
        if not .8 <= self.min_rectangle_support <= 1:
            raise ValueError('Rectangle support must be at least 80%')
        if min(self.min_patch_area_m2, self.min_rectangle_area_m2, self.min_rectangle_extent_m, self.fit_grid_m) <= 0:
            raise ValueError('Positive patch, rectangle and fitting resolutions required')
        if self.fit_min_votes < 3 or self.max_rectangles_per_patch < 1:
            raise ValueError('Insufficient fitting votes or rectangle budget')


def say(s):
    print(s, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8_000_000), b''):
            h.update(chunk)
    return h.hexdigest()


def rigid(matrix):
    a = np.asarray(matrix, float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError('Invalid transform')
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid homogeneous transform')
    if not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(a[:3, :3]), 1):
        raise ValueError('Transform must be rigid: no scaling or reflection')
    return a


def polys(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == 'Polygon':
        return [shape]
    return [p for g in shape.geoms for p in polys(g)]


def extract_patches(payload, cfg):
    """Only axis-compatible hypotheses; never flatten oblique seed geometry."""
    patches, exclusions = [], []
    for item in payload['parts']:
        m = trimesh.Trimesh(item['v'], item['f'], process=False)
        if not np.isfinite(m.vertices).all():
            raise ValueError('Non-finite candidate coordinates')
        groups = defaultdict(list)
        for i, (n, area) in enumerate(zip(m.face_normals, m.area_faces)):
            if area < 1e-10:
                continue
            axis = int(np.argmax(abs(n)))
            # Caller provides the rectilinear frame; do not rotate a real slope.
            if abs(n[axis]) < 1-1e-7:
                exclusions.append({'object': item['name'], 'triangle': i,
                                   'reason': 'non_rectilinear_candidate', 'area_m2': float(area)})
                continue
            d = float(m.triangles[i, :, axis].mean())
            groups[(axis, round(d, 6))].append(i)
        for (axis, d), ids in sorted(groups.items()):
            other = [k for k in range(3) if k != axis]
            polygon = union_all([Polygon(t[:, other]) for t in m.triangles[ids]]).buffer(0)
            if polygon.area < cfg.min_patch_area_m2:
                exclusions.append({'object': item['name'], 'reason': 'small_candidate_patch', 'area_m2': float(polygon.area)})
                continue
            patches.append({'id': f'p{len(patches):04d}', 'object': item['name'],
                            'kind': item['kind'], 'axis': axis, 'axes': other,
                            'original_d': d, 'd': d, 'polygon': polygon})
    return patches, exclusions


def grid(polygon, step):
    """Candidate boundary coordinates are grid lines, preserving seed openings."""
    rings = [np.asarray(r.coords) for p in polys(polygon) for r in [p.exterior, *p.interiors]]
    coords = np.vstack(rings)
    lo, hi = np.array(polygon.bounds[:2]), np.array(polygon.bounds[2:])
    edges = []
    for k in range(2):
        values = np.r_[np.arange(lo[k], hi[k], step), coords[:, k], hi[k]]
        values = np.unique(np.round(values, 7))
        # Avoid pathological nearly-coincident lines without changing endpoints.
        values = values[np.r_[True, np.diff(values) > 1e-6]]
        edges.append(values)
    xs, ys = edges
    x, y = np.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2)
    domain = contains_xy(polygon, x, y)
    area = np.diff(ys)[:, None]*np.diff(xs)[None, :]
    return xs, ys, np.stack((x, y), -1), domain, area


def xyz_at(xy, patch, d=None):
    out = np.empty((len(xy), 3))
    out[:, patch['axis']] = patch['d'] if d is None else d
    out[:, patch['axes']] = xy
    return out


def load_reference(scan, base, bounds):
    path = (base/scan['path']).resolve()
    matrix = rigid(scan['scan_to_model'])
    chunks, total = [], 0
    with laspy.open(path) as reader:
        for c in reader.chunk_iterator(2_000_000):
            q = np.column_stack((c.x, c.y, c.z)) @ matrix[:3, :3].T + matrix[:3, 3]
            total += len(q)
            if not np.isfinite(q).all():
                raise ValueError(f'Non-finite LAS: {path}')
            inside = np.all((q >= bounds[0]) & (q <= bounds[1]), axis=1)
            chunks.append(q[inside])
    points = np.concatenate(chunks)
    if not len(points):
        raise ValueError(f'Empty reference crop; check transform for {path}')
    say(f'{scan["id"]}: {total:,} input returns; {len(points):,} retained in candidate ROI')
    return {'id': scan['id'], 'points': points, 'tree': cKDTree(points, leafsize=32, compact_nodes=False),
            'provenance': {'id': scan['id'], 'path': str(path), 'sha256': sha(path),
                           'input_points': total, 'roi_points': len(points), 'scan_to_model': matrix.tolist()}}


def normal_votes(reference, query, axis, cfg):
    values = []
    for start in range(0, len(query), 10_000):
        q = query[start:start+10_000]
        distance, ids = reference['tree'].query(q, k=24, workers=8)
        near = distance[:, 0] <= cfg.association_m
        q, ids = q[near], ids[near]
        if not len(q):
            continue
        neighbors = reference['points'][ids]
        centred = neighbors-neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', centred, centred)/24
        eigen, vectors = np.linalg.eigh(covariance)
        aligned = abs(vectors[:, axis, 0]) >= np.cos(np.deg2rad(cfg.normal_angle_deg))
        planar = eigen[:, 0]/np.maximum(eigen.sum(axis=1), 1e-12) <= cfg.max_normal_variation
        values.extend(neighbors[aligned & planar, 0, axis].tolist())
    return np.asarray(values)


def robust_position(votes, candidate, cfg):
    """Dominant near-seed depth mode, then median; never pool opposite faces."""
    v = np.asarray(votes)
    v = v[abs(v-candidate) <= cfg.association_m]
    if len(v) < cfg.fit_min_votes:
        return None
    histogram, edges = np.histogram(v-candidate, bins=np.arange(-cfg.association_m, cfg.association_m+.002, .002))
    smooth = ndimage.gaussian_filter1d(histogram.astype(float), 1)
    mode = candidate + (edges[np.argmax(smooth)]+edges[np.argmax(smooth)+1])/2
    inliers = v[abs(v-mode) <= .015]
    if len(inliers) < cfg.fit_min_votes:
        return None
    position = float(np.median(inliers))
    if abs(position-candidate) > cfg.max_fit_shift_m:
        return None
    return {'position_m': position, 'votes': len(v), 'mode_votes': len(inliers),
            'all_associated_p95_residual_mm': float(np.percentile(abs(v-position), 95)*1000),
            'mode_mad_mm': float(np.median(abs(inliers-position))*1000)}


def fit_planes(patches, references, cfg):
    positions = defaultdict(list)
    for i, p in enumerate(patches):
        _, _, xy, domain, _ = grid(p['polygon'], cfg.fit_grid_m)
        query = xyz_at(xy[domain], p)
        fits = {}
        for ref in references:
            fit = robust_position(normal_votes(ref, query, p['axis'], cfg), p['d'], cfg)
            if fit:
                fits[ref['id']] = fit
        p['fits'] = fits
        if fits:
            values = [v['position_m'] for v in fits.values()]
            p['scan_disagreement_mm'] = (max(values)-min(values))*1000
            # Disagreement is flagged, not corrected by another scan registration.
            p['proposal_d'] = float(np.median(values))
            positions[(p['axis'], p['original_d'])].append((p['proposal_d'], p['polygon'].area))
        if i % 25 == 0:
            say(f'Plane fit {i+1}/{len(patches)}')
    maps = [{d: d for axis, d in {(p['axis'], p['original_d']) for p in patches} if axis == k} for k in range(3)]
    conflicts = []
    for (axis, old), votes in positions.items():
        v = np.array([a for a, _ in votes]); weight = np.array([w for _, w in votes])
        if np.ptp(v) > cfg.max_axis_plane_spread_m:
            conflicts.append({'axis': axis, 'coordinate_m': old, 'reason': 'coplanar_fit_disagreement', 'spread_mm': float(np.ptp(v)*1000)})
            continue
        order = np.argsort(v)
        maps[axis][old] = float(v[order[np.searchsorted(np.cumsum(weight[order]), weight.sum()/2)]])
    # Shared coordinate constraints cannot invert a wall or collapse a thin part.
    for axis, mapping in enumerate(maps):
        keys = sorted(mapping)
        while True:
            bad = [(a, b) for a, b in zip(keys, keys[1:]) if mapping[b]-mapping[a] < min((b-a)*.2, .001)]
            if not bad:
                break
            changed = False
            for a, b in bad:
                if mapping[a] != a or mapping[b] != b:
                    conflicts.append({'axis': axis, 'coordinates_m': [a, b], 'reason': 'fit_would_invert_or_collapse_interval'})
                    mapping[a], mapping[b] = a, b
                    changed = True
            if not changed:
                raise ValueError('Unresolvable coordinate ordering')
    for p in patches:
        p['d'] = maps[p['axis']][p['original_d']]
        # Interpolate DISPLACEMENT rather than clamp outside the plane map.
        def warp_delta(x, y, z=None):
            result = []
            for values, axis in zip((x, y), p['axes']):
                keys = sorted(maps[axis])
                delta = [maps[axis][k]-k for k in keys]
                result.append(np.asarray(values)+(np.interp(values, keys, delta, left=0, right=0) if keys else 0))
            return tuple(result)
        p['polygon'] = transform_polygon(warp_delta, p['polygon'])
    return conflicts, maps


def largest_rectangle(mask, xs, ys):
    """Maximum physical area rectangle in a variable-width binary grid."""
    heights = np.zeros(mask.shape[1])
    best = (0., 0, 0, 0, 0)
    for y in range(mask.shape[0]):
        heights = np.where(mask[y], heights+ys[y+1]-ys[y], 0)
        stack = []
        for x in range(mask.shape[1]+1):
            h = heights[x] if x < mask.shape[1] else 0
            start = x
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                area = sh*(xs[x]-xs[sx])
                if area > best[0]+1e-12:
                    y0 = int(np.argmin(abs(ys-(ys[y+1]-sh))))
                    best = (float(area), y0, y+1, sx, x)
                start = sx
            stack.append((start, h))
    return best


def rectangle_partition(support, domain, xs, ys, cfg):
    """Bounded continuity + few large rectangles; source holes are hard vetoes."""
    weights = np.diff(ys)[:, None]*np.diff(xs)[None, :]
    mask = support & domain
    if cfg.allow_continuity and cfg.close_cells:
        padded = np.pad(mask, cfg.close_cells, mode='edge')
        filled = ndimage.binary_closing(padded, structure=np.ones((3, 3)), iterations=cfg.close_cells)
        mask = (mask | filled[cfg.close_cells:-cfg.close_cells, cfg.close_cells:-cfg.close_cells]) & domain
    output = []
    for _ in range(cfg.max_rectangles_per_patch):
        area, y0, y1, x0, x1 = largest_rectangle(mask, xs, ys)
        if area < cfg.min_rectangle_area_m2:
            break
        mask[y0:y1, x0:x1] = False
        w = weights[y0:y1, x0:x1]
        fraction = float((w*support[y0:y1, x0:x1]).sum()/w.sum())
        if min(xs[x1]-xs[x0], ys[y1]-ys[y0]) < cfg.min_rectangle_extent_m or fraction < cfg.min_rectangle_support:
            continue
        output.append({'bounds': [float(xs[x0]), float(ys[y0]), float(xs[x1]), float(ys[y1])],
                       'grid_supported_fraction': fraction, 'grid_unknown_area_m2': float(w.sum()*(1-fraction))})
    return output


def distribution(distances, weights=None):
    d = np.asarray(distances)
    w = np.ones(len(d)) if weights is None else np.asarray(weights)
    if not len(d):
        return None
    order = np.argsort(d); cumulative = np.cumsum(w[order])/w.sum()
    return {'median_mm': float(d[order[np.searchsorted(cumulative, .5)]]*1000),
            'p95_mm': float(d[order[np.searchsorted(cumulative, .95)]]*1000),
            'max_sample_mm': float(d.max()*1000),
            **{f'within_{t}mm_pct': float(w[d <= t/1000].sum()/w.sum()*100) for t in [10, 20, 30, 50]}}


def run(manifest_path, output):
    started = time.time()
    manifest_path = Path(manifest_path).resolve(); base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    cfg = Settings(**manifest.get('settings', {})); cfg.validate()
    if manifest.get('units') != 'metres':
        raise ValueError('Explicit units=metres required')
    output = Path(output).resolve()
    if (output/'model.build.json').exists():
        raise FileExistsError('Refusing to overwrite existing reconstruction')
    output.mkdir(parents=True, exist_ok=True)
    model_path = (base/manifest['candidate_model']).resolve()
    payload = json.loads(model_path.read_text())
    patches, exclusions = extract_patches(payload, cfg)
    vertices = np.concatenate([p['v'] for p in payload['parts']])
    bounds = [vertices.min(axis=0)-.2, vertices.max(axis=0)+.2]
    references = [load_reference(s, base, bounds) for s in manifest['scans']]
    conflicts, maps = fit_planes(patches, references, cfg)
    rng = np.random.default_rng(cfg.seed)
    parts, records, sample_d, sample_weights = [], [], [], []
    display = []
    for ref in references:
        n = min(160_000//len(references), len(ref['points']))
        display.append(ref['points'][rng.choice(len(ref['points']), n, replace=False)])
    np.savez_compressed(output/'scan_display.npz', points=np.vstack(display))
    for i, p in enumerate(patches):
        record = {k: v for k, v in p.items() if k != 'polygon'}
        record['candidate_area_m2'] = float(p['polygon'].area)
        if not p['fits']:
            record.update(status='rejected_no_plane_evidence', rectangles=[])
            records.append(record); continue
        xs, ys, xy, domain, weights = grid(p['polygon'], cfg.grid_m)
        distances = np.full(domain.shape, np.inf)
        q = xyz_at(xy[domain], p)
        distances[domain] = np.min([ref['tree'].query(q, workers=8)[0] for ref in references], axis=0)
        if cfg.allow_continuity:
            support = distances <= cfg.support_m
        else:
            radius = .5*np.sqrt(np.diff(ys)[:,None]**2+np.diff(xs)[None,:]**2)
            # Nearest-set distance is 1-Lipschitz: bound the ENTIRE cell.
            # Rectangles are exact unions of admitted cells, without filling.
            support = distances+radius <= cfg.support_m-.000002
        rectangles = rectangle_partition(support, domain, xs, ys, cfg)
        record['candidate_grid_support_pct'] = float((weights*support).sum()/(weights*domain).sum()*100)
        kept = []
        for r in rectangles:
            x0, y0, x1, y1 = r['bounds']
            poly = box(x0, y0, x1, y1)
            if poly.difference(p['polygon']).area > 1e-7:
                # For non-orthogonal candidate boundaries, do not grow a rectangle.
                continue
            count = max(300, int(poly.area*1500))
            uv = rng.uniform([x0, y0], [x1, y1], (count, 2))
            sample = xyz_at(uv, p)
            ds = {ref['id']: ref['tree'].query(sample, workers=8)[0] for ref in references}
            nearest = np.min(list(ds.values()), axis=0)
            support_fraction = float(np.mean(nearest <= cfg.support_m))
            if support_fraction < cfg.min_rectangle_support:
                continue
            corners = xyz_at(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]), p)
            # Probe edges separately; this does not certify unsampled interiors.
            edge_uv = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/.01)+1)) for a,b in zip(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]),np.array([[x1,y0],[x1,y1],[x0,y1],[x0,y0]]))])
            edge_q = xyz_at(edge_uv, p)
            boundary_max = float(np.min([ref['tree'].query(edge_q, workers=8)[0] for ref in references],axis=0).max()*1000)
            if not cfg.allow_continuity and (nearest.max() > cfg.support_m+1e-7 or boundary_max > cfg.support_m*1000+.0001):
                raise ValueError('Strict rectangle failed exported surface check')
            status = 'regularized_continuity' if np.any(nearest > cfg.support_m) or boundary_max > cfg.support_m*1000 else 'supported_samples'
            if not cfg.allow_continuity:
                status = 'bounded_scan_support'
            name = f'{p["object"]}__{p["id"]}__r{len(kept):02d}'
            colour = [228,196,132] if status == 'regularized_continuity' else [174,202,211]
            if p['kind'] == 'floor':
                colour = [207,200,187] if status == 'regularized_continuity' else [190,207,201]
            parts.append({'name': name, 'kind': p['kind'], 'source_object': p['object'],
                          'v': np.round(corners,8).tolist(), 'f': [[0,1,2],[0,2,3]], 'quads': [[0,1,2,3]],
                          'colour': colour, 'evidence_status': status, 'level': manifest.get('level',0),
                          'support_cutoff_mm': cfg.support_m*1000, 'supported_sample_pct':support_fraction*100})
            r.update(name=name, area_m2=float(poly.area), evidence_status=status,
                     support=distribution(nearest), per_scan={key:distribution(v) for key,v in ds.items()},
                     boundary_max_mm=boundary_max, samples=count)
            kept.append(r); sample_d.append(nearest); sample_weights.append(np.full(count,poly.area/count))
        record.update(rectangles=kept, status='retained' if kept else 'rejected_no_coherent_rectangle')
        records.append(record)
        if i%20 == 0:
            say(f'Rectangle rebuild {i+1}/{len(patches)}; {len(parts)} quads')
    if not parts:
        raise ValueError('No supported rectangles; input frame or candidates may be wrong')
    summary = {'source_objects': len(payload['parts']), 'candidate_patches':len(patches),
               'retained_source_objects':len({p['source_object'] for p in parts}), 'rectangles':len(parts),
               'retained_area_m2':float(np.sum(np.concatenate(sample_weights))),
               'area_weighted_scan_support':distribution(np.concatenate(sample_d),np.concatenate(sample_weights)),
               'continuity_rectangles':sum(p['evidence_status']=='regularized_continuity' for p in parts),
               'rejected_patches':sum(not p['rectangles'] for p in records),
               'elapsed_seconds':time.time()-started}
    audit = {'schema':'rectangular-rebuild-v1', 'label':manifest['label'], 'settings':asdict(cfg),
             'manifest':manifest, 'candidate_model_sha256':sha(model_path),
             'references':[r['provenance'] for r in references], 'reference_crop_bounds':np.asarray(bounds).tolist(),
             'summary':summary, 'plane_conflicts':conflicts, 'excluded_candidate_geometry':exclusions,
             'coordinate_maps':[{str(k):v for k,v in m.items()} for m in maps], 'patches':records,
             'strict_cell_distance_bound_mm':cfg.support_m*1000 if not cfg.allow_continuity else None,
             'limitations':['Candidate-guided; no discovery of entirely missing rooms or openings.',
                            'No trajectory/free-space or RGB door classification in this stage.',
                            'Missing returns are unknown, not proof of an opening.',
                            ('Continuity rectangles contain inferred area beyond 50 mm; not strict overlap-only.' if cfg.allow_continuity else 'Strict rectangles remain within a conservative nearest-return envelope; this does not establish wall identity.'),
                            'Distances are model-to-scan agreement, not independent dimensional accuracy.',
                            'Only axis-aligned candidates in a declared common frame are processed.']}
    (output/'model.build.json').write_text(json.dumps({'label':manifest['label'],'parts':parts}))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    say(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.manifest,args.out)
