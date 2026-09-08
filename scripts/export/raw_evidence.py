"""Shared raw-return evidence for one property, cached once per work folder.

Streams every LAS return of the declared scans into the model frame and keeps
one original return per occupied 10 mm voxel inside the model bounds. The
scans come from either a flow manifest (``configs/architectural_flow/*.json``)
or a plain support list ``{"scans": [{"id", "path", "scan_to_model"}]}``.
The cache records file sizes and mtimes so a stale cache is detected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from run_architectural_flow import read_registered_points, rigid


def scan_list(spec):
    spec = Path(spec).resolve()
    data = json.loads(spec.read_text())
    if data.get('candidate_model'):
        from run_architectural_flow import read_manifest
        manifest = read_manifest(spec)
        matrix = np.asarray(manifest['model_to_building'])
        return [{**s, 'scan_to_model': (matrix @ np.asarray(s['scan_to_model'])).tolist()} for s in manifest['scans']]
    scans = []
    for s in data['scans']:
        path = Path(s['path'])
        if not path.is_absolute():
            path = (spec.parent / path).resolve()
        scans.append({'id': s['id'], 'path': str(path), 'scan_to_model': rigid(s['scan_to_model']).tolist()})
    return scans


def load_evidence(spec, bounds, cache, voxel_m=.01):
    cache = Path(cache)
    scans = scan_list(spec)
    stamp = [[s['id'], Path(s['path']).stat().st_size, Path(s['path']).stat().st_mtime_ns] for s in scans]
    bounds = np.asarray(bounds, float)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        meta = json.loads(str(data['meta']))
        if meta['stamp'] == stamp and meta['voxel_m'] == voxel_m and np.all(np.asarray(meta['bounds'])[0] <= bounds[0] + 1e-9) \
                and np.all(np.asarray(meta['bounds'])[1] >= bounds[1] - 1e-9):
            return data['p'], meta['provenance']
    # Cache a wider box so later revisions that grow by a few centimetres reuse it.
    bounds = np.array([bounds[0] - .7, bounds[1] + .7])
    p, provenance = read_registered_points(scans, bounds, voxel_m=voxel_m)
    meta = {'stamp': stamp, 'voxel_m': voxel_m, 'bounds': bounds.tolist(), 'provenance': provenance}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, p=p.astype(np.float64), meta=json.dumps(meta))
    return p, provenance


def model_bounds(parts, margin=.3):
    xyz = np.vstack([np.asarray(p['v']) for p in parts])
    return np.array([xyz.min(0) - margin, xyz.max(0) + margin])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--model', required=True)
    parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    parts = json.loads(Path(args.model).read_text())['parts']
    p, provenance = load_evidence(args.spec, model_bounds(parts), args.cache)
    print(json.dumps({'points': int(len(p)), 'scans': [(s['id'], s['evidence_points']) for s in provenance]}), flush=True)


if __name__ == '__main__':
    main()
