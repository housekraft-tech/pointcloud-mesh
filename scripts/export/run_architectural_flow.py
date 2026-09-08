"""Property-independent registration, architectural refinement and QA runner.

Input manifests declare CAD hypotheses, raw LAS registrations, and level frames.
An existing isolated-scan reconstruction supplies hypotheses; raw LAS validates
their supported surfaces. Native export is a separate optional final stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def rigid(matrix):
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('A finite 4 x 4 transform is required')
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError('Invalid transform last row')
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('Registration may not scale or shear survey coordinates')
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError('Registration may not reflect survey coordinates')
    return value


def transform_part(part, matrix, name=None, level=None):
    matrix = rigid(matrix)
    vertices = np.asarray(part['v'], dtype=float)
    result = {**part, 'v': (vertices @ matrix[:3, :3].T + matrix[:3, 3]).tolist()}
    if name is not None:
        result['name'] = name
    if level is not None:
        result['level'] = level
    if part.get('wall_plane_pair'):
        pairs = []
        for plane in part['wall_plane_pair']:
            normal = matrix[:3, :3] @ np.asarray(plane['normal'], dtype=float)
            pairs.append({**plane, 'normal': normal.tolist(),
                          'offset_m': float(plane['offset_m'] + normal @ matrix[:3, 3])})
        result['wall_plane_pair'] = pairs
    return result


def validate_parts(parts):
    names = set()
    if not parts:
        raise ValueError('Model contains no parts')
    for part in parts:
        name = part.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f'Missing or duplicate part name: {name}')
        names.add(name)
        vertices = np.asarray(part['v'], dtype=float)
        faces = np.asarray(part['f'])
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f'Invalid coordinates: {name}')
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f'Triangular faces required: {name}')
        if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError(f'Invalid vertex index: {name}')
    return names


def validate_bounded_walls(parts):
    """Existing geometry is not automatically evidence; reject opaque inputs.

    A current model must be an audited overlap-baseline product. Omit that
    optional field to have this runner clip fresh hypotheses to raw evidence.
    """
    for part in parts:
        if not part.get('kind', '').startswith('wall'):
            continue
        if (not part.get('open_surface') or part.get('support_cutoff_mm', float('inf')) > 50
                or part.get('evidence_status') != 'bounded_lidar_baseline'):
            raise ValueError(f'{part["name"]}: current wall is not an audited bounded LiDAR baseline; '
                             'omit current_model to clip hypotheses first')


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('units') != 'metres':
        raise ValueError('Declare units as metres; implicit unit conversion is forbidden')
    base = path.parent
    manifest['candidate_model'] = str((base / manifest['candidate_model']).resolve())
    if not Path(manifest['candidate_model']).is_file():
        raise FileNotFoundError(manifest['candidate_model'])
    for field in ('current_model', 'source_native', 'source_native_audit'):
        if manifest.get(field):
            manifest[field] = str((base / manifest[field]).resolve())
            if not Path(manifest[field]).is_file():
                raise FileNotFoundError(manifest[field])
    scans = manifest.get('scans', [])
    if not scans or len({s['id'] for s in scans}) != len(scans):
        raise ValueError('At least one uniquely named registered scan is required')
    for scan in scans:
        scan['path'] = str((base / scan['path']).resolve())
        if not Path(scan['path']).is_file():
            raise FileNotFoundError(scan['path'])
        scan['scan_to_model'] = rigid(scan['scan_to_model']).tolist()
    manifest['model_to_building'] = rigid(manifest.get('model_to_building', np.eye(4))).tolist()
    manifest['level'] = int(manifest.get('level', 0))
    if manifest.get('wall_metadata'):
        metadata_path = (base / manifest['wall_metadata']).resolve()
        metadata = json.loads(metadata_path.read_text())
        candidate_parts = json.loads(Path(manifest['candidate_model']).read_text())['parts']
        hints = dict(manifest.get('expected_thickness_by_name', {}))
        for row in metadata.get('walls', []):
            source_id = row.get('wall_id', row.get('wall'))
            thickness_mm = row.get('thickness_mm')
            if source_id is None or thickness_mm is None:
                continue
            for part in candidate_parts:
                if part['name'] == source_id or part['name'].startswith(source_id + '_'):
                    hints[part['name']] = float(thickness_mm) / 1000
        manifest['expected_thickness_by_name'] = hints
        manifest['wall_metadata'] = str(metadata_path)
    manifest['manifest_path'] = str(path)
    return manifest


def read_registered_points(scans, bounds, voxel_m=0.02):
    """Stream every return, retain a representative per occupied voxel.

    Occupancy sampling bounds memory and avoids repeat-visit count bias. Raw
    point coordinates are retained, not voxel centres; audit labels the sample.
    """
    if not 0 < voxel_m <= 0.05:
        raise ValueError('Evidence voxel size must be positive and at most 50 mm')
    all_points = []
    records = []
    for scan in scans:
        matrix = rigid(scan['scan_to_model'])
        selected = []
        total = 0
        roi_count = 0
        with laspy.open(scan['path']) as reader:
            for block in reader.chunk_iterator(1_000_000):
                points = np.column_stack((block.x, block.y, block.z))
                if not np.isfinite(points).all():
                    raise ValueError(f'Non-finite LAS coordinates in {scan["id"]}')
                points = points @ matrix[:3, :3].T + matrix[:3, 3]
                total += len(points)
                points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
                roi_count += len(points)
                if not len(points):
                    continue
                _, ids = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0, return_index=True)
                selected.append(points[ids])
        if not selected:
            raise ValueError(f'{scan["id"]}: no raw points in model ROI; check registration')
        points = np.concatenate(selected)
        _, ids = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0, return_index=True)
        points = points[ids]
        all_points.append(points)
        stat = Path(scan['path']).stat()
        records.append({**scan, 'input_returns': total, 'roi_returns': roi_count,
                        'evidence_points': len(points), 'voxel_m': voxel_m,
                        'file_size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'sampling': 'All returns streamed, one original return per occupied voxel'})
        print(f'{scan["id"]}: {total:,} returns; {len(points):,} occupied evidence voxels', flush=True)
    combined = np.concatenate(all_points)
    _, ids = np.unique(np.floor(combined / voxel_m).astype(np.int64), axis=0, return_index=True)
    return combined[ids], records


def sample_surface(parts, maximum=100000):
    """Area-proportional deterministic interior samples, including all classes."""
    meshes = [trimesh.Trimesh(p['v'], p['f'], process=False) for p in parts]
    areas = np.array([mesh.area for mesh in meshes])
    rng = np.random.default_rng(9210)
    samples = []
    weights = []
    for mesh, area in zip(meshes, areas):
        if area <= 0:
            continue
        count = max(20, int(maximum * area / areas.sum()))
        ids = rng.choice(len(mesh.faces), size=count, p=mesh.area_faces / area)
        uv = rng.random((count, 2))
        folded = uv.sum(axis=1) > 1
        uv[folded] = 1 - uv[folded]
        tri = mesh.triangles[ids]
        samples.append(tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0]))
        weights.append(np.full(count, area / count))
    if not samples:
        raise ValueError('No positive-area geometry to sample')
    return np.concatenate(samples), np.concatenate(weights)


def distance_summary(distances, weights=None):
    distances = np.asarray(distances)
    result = {'samples': len(distances), 'p50_mm': float(np.percentile(distances, 50) * 1000),
              'p95_mm': float(np.percentile(distances, 95) * 1000)}
    for threshold in (10, 20, 50):
        result[f'within_{threshold}mm_percent'] = float(np.average(distances <= threshold / 1000, weights=weights) * 100)
    return result


def compare_coverage(before, after, points):
    """Report fit and missing coverage separately; neither is semantic proof."""
    from inspect_scan_coverage import distance_to_parts
    ids = np.linspace(0, len(points) - 1, min(len(points), 150000), dtype=int)
    raw = points[ids].astype(np.float32)
    tree = cKDTree(points)
    rows = {}
    distances = {}
    for key, parts in [('before', before), ('after', after)]:
        surface, weight = sample_surface(parts)
        model_to_raw = tree.query(surface, workers=8)[0]
        raw_to_model = distance_to_parts(raw, {'parts': parts})
        rows[key] = {'model_to_scan': distance_summary(model_to_raw, weight),
                     'scan_to_model': distance_summary(raw_to_model),
                     'surface_area_m2': float(weight.sum())}
        distances[key] = raw_to_model
    rows['previously_represented_scan_samples_lost_over_50mm'] = int(np.sum(
        (distances['before'] <= 0.05) & (distances['after'] > 0.05)))
    rows['newly_represented_scan_samples_within_50mm'] = int(np.sum(
        (distances['before'] > 0.05) & (distances['after'] <= 0.05)))
    rows['interpretation'] = ('Diagnostic distances to occupied raw samples. Unmatched points include furniture, '
                              'door leaves and scan noise; a nearby return does not certify a wall or site dimensions.')
    return rows, raw, distances


def render_review(parts, raw, distances, out, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(17, 12), layout='constrained')
    fig.suptitle(label + ' | shared reconstruction review', fontsize=17)
    for index, axes, title in [(1, (0, 1), 'Top'), (2, (0, 2), 'Front'), (3, (1, 2), 'Side')]:
        ax = fig.add_subplot(2, 2, index)
        q = raw[::3]
        d = distances['after'][::3]
        ax.scatter(q[:, axes[0]], q[:, axes[1]], s=0.6,
                   c=np.where(d <= .05, '#89999c', '#c65b47'), rasterized=True)
        ax.set_aspect('equal'); ax.set_title(title + ': gray = within 50 mm; red = unmatched scan')
        ax.set_xlabel('XYZ'[axes[0]] + ' (m)'); ax.set_ylabel('XYZ'[axes[1]] + ' (m)')
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    displayed = [p for p in parts if 'ceiling' not in p.get('kind', '') and not p.get('reference_only')]
    all_vertices = []
    for p in displayed:
        v = np.asarray(p['v']); f = np.asarray(p['f'])
        all_vertices.append(v)
        ax.add_collection3d(Poly3DCollection(v[f], facecolor=np.array(p.get('colour', [195, 188, 166])) / 255,
                                             edgecolor='none', linewidth=0))
    if all_vertices:
        v = np.vstack(all_vertices); lo, hi = v.min(0), v.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, .1))
    ax.view_init(30, -62); ax.set_title('3D geometry preview; native SketchUp views exported separately')
    fig.savefig(out / 'review_top_front_side_3d.png', dpi=130)
    plt.close(fig)


def run(manifest_paths, out, evidence_voxel_m=.02):
    from architectural_surface_refinement import refine_parts, supported_candidate_replacement, regularize_planar_fragments
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    manifests = [read_manifest(path) for path in manifest_paths]
    if len({m['level'] for m in manifests}) != len(manifests):
        raise ValueError('Each level must occur once; put multiple scans in the same level manifest')
    out.mkdir(parents=True, exist_ok=True)
    combined = []
    reports = []
    reference_names = set()
    for manifest in manifests:
        level = manifest['level']
        source_path = Path(manifest['candidate_model'])
        candidates = json.loads(source_path.read_text())['parts']
        validate_parts(candidates)
        candidates = [{**p, 'level': level} for p in candidates]
        if manifest.get('current_model'):
            parts = json.loads(Path(manifest['current_model']).read_text())['parts']
        else:
            from lidar_support_baseline import main as bounded_surfaces
            bounded_dir = out / f'L{level}_bounded_input'
            bounded_surfaces(manifest['manifest_path'], bounded_dir)
            parts = json.loads((bounded_dir / 'model.build.json').read_text())['parts']
        validate_parts(parts)
        validate_bounded_walls(parts)
        parts = [{**p, 'level': level} for p in parts]
        reference_names.update(p['name'] for p in parts)
        vertices = np.vstack([p['v'] for p in parts])
        bounds = np.array([vertices.min(0) - .25, vertices.max(0) + .25])
        points, provenance = read_registered_points(manifest['scans'], bounds, evidence_voxel_m)
        tree = cKDTree(points)
        current_by_name = {p['name']: p for p in parts}
        candidate_by_name = {p['name']: p for p in candidates}
        selected = []
        choices = []
        smoothing = []
        for index, part in enumerate(parts, 1):
            chosen = part
            if part.get('kind','').startswith('wall'):
                choices.append({'name': part['name'], 'accepted': False,
                                'reason': 'strict_wall_policy_retains_bounded_scan_surfaces_no_full_solid_completion'})
            elif manifest.get('current_model') and part['name'] in candidate_by_name:
                chosen, choice = supported_candidate_replacement(part, candidate_by_name[part['name']], tree)
                if choice['accepted'] and choice['evidence']['max_mm'] > 50:
                    chosen = part
                    choice.update(accepted=False, reason='strict_50mm_sample_envelope_failed')
                choices.append(choice)
            # Incomplete scan patches remain surfaces. Smooth only planar
            # architectural classes, never stair treads or unclassified mesh.
            if part.get('kind','').startswith(('wall','floor','beam','ceiling')) and not trimesh.Trimesh(chosen['v'],chosen['f'],process=False).is_watertight:
                chosen, smooth = regularize_planar_fragments(chosen, tree)
                smoothing.append(smooth)
            chosen['level'] = level
            selected.append(chosen)
            print(f'L{level} refinement {index}/{len(parts)} {part["name"]}', flush=True)
        refined, audit = refine_parts(selected,
                                     expected_thickness_by_name=manifest.get('expected_thickness_by_name'),
                                     raw_points_by_level={level: tree})
        from floor_surface_cleanup import consolidate_floor_surfaces
        refined, floor_cleanup = consolidate_floor_surfaces(refined, tree)
        audit['floor_surface_cleanup'] = floor_cleanup
        audit['candidate_replacement'] = choices
        audit['fragment_regularization'] = smoothing
        audit['new_candidate_identities_not_inserted'] = sorted(set(candidate_by_name) - set(current_by_name))
        audit['wall_policy'] = 'Retain bounded scan surface geometry; never restore a complete wall solid from a candidate'
        validate_parts(refined)
        coverage, raw, distances = compare_coverage(parts, refined, points)
        level_dir = out / f'L{level}'
        level_dir.mkdir()
        report = {'level': level, 'input_manifest': manifest, 'scans': provenance,
                  'source_model_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  'refinement': audit, 'coverage': coverage, 'site_accuracy_certified': False}
        (level_dir / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (level_dir / 'model.build.json').write_text(json.dumps({'label': manifest['label'], 'parts': refined}), encoding='utf-8')
        render_review(refined, raw, distances, level_dir, manifest['label'])
        for part in refined:
            combined.append(transform_part(part, manifest['model_to_building'],
                                           name=f'L{level} {part["name"]}', level=level))
        reports.append(report)
    validate_parts(combined)
    label = manifests[0]['label'] if len(manifests) == 1 else 'Architectural reconstruction | complete building'
    (out / 'model.build.json').write_text(json.dumps({'label': label, 'parts': combined}), encoding='utf-8')
    native_sources = {m['source_native'] for m in manifests if m.get('source_native')}
    if native_sources:
        if len(native_sources) != 1 or not all(m.get('source_native_audit') for m in manifests):
            raise ValueError('Native append requires one shared source native and its audit on each manifest')
        native_audit = json.loads(Path(manifests[0]['source_native_audit']).read_text())
        names = {p['name'] for p in native_audit['groups']}
        if not reference_names.issubset(names):
            raise ValueError(f'Native reference names absent: {sorted(reference_names - names)}')
        additions = {'label': label + ' | Astra shared flow', 'source_native': next(iter(native_sources)),
                     'expected_base_groups': len(names), 'reference_source_names': sorted(reference_names),
                     'parts': [{**p, 'name': 'Refined ' + p['name']} for p in combined],
                     'render_image_names': ['3D','Top','Front','Side'],
                     'source_label': 'Shared registered raw-scan refinement with explicit support and coverage audit',
                     'note': 'Architectural candidates are gated against registered raw samples. Incomplete patches remain surfaces. Prior objects retained as hidden references. Site accuracy unverified.'}
        (out / 'additions.build.json').write_text(json.dumps(additions), encoding='utf-8')
    scene = trimesh.Scene()
    for part in combined:
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        mesh.visual.face_colors = np.r_[part.get('colour', [195, 188, 166]), 255]
        scene.add_geometry(mesh, node_name=part['name'], geom_name=part['name'])
    scene.export(out / 'model.glb')
    result = {'label': label, 'levels': [r['level'] for r in reports], 'parts': len(combined),
              'engine': 'architectural_surface_refinement.refine_parts',
              'manifests': [m['manifest_path'] for m in manifests],
              'site_accuracy_certified': False,
              'native_export_pending': True,
              'outputs': ['model.build.json', 'model.glb', 'L*/audit.json', 'L*/review_top_front_side_3d.png']}
    (out / 'flow_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


def export_native(out, name, backend='sdk'):
    from export_observed_recovery import main as append_native
    out = Path(out).resolve()
    if Path(name).name != name or not name.lower().endswith('.skp'):
        raise ValueError('Native name must be a .skp filename')
    if (out / name).exists():
        raise FileExistsError('Use a new native filename; previous handovers are preserved')
    if backend=='sdk':
        from native_sdk_export import export as sdk_export
        payload=out/('additions.build.json' if (out/'additions.build.json').is_file() else 'model.build.json')
        sdk_export(payload,out/name)
    elif (out / 'additions.build.json').is_file():
        append_native('soulace', folder_override=str(out), native_name=name)
        audit = json.loads((out / 'native_audit.json').read_text())
        if not audit.get('addition_area_pass'):
            raise ValueError('Native addition area verification failed')
    else:
        from export_lidar_baseline_skp import main as build_native
        build_native(out, name)
    if not (out / name).is_file():
        raise RuntimeError('Native bridge did not produce the requested SketchUp file')
    report_path = out / 'flow_report.json'
    report = json.loads(report_path.read_text())
    report['native_export_pending'] = False
    report['native_backend'] = backend
    report['native_file'] = str(out / name)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', help='Repeat for levels in the same building')
    parser.add_argument('--out', required=True)
    parser.add_argument('--evidence-voxel-mm', type=float, default=20)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--native-name', help='Native .skp export after processing using the chosen backend')
    parser.add_argument('--native-backend',choices=['sdk','bridge'],default='sdk',help='Installed native file library (default) or interactive Ruby bridge')
    parser.add_argument('--export-only', action='store_true', help='Export an already processed output through the native bridge')
    args = parser.parse_args()
    if args.export_only:
        if not args.native_name:
            parser.error('--export-only requires --native-name')
        export_native(args.out, args.native_name,args.native_backend)
    elif not args.manifest:
        parser.error('--manifest is required unless --export-only is selected')
    elif args.preflight:
        for path in args.manifest:
            manifest = read_manifest(path)
            names = validate_parts(json.loads(Path(manifest['candidate_model']).read_text())['parts'])
            print(json.dumps({'label': manifest['label'], 'level': manifest['level'], 'parts': len(names),
                              'registered_scans': len(manifest['scans'])}), flush=True)
    else:
        run(args.manifest, args.out, args.evidence_voxel_mm / 1000)
        if args.native_name:
            export_native(args.out, args.native_name,args.native_backend)


if __name__ == '__main__':
    main()
