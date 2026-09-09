"""List native groups that duplicate a rebuilt exterior plane, so they can be hidden.

Observed faces, scan surfaces and speckle fragments that sit within
``tolerance`` of a rebuilt wall plane over most of their area carry no
information the rebuilt plane lacks; in a preview and in SketchUp they only
draw extra outlines. They are hidden as references, not deleted: the
assembler keeps every native group in the file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from wall_surface_cleanup import scene_of, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='reopened_visible.build.json of the checked model')
    parser.add_argument('--fragment', action='append', required=True, help='rebuilt parts (patches) that supersede duplicates')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tolerance', type=float, default=.03)
    parser.add_argument('--fraction', type=float, default=.7)
    parser.add_argument('--kinds', default='wall_face_observed,wall_scan_surface,wall_inferred,wall_measured,parapet')
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())['parts']
    rebuilt = [p for f in args.fragment for p in json.loads(Path(f).read_text())['parts']]
    replaced = set()
    for p in rebuilt:
        replaced.update(p.get('source_group_names', []))
    scene = scene_of(rebuilt)
    kinds = tuple(k.strip() for k in args.kinds.split(','))
    hide, report = [], []
    for part in model:
        if part['name'] in replaced or not part.get('kind', '').startswith(kinds):
            continue
        mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
        if mesh.area < 1e-4:
            continue
        points, _ = trimesh.sample.sample_surface(mesh, int(np.clip(mesh.area * 400, 200, 4000)), seed=0)
        near = float((distances(scene, np.asarray(points, np.float32)) <= args.tolerance).mean())
        row = {'name': part['name'], 'kind': part['kind'], 'area_m2': float(mesh.area), 'covered_fraction': near}
        if near >= args.fraction:
            hide.append(part['name']); row['decision'] = 'hidden_duplicate'
        else:
            row['decision'] = 'kept'
        report.append(row)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'hide.json').write_text(json.dumps(hide, indent=2))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'hidden': len(hide), 'hidden_area_m2': round(sum(r['area_m2'] for r in report if r['decision'] == 'hidden_duplicate'), 2),
                      'kept': sum(r['decision'] == 'kept' for r in report)}))


if __name__ == '__main__':
    main()
