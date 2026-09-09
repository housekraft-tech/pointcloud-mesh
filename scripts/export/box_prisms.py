"""Replace beam and column fragments by their axis-aligned box (declared inference).

Beams and columns are rectangular prisms; the scan fragments that represent
them are what make a wall's thickness step. Each such part becomes the solid
box of its extent, so the step is kept with sharp corners. Parts whose extent
is implausible for a prism (over ``max_span_m`` in two axes) are left alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kinds', default='beam,column')
    parser.add_argument('--max-span-m', type=float, default=1.)
    args = parser.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(','))
    parts = json.loads(Path(args.model).read_text())['parts']
    out_parts, report = [], []
    for part in parts:
        if not part.get('kind', '').startswith(kinds):
            continue
        v = np.asarray(part['v'], float)
        lo, hi = v.min(0), v.max(0)
        span = hi - lo
        if (span > args.max_span_m).sum() >= 2:
            report.append({'part': part['name'], 'decision': 'kept', 'span_m': span.tolist()}); continue
        box = trimesh.creation.box(extents=np.maximum(span, .01), transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
        out_parts.append({**{k: x for k, x in part.items() if k != 'planar_loops'},
                          'name': part['name'] + ' - box - INFERRED', 'v': box.vertices.tolist(), 'f': box.faces.tolist(),
                          'source_group_names': [part['name']], 'merge_coplanar_faces': True,
                          'evidence_status': 'prism_boxed_to_extent_INFERRED', 'completion_is_measured': False})
        report.append({'part': part['name'], 'decision': 'boxed', 'span_m': span.tolist()})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'boxed': len(out_parts), 'kept': sum(r['decision'] == 'kept' for r in report)}))


if __name__ == '__main__':
    main()
