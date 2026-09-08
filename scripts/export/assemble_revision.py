"""Assemble bounded fixes into a new native revision of a checked model.

Fragments (junction fix, wall cleanup, observed faces) each carry parts whose
``source_group_names`` name what they replace. Replacements may chain (a wall
seated on its floor and then cleaned); only the final part of each chain is
exported, every replaced native group is hidden as a reference, and nothing
in the source file is altered. The saved file is reopened and audited by the
native SDK exporter, then rendered from the reopened geometry.
"""
import argparse
import json
from pathlib import Path

import trimesh

from native_sdk_export import export
from render_native_geometry import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, help='folder of the checked model being revised')
    parser.add_argument('--fragment', action='append', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--native-name', required=True)
    parser.add_argument('--label', required=True); parser.add_argument('--note', required=True)
    args = parser.parse_args()
    previous = Path(args.previous).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((previous / 'native_sdk_audit.json').read_text())
    native_names = {g['name'] for g in audit['groups']}
    source_native = audit['path']
    chain = {}
    for fragment in args.fragment:
        for part in json.loads(Path(fragment).read_text())['parts']:
            chain[part['name']] = part
    replaced = {s for p in chain.values() for s in p.get('source_group_names', [])}
    final = [p for name, p in chain.items() if name not in replaced]
    references = set()
    for part in final:
        sources = collect(chain, part, native_names)
        part['source_group_names'] = sorted(sources)
        references |= sources
    for part in final:
        if part['name'] in native_names:
            raise ValueError(f'Name collides with a native group: {part["name"]}')
        trimesh.Trimesh(part['v'], part['f'], process=False)
    previous_payload = json.loads((previous / 'additions.build.json').read_text())
    payload = {'label': args.label, 'source_native': source_native, 'expected_base_groups': len(native_names),
               'reference_source_names': sorted(references), 'parts': final,
               'render_image_names': previous_payload.get('render_image_names', ['3D', 'Top', 'Front', 'Side']),
               'source_label': 'Bounded junction, wall-artifact and observed-face revision of the checked native model',
               'note': args.note}
    (out / 'additions.build.json').write_text(json.dumps(payload, separators=(',', ':')))
    print(json.dumps({'new_parts': len(final), 'hidden_references': len(references),
                      'kinds': sorted({p['kind'] for p in final})}, indent=2), flush=True)
    export(out / 'additions.build.json', out / args.native_name)
    parts = json.loads((out / 'reopened_visible.build.json').read_text())['parts']
    render(parts, out / 'checked_house_3d.png', args.label + ' | complete house')
    floors = [p for p in parts if 'floor' in p['kind'] or p['kind'] == 'ground_single_plane']
    if floors:
        render(floors, out / 'checked_floor_planes.png', args.label + ' | floor planes')
    levels = sorted({p['level'] for p in parts})
    if len(levels) > 1:
        for level in levels:
            render([p for p in parts if p['level'] == level], out / f'checked_l{level}_3d.png', args.label + f' | level {level}')
    walls = [p for p in parts if p['kind'].startswith(('wall', 'parapet'))]
    render(walls, out / 'checked_walls_3d.png', args.label + ' | wall faces only', offset=(-1.3, -1.5, 1.0))
    (out / 'flow_report.json').write_text(json.dumps({
        'label': args.label, 'previous': str(previous), 'native_file': str(out / args.native_name),
        'native_backend': 'sdk', 'native_checks': 'native_sdk_audit.json', 'new_parts': len(final),
        'hidden_references': sorted(references), 'fragments': [str(Path(f).resolve()) for f in args.fragment],
        'site_accuracy_certified': False,
        'preview_type': 'Native-geometry CPU render, not UI screenshot'}, indent=2))


def collect(chain, part, native_names):
    found = set(); stack = list(part.get('source_group_names', []))
    while stack:
        name = stack.pop()
        if name in native_names:
            found.add(name)
        elif name in chain:
            stack.extend(chain[name].get('source_group_names', []))
        else:
            raise ValueError(f'Unknown replaced part: {name}')
    return found


if __name__ == '__main__':
    main()
