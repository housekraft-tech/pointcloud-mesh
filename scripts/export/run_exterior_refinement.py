"""Reusable exterior-only refinement for any checked, registered building model.

Upstream run_architectural_flow.py supplies wall/floor parts in metres with
gravity aligned to Z, and raw_evidence.py caches the raw returns of the same
property in that frame. Exterior faces are selected in the scan (open air in
front of the face), only those planes are regularized within 50 mm, and the
revision is assembled, reopened and audited natively. Every unselected native
group must read back unchanged. Review faces are reported, never modified.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from render_native_geometry import render
from select_exterior_walls import Occupancy, load_returns, select, summary

EXTERIOR_VIEWS = {'south_west': (-1.3, -1.5, 1.), 'south_east': (1.3, -1.5, 1.),
                  'north_east': (1.3, 1.5, 1.), 'north_west': (-1.3, 1.5, 1.)}


def render_exterior_views(parts, out, label, prefix):
    for name, offset in EXTERIOR_VIEWS.items():
        render(parts, out / f'{prefix}_{name}.png', f'{label} | exterior from {name.replace("_", "-")}', offset=offset)


def selection_preview(parts, audit, out, label):
    """Colour every wall part by its strongest classification so the selection can be judged from outside."""
    colour = {'exterior_face': [204, 82, 60], 'partly_exterior_face': [232, 170, 60], 'interior_face': [150, 150, 150]}
    rank = {'exterior_face': 2, 'partly_exterior_face': 1, 'interior_face': 0}
    best = {}
    for row in audit:
        if rank[row['classification']] > rank.get(best.get(row['wall']), -1):
            best[row['wall']] = row['classification']
    painted = []
    for part in parts:
        if part['name'] in best:
            painted.append({**part, 'colour': colour[best[part['name']]]})
        elif part.get('kind', '').startswith(('wall', 'parapet')):
            painted.append({**part, 'colour': [120, 120, 120]})
        else:
            painted.append({**part, 'colour': [225, 222, 212]})
    render_exterior_views(painted, out, f'{label} | red exterior, amber partly exterior, grey interior', 'selection')


def stage_selection_early(out, parts, args):
    """Selection file for the regularize stage: every wall in all-walls mode, else the scan selection."""
    if args.all_walls:
        names = [p['name'] for p in parts if p.get('kind', '').startswith(('wall', 'parapet'))]
        (out / 'selection_all_walls.json').write_text(json.dumps(names, indent=2))
        return str(out / 'selection_all_walls.json')
    return str(out / 'selection.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, help='folder of the checked model being revised')
    parser.add_argument('--evidence-cache', required=True, help='raw_evidence.py npz for the same property frame')
    parser.add_argument('--out', required=True)
    parser.add_argument('--native-name', default='Exterior_refined.skp')
    parser.add_argument('--label', default='Exterior wall refinement')
    parser.add_argument('--skip-export', action='store_true', help='Stop after selection and regularization')
    parser.add_argument('--reuse', action='store_true', help='Keep an existing selection.json and walls/patches.build.json in --out')
    parser.add_argument('--all-walls', action='store_true',
                        help='Apply regularize, rectangle and rectilinear stages to every wall plane, interior faces included')
    parser.add_argument('--box-prisms', action='store_true', help='Replace beam and column fragments by their axis-aligned boxes')
    parser.add_argument('--rectilinear-slabs', action='store_true',
                        help='Square floor and ceiling outlines, re-extruding solid slabs (declared inference)')
    parser.add_argument('--hide-kinds', default='',
                        help='Comma-separated kind substrings to hide, e.g. beam,plinth,junction_patch,wall_face_observed,floor_clean')
    parser.add_argument('--rectilinear', action='store_true',
                        help='Square every exterior ring and merge coplanar planes into rectangular blocks (declared inference)')
    parser.add_argument('--hide-duplicates', action='store_true',
                        help='Hide native faces that lie within 30 mm of a rebuilt plane over 70 %% of their area')
    parser.add_argument('--complete-rectangles', action='store_true',
                        help='Complete exterior planes to level rectangles where the scan shows nothing (declared inference)')
    parser.add_argument('--recover-recessed', action='store_true',
                        help='Add recessed/stepped faces the model lacks behind exterior voids (recover_recessed_faces.py)')
    args = parser.parse_args()
    previous = Path(args.previous).resolve(); out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_export and (out / args.native_name).exists():
        raise FileExistsError(f'{out / args.native_name} exists; the native exporter never overwrites a checked model. '
                              'Choose another --native-name or --out.')
    model = previous / 'reopened_visible.build.json'
    parts = json.loads(model.read_text())['parts']
    scripts = Path(__file__).parent
    reusable = args.reuse and (out / 'selection.json').exists() and (out / 'selection_audit.json').exists()
    if reusable:
        selection = json.loads((out / 'selection.json').read_text())
        selection_audit = json.loads((out / 'selection_audit.json').read_text())
        counts = summary(selection, selection_audit)
        print('Reusing selection in', out, flush=True)
        if not (out / 'walls' / 'patches.build.json').exists():
            subprocess.run([sys.executable, str(scripts / 'regularize_wall_surfaces.py'), '--model', str(model),
                            '--out', str(out / 'walls'), '--selection', stage_selection_early(out, parts, args),
                            '--evidence-cache', args.evidence_cache], check=True)
    else:
        occupancy = Occupancy(load_returns(args.evidence_cache))
        selection, selection_audit = select(parts, occupancy)
        counts = summary(selection, selection_audit)
        print(json.dumps(counts), flush=True)
        if not selection['walls']:
            raise ValueError('No confident exterior faces; check the evidence frame and the wall parts')
        (out / 'selection.json').write_text(json.dumps(selection, indent=2))
        (out / 'selection_audit.json').write_text(json.dumps(selection_audit, indent=2))
        selection_preview(parts, selection_audit, out, args.label)
        subprocess.run([sys.executable, str(scripts / 'regularize_wall_surfaces.py'), '--model', str(model),
                        '--out', str(out / 'walls'), '--selection', stage_selection_early(out, parts, args),
                        '--evidence-cache', args.evidence_cache], check=True)
    fragments = [str(out / 'walls' / 'patches.build.json')]
    # All-walls mode: the regularize, rectangle and rectilinear stages take every
    # wall plane (interior faces too); recessed-face recovery stays exterior-only.
    stage_selection = str(out / 'selection.json')
    if args.all_walls:
        names = [p['name'] for p in parts if p.get('kind', '').startswith(('wall', 'parapet'))]
        (out / 'selection_all_walls.json').write_text(json.dumps(names, indent=2))
        stage_selection = str(out / 'selection_all_walls.json')
    if args.recover_recessed:
        if not (args.reuse and (out / 'recessed' / 'recessed_faces.build.json').exists()):
            subprocess.run([sys.executable, str(scripts / 'recover_recessed_faces.py'), '--model', str(model),
                            '--selection', str(out / 'selection.json'), '--evidence-cache', args.evidence_cache,
                            '--out', str(out / 'recessed')], check=True)
        if json.loads((out / 'recessed' / 'recessed_faces.build.json').read_text())['parts']:
            fragments.append(str(out / 'recessed' / 'recessed_faces.build.json'))
    if args.complete_rectangles:
        command = [sys.executable, str(scripts / 'complete_exterior_rectangles.py'), '--model', str(model),
                   '--patches', str(out / 'walls' / 'patches.build.json'), '--selection', stage_selection,
                   '--evidence-cache', args.evidence_cache, '--out', str(out / 'rectangles')]
        if len(fragments) > 1:
            command.extend(['--recessed', fragments[1]])
        if not (args.reuse and (out / 'rectangles' / 'patches.build.json').exists()):
            subprocess.run(command, check=True)
        if json.loads((out / 'rectangles' / 'patches.build.json').read_text())['parts']:
            fragments.append(str(out / 'rectangles' / 'patches.build.json'))
    if args.rectilinear:
        command = [sys.executable, str(scripts / 'rectilinear_exterior.py'), '--selection', stage_selection,
                   '--out', str(out / 'rectilinear')]
        for fragment in fragments:
            command.extend(['--fragment', fragment])
        if not (args.reuse and (out / 'rectilinear' / 'patches.build.json').exists()):
            subprocess.run(command, check=True)
        fragments.append(str(out / 'rectilinear' / 'patches.build.json'))
    if args.rectilinear_slabs:
        if not (args.reuse and (out / 'slabs' / 'patches.build.json').exists()):
            subprocess.run([sys.executable, str(scripts / 'rectilinear_slabs.py'), '--model', str(model),
                            '--out', str(out / 'slabs')], check=True)
        if json.loads((out / 'slabs' / 'patches.build.json').read_text())['parts']:
            fragments.append(str(out / 'slabs' / 'patches.build.json'))
    if args.box_prisms:
        if not (args.reuse and (out / 'prisms' / 'patches.build.json').exists()):
            subprocess.run([sys.executable, str(scripts / 'box_prisms.py'), '--model', str(model), '--out', str(out / 'prisms')], check=True)
        if json.loads((out / 'prisms' / 'patches.build.json').read_text())['parts']:
            fragments.append(str(out / 'prisms' / 'patches.build.json'))
    hide_files = []
    if args.hide_kinds:
        # Only rectangular planes and stairs remain visible: every other kind is hidden as a reference.
        replaced_now = {s for f in fragments for p in json.loads(Path(f).read_text())['parts'] for s in p.get('source_group_names', [])}
        kinds = tuple(k.strip() for k in args.hide_kinds.split(',') if k.strip())
        names = [p['name'] for p in parts if p['name'] not in replaced_now and any(k in p.get('kind', '') for k in kinds)]
        (out / 'hide_kinds.json').write_text(json.dumps(names, indent=2))
        if names:
            hide_files.append(str(out / 'hide_kinds.json'))
        print(json.dumps({'hidden_by_kind': len(names)}), flush=True)
    if args.hide_duplicates:
        command = [sys.executable, str(scripts / 'hide_duplicate_faces.py'), '--model', str(model), '--out', str(out / 'hide')]
        for fragment in fragments:
            command.extend(['--fragment', fragment])
        subprocess.run(command, check=True)
        if json.loads((out / 'hide' / 'hide.json').read_text()):
            hide_files.append(str(out / 'hide' / 'hide.json'))
    if args.skip_export:
        return
    audit = json.loads((previous / 'native_sdk_audit.json').read_text())
    command = [sys.executable, str(scripts / 'assemble_revision.py'), '--previous', str(previous), '--out', str(out),
               '--native-name', args.native_name, '--label', args.label,
               '--note', 'Exterior faces selected in the raw returns (open air in front of the face); only those '
                         'planes regularized within 50 mm. Filled scan gaps are inferred continuity, not measured '
                         'area. Protected openings, interior faces, floors, stairs and ceilings retained.']
    for fragment in fragments:
        command.extend(['--fragment', fragment])
    for path in hide_files:
        command.extend(['--hide', path])
    for group in audit['groups']:
        if 'ceiling' in group['kind'] and not group['hidden']:
            command.extend(['--show', group['name']])
    subprocess.run(command, check=True)
    after = json.loads((out / 'native_sdk_audit.json').read_text())
    groups = {g['name']: g for g in after['groups']}
    replaced = {s for p in json.loads((out / 'walls' / 'patches.build.json').read_text())['parts']
                for s in p['source_group_names']}
    for path in hide_files:
        replaced.update(json.loads(Path(path).read_text()))
    unchanged = []
    for before in audit['groups']:
        if before['name'] in replaced:
            continue
        saved = groups[before['name']]
        if saved['hidden'] != before['hidden'] or saved['faces'] != before['faces'] or abs(saved['area_m2'] - before['area_m2']) > 1e-9:
            raise ValueError(f'Unselected group changed: {before["name"]}')
        unchanged.append(before['name'])
    reopened = json.loads((out / 'reopened_visible.build.json').read_text())['parts']
    render_exterior_views(reopened, out, args.label, 'review_exterior')
    (out / 'scope_verification.json').write_text(json.dumps({
        'unselected_groups_preserved': len(unchanged), 'replaced_wall_groups': sorted(replaced),
        'selection': counts, 'property_specific_wall_list_used': False, 'site_accuracy_certified': False}, indent=2))
    print(f'Verified {len(unchanged)} unselected native groups unchanged', flush=True)


if __name__ == '__main__':
    main()
