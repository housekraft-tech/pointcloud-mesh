"""Create an interior-scan SKP with wall-side audit flags and unchanged geometry."""
import argparse
import ctypes as C
import json
from pathlib import Path
import re

import numpy as np

from native_sdk_export import SDK, Color, snapshots_match
from render_native_geometry import render


STYLES = {
    'no_substantial_opposite_face': ('SINGLE-SIDED / review', [205, 90, 80]),
    'partial_opposite_face': ('PARTIAL OTHER FACE / review', [225, 170, 65]),
    'substantial_opposite_face_candidate': ('PAIRED FACE CANDIDATE', [105, 170, 145]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='Interior scan')
    parser.add_argument('--native-name', default='Interior_wall_audit.skp')
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    destination = out / args.native_name
    if destination.parent != out or destination.suffix.lower() != '.skp':
        raise ValueError('--native-name must be a .skp filename without directories')
    if destination.exists():
        raise FileExistsError(destination)
    audit = json.loads(Path(args.audit).read_text())
    rows = {r['wall']: r for r in audit['walls']}
    sdk = SDK(); model = sdk.load(args.source)
    before = sdk.snapshot(model)
    names = {r['name'] for r in before}
    if not set(rows) <= names:
        raise ValueError('Audit does not match source native file')
    renames, hidden = {}, []
    for group in sdk.groups(model):
        name = sdk.name(group)
        kind = sdk.attribute(group, 'CoverageBaseline', 'kind', '')
        drawing = sdk.api.SUGroupToDrawingElement(group)
        if kind == 'scan_detail':
            sdk.call('SUDrawingElementSetHidden', drawing, C.c_bool(True))
            hidden.append(name)
        if name not in rows:
            continue
        row = rows[name]; label, rgb = STYLES[row['status']]
        match = re.search(r'wall_(\d+)', name)
        identifier = match.group(1) if match else str(len(renames) + 1)
        renamed = f'Wall {identifier} | {label} | thickness unverified'
        renames[name] = renamed
        sdk.call('SUGroupSetName', group, renamed.encode())
        for key, value in {
            'source_group_name': name,
            'scan_access': 'interior_only_apartment',
            'side_status': row['status'],
            'opposite_coverage_fraction': float(row['opposite_coverage_fraction']),
            'same_group_opposite_coverage_fraction': float(row['same_group_opposite_coverage_fraction']),
            'physical_thickness_verified': False,
            'perimeter_or_internal_identity': 'requires_review',
            'interpretation': 'Unscanned exterior faces can legitimately be absent. Internal partitions need both room-side faces checked. Geometric overlap is not a thickness measurement.',
        }.items():
            sdk.set_attribute(group, 'WallSideAudit', key, value)
        material = sdk.new('Material')
        sdk.call('SUMaterialSetName', material, ('Wall audit ' + identifier).encode())
        colour = Color(*rgb, 255)
        sdk.call('SUMaterialSetColor', material, C.byref(colour))
        sdk.call('SUModelAddMaterials', model, C.c_size_t(1), (type(material) * 1)(material))
        sdk.call('SUDrawingElementSetMaterial', drawing, material)
        for face in sdk.sequence(sdk.getref('SUGroupGetEntities', group), 'SUEntities', 'Faces'):
            sdk.call('SUFaceSetFrontMaterial', face, material)
            sdk.call('SUFaceSetBackMaterial', face, material)
    after = sdk.snapshot(model); lookup = {r['name']: r for r in after}
    for old in before:
        new = lookup[renames.get(old['name'], old['name'])]
        assert old['faces'] == new['faces']
        assert abs(old['area_m2'] - new['area_m2']) < 1e-9
        assert np.allclose(old['transform'], new['transform'], atol=1e-12, rtol=0)
    sdk.metric_display(model)
    parts = sdk.visible_geometry(model)
    sdk.scenes(model, parts)
    sdk.call('SUModelSaveToFile', model, str(destination).encode())
    sdk.call('SUModelRelease', C.byref(model))
    saved = sdk.load(destination); reopened = sdk.snapshot(saved)
    assert snapshots_match(after, reopened)
    for group in sdk.groups(saved):
        if sdk.name(group) in renames.values():
            assert sdk.attribute(group, 'WallSideAudit', 'scan_access', '') == 'interior_only_apartment'
            assert sdk.attribute(group, 'WallSideAudit', 'side_status', '') in STYLES
    (out / 'native_audit.json').write_text(json.dumps({
        'source': str(Path(args.source).resolve()), 'path': str(destination),
        'geometry_preserved': True, 'native_roundtrip_pass': True,
        'scan_access': 'interior_only_apartment', 'counts': audit['counts'],
        'renamed_groups': renames, 'hidden_scan_references': hidden,
        'site_accuracy_certified': False, 'groups': reopened}, indent=2))
    (out / 'wall_side_audit.json').write_text(json.dumps(audit, indent=2))
    (out / 'reopened_visible.build.json').write_text(json.dumps({'parts': parts}))
    sdk.call('SUModelRelease', C.byref(saved)); sdk.api.SUTerminate()
    render(parts, out / 'wall_audit.png', args.label + ' | red: predominantly single-sided; amber: partial; green: pair candidate')
    (out / 'README.txt').write_text(
        'INTERIOR-ONLY APARTMENT SCAN\n'
        'Red: less than 10% opposite-face overlap; predominantly single-sided.\n'
        'Amber: 10-80% overlap; partial opposite face.\n'
        'Green: at least 80% overlap; possible paired faces, physical thickness unverified.\n'
        'All wall groups are flagged and named in SketchUp Outliner.\n'
        'Perimeter faces may correctly be absent because exteriors were not scanned.\n'
        'Perimeter versus internal identity still requires review; no back faces were invented.\n'
        'Unclassified scan detail remains in the file, hidden.\n')
    print(json.dumps({'path': str(destination), 'bytes': destination.stat().st_size,
                      'counts': audit['counts'], 'geometry_preserved': True}))


if __name__ == '__main__':
    main()
