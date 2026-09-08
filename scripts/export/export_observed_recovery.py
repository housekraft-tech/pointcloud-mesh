"""Append recovered evidence while retaining the earlier native CAD unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]


def main(project,folder_override=None,native_name=None):
    folder=Path(folder_override).resolve() if folder_override else ROOT/f'output_final/scan_recovered_v1/{project}'
    additions=folder/'additions.build.json'
    if project=='engrance':
        payload=json.loads((folder/'model.build.json').read_text())
        baseline=json.loads((ROOT/'output_final/coverage_restored_v1/engrance/model.build.json').read_text())
        assert payload['parts'][:64]==baseline['parts']
        data={'label':payload['label'],'source_native':str(ROOT/'output_final/coverage_restored_v1/engrance/Engrance_LiDAR_coverage_first.skp'),
              'expected_base_groups':64,'parts':payload['parts'][64:]}
        additions.write_text(json.dumps(data,separators=(',',':')))
        native=folder/'Engrance_scan_recovered.skp'
    else:
        data=json.loads(additions.read_text());native=folder/(native_name or 'Soulace_stairs_and_lower_floor.skp')
    result=invoke(f'load {json.dumps((ROOT/"scripts/export/ruby/pcm_coverage_baseline.rb").as_posix())}; PCMCoverageBaseline.append({json.dumps(additions.as_posix())},{json.dumps(native.as_posix())})')
    print(result,flush=True)
    audit=json.loads((folder/'native_audit.json').read_text());groups={g['name']:g for g in audit['groups']};checks=[]
    for part in data['parts']:
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False);expected=float(mesh.area);observed=groups[part['name']]['area_m2']
        checks.append({'name':part['name'],'source_triangles':len(mesh.faces),'native_faces':groups[part['name']]['faces'],
                       'source_area_m2':expected,'native_area_m2':observed,'delta_m2':observed-expected,
                       'area_pass':abs(observed-expected)<max(.000025,expected*1e-5)})
    audit['addition_area_checks']=checks;audit['addition_area_pass']=all(c['area_pass'] for c in checks)
    (folder/'native_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('project',choices=['engrance','soulace']);ap.add_argument('--folder');ap.add_argument('--native-name');args=ap.parse_args();main(args.project,args.folder,args.native_name)
