"""Export any rectangular rebuild and verify native per-object surface areas."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import trimesh
from skp_client import rb

ROOT = Path(__file__).resolve().parents[2]


def invoke(code):
    result = rb(code,timeout=1200)
    if not result.get('ok'):
        raise RuntimeError(result)
    value = result['result']
    for _ in range(3):
        if not isinstance(value,str):
            break
        try:
            value=json.loads(value)
        except ValueError:
            break
    return value


def main(folder, name, verify_only=False):
    folder = Path(folder).resolve()
    native = folder/name
    if native.exists() and not verify_only:
        raise FileExistsError(native)
    source = folder/'model.build.json'
    ruby = ROOT/'scripts/export/ruby/pcm_rectangular.rb'
    build = {'path':str(native),'verification_only':True} if verify_only else invoke(f'load {json.dumps(ruby.as_posix())}; PCMRectangular.build({json.dumps(source.as_posix())},{json.dumps(native.as_posix())})')
    if not isinstance(build,dict):
        raise RuntimeError(build)
    if not verify_only:
        (folder/'native_build_result.json').write_text(json.dumps(build,indent=2))
    elif (folder/'native_build_result.json').exists():
        build=json.loads((folder/'native_build_result.json').read_text())
    print(json.dumps(build),flush=True)
    audit_path=folder/'native_audit.json'
    result = invoke(f'''
      raise 'Reopen failed' unless Sketchup.open_file({json.dumps(native.as_posix())});
      m=Sketchup.active_model;
      rows=m.entities.grep(Sketchup::Group).map do |g|
        faces=g.entities.grep(Sketchup::Face);
        {{name:g.name,faces:faces.length,area_m2:faces.sum{{|f|f.area}}/(39.37007874015748**2)}}
      end;
      File.write({json.dumps(audit_path.as_posix())},JSON.pretty_generate({{path:m.path, scenes:m.pages.map{{|p|p.name}}, groups:rows}}));
      {{groups:rows.length,audit:{json.dumps(audit_path.as_posix())}}}.to_json
    ''')
    if not isinstance(result,dict):
        raise RuntimeError(result)
    measured=json.loads(audit_path.read_text())
    expected=defaultdict(float)
    for p in json.loads(source.read_text())['parts']:
        expected[p['source_object']]+=float(trimesh.Trimesh(p['v'],p['f'],process=False).area)
    failures=[]
    for g in measured['groups']:
        delta=g['area_m2']-expected[g['name']]
        g['area_delta_m2']=delta
        if abs(delta)>max(.000025,expected[g['name']]*1e-5):
            failures.append(g)
    if set(expected)!={g['name'] for g in measured['groups']}:
        failures.append({'group_name_mismatch':True})
    measured.update(build=build,area_consistency_pass=not failures,failures=failures)
    (folder/'native_audit.json').write_text(json.dumps(measured,indent=2))
    if failures:
        raise ValueError(f'Native audit failures: {failures[:3]}')
    print(f'Native audit passed: {native}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',required=True);ap.add_argument('--name',default='Rectangular_rebuild.skp')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();main(args.folder,args.name,args.verify_only)
