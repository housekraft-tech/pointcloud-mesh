"""Export the generic coverage-first baseline and audit native surface areas."""
import argparse
import json
from pathlib import Path
import trimesh
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]


def main(folder,name):
    folder=Path(folder).resolve();source=folder/'model.build.json';native=folder/name
    ruby=ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb'
    result=invoke(f'load {json.dumps(ruby.as_posix())}; PCMCoverageBaseline.build({json.dumps(source.as_posix())},{json.dumps(native.as_posix())})')
    if not isinstance(result,dict):
        raise RuntimeError(result)
    audit=json.loads((folder/'native_audit.json').read_text())
    expected={p['name']:float(trimesh.Trimesh(p['v'],p['f'],process=False).area) for p in json.loads(source.read_text())['parts']}
    errors=[]
    if set(expected)!={p['name'] for p in audit['groups']}:
        errors.append({'group_names_differ':True})
    for group in audit['groups']:
        delta=group['area_m2']-expected[group['name']]
        group['area_delta_m2']=delta
        if abs(delta)>max(.000025,expected[group['name']]*.00001):
            errors.append(group)
    audit.update(area_consistency_pass=not errors,errors=errors)
    (folder/'native_audit.json').write_text(json.dumps(audit,indent=2))
    if errors:
        raise ValueError(errors[:3])
    print(json.dumps(result),flush=True)
    print('Native area consistency passed',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folder',required=True);ap.add_argument('--name',required=True)
    args=ap.parse_args();main(args.folder,args.name)
