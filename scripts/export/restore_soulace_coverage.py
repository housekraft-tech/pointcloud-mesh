"""Restore ALL earlier LiDAR-supported Soulace faces, in measured level frames."""
import json
from pathlib import Path
import numpy as np
from export_rectangular_skp import invoke
from rectangular_rebuild import rigid

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'output_final/coverage_restored_v1/soulace'
    out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'output_final/soulace_overlap_only_50mm/release/release_audit.json').read_text())
    current=json.loads((ROOT/'output_final/rectangular_rebuild_v1/soulace_entire_house/assembly_audit.json').read_text())
    groups={};transforms={}
    original_offsets=[0,3.2,6.5]
    for lv,new in zip(previous['levels'],current['levels']):
        level=lv['level']
        undo=np.eye(4);undo[2,3]=-original_offsets[level]
        transforms[str(level)]=rigid(np.asarray(new['local_to_common'])@undo).tolist()
        for item in lv['objects']:
            if 'area_m2' in item:
                groups[f'L{level}_{item["name"]}']={'level':level,'kind':item['kind']}
    source=ROOT/'output_final/soulace_overlap_only_50mm/release/Soulace_overlap_only.skp'
    config={'label':'Soulace - earlier LiDAR coverage restored','native_source':source.as_posix(),
            'groups':groups,'source_to_common':transforms,
            'retained_surface_area_m2':sum(l['area_weighted_sample_support']['area_m2'] for l in previous['levels']),
            'rectangular_surface_area_m2':current['summary']['retained_area_m2'],
            'restored_groups_vs_rectangles':len(groups)-current['summary']['retained_source_objects'],
            'note':'No source surfaces added or removed. Only rigid level-frame placement and display settings changed.'}
    path=out/'restore_manifest.json';path.write_text(json.dumps(config,indent=2))
    ruby=ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb'
    native=out/'Soulace_LiDAR_coverage_restored.skp'
    result=invoke(f'load {json.dumps(ruby.as_posix())}; PCMCoverageBaseline.restore({json.dumps(path.as_posix())},{json.dumps(native.as_posix())})')
    if not isinstance(result,dict):
        raise RuntimeError(result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
