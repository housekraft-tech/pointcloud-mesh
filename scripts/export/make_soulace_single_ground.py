"""User-requested simplified ground: one horizontal SketchUp face, no holes.

This is deliberately a modelling simplification, not a scan-only reconstruction.
The raised interior and all stairs remain untouched. Superseded ground stays as
hidden reference in a new native file; no existing handover is overwritten.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output_final/soulace_clean_floor_stairs_v2'
OUT=ROOT/'output_final/soulace_single_ground_plane_v3'
NATIVE='Soulace_one_ground_plane.skp'


def single_ground_rectangle(vertices,datum):
    """A single continuous sheet covering the prior ground's XY envelope."""
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3:
        raise ValueError('Expected at least three XYZ vertices')
    if not np.isfinite(vertices).all() or not np.isfinite(datum):
        raise ValueError('Nonfinite coordinate')
    lo=vertices[:,:2].min(0);hi=vertices[:,:2].max(0)
    if np.any(hi-lo<=0):raise ValueError('Degenerate ground envelope')
    return np.array([[lo[0],lo[1],datum],[hi[0],lo[1],datum],
                     [hi[0],hi[1],datum],[lo[0],hi[1],datum]])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/NATIVE).exists():raise FileExistsError('Use a new revision; native handover exists')
    floors=json.loads((SOURCE/'floors.build.json').read_text())['parts']
    vertices=np.vstack([p['v'] for p in floors])
    scan=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
    p=scan['p'];parking=(scan['horizontal']&(p[:,0]>3)&(p[:,0]<5)&(p[:,1]>-3)&(p[:,1]<-1)&(p[:,2]>-.65)&(p[:,2]<-.4))
    q=p[parking]
    if len(q)<100:raise ValueError('Parking datum has insufficient source evidence')
    datum=round(float(np.median(q[:,2])),4)
    outline=single_ground_rectangle(vertices,datum)
    prior=json.loads((SOURCE/'native_audit.json').read_text())
    ref_names=sorted(set(prior['reference_only_groups']+[p['name'] for p in floors]))
    config={
        'source_native':str(SOURCE/'Soulace_clean_floor_and_upper_stairs.skp'),
        'destination':str(OUT/NATIVE),'expected_base_groups':len(prior['groups']),
        'reference_source_names':ref_names,'outline_m':outline.tolist(),
        'group_name':'Ground - ONE CONTINUOUS PLANE (simplified)',
        'kind':'floor_single_plane','colour':[85,161,187],
        'label':'Soulace | single continuous ground plane',
        'datum_m':datum,'parking_datum_samples':len(q),
        'area_m2':float(np.prod(np.ptp(outline[:,:2],axis=0))),
        'original_cyan_vertex_z_min_max_m':[float(vertices[:,2].min()),float(vertices[:,2].max())],
        'original_cyan_vertex_adjustment_min_max_mm':(np.array([(datum-vertices[:,2]).min(),(datum-vertices[:,2]).max()])*1000).tolist(),
        'modelling_choice':'One horizontal rectangular sheet at the sampled parking datum, covering the prior cyan XY envelope and continuing below the raised house. All ground holes and patch seams removed.',
        'interior_and_stairs_unchanged':True,'reference_geometry_retained':True,
        'scan_only_reconstruction':False,'survey_accuracy_claimed':False,
        'note':'USER-REQUESTED SIMPLIFICATION: single horizontal ground sheet. Rectangular XY envelope and filled/occluded areas are modelling assumptions, not new scan evidence. Ground falls/level changes intentionally replaced; raised interior and stairs unchanged.'}
    (OUT/'single_ground.config.json').write_text(json.dumps(config,indent=2))
    (OUT/'modelled_plane.build.json').write_text(json.dumps({'parts':[{'name':config['group_name'],'kind':config['kind'],
        'level':0,'colour':config['colour'],'v':outline.tolist(),'f':[[0,1,2],[0,2,3]]}]},separators=(',',':')))
    print(json.dumps({k:config[k] for k in ['datum_m','area_m2','parking_datum_samples','modelling_choice']},indent=2))
    return config


def export():
    config=prepare()
    ruby=ROOT/'scripts/export/ruby/pcm_single_ground.rb'
    result=invoke(f'load {json.dumps(ruby.as_posix())}; PCMSingleGround.build({json.dumps((OUT/"single_ground.config.json").as_posix())})')
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.export:export()
    else:prepare()
