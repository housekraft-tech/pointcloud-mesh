"""Assemble level models using their recorded rigid frames, not nominal heights."""
import argparse
import json
from pathlib import Path
import numpy as np
from rectangular_rebuild import rigid


def main(folders, out):
    folders=[Path(p).resolve() for p in folders];out=Path(out).resolve()
    if (out/'model.build.json').exists():
        raise FileExistsError(out/'model.build.json')
    out.mkdir(parents=True,exist_ok=True)
    audits=[json.loads((p/'audit.json').read_text()) for p in folders]
    common=rigid(audits[0]['manifest']['scans'][0]['scan_to_model'])
    output=[];levels=[];clouds=[]
    for folder,audit in zip(folders,audits):
        local=rigid(audit['manifest']['scans'][0]['scan_to_model'])
        change=rigid(common@np.linalg.inv(local))
        level=audit['manifest']['level']
        data=json.loads((folder/'model.build.json').read_text())
        for part in data['parts']:
            v=np.asarray(part['v'])@change[:3,:3].T+change[:3,3]
            output.append({**part,'v':np.round(v,8).tolist(),'name':f'L{level}_{part["name"]}',
                           'source_object':f'L{level}_{part["source_object"]}','level':level})
        raw=np.load(folder/'scan_display.npz')['points']
        clouds.append(raw@change[:3,:3].T+change[:3,3])
        levels.append({'level':level,'source_folder':str(folder),'local_to_common':change.tolist(),
                       'local_zero_elevation_in_common_m':float(change[2,3]),'summary':audit['summary'],
                       'strict_cell_distance_bound_mm':audit['strict_cell_distance_bound_mm']})
    total=sum(l['summary']['retained_area_m2'] for l in levels)
    support={f'within_{t}mm_pct':sum(l['summary']['area_weighted_scan_support'][f'within_{t}mm_pct']*l['summary']['retained_area_m2'] for l in levels)/total for t in [10,20,30,50]}
    record={'label':'Soulace entire house - strict rectangular support','levels':levels,
            'summary':{'rectangles':len(output),'retained_area_m2':total,'area_weighted_support':support,
                       'retained_source_objects':len({p['source_object'] for p in output})},
            'assembly':'Recorded raw-to-local frames mapped into the L0 frame. No nominal storey offsets or new registration.',
            'limitations':['Incomplete candidate-guided surface model, not watertight solids.',
                           'No inferred continuity; nearest-return support does not establish architectural identity.',
                           'No independent room-dimension accuracy certification.']}
    (out/'model.build.json').write_text(json.dumps({'label':record['label'],'parts':output}))
    (out/'assembly_audit.json').write_text(json.dumps(record,indent=2))
    np.savez_compressed(out/'scan_display.npz',points=np.concatenate(clouds))
    print(json.dumps(record['summary'],indent=2))
    print('Storey local-zero offsets:',[l['local_zero_elevation_in_common_m'] for l in levels])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--levels',nargs='+',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();main(args.levels,args.out)
