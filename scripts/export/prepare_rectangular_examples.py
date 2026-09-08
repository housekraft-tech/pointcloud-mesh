"""Explicit project adapters: write manifests, never change the shared engine."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output_final/rectangular_rebuild_v1'


def rotation(yaw):
    angle = np.deg2rad(-yaw)
    c,s = np.cos(angle),np.sin(angle)
    r = np.eye(4)
    r[:2,:2] = [[c,-s],[s,c]]
    return r


def main():
    OUT.mkdir(exist_ok=True)
    frame = json.loads((ROOT/'cage_output/scores/variants.json').read_text())
    k = rotation(frame['yaw_deg'])
    k[:2,3] = -k[:2,:2]@np.asarray(frame['centre_xy'])
    registration = json.loads((ROOT/'output2/koushik_all/skeleton_3d/annotated/scan_transform.json').read_text())
    m = k@np.asarray(registration['transform'])
    muj = {'label':'Koushik - Mujammel rectangular rebuild', 'units':'metres', 'level':0,
           'candidate_model':str(ROOT/'output_final/mujammel_asbuilt_final_v6/Mujammel_asbuilt.build.json'),
           'scans':[{'id':'mujammel','path':str(ROOT/'mujammelexport.las'),'scan_to_model':m.tolist()},
                    {'id':'koushik','path':str(ROOT/'koushikexport.las'),'scan_to_model':k.tolist()}],
           'settings':{}, 'adapter_note':'Recorded Mujammel-to-Koushik registration, then the existing stable CAD frame; no new ICP or bias adjustment.'}
    old = json.loads((ROOT/'output_final/soulace_overlap_audit_20260903/Soulace_L0_overlap.json').read_text())['transform']
    r = rotation(old['yaw_deg'])
    r[2,3] = -old['las_z_percentile_0_5_m']-old['source_floor_z_m']
    soul = {'label':'Soulace L0 rectangular rebuild', 'units':'metres', 'level':0,
            'candidate_model':str(ROOT/'output_final/soulace_asbuilt_v2/Soulace_L0_ground_asbuilt.build.json'),
            'scans':[{'id':'soulace_l0','path':str(ROOT/'output_final/soulace_L0/lidar/L0.las'),'scan_to_model':r.tolist()}],
            'settings':{}, 'adapter_note':'Reuse the recorded L0 raw-to-CAD frame from the previous full LAS audit; no new alignment.'}
    examples=[('mujammel',muj),('soulace_l0',soul)]
    stems=['Soulace_L0_ground','Soulace_L1_first','Soulace_L2_second']
    for level in [1,2]:
        tr=json.loads((ROOT/f'output_final/soulace_overlap_audit_20260903/Soulace_L{level}_overlap.json').read_text())['transform']
        matrix=rotation(tr['yaw_deg']);matrix[2,3]=-tr['las_z_percentile_0_5_m']-tr['source_floor_z_m']
        item={'label':f'Soulace L{level} rectangular rebuild','units':'metres','level':level,
              'candidate_model':str(ROOT/f'output_final/soulace_asbuilt_v2/{stems[level]}_asbuilt.build.json'),
              'scans':[{'id':f'soulace_l{level}','path':str(ROOT/f'output_final/soulace_L{level}/lidar/L{level}.las'),'scan_to_model':matrix.tolist()}],
              'settings':{},'adapter_note':'Reuse recorded raw-to-CAD transform; no new alignment.'}
        examples.append((f'soulace_l{level}',item))
    for name,item in list(examples):
        strict={**item,'label':item['label']+' - strict support',
                'settings':{'allow_continuity':False,'grid_m':.025}}
        examples.append((name+'_strict',strict))
    for name,manifest in examples:
        path=OUT/f'{name}.manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise FileExistsError(path)
        path.write_text(json.dumps(manifest,indent=2))
        print(path)


if __name__ == '__main__':
    main()
