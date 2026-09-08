"""Separate stair-side scan skins for inspection, without deleting any triangle."""
import json
from pathlib import Path
import numpy as np
from recover_mesh_evidence import mesh_part

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'output_final/scan_recovered_v1/soulace'


def main():
    if list(FOLDER.glob('*.skp')):raise FileExistsError('Do not modify an exported source')
    source=FOLDER/'additions.build.json';data=json.loads(source.read_text())
    if any(p['kind']=='stair_context' for p in data['parts']):return
    parts=[];before=0;after=0
    for part in data['parts']:
        before+=len(part['f'])
        if part['kind']!='stair_observed':parts.append(part);after+=len(part['f']);continue
        tri=np.array(part['v'])[np.array(part['f'])]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-15)
        context=(abs(normal[:,0])>.72)|((tri.mean(axis=1)[:,1]>5.42)&(abs(normal[:,1])>.72))
        for mask,kind,colour in [(~context,'stair_observed',[216,133,64]),(context,'stair_context',[155,164,167])]:
            if not mask.any():continue
            name=part['name'] if kind=='stair_observed' else part['name'].replace('surfaces','side-context scan skins')
            p=mesh_part(tri[mask],name,kind,colour);p['level']=part['level'];parts.append(p);after+=len(p['f'])
    assert before==after
    data['parts']=parts;data['stair_context_partition']='View-only classification by normals: no triangles removed; context is retained on separate tags.'
    source.write_text(json.dumps(data,separators=(',',':')))
    audit=json.loads((FOLDER/'recovery_audit.json').read_text());audit['context_partition_preserved_triangle_count']=before
    (FOLDER/'recovery_audit.json').write_text(json.dumps(audit,indent=2));print('Preserved',before,'triangles; now',len(parts),'parts',flush=True)


if __name__=='__main__':main()
