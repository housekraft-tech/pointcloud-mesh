"""Summarize already verified native files and compose their real viewport images."""
import json
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/scan_recovered_v1'


def main():
    rows=[]
    for project in ['engrance','soulace']:
        folder=OUT/project;audit=json.loads((folder/'native_audit.json').read_text())
        assert audit['addition_area_pass'] and audit['native_reopen_verified']
        assert audit['existing_geometry_preserved']
        path=Path(audit['path'])
        rows.append({'project':project,'path':str(path),'bytes':path.stat().st_size,'groups':len(audit['groups']),
                     'faces':sum(g['faces'] for g in audit['groups']),'scenes':len(audit['scenes']),
                     'native_reopen_verified':True,'added_surface_area_checks_passed':True,'baseline_preserved':True})
    (OUT/'handover_verification.json').write_text(json.dumps(rows,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(17,8),layout='constrained')
    for a,path,title in zip(ax,[OUT/'soulace/detail_observed_stairs_3d.png',OUT/'soulace/detail_ground_and_floors_3d.png'],
                             ['Soulace: observed stair surfaces','Soulace: interior floor + recovered lower floor (blue)']):
        a.imshow(Image.open(path));a.set_title(title,fontsize=13);a.axis('off')
    fig.suptitle('Actual SketchUp views | stair-side skins hidden for inspection; geometry retained',fontsize=16)
    fig.savefig(OUT/'soulace/Soulace_stairs_and_floors_review.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
