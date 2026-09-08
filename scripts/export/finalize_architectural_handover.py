"""Summarize completed native checks and validate the parking/floor contacts."""
from pathlib import Path
import json
import numpy as np
import shapely

ROOT=Path(__file__).resolve().parents[2]


def main():
    soulace=ROOT/'output_final/soulace_architectural_v9'
    flat=ROOT/'output_final/architectural_flow_astra/koushik_mujammel_final'
    native=json.loads((soulace/'native_sdk_audit.json').read_text())
    floor=json.loads((soulace/'floor_geometry_checks.json').read_text())
    policy=json.loads((soulace/'visibility_policy.json').read_text())
    groups={g['name']:g for g in native['groups']}
    assert native['reopened_geometry_and_visibility_pass'] and native['native_area_pass']
    assert floor['all_valid_closed'] and not floor['coplanar_floor_overlaps'] and floor['unintended_pore_holes']==0
    assert all(groups[n]['hidden'] and groups[n]['reference_only'] for n in policy['paired_solid_references'])
    assert all(groups[n]['hidden'] for n in policy['demoted_stair_wall_hypotheses'])
    ground=groups['Ground - ONE CONTINUOUS PLANE (simplified)']
    assert ground['faces']==1 and not ground['hidden']
    parts=json.loads((soulace/'reopened_visible.build.json').read_text())['parts']
    triangles=[]
    for p in parts:
        if p['kind']=='floor_wall_joined' and p['level']==0:
            t=np.asarray(p['v'])[np.asarray(p['f'])]
            triangles.extend(t[np.all(abs(t[:,:,2])<1e-7,axis=1),:,:2])
    region=shapely.union_all(shapely.polygons(np.asarray(triangles)))
    contacts=[]
    for part in parts:
        if part['kind']!='plinth_observed_surface':continue
        v=np.asarray(part['v']);top=v[abs(v[:,2])<1e-7]
        gap=shapely.distance(shapely.points(top[:,:2]),region)
        contacts.append({'name':part['name'],'top_vertex_to_slab_xy_gap_max_mm':float(gap.max()*1000),
                         'base_to_ground_datum_gap_mm':abs(float(v[:,2].min())+.5334)*1000})
    assert len(contacts)==2 and all(c['top_vertex_to_slab_xy_gap_max_mm']<1e-4 and c['base_to_ground_datum_gap_mm']<1e-4 for c in contacts)
    report={'native_file':native['path'],'verified_native_groups':len(groups),
        'unsupported_paired_wall_solids_hidden':len(policy['paired_solid_references']),
        'stair_conflicting_wall_hypotheses_hidden':policy['demoted_stair_wall_hypotheses'],
        'single_cyan_ground_faces':ground['faces'],'floor_solids_checked':len(floor['finished_floors']),
        'coplanar_floor_overlaps':0,'small_floor_pores_remaining':0,'parking_contacts':contacts,
        'site_accuracy_certified':False,'preview_type':'CPU render of native-model geometry; not UI screenshot',
        'limitations':'Larger modeled floor openings and scan-unsupported gaps remain. Scan proximity does not certify wall identity or ±10mm site dimensions.'}
    (soulace/'handover_checks.json').write_text(json.dumps(report,indent=2))
    flat_native=json.loads((flat/'native_sdk_audit.json').read_text())
    flat_floor=json.loads((flat/'floor_cleanup_audit.json').read_text())
    assert flat_native['native_area_pass'] and flat_native['reopened_geometry_and_visibility_pass']
    assert sum(r['degenerate_triangles'] for r in flat_floor['after']['floors'])==0
    flow=json.loads((flat/'flow_report.json').read_text())
    flow.update(native_export_pending=False,native_file=flat_native['path'],native_backend='sdk',
                native_checks='native_sdk_audit.json',preview_type='Native-geometry CPU render, not UI screenshot')
    (flat/'flow_report.json').write_text(json.dumps(flow,indent=2))
    print(json.dumps({'soulace':report,'flat_native':flat_native['path'],'flat_numerical_floor_defects':0},indent=2))

if __name__=='__main__':main()
