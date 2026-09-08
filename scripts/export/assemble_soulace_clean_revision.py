"""Keep previous geometry as reference, append clean floor/upper-stair surfaces."""
import json
from clean_soulace_surfaces import ROOT,OUT


def main():
    if list(OUT.glob('Soulace*.skp')):raise FileExistsError('Native revision already exists; use a new revision')
    prior=ROOT/'output_final/scan_recovered_v1/soulace'
    old=json.loads((prior/'additions.build.json').read_text())['parts']
    reference=[p['name'] for p in old if 'floor' in p['kind'] or (p.get('level',0)>0 and 'stair' in p['kind'])]
    parts=json.loads((OUT/'floors.build.json').read_text())['parts']+json.loads((OUT/'upper_stairs.build.json').read_text())['parts']
    payload={'label':'Soulace | planar ground and measured upper stairs','source_native':str(prior/'Soulace_stairs_and_lower_floor.skp'),
             'expected_base_groups':302,'reference_source_names':reference,'parts':parts,
             'source_label':'Measured plane reconstruction; explicit small-gap interpolation; see scan_proximity_audit.json',
             'small_hole_interpolation':True,
             'note':'Clean planar ground / upper stairs. Previous 302 groups preserved; superseded rough floor and upper staircase retained as hidden reference. Small sampling gaps interpolated, larger voids retained. Not certified site dimensions.'}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    print({'new_parts':len(parts),'new_triangles':sum(len(p['f']) for p in parts),'hidden_original_reference':reference})


if __name__=='__main__':main()
