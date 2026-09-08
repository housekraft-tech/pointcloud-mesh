"""Combine continuous top walls and the clean roof staircase in a new native file."""
import json
from pathlib import Path
from shapely.geometry import box,Point
from shapely.ops import unary_union
from close_soulace_top_floor import ROOT,OUT,PRIOR


def main():
    if list(OUT.glob('Soulace*.skp')):raise FileExistsError('Native revision exists')
    walls=json.loads((OUT/'walls.build.json').read_text())['parts']
    stairs=json.loads((OUT/'roof_stair.build.json').read_text())['parts']
    old=json.loads((PRIOR/'native_audit.json').read_text())
    names=[g['name'] for g in old['groups']]
    references=set(old['reference_only_groups'])
    references.update(n for n in names if n.startswith('L2_wall_') or n.startswith('L2_parapet_') or n.startswith('L2 modeled '))
    references.add('Top-floor roof stair - crosswise scan reference')
    parts=walls+stairs
    payload={'label':'Soulace | continuous top-floor walls and clean roof staircase',
             'source_native':str(PRIOR/'Soulace_top_floor_wall_revision.skp'),'expected_base_groups':370,
             'reference_source_names':sorted(references),'parts':parts,
             'render_image_names':['3D','L2 3D','L2 Top','L2 Front','L2 Side'],
             'source_label':'Measured visible wall planes and stair riser/tread evidence; explicitly modeled closure in occluded areas',
             'small_hole_interpolation':True,
             'note':'21 continuous top-floor wall/parapet assemblies and a clean 15-level roof stair. Ten wall thicknesses retain unverified prior model values; 11 use paired faces. Wall junction closure and concealed surfaces are modeled. Doors/windows preserved. Roof stair lower undersides and landing closure inferred; upper tread heights estimated from observed riser tops. Not certified perfect or ±10 mm site accuracy. Cyan ground and lower-storey geometry unchanged.'}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    profiles=json.loads((OUT/'wall_profiles.json').read_text());footprints=[]
    for p in profiles:
        r=p['run'];a,b=r['along'];c,d=r['cross']
        footprints.append(box(c,a,d,b) if r['axis']==0 else box(a,c,b,d))
    allwalls=unary_union(footprints);domain=box(*allwalls.bounds).buffer(1)
    free=domain.difference(allwalls)
    rooms=[p for p in free.geoms if p.area>.25 and not p.boundary.intersects(domain.boundary)]
    core=next((p for p in rooms if p.contains(Point(-7,0))),None)
    if core is None:raise AssertionError('Large room still leaks through its wall boundary')
    qa={'doorways_temporarily_bridged_for_connectivity_test_only':True,'bounded_wall_layout_cells':len(rooms),
        'large_outer_room_boundary_closed':True,'large_outer_room_area_m2':core.area,
        'bounded_cell_areas_m2':sorted([p.area for p in rooms],reverse=True),
        'note':'Plan connectivity, not a claim that real door/window openings are filled or absolute dimensions verified.'}
    (OUT/'wall_connectivity_check.json').write_text(json.dumps(qa,indent=2))
    print({'parts':len(parts),'new_solids':len(parts),'prior_groups':370,'references':len(references),**qa},flush=True)


if __name__=='__main__':main()
