"""Check paired native wall faces and floor contacts, not just closed solids."""
import argparse
import json
import numpy as np
import shapely
from shapely.geometry import Point,LineString,box
from shapely.ops import unary_union
import trimesh
from check_soulace_floor_junctions import ROOT,SOURCE,OUT
from build_soulace_wall_floor_planes import polygons,physical_wall_bases
from export_rectangular_skp import invoke
from verify_observed_recovery import main as verify_native

NATIVE='Soulace_complete_wall_planes_and_floor_junctions.skp'


def contact_check(parts,label):
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.array(cfg['source_to_common']['2']);tops=[]
    for p in parts:
        if p['kind']!='floor_wall_joined':continue
        v=(np.array(p['v'])-matrix[:3,3])@matrix[:3,:3];v[:,2]-=6.5
        tri=v[np.array(p['f'])];top=tri[np.all(abs(tri[:,:,2])<1e-6,axis=1)]
        assert len(top)>0
        tops.extend(shapely.polygons(top[:,:,:2]))
    floor=shapely.union_all(tops)
    protected=shapely.from_wkt((OUT/'protected_floor_voids.wkt').read_text())
    assert floor.intersection(protected.buffer(-.0002)).area<1e-8
    profiles=json.loads((SOURCE/'wall_profiles.json').read_text());footprints=[]
    for p in profiles:
        r=p['run'];a,b=r['along'];c,d=r['cross'];footprints.append(box(c,a,d,b) if r['axis']==0 else box(a,c,b,d))
    walls=unary_union(footprints);rows=[];unjoined=[]
    for p in profiles:
        r=p['run'];axis=r['axis'];along=1-axis;a,b=r['along'];profile=shapely.from_wkt(p['polygon_wkt'])
        for side,d in zip((-1,1),r['cross']):
            gaps=[];voids=0;without_floor=0;pending=[]
            for u in np.arange(a+.15,b-.149,.05):
                if not profile.covers(Point(u,.03)):continue
                xy=np.zeros(2);xy[axis]=d;xy[along]=u;near=xy.copy();near[axis]+=side*.01
                immediately_outside=xy.copy();immediately_outside[axis]+=side*.0002
                if walls.contains(Point(immediately_outside)) or walls.contains(Point(near)):continue
                if protected.buffer(.0002).contains(Point(near)):voids+=1;continue
                end=xy.copy();end[axis]+=side*.65
                line=LineString([xy,end]);found=line.intersection(floor)
                if found.is_empty:without_floor+=1;continue
                gap=Point(xy).distance(found)
                if gap>.001:
                    pending.append({'xy':xy.tolist(),'gap_mm':gap*1000})
                else:gaps.append(gap)
            row={'wall':r['name'],'side':side,'joined_samples':len(gaps),'max_joined_gap_mm':float(max(gaps,default=0)*1000),
                 'intentional_void_samples':voids,'no_floor_in_search_direction':without_floor,'noncontact_candidates':pending}
            rows.append(row);unjoined.extend([{**x,'wall':r['name'],'side':side} for x in pending])
    bases,thresholds=physical_wall_bases(profiles);door_checks=[]
    sealed=unary_union([floor,bases]);small_holes=[]
    for poly in polygons(sealed):
        for hole in poly.interiors:
            h=shapely.Polygon(hole)
            if .005<h.area<.20 and h.intersection(protected).area<h.area*.9:
                small_holes.append({'area_m2':h.area,'bounds':h.bounds})
    assert not small_holes,small_holes
    for t in thresholds:
        t=t.difference(bases).difference(protected)
        if t.area<.005 or t.buffer(.06).intersection(floor).area<.005:continue
        missing=t.difference(floor.buffer(.0002)).area
        door_checks.append({'threshold_area_m2':t.area,'missing_floor_m2':missing})
        assert missing<.00001,door_checks[-1]
    # Explicit regression checks on the visible gaps reported by the user.
    must_join=[('wall_02',1),('wall_07',1),('wall_01',-1),('wall_03',1),('parapet_10',-1)]
    for name,side in must_join:
        row=next(r for r in rows if r['wall']==name and r['side']==side)
        assert row['joined_samples']>=20,row
        assert row['max_joined_gap_mm']<.2,row
        assert not row['noncontact_candidates'],row
    report={'source':label,'floor_top_area_m2':floor.area,'protected_void_overlap_m2':floor.intersection(protected.buffer(-.0002)).area,
            'contact_sampling_m':.05,'joined_samples':sum(r['joined_samples'] for r in rows),
            'max_joined_gap_mm':max(r['max_joined_gap_mm'] for r in rows),'faces':rows,
            'doorway_floor_checks':door_checks,
            'remaining_small_internal_sampling_holes':small_holes,
            'remaining_noncontact_candidates':unjoined,'note':'Contact accuracy is internal CAD consistency, not independent survey accuracy. Exterior/void sides without adjacent modeled floors are not filled.'}
    (OUT/f'{label}_junction_contacts.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k not in ('faces','remaining_noncontact_candidates')},flush=True)
    print('Remaining candidates by wall:',{r['wall']+str(r['side']):[len(r['noncontact_candidates']),max(x['gap_mm'] for x in r['noncontact_candidates'])] for r in rows if r['noncontact_candidates']},flush=True)
    return report


def offline():
    data=json.loads((OUT/'additions.build.json').read_text());expected=json.loads((OUT/'plane_build_audit.json').read_text())
    assert len([p for p in data['parts'] if p['kind']=='wall_paired_planes'])==21
    for p in data['parts']:
        mesh=trimesh.Trimesh(p['v'],p['f'],process=True)
        assert mesh.is_watertight,p['name']
        if p['kind']=='wall_paired_planes':
            old=next(x for x in json.loads((SOURCE/'walls.build.json').read_text())['parts'] if x['name']==p['name'].replace('L2 paired wall planes ','L2 continuous '))
            assert p['v']==old['v'] and p['f']==old['f']
    return contact_check(data['parts'],'build')


def native():
    verify_native('soulace',folder_override=str(OUT),native_name=NATIVE)
    target=OUT/'native_plane_checks.json'
    code='''
      m=Sketchup.active_model;groups=m.entities.grep(Sketchup::Group);
      selected=groups.select{|g|['wall_paired_planes','floor_wall_joined'].include?(g.get_attribute('CoverageBaseline','kind',''))};
      rows=selected.map do |g|
        faces=g.entities.grep(Sketchup::Face);
        sides=['A','B'].map do |side|
          sf=faces.select{|f| f.get_attribute('WallPlane','side','')==side};
          {side:side,faces:sf.length,area_m2:sf.sum{|f|f.area}/(PCMCoverageBaseline::SCALE**2)}
        end;
        vertices=[];triangles=[];
        if g.get_attribute('CoverageBaseline','kind','')=='floor_wall_joined'
          faces.each do |f|
            mesh=f.mesh(0);start=vertices.length;
            vertices.concat(mesh.points.map{|p|p.transform(g.transformation).to_a.map{|c|c/PCMCoverageBaseline::SCALE}});
            mesh.polygons.each do |poly|
              ids=poly.map{|i|start+i.abs-1};(1...ids.length-1).each{|i|triangles<<[ids[0],ids[i],ids[i+1]]};
            end;
          end;
        end;
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),manifold:g.manifold?,volume_m3:g.volume/(PCMCoverageBaseline::SCALE**3),
         faces:faces.length,hidden:g.hidden?,sides:sides,v:vertices,f:triangles}
      end;
      ground=groups.find{|g|g.name==GROUND};raise 'Ground missing' unless ground;
      payload={path:m.path,parts:rows,ground_faces:ground.entities.grep(Sketchup::Face).length,
        ground_edges:ground.entities.grep(Sketchup::Edge).length,ground_loops:ground.entities.grep(Sketchup::Face).map{|f|f.loops.length}};
      File.write(TARGET,JSON.generate(payload));{parts:rows.length,manifold:rows.count{|r|r[:manifold]},wall_side_faces:rows.sum{|r|r[:sides].sum{|s|s[:faces]}}}.to_json
    '''
    ground=json.loads((ROOT/'output_final/soulace_single_ground_plane_v3/single_ground.config.json').read_text())
    print(invoke(code.replace('GROUND',json.dumps(ground['group_name'])).replace('TARGET',json.dumps(target.as_posix()))),flush=True)
    data=json.loads(target.read_text());audit=json.loads((OUT/'plane_build_audit.json').read_text())
    expected={p['name']:p for p in audit['parts']}
    assert len(data['parts'])==len(expected)
    for p in data['parts']:
        assert p['manifold'] and not p['hidden'],p['name']
        assert abs(p['volume_m3']-expected[p['name']]['volume_m3'])<.0001,p['name']
        if p['kind']=='wall_paired_planes':
            a,b=p['sides'];assert a['faces']>0 and b['faces']>0
            assert abs(a['area_m2']-b['area_m2'])<1e-5,p['name']
    assert data['ground_faces']==1 and data['ground_edges']==4 and data['ground_loops']==[1]
    report=contact_check(data['parts'],'native')
    summary=json.loads((OUT/'native_audit.json').read_text());assert summary['addition_area_pass']
    summary.update(paired_wall_planes_verified=21,closed_new_solids_verified=len(data['parts']),wall_floor_contact_verified=True,
                   tested_joined_samples=report['joined_samples'],max_tested_joined_gap_mm=report['max_joined_gap_mm'],ground_single_face_unchanged=True)
    (OUT/'native_audit.json').write_text(json.dumps(summary,indent=2));print('Native paired planes, solids and floor contacts verified.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--native',action='store_true');args=ap.parse_args()
    if args.native:native()
    else:offline()
