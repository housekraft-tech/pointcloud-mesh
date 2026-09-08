"""Read-only native floor/wall extraction for the Soulace junction diagnosis."""
import json
from pathlib import Path
import numpy as np
from export_rectangular_skp import invoke
import shapely
from shapely.geometry import box,Point,LineString,Polygon
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output_final/soulace_top_floor_closed_v6'
OUT=ROOT/'output_final/soulace_wall_floor_junctions_v7'


def extract():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'native_junction_source.json'
    if target.exists():
        return json.loads(target.read_text())
    code='''
      m=Sketchup.active_model;
      raise 'Expected v6 source model; no model switch attempted' unless File.basename(m.path)=='Soulace_closed_top_floor_and_clean_stairs.skp';
      groups=m.entities.grep(Sketchup::Group);
      summaries=groups.map do |g|
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),level:g.get_attribute('CoverageBaseline','level',0),
         reference:g.get_attribute('CoverageBaseline','reference_only',false),hidden:g.hidden?,
         bounds:[g.bounds.min.to_a,g.bounds.max.to_a].map{|p|p.map{|c|c/39.37007874015748}},transform:g.transformation.to_a}
      end;
      selected=groups.select do |g|
        kind=g.get_attribute('CoverageBaseline','kind','');
        g.get_attribute('CoverageBaseline','level',0)==2 && !g.get_attribute('CoverageBaseline','reference_only',false) &&
          (kind.include?('floor') || kind=='wall_solid_continuous')
      end;
      parts=selected.map do |g|
        vs=[];fs=[];ids={};
        g.entities.grep(Sketchup::Face).each do |face|
          mesh=face.mesh(0); local=[];
          mesh.points.each do |p|
            xyz=p.transform(g.transformation).to_a.map{|c|c/39.37007874015748};key=xyz.map{|c|(c*1e8).round};
            id=ids[key];if id.nil?;id=vs.length;ids[key]=id;vs<<xyz;end;local<<id;
          end;
          mesh.polygons.each do |poly|
            poly=poly.map{|i|local[i.abs-1]};(1...poly.length-1).each{|i|fs<<[poly[0],poly[i],poly[i+1]]};
          end;
        end;
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),level:2,v:vs,f:fs}
      end;
      payload={path:m.path,modified:m.modified?,parts:parts,groups:summaries};
      File.write(TARGET,JSON.generate(payload));{path:m.path,parts:parts.length}.to_json
    '''.replace('TARGET',json.dumps(target.as_posix()))
    print(invoke(code),flush=True)
    return json.loads(target.read_text())


def main():
    data=extract()
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.array(cfg['source_to_common']['2'])
    floors=[];plane_z=[];floor_rows=[]
    for p in data['parts']:
        if 'floor' not in p['kind']:continue
        v=(np.array(p['v'])-matrix[:3,3])@matrix[:3,:3];v[:,2]-=6.5
        tri=v[np.array(p['f'])];top=tri[(np.ptp(tri[:,:,2],axis=1)<1e-6)&(abs(tri[:,:,2].mean(1))<.005)]
        shapes=shapely.polygons(top[:,:,:2]);poly=shapely.union_all(shapes)
        floors.append(poly);plane_z.extend(top[:,:,2].ravel().tolist())
        floor_rows.append({'name':p['name'],'area_m2':poly.area,'wkt':poly.wkt})
    floor=unary_union(floors)
    profiles=json.loads((SOURCE/'wall_profiles.json').read_text())
    footprints=[]
    for p in profiles:
        r=p['run'];a,b=r['along'];c,d=r['cross'];footprints.append(box(c,a,d,b) if r['axis']==0 else box(a,c,b,d))
    allwalls=unary_union(footprints);rows=[];lines=[]
    for p in profiles:
        r=p['run'];profile=shapely.from_wkt(p['polygon_wkt']);axis=r['axis'];along=1-axis;a,b=r['along']
        for side,d in zip((-1,1),r['cross']):
            samples=[];pts=[]
            for u in np.arange(a+.15,b-.149,.1):
                if not profile.covers(Point(u,.03)):continue
                xy=np.zeros(2);xy[axis]=d;xy[along]=u
                near=xy.copy();near[axis]+=side*.008
                if allwalls.contains(Point(near)):continue
                end=xy.copy();end[axis]+=side*.65
                intersection=LineString([xy,end]).intersection(floor)
                if intersection.is_empty:continue
                gap=Point(xy).distance(intersection)
                samples.append(gap);pts.append([*xy,gap])
            if samples:
                rows.append({'wall':r['name'],'side':side,'cross_coordinate_m':d,'sample_count':len(samples),
                             'median_gap_mm':float(np.median(samples)*1000),'p95_gap_mm':float(np.percentile(samples,95)*1000),
                             'max_gap_mm':float(max(samples)*1000),'samples_over_10mm':int((np.array(samples)>.01).sum()),
                             'sample_xy_gap_m':pts})
                lines.extend(pts)
    raw=np.load(ROOT/'output_final/soulace_top_floor_walls_v5/local_raw.npy',mmap_mode='r')
    raw=raw[(abs(raw[:,2])<.12)]
    for r in floor_rows:
        poly=shapely.from_wkt(r['wkt']).buffer(-.15)
        p=raw[shapely.contains_xy(poly,raw[:,0],raw[:,1])]
        if len(p):r.update(raw_near_floor_points=len(p),raw_z_p05_median_p95_mm=(np.percentile(p[:,2],[5,50,95])*1000).tolist())
    report={'native_source':data['path'],'native_model_modified':data['modified'],
            'floor_top_local_z_range_m':[min(plane_z),max(plane_z)],'wall_bottom_local_z_m':0,
            'floor_top_union_area_m2':floor.area,'wall_face_gap_sampling_m':.1,'max_search_gap_m':.65,
            'note':'Horizontal ray gaps to existing top faces, excluding wall intersections and door openings; not all samples denote intended floor adjacency.',
            'faces':rows,'floors':floor_rows}
    (OUT/'junction_diagnosis.json').write_text(json.dumps(report,indent=2))
    (OUT/'existing_floor_outline.wkt').write_text(floor.wkt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    fig,ax=plt.subplots(figsize=(14,10),layout='constrained')
    for poly in getattr(floor,'geoms',[floor]):
        ax.add_patch(Patch(np.array(poly.exterior.coords),facecolor='#bdc6c8',edgecolor='none'))
        for hole in poly.interiors:ax.add_patch(Patch(np.array(hole.coords),facecolor='white',edgecolor='none'))
    for p,foot in zip(profiles,footprints):
        ax.add_patch(Patch(np.array(foot.exterior.coords),facecolor='#d9c69f',edgecolor='#776952',lw=.5))
        c=foot.centroid;ax.text(c.x,c.y,p['run']['name'].replace('wall_','w').replace('parapet_','p'),fontsize=7)
    q=np.array(lines);sc=ax.scatter(q[:,0],q[:,1],c=np.minimum(q[:,2]*1000,300),s=8,cmap='inferno',vmin=0,vmax=300)
    fig.colorbar(sc,ax=ax,label='Gap to existing floor top along wall normal (mm; capped at 300)')
    ax.set_aspect('equal');ax.autoscale_view();ax.set_title('Native v6 wall-floor junctions: lateral gaps, not a height offset');ax.set_xlabel('L2 local X (m)');ax.set_ylabel('L2 local Y (m)')
    fig.savefig(OUT/'junction_gaps_before.png',dpi=160);plt.close(fig)
    for r in rows:print({k:v for k,v in r.items() if k!='sample_xy_gap_m'},flush=True)
    for r in floor_rows:print({k:v for k,v in r.items() if k!='wkt'},flush=True)
    print('Native floor top / wall base local Z:',report['floor_top_local_z_range_m'],0,flush=True)


if __name__=='__main__':main()
