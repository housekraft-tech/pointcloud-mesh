"""Wall-side planes and floor contacts derived from the native v6 geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import box,Polygon,Point,LineString
from shapely.ops import unary_union
import trimesh
from check_soulace_floor_junctions import ROOT,SOURCE,OUT


def polygons(g):
    if g.is_empty:return []
    if g.geom_type=='Polygon':return [g]
    return [p for p in g.geoms if p.geom_type=='Polygon']


def close_small_contacts(floor,walls,protected,max_gap=.03):
    """Close local wall-side strips only; never spread into protected voids."""
    near=floor.buffer(max_gap,join_style=2).intersection(walls.buffer(max_gap,join_style=2))
    return unary_union([floor,near]).difference(walls).difference(protected)


def physical_wall_bases(profiles):
    """Actual wall cross-section just above floor, excluding walk-through cuts."""
    bases=[];thresholds=[]
    for p in profiles:
        r=p['run'];axis=r['axis'];a,b=r['along'];c,d=r['cross']
        profile=shapely.from_wkt(p['polygon_wkt']);line=LineString([(a,.001),(b,.001)])
        for shapes,output in [(line.intersection(profile),bases),(line.difference(profile),thresholds)]:
            for item in getattr(shapes,'geoms',[shapes]):
                if item.geom_type!='LineString' or item.length<.001:continue
                aa,_,bb,_=item.bounds
                output.append(box(c,aa,d,bb) if axis==0 else box(aa,c,bb,d))
    return unary_union(bases),thresholds


def top_polygon(part):
    tri=np.asarray(part['v'])[np.asarray(part['f'])]
    z=tri[:,:,2].max()
    top=tri[np.all(abs(tri[:,:,2]-z)<1e-6,axis=1)]
    return shapely.union_all(shapely.polygons(top[:,:,:2]))


def inputs():
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.array(cfg['source_to_common']['2'])
    profiles=json.loads((SOURCE/'wall_profiles.json').read_text())
    footprints=[]
    for p in profiles:
        r=p['run'];a,b=r['along'];c,d=r['cross'];footprints.append(box(c,a,d,b) if r['axis']==0 else box(a,c,b,d))
    allwalls=unary_union(footprints)
    original=json.loads((ROOT/'output_final/soulace_asbuilt_v2/Soulace_L2_second_asbuilt.build.json').read_text())['parts']
    old_floors={p['name']:top_polygon(p) for p in original if p['kind']=='floor'}
    current=shapely.from_wkt((OUT/'existing_floor_outline.wkt').read_text())
    raw=np.load(ROOT/'output_final/soulace_top_floor_walls_v5/local_raw.npy',mmap_mode='r')
    raw=raw[abs(raw[:,2])<.08]
    return matrix,profiles,footprints,allwalls,old_floors,current,raw


def diagnose():
    matrix,profiles,footprints,allwalls,old,current,raw=inputs()
    free=box(*allwalls.bounds).buffer(1).difference(allwalls)
    domain=box(*allwalls.bounds).buffer(1)
    cells=[p for p in polygons(free) if p.area>.25 and p.boundary.distance(domain.boundary)>.01]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    fig,axes=plt.subplots(1,2,figsize=(20,10),layout='constrained')
    for ax in axes:
        p=raw[::10];ax.scatter(p[:,0],p[:,1],s=.3,c='#acb7bc')
        for f in footprints:ax.add_patch(Patch(np.array(f.exterior.coords),facecolor='#d4b47a',edgecolor='black',lw=.4))
        ax.set_aspect('equal');ax.set_xlim(-9.8,6.5);ax.set_ylim(-4.5,7.1)
    records=[]
    for i,(name,g) in enumerate(old.items()):
        for p in polygons(g):
            xy=np.array(p.exterior.coords);axes[0].plot(*xy.T,color=plt.cm.tab20(i%20),lw=1)
            for h in p.interiors:axes[0].plot(*np.array(h.coords).T,color=plt.cm.tab20(i%20),lw=1)
        c=g.representative_point();axes[0].text(c.x,c.y,name.replace('floor_',''),fontsize=8)
        records.append({'old_floor':name,'area_m2':g.area,'bounds':g.bounds})
    for i,p in enumerate(cells):
        xy=np.array(p.exterior.coords);axes[1].plot(*xy.T,lw=2);c=p.representative_point();axes[1].text(c.x,c.y,f'cell {i}\n{p.area:.1f} m²')
        pts=raw[shapely.contains_xy(p,raw[:,0],raw[:,1])]
        records.append({'cell':i,'area_m2':p.area,'current_overlap_m2':p.intersection(current).area,'bounds':p.bounds,
                        'z_p05_p50_p95_mm':(np.percentile(pts[:,2],[5,50,95])*1000).tolist() if len(pts) else [],'raw_points':len(pts)})
    axes[0].set_title('Original unfiltered floor proposals over raw floor returns');axes[1].set_title('Current wall-bounded cells: stairwell must remain open')
    fig.savefig(OUT/'floor_domain_diagnostics.png',dpi=140);plt.close(fig)
    (OUT/'floor_domain_diagnostics.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2),flush=True)


def rectangular_floor_domain(profiles,footprints,old,current,raw,matrix):
    """Evidence-selected rectangular cells, bounded by real wall-side coordinates.

    No global rectangle is blindly filled. Large unsupported cells and the
    measured stair opening remain empty. Support votes select *regions*; sparse
    point holes are not subsequently punched through accepted floor planes.
    """
    walls=unary_union(footprints)
    domain=box(*walls.bounds).buffer(1)
    rooms=[p for p in polygons(domain.difference(walls)) if p.area>.25 and p.boundary.distance(domain.boundary)>.01]
    stair_room=next(p for p in rooms if p.contains(Point(-3.8,4)))
    stairs=json.loads((ROOT/'output_final/soulace_clean_floor_stairs_v2/upper_stairs.build.json').read_text())['parts']
    landing=next(p for p in stairs if p['name']=='Upper floor arrival landing - measured plane')
    lv=(np.array(landing['v'])-matrix[:3,3])@matrix[:3,:3];lv[:,2]-=6.5
    landing_poly=top_polygon({**landing,'v':lv.tolist()})
    # The entire upper opening beyond the observed arrival landing is protected,
    # including strips beside the lower flights. It is not a lost floor patch.
    landing_edge=float(lv[:,1].max())
    stair_void=stair_room.intersection(box(-20,landing_edge,20,20))
    protected=[stair_void];void_reports=[{'reason':'measured_stairwell','wkt':stair_void.wkt}]
    accepted_rooms=[]
    for room in rooms:
        overlap=room.intersection(current).area/room.area
        if room==stair_room:accepted_rooms.append(room.difference(stair_void));continue
        if overlap<.15:
            protected.append(room);void_reports.append({'reason':'large_unsupported_enclosed_void','wkt':room.wkt,'existing_floor_fraction':overlap})
        else:accepted_rooms.append(room)
    protected=unary_union(protected)
    xs=[];ys=[]
    for f in footprints:
        a,b,c,d=f.bounds;xs.extend([a,c]);ys.extend([b,d])
    for f in old.values():
        a,b,c,d=f.bounds;xs.extend([a,c]);ys.extend([b,d])
    a,b,c,d=current.bounds;xs.extend([a,c]);ys.extend([b,d]);ys.append(landing_edge)
    xs=np.unique(np.round(xs,9));ys=np.unique(np.round(ys,9))
    # Sampling occupancy is counted once per 40 mm XY cell, not by repeated SLAM
    # visits. This prevents a scanned vertical edge from masquerading as a floor.
    p=raw[(raw[:,2]>-.075)&(raw[:,2]<.020)]
    keys=np.floor(p[:,:2]/.04).astype(int);_,idx=np.unique(keys,axis=0,return_index=True);p=p[idx]
    hist=np.histogram2d(p[:,0],p[:,1],bins=[xs,ys])[0]
    cells=[];scores=[]
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            area=(xs[i+1]-xs[i])*(ys[j+1]-ys[j])
            if area<1e-6:continue
            poly=box(xs[i],ys[j],xs[i+1],ys[j+1]);support=hist[i,j]*.04**2/area
            if support<.10:continue
            old_fraction=poly.intersection(current).area/area
            if support>=.30 or (support>=.12 and old_fraction>.18):
                cells.append(poly);scores.append({'bounds':poly.bounds,'support_fraction':min(support,1.),'old_fraction':old_fraction})
    region=unary_union(cells+accepted_rooms)
    # Retain small connected near-wall strips of the same accepted region; do
    # not restore isolated old fragments outside the modeled wall perimeter.
    region=region.difference(walls).difference(protected)
    # Doorways remain openings in the wall, and gain the adjoining floor plane.
    thresholds=[]
    for p in profiles:
        r=p['run'];axis=r['axis'];a,b=r['along'];c,d=r['cross'];profile=shapely.from_wkt(p['polygon_wkt'])
        section=LineString([(a,.001),(b,.001)]).difference(profile)
        for line in getattr(section,'geoms',[section]):
            if line.geom_type!='LineString' or line.length<.15:continue
            aa,_,bb,_=line.bounds
            cut=box(c,aa,d,bb) if axis==0 else box(aa,c,bb,d)
            if cut.buffer(.03).intersection(region).area>.002:thresholds.append(cut)
    region=shapely.set_precision(unary_union([region,*thresholds]),.0001)
    region=region.buffer(.003,join_style=2).buffer(-.003,join_style=2)
    # Temporarily include wall footprints while classifying holes: a tiny lost
    # floor patch touching a wall is still a sampling gap, not a large room void.
    sealed=unary_union([region,walls])
    region=unary_union([Polygon(p.exterior,[h for h in p.interiors if Polygon(h).area>.20]) for p in polygons(sealed)])
    region=region.difference(walls).difference(protected)
    # Avoid thin uncertain exterior slivers disconnected from any useful floor.
    region=unary_union([p for p in polygons(region) if p.area>.65])
    return region,protected,accepted_rooms,{'protected_voids':void_reports,'landing_edge_local_y_m':landing_edge,
            'accepted_grid_cells':len(cells),'cell_scores':scores,'room_count':len(accepted_rooms),
            'prior_floor_area_m2':current.area,'proposed_floor_area_m2':region.area,'protected_void_area_m2':protected.area}


def candidates():
    matrix,profiles,footprints,allwalls,old,current,raw=inputs()
    region,protected,rooms,audit=rectangular_floor_domain(profiles,footprints,old,current,raw,matrix)
    (OUT/'proposed_floor_outline.wkt').write_text(region.wkt)
    (OUT/'protected_floor_voids.wkt').write_text(protected.wkt)
    (OUT/'floor_region_audit.json').write_text(json.dumps(audit,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    fig,axes=plt.subplots(1,2,figsize=(20,10),layout='constrained')
    for ax,g,title in zip(axes,[current,region],['Existing floor gaps','Wall-plane-bounded floor proposal']):
        for poly in polygons(g):
            ax.add_patch(Patch(np.array(poly.exterior.coords),facecolor='#a9c5ce',edgecolor='#68808a',lw=.4))
            for h in poly.interiors:ax.add_patch(Patch(np.array(h.coords),facecolor='white',edgecolor='#68808a',lw=.4))
        for f in footprints:ax.add_patch(Patch(np.array(f.exterior.coords),facecolor='#d5bd94',edgecolor='#77694c',lw=.5))
        for f in polygons(protected):ax.add_patch(Patch(np.array(f.exterior.coords),facecolor='none',edgecolor='#c55050',hatch='//',lw=1))
        ax.set_aspect('equal');ax.set_xlim(-9.8,6.6);ax.set_ylim(-4.8,6.9);ax.set_title(title)
    fig.savefig(OUT/'floor_plane_proposal.png',dpi=150);plt.close(fig)
    print({k:v for k,v in audit.items() if k not in ('cell_scores','protected_voids')},flush=True)


def build(refine=False,amend=False):
    if list(OUT.glob('*.skp')) and not refine:raise FileExistsError('Use a new native revision')
    if (OUT/'Soulace_complete_wall_planes_and_floor_junctions.skp').exists() and not amend:raise FileExistsError('Final native revision already exists')
    matrix,profiles,footprints,allwalls,old,current,raw=inputs()
    region=shapely.from_wkt((OUT/'proposed_floor_outline.wkt').read_text())
    protected=shapely.from_wkt((OUT/'protected_floor_voids.wkt').read_text())
    if refine:
        before=region.area;bases,thresholds=physical_wall_bases(profiles)
        supported_thresholds=[t for t in thresholds if t.buffer(.06).intersection(region).area>.005]
        region=unary_union([region,*supported_thresholds]).difference(protected)
        region=close_small_contacts(region,bases,protected)
        sealed=unary_union([region,bases])
        region=unary_union([Polygon(p.exterior,[h for h in p.interiors if Polygon(h).area>.20]) for p in polygons(sealed)])
        region=region.difference(bases).difference(protected)
        (OUT/'refined_floor_outline.wkt').write_text(region.wkt)
        (OUT/'local_contact_refinement.json').write_text(json.dumps({'max_contact_closure_m':.03,'added_area_m2':region.area-before,
            'doorway_floor_sections_restored':len(supported_thresholds),'protected_void_overlap_m2':region.intersection(protected).area},indent=2))
    assert region.intersection(protected).area<1e-8
    prior=json.loads((SOURCE/'native_audit.json').read_text())
    references=set(prior['reference_only_groups'])
    references.update(p['name'] for p in json.loads((OUT/'native_junction_source.json').read_text())['parts'])
    source_walls=json.loads((SOURCE/'walls.build.json').read_text())['parts']
    parts=[];audit=[]
    # Exact copies of the existing wall geometry, with native coplanar face
    # consolidation requested. No measured wall face or opening is moved.
    for source,p in zip(source_walls,profiles):
        r=p['run'];axis=r['axis'];name=source['name'].replace('L2 continuous ','L2 paired wall planes ')
        direction=matrix[:3,axis];origin=matrix[:3,3]+matrix[:3,:3]@np.array([0,0,6.5])
        planes=[]
        for side,c in zip(('A','B'),r['cross']):
            measured=any(abs(c-d)<1e-5 for d in r['faces'])
            planes.append({'side':side,'normal':direction.tolist(),'offset_m':float(direction@origin+c),
                           'source':'observed_wall_face' if measured else 'inferred_back_face_thickness_unverified'})
        part={**source,'name':name,'kind':'wall_paired_planes','merge_coplanar_faces':True,'wall_plane_pair':planes}
        parts.append(part)
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
        audit.append({'name':name,'kind':part['kind'],'volume_m3':abs(mesh.volume),'area_m2':mesh.area,'plane_pair':planes,
                      'unchanged_geometry_from':source['name']})
    # A constant finish-floor plane joins the wall bases. Retain the existing
    # 100 mm nominal backing, explicitly modeled rather than surveyed thickness.
    floor_polys=polygons(shapely.set_precision(region,.0001))
    floor_polys=[p for p in floor_polys if p.area>.0025]
    assert abs(sum(p.area for p in floor_polys)-region.area)<.005
    for i,poly in enumerate(sorted(floor_polys,key=lambda p:-p.area),1):
        mesh=trimesh.creation.extrude_polygon(poly,height=.1,engine='earcut');mesh.vertices[:,2]-=.1
        mesh.merge_vertices();mesh.fix_normals()
        if not mesh.is_watertight:raise ValueError('Floor is not closed')
        common=(mesh.vertices+[0,0,6.5])@matrix[:3,:3].T+matrix[:3,3]
        name=f'L2 floor plane {i:02d} - joined to wall faces'
        part={'name':name,'kind':'floor_wall_joined','level':2,'colour':[174,174,166],
              'v':np.round(common,8).tolist(),'f':mesh.faces.tolist(),'merge_coplanar_faces':True,'construction_solid':True}
        parts.append(part);audit.append({'name':name,'kind':part['kind'],'volume_m3':abs(mesh.volume),'area_m2':mesh.area,
                                        'floor_top_area_m2':poly.area,'floor_top_local_z_m':0,'backing_thickness_m':.1,'backing_thickness_verified':False,'polygon_wkt':poly.wkt})
    payload={'label':'Soulace | inner and outer wall planes with joined floors',
             'source_native':str(SOURCE/'Soulace_closed_top_floor_and_clean_stairs.skp'),'expected_base_groups':392,
             'reference_source_names':sorted(references),'parts':parts,
             'render_image_names':['3D','L2 3D','L2 Top','L2 Front','L2 Side'],
             'source_label':'Paired planar wall faces retained from v6; evidence-selected rectangular floor regions joined to wall faces',
             'note':'Wall face positions, doors/windows, roof stair, lower floors and cyan ground are unchanged. Ten wall thicknesses remain unverified. Floor infill is modeled from wall boundaries and existing/raw floor support; stairwell and unsupported central void are not filled. Existing L2 elevation and 100 mm modeled floor backing retained, not independently certified.'}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    (OUT/'plane_build_audit.json').write_text(json.dumps({'parts':audit,'protected_void_overlap_m2':region.intersection(protected).area,
        'existing_geometry_preserved':True,'wall_plane_positions_unchanged':True,'floor_elevation_unchanged':True,'not_certified_site_accuracy':True},indent=2))
    print({'new_wall_solids':len(source_walls),'new_floor_solids':len(parts)-len(source_walls),'references':len(references),'triangles':sum(len(p['f']) for p in parts)},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--diagnose',action='store_true');ap.add_argument('--build',action='store_true');ap.add_argument('--refine',action='store_true');ap.add_argument('--amend',action='store_true');args=ap.parse_args()
    if args.diagnose:diagnose()
    elif args.build:build(refine=args.refine,amend=args.amend)
    else:candidates()
