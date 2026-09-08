"""Reusable, conservative architectural refinement of registered CAD candidates.

Coordinates are metres.  This stage does not discover walls or turn scan
fragments into guessed boxes.  Wall face pairs use planar face area rather
than object bounds; profiles and pilasters remain part of the original mesh.
Floor unions preserve every existing void and elevation.  Optional scan trees
produce proximity evidence, never a claim of survey certification.
"""
from copy import deepcopy
import numpy as np
import shapely
from shapely.geometry import Polygon
import trimesh


def _query(reference, points):
    if reference is None:
        return None
    if isinstance(reference, dict):
        reference = reference.get("tree", reference.get("points"))
    if not hasattr(reference, "query"):
        from scipy.spatial import cKDTree
        reference = cKDTree(reference)
    return np.asarray(reference.query(points, workers=8)[0])


def _samples(triangles, limit=2400):
    """Area-weighted deterministic triangle samples; invariant to triangulation."""
    if not len(triangles):
        return np.empty((0, 3))
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    count = min(limit, max(200, int(area.sum()*120)))
    rng = np.random.default_rng(91023)
    ids = rng.choice(len(triangles), count, p=area/area.sum())
    uv = rng.random((count, 2)); uv[uv.sum(1)>1] = 1-uv[uv.sum(1)>1]
    selected = triangles[ids]
    return selected[:, 0]+uv[:, :1]*(selected[:, 1]-selected[:, 0])+uv[:, 1:]*(selected[:, 2]-selected[:, 0])


def proximity_evidence(triangles, reference):
    sample = _samples(np.asarray(triangles))
    distances = _query(reference, sample)
    if distances is None:
        return {"status": "raw_not_supplied", "samples": 0}
    return {"status": "raw_proximity_checked", "samples": len(distances),
            "within_10mm": float(np.mean(distances<=.01)),
            "within_30mm": float(np.mean(distances<=.03)),
            "within_50mm": float(np.mean(distances<=.05)),
            "median_mm": float(np.median(distances)*1000),
            "p95_mm": float(np.percentile(distances,95)*1000),
            "max_mm": float(distances.max()*1000)}


def wall_face_pair(part, expected_thickness=None, reference=None):
    """Extract two substantial parallel vertical faces, ignoring profile extent.

    expected_thickness is a candidate prior in metres, not survey evidence.
    Main-face offsets are area-ranked among vertical triangle plane clusters.
    """
    mesh = trimesh.Trimesh(part['v'], part['f'], process=False)
    normals = mesh.face_normals.copy(); vertical = np.abs(normals[:, 2])<.015
    if not vertical.any():
        return None
    normal = normals[np.flatnonzero(vertical)[np.argmax(mesh.area_faces[vertical])]]
    if normal[np.argmax(np.abs(normal))]<0: normal = -normal
    aligned = np.abs(normals@normal)>1-1e-7
    offsets = mesh.triangles_center@normal
    bins = np.round(offsets[aligned], 6)
    planes=[]
    for offset in np.unique(bins):
        ids=np.flatnonzero(aligned & (np.abs(offsets-offset)<2e-6))
        planes.append((float(np.average(offsets[ids], weights=mesh.area_faces[ids])),
                       float(mesh.area_faces[ids].sum()), ids))
    choices=[]
    for i,a in enumerate(planes):
        for b in planes[i+1:]:
            gap=b[0]-a[0]
            if gap<.05 or gap>.65: continue
            score=min(a[1],b[1])+.12*(a[1]+b[1])
            if expected_thickness:
                score /= 1+(abs(gap-expected_thickness)/.01)**2
            choices.append((score,a,b))
    if not choices: return None
    _,a,b=max(choices,key=lambda row:row[0])
    evidence=[proximity_evidence(mesh.triangles[item[2]],reference) for item in (a,b)]
    supported=all(e.get('within_30mm',0)>=.7 and e.get('within_50mm',0)>=.9 for e in evidence)
    return {"normal":normal.tolist(),"offsets_m":[a[0],b[0]],
            "thickness_m":b[0]-a[0],"face_areas_m2":[a[1],b[1]],
            "whole_object_cross_extent_m":float(np.ptp(np.asarray(part['v'])@normal)),
            "opposing_face_proximity_supported":supported,
            "thickness_verified":False,
            "status":"opposing_raw_proximity_supported" if supported else "candidate_face_pair_unverified",
            "side_evidence":evidence,
            "note":"Planar candidates with proximity checks; semantic wall identity and physical thickness are not independently certified."}


def _polygons(geometry):
    if geometry.is_empty:return []
    if geometry.geom_type=='Polygon':return [geometry]
    return [part for g in geometry.geoms for part in _polygons(g)]


def floor_top(part):
    vertices=np.asarray(part['v']); triangles=vertices[np.asarray(part['f'])]
    z=float(vertices[:,2].max())
    selected=triangles[np.all(np.abs(triangles[:,:,2]-z)<1e-6,axis=1)]
    if not len(selected):return None
    region=shapely.union_all(shapely.polygons(selected[:,:,:2]))
    return region,z,float(vertices[:,2].min())


def supported_candidate_replacement(current, candidate, reference,
                                    minimum_within_50mm=.90, minimum_within_30mm=.70):
    """Gate an existing registered candidate; never create a new object identity.

    Returns (selected_part, audit). Failure retains current geometry.  Passing
    means a sampled modeled continuation, not a fully observed construction
    solid.  Callers must retain the original as a recoverable reference.
    """
    mesh=trimesh.Trimesh(candidate['v'],candidate['f'],process=False)
    evidence=proximity_evidence(mesh.triangles,reference)
    accepted=(evidence.get('within_50mm',0)>=minimum_within_50mm and
              evidence.get('within_30mm',0)>=minimum_within_30mm)
    row={'name':current['name'],'candidate_name':candidate['name'],'accepted':accepted,
         'method':'registered_candidate_raw_proximity_gate','evidence':evidence,
         'thresholds':{'within_50mm':minimum_within_50mm,'within_30mm':minimum_within_30mm},
         'reason':'sampled_support_pass' if accepted else 'insufficient_raw_support_keep_current'}
    if not accepted:return deepcopy(current),row
    result={**deepcopy(candidate),'name':current['name'],
            'evidence_status':'sampled_supported_candidate_continuation',
            'support_evidence':evidence,'merge_coplanar_faces':True,
            'thickness_verified':False}
    return result,row


def regularize_planar_fragments(part, reference, maximum_boundary_shift=.015,
                               support_cutoff=.05, pore_area=.005):
    """Clean bounded scan-fragment edges while retaining genuine larger voids.

    Only small local closing/simplification, with strict sampled checks of
    the proposed surface and boundaries. No bounding boxes or new wall runs.
    Ineligible/failed patches are retained exactly. Original reference should
    also remain in the native document, hidden from the refined view.
    """
    from filter_soulace_overlap import mesh_from_polygon
    if reference is None:return deepcopy(part),{'name':part['name'],'changed':False,'reason':'raw_not_supplied'}
    source=trimesh.Trimesh(part['v'],part['f'],process=False)
    pieces=[];rows=[]
    for poly,origin,u,v in _all_planar_patches(source):
        radius=maximum_boundary_shift/2
        proposed=poly.buffer(radius,join_style=2).buffer(-radius,join_style=2)
        proposed=proposed.simplify(maximum_boundary_shift/3,preserve_topology=True)
        proposed=shapely.union_all([Polygon(p.exterior,[h for h in p.interiors if Polygon(h).area>pore_area])
                                   for p in _polygons(proposed)])
        # Retain all large source holes, even if a narrow bridge could close.
        holes=[Polygon(h) for p in _polygons(poly) for h in p.interiors if Polygon(h).area>pore_area]
        if holes:proposed=proposed.difference(shapely.union_all(holes))
        # Limit every boundary displacement, including corners of tiny islands.
        delta=proposed.symmetric_difference(poly).area
        shift=float(proposed.boundary.hausdorff_distance(poly.boundary))
        record={'source_area_m2':float(poly.area),'proposed_area_m2':float(proposed.area),
                'changed_area_m2':float(delta),'boundary_hausdorff_m':shift,'accepted':False}
        candidate=mesh_from_polygon(proposed,origin,u,v) if not proposed.is_empty else None
        if candidate is not None and delta>1e-9 and shift<=maximum_boundary_shift+1e-7:
            boundary=[]
            for polygon in _polygons(proposed):
                for ring in [polygon.exterior,*polygon.interiors]:
                    coords=np.asarray(ring.coords)
                    for a,b in zip(coords[:-1],coords[1:]):
                        boundary.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.005)+1)))
            xy=np.asarray(boundary)
            boundary3=origin+xy[:,:1]*u+xy[:,1:]*v
            evidence=proximity_evidence(candidate.triangles,reference)
            boundary_max=float(_query(reference,np.vstack([candidate.vertices,boundary3])).max())
            record.update(evidence=evidence,boundary_max_mm=boundary_max*1000)
            record['accepted']=evidence['max_mm']<=support_cutoff*1000 and boundary_max<=support_cutoff
        pieces.append(candidate if record['accepted'] else mesh_from_polygon(poly,origin,u,v))
        rows.append(record)
    changed=sum(row['accepted'] for row in rows)
    if not changed:return deepcopy(part),{'name':part['name'],'changed':False,'patches':rows}
    mesh=trimesh.util.concatenate([p for p in pieces if p is not None])
    mesh.merge_vertices(digits_vertex=8);mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices()
    result={**deepcopy(part),'v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),
            'merge_coplanar_faces':True,'evidence_status':'locally_regularized_scan_fragment',
            'support_cutoff_mm':support_cutoff*1000,'construction_solid':False}
    return result,{'name':part['name'],'changed':True,'changed_patches':changed,
                   'before_triangles':len(source.faces),'after_triangles':len(mesh.faces),
                   'maximum_boundary_shift_m':maximum_boundary_shift,'patches':rows}


def _all_planar_patches(mesh):
    """Unlike the legacy clipping adapter, retain sub-0.02 m² faces too."""
    from collections import defaultdict
    groups=defaultdict(list)
    for i,(normal,centre) in enumerate(zip(mesh.face_normals,mesh.triangles_center)):
        if mesh.area_faces[i]<1e-12:continue
        normal=normal.copy()
        if normal[np.argmax(np.abs(normal))]<0:normal=-normal
        key=tuple(np.round(normal,8))+(round(float(normal@centre),7),)
        groups[key].append(i)
    for ids in groups.values():
        normal=mesh.face_normals[ids[0]]
        seed=np.eye(3)[np.argmin(np.abs(normal))]
        u=np.cross(normal,seed);u/=np.linalg.norm(u);v=np.cross(normal,u)
        origin=mesh.triangles[ids[0],0].copy();triangles=mesh.triangles[ids]
        if np.max(np.abs((triangles-origin)@normal))>1e-7:
            raise ValueError('Non-coplanar source triangles')
        projected=np.stack(((triangles-origin)@u,(triangles-origin)@v),axis=-1)
        yield shapely.union_all(shapely.polygons(projected)),origin,u,v


def stair_wall_conflicts(walls, stair_parts, clearance=.08, edge_margin=.04):
    """Find wall candidates that occupy observed walking-surface interiors.

    This is an interference diagnostic, not permission to delete a real wall.
    Callers must review scan support/identity before demoting a hypothesis.
    """
    walking=[]
    for part in stair_parts:
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
        ids=np.flatnonzero(np.abs(mesh.face_normals[:,2])>.99999)
        for z in np.unique(np.round(mesh.triangles_center[ids,2],6)):
            tri=mesh.triangles[ids[np.abs(mesh.triangles_center[ids,2]-z)<1e-6]]
            region=shapely.union_all(shapely.polygons(tri[:,:,:2])).buffer(-edge_margin)
            if region.is_empty:continue
            xmin,ymin,xmax,ymax=region.bounds
            x,y=np.meshgrid(np.arange(xmin,xmax,.04),np.arange(ymin,ymax,.04))
            xy=np.column_stack((x.ravel(),y.ravel()));xy=xy[shapely.contains_xy(region,xy[:,0],xy[:,1])]
            walking.append(np.column_stack((xy,np.full(len(xy),z+clearance))))
    if not walking:return []
    query=np.vstack(walking);rows=[]
    for part in walls:
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
        inside=np.all((query>mesh.bounds[0])&(query<mesh.bounds[1]),axis=1)
        if not inside.any() or not mesh.is_watertight:continue
        probes=query[inside];collision=np.zeros(len(probes),dtype=bool)
        for z in np.unique(probes[:,2]):
            lines=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
            if not len(lines):continue
            segments=shapely.linestrings(np.round(lines[:,:,:2],8))
            region=shapely.union_all(shapely.polygonize(segments).geoms)
            at_z=probes[:,2]==z
            collision[at_z]=shapely.contains_xy(region,probes[at_z,0],probes[at_z,1])
        if collision.any():
            q=query[inside][collision]
            rows.append({'name':part['name'],'interference_samples':len(q),
                         'approximate_walking_area_m2':float(len(q)*.04**2),
                         'bounds_m':[q.min(0).tolist(),q.max(0).tolist()],
                         'clearance_probe_m':clearance,'requires_evidence_review':True})
    return rows


def refine_parts(parts, *, expected_thickness_by_name=None,
                 transforms_by_level=None, raw_points_by_level=None):
    """Return (parts, audit), preserving walls and consolidating coplanar floors.

    Inputs are ordinary {name, kind, v, f, level?} build parts in one frame.
    raw_points_by_level maps level -> scipy KDTree, query-compatible tree,
    ndarray, or {'tree': tree}. expected thickness maps name -> metres.
    Optional transforms apply after refinement; scan trees must be in input
    coordinates. A consolidated floor records source_group_names for native
    replacement. No exterior dilation, pore filling or wall run insertion.
    """
    expected_thickness_by_name=expected_thickness_by_name or {}
    raw_points_by_level=raw_points_by_level or {}
    out=[]; walls=[]; floors={}; unchanged=[]
    for original in parts:
        part=deepcopy(original); level=part.get('level',0); kind=part.get('kind','')
        if kind.startswith('wall'):
            pair=wall_face_pair(part,expected_thickness_by_name.get(part['name']),raw_points_by_level.get(level))
            if pair:
                part['wall_plane_pair']=[{'side':side,'normal':pair['normal'],'offset_m':offset,
                   'source':pair['status']} for side,offset in zip(('A','B'),pair['offsets_m'])]
                part.update(modeled_thickness_m=pair['thickness_m'],thickness_verified=False,
                            measured_face_positions_m=[],wall_face_evidence=pair,merge_coplanar_faces=True)
                walls.append({'name':part['name'],**pair})
            out.append(part)
        elif kind in ('floor','floor_wall_joined') and not part.get('reference_only'):
            top=floor_top(part)
            mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
            if top is None or not mesh.is_watertight:
                out.append(part);unchanged.append({'name':part['name'],'reason':'floor_not_closed_horizontal_solid'});continue
            region,z,bottom=top
            key=(level,round(z,6),round(bottom,6))
            floors.setdefault(key,[]).append((part,region))
        else:
            out.append(part)
    floor_audit=[]
    for (level,z,bottom),items in floors.items():
        if len(items)==1:
            part=items[0][0];part['merge_coplanar_faces']=True;out.append(part);continue
        region=shapely.union_all([row[1] for row in items])
        names=[row[0]['name'] for row in items]
        for i,poly in enumerate(sorted(_polygons(region),key=lambda p:-p.area),1):
            if poly.area<1e-10:continue
            mesh=trimesh.creation.extrude_polygon(poly,height=z-bottom,engine='earcut')
            mesh.vertices[:,2]+=bottom;mesh.merge_vertices();mesh.fix_normals()
            if not mesh.is_watertight:raise ValueError('Consolidated floor is not closed')
            part={'name':f'L{level} consolidated floor {z:.4f}m {i:02d}',
                  'kind':'floor_wall_joined','level':level,'v':mesh.vertices.tolist(),
                  'f':mesh.faces.tolist(),'colour':items[0][0].get('colour',[174,174,166]),
                  'merge_coplanar_faces':True,'construction_solid':True,'source_group_names':names,
                  'backing_thickness_verified':False}
            out.append(part)
        floor_audit.append({'level':level,'top_z_m':z,'bottom_z_m':bottom,
                            'source_group_names':names,'source_top_sum_m2':sum(p.area for _,p in items),
                            'union_area_m2':region.area,'voids_preserved':sum(len(p.interiors) for p in _polygons(region)),
                            'added_area_m2':0.0,'method':'exact_coplanar_union_no_dilation'})
    if transforms_by_level:
        for part in out:
            matrix=np.asarray(transforms_by_level[part.get('level',0)])
            part['v']=(np.asarray(part['v'])@matrix[:3,:3].T+matrix[:3,3]).tolist()
            for plane in part.get('wall_plane_pair',[]):
                normal=matrix[:3,:3]@plane['normal']
                plane['normal']=normal.tolist();plane['offset_m']+=float(normal@matrix[:3,3])
    return out,{'input_parts':len(parts),'output_parts':len(out),'wall_face_pairs':walls,
                'floor_consolidation':floor_audit,'unchanged':unchanged,
                'new_wall_runs':0,'site_accuracy_certified':False,
                'limitations':'Candidate completeness is unchanged. Proximity alone does not prove wall semantics, hidden backs, or full raw-scan coverage.'}
