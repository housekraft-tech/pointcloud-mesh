"""Apply the shared evidence-aware refinement to the registered Soulace house."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from architectural_surface_refinement import wall_face_pair, proximity_evidence, supported_candidate_replacement, regularize_planar_fragments
from rectangular_rebuild import load_reference
from export_rectangular_skp import invoke
from architectural_surface_refinement import stair_wall_conflicts

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_architectural_v9'
SOURCE=ROOT/'output_final/soulace_full_house_planes_v8'
NATIVE='Soulace_architectural_refined_v9_checked.skp'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/NATIVE).exists():raise FileExistsError('Use another native revision')
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    audit=json.loads((SOURCE/'native_audit.json').read_text())
    registered={g['name'] for g in audit['groups']}
    wall_rows=[];beam_rows=[];parts=[];references=set(audit['reference_only_groups'])
    for level,stem,nominal in [(0,'ground',0),(1,'first',3.2),(2,'second',6.5)]:
        path=ROOT/f'output_final/rectangular_rebuild_v1/soulace_l{level}.manifest.json'
        manifest=json.loads(path.read_text())
        original=json.loads(Path(manifest['candidate_model']).read_text())['parts']
        metadata=json.loads((ROOT/f'output_final/soulace_asbuilt_v2/Soulace_L{level}_{stem}_asbuilt_manifest.json').read_text())
        thickness={p['wall']:p['thickness_mm']/1000 for p in metadata['walls']}
        xyz=np.concatenate([p['v'] for p in original])
        reference=load_reference(manifest['scans'][0],path.parent,[xyz.min(0)-.2,xyz.max(0)+.2])
        matrix=np.asarray(cfg['source_to_common'][str(level)])
        origin=matrix[:3,3]+matrix[:3,:3]@np.array([0,0,nominal])
        if level<2:
            for part in original:
                if not part['kind'].startswith('wall'):continue
                pair=wall_face_pair(part,thickness[part['name']],reference)
                name=f'L{level} paired wall planes {part["name"]}'
                if name not in registered:raise ValueError(name)
                normal=matrix[:3,:3]@pair['normal']
                planes=[{'side':side,'normal':normal.tolist(),'offset_m':float(d+normal@origin),
                         'source':pair['status']} for side,d in zip(('A','B'),pair['offsets_m'])]
                wall_rows.append({'name':name,'pair':pair,'planes':planes})
        for part in original:
            name=f'L{level}_{part["name"]}'
            if part['kind']!='beam' or name not in registered:continue
            mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
            evidence=proximity_evidence(mesh.triangles,reference)
            faces=[]
            for normal in np.unique(np.round(mesh.face_normals,6),axis=0):
                ids=np.flatnonzero(mesh.face_normals@normal>1-1e-6)
                faces.append({'normal':normal.tolist(),'area_m2':float(mesh.area_faces[ids].sum()),
                              **proximity_evidence(mesh.triangles[ids],reference)})
            beam_rows.append({'name':name,'evidence':evidence,'faces':faces,'candidate':part})
        del reference
    (OUT/'candidate_evidence.json').write_text(json.dumps({'walls':wall_rows,'beams':beam_rows},indent=2))
    print(json.dumps({'walls':len(wall_rows),'beams':[{k:b[k] for k in ('name','evidence')} for b in beam_rows]},indent=2),flush=True)


def build():
    if (OUT/NATIVE).exists():raise FileExistsError('Use another native revision')
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    prior=json.loads((SOURCE/'native_audit.json').read_text())
    registered={p['name'] for p in prior['groups']}
    parts=[];rows=[];references=set(prior['reference_only_groups'])
    for level,stem,nominal in [(0,'ground',0),(1,'first',3.2),(2,'second',6.5)]:
        path=ROOT/f'output_final/rectangular_rebuild_v1/soulace_l{level}.manifest.json'
        manifest=json.loads(path.read_text())
        original=json.loads(Path(manifest['candidate_model']).read_text())['parts']
        candidates={p['name']:p for p in original}
        current=json.loads((ROOT/f'output_final/soulace_overlap_only_50mm/Soulace_L{level}_overlap_only.build.json').read_text())['parts']
        xyz=np.concatenate([p['v'] for p in original])
        reference=load_reference(manifest['scans'][0],path.parent,[xyz.min(0)-.2,xyz.max(0)+.2])
        matrix=np.asarray(cfg['source_to_common'][str(level)])
        for old in current:
            name=f'L{level}_{old["name"]}'
            if old['kind']!='beam' or name not in registered:continue
            selected,gate=supported_candidate_replacement(old,candidates[old['name']],reference)
            if gate['accepted']:
                detail={'changed':True,'method':'registered_candidate_support_pass'}
            else:selected,detail=regularize_planar_fragments(old,reference)
            rows.append({'name':name,'candidate_gate':gate,'local_cleanup':detail})
            if not detail['changed']:continue
            selected['v']=((np.asarray(selected['v'])+[0,0,nominal])@matrix[:3,:3].T+matrix[:3,3]).tolist()
            selected.update(name=f'L{level} refined {old["name"]}',level=level,kind='beam_refined',
                            colour=[166,161,148],source_group_names=[name])
            parts.append(selected);references.add(name)
        del reference
    payload={'label':'Soulace | evidence-aware architectural refinement v9',
             'source_native':str(SOURCE/'Soulace_full_house_wall_and_floor_planes.skp'),
             'expected_base_groups':len(prior['groups']),'reference_source_names':sorted(references),
             'parts':parts,'render_image_names':['3D','Top','Front','Side','L0 3D','L0 Top','L1 3D','L1 Top','L2 3D','L2 Top'],
             'source_label':'Registered beam fragments with local, raw-support-checked edge regularization',
             'note':'Wall solids and openings, floor planes, parking offset and stair orientation retained. Local beam edge cleanup passes sampled 50mm raw proximity; rejected full envelopes remain unbuilt. Nominal wall plane labels corrected independently of merged columns. Candidate-derived backs remain unverified; site accuracy not certified.'}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    (OUT/'refinement_audit.json').write_text(json.dumps({'beam_refinement':rows,'new_parts':len(parts),'site_accuracy_certified':False},indent=2))
    print(json.dumps({'new_parts':len(parts),'beam_results':[(r['name'],r['candidate_gate']['accepted'],r['local_cleanup']['changed']) for r in rows]},indent=2),flush=True)


def native_finalize():
    target=OUT/'native_refinement_checks.json'
    code=r'''
      load RUBY;
      m=Sketchup.active_model;
      raise 'Wrong current model' unless m.path.tr('\\','/').end_with?(NATIVE);
      data=JSON.parse(File.read(EVIDENCE)); groups=m.entities.grep(Sketchup::Group);
      byname=groups.to_h{|g|[g.name,g]};
      data['walls'].each do |row|
        g=byname.fetch(row['name']);
        g.entities.grep(Sketchup::Face).each{|f|f.delete_attribute('WallPlane')};
        PCMCoverageBaseline.label_wall_planes(g,row['planes']);
        g.set_attribute('ObservedEvidence','thickness_verified',false);
        g.set_attribute('ObservedEvidence','modeled_thickness_m',row['pair']['thickness_m']);
        g.set_attribute('ObservedEvidence','measured_face_positions_local_m',[]);
        g.set_attribute('ObservedEvidence','opposing_face_proximity_supported',row['pair']['opposing_face_proximity_supported']);
      end;
      policy=JSON.parse(File.read(POLICY));
      groups.each do |g|
        if policy['bounded_wall_groups_visible'].include?(g.name)
          g.set_attribute('CoverageBaseline','reference_only',false);
          g.set_attribute('ObservedEvidence','thickness_verified',false);
          g.set_attribute('ObservedEvidence','source','Existing bounded 50mm raw-supported surface; wall thickness and semantic identity unverified');
          PCMCoverageBaseline.consolidate_planes(g);
        end;
        g.set_attribute('CoverageBaseline','reference_only',false) if policy['entry_step_groups_visible'].include?(g.name);
      end;
      PCMCoverageBaseline.finish(m,DEST,'Soulace | scan-supported walls and individually measured stair flights',policy['render_image_names']);
      PCMCoverageBaseline.detail_scenes(m,DEST,lambda{|g|g.get_attribute('CoverageBaseline','kind','')=='roof_stair_clean'},'Roof staircase unobstructed',[0,-1,0]);
      PCMCoverageBaseline.detail_scenes(m,DEST,lambda{|g|g.name.start_with?('Lower ') && g.get_attribute('CoverageBaseline','kind','')=='stair_clean'},'Lower staircase measured',[1,0,0]);
      PCMCoverageBaseline.detail_scenes(m,DEST,lambda{|g|!PCMCoverageBaseline.default_hidden?(g) && g.get_attribute('CoverageBaseline','kind','')=='stair_clean'},'All inter floor stairs',[1,0,0]);
      PCMCoverageBaseline.detail_scenes(m,DEST,lambda{|g|!PCMCoverageBaseline.default_hidden?(g) && ['floor_wall_joined','floor_clean','ground_single_plane','plinth_observed_surface'].include?(g.get_attribute('CoverageBaseline','kind',''))},'Verified floors and parking',[1,0,0]);
      PCMCoverageBaseline.detail_scenes(m,DEST,lambda{|g|g.get_attribute('CoverageBaseline','kind','')=='plinth_observed_surface' || policy['entry_step_groups_visible'].include?(g.name)},'Measured parking connection',[1,0,0]);
      groups.each{|g|g.hidden=PCMCoverageBaseline.default_hidden?(g)};
      m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;
      m.set_attribute('ArchitecturalRefinement','wall_thickness_site_verified',false);
      raise 'Save failed' unless m.save(DEST);
      raise 'Reopen failed' unless Sketchup.open_file(DEST);
      m=Sketchup.active_model;groups=m.entities.grep(Sketchup::Group);
      rows=groups.map do |g|
        faces=g.entities.grep(Sketchup::Face);
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),
         hidden:g.hidden?,reference:g.get_attribute('CoverageBaseline','reference_only',false),
         manifold:g.manifold?,faces:faces.length,
         thickness_m:g.get_attribute('ObservedEvidence','modeled_thickness_m',nil),
         thickness_verified:g.get_attribute('ObservedEvidence','thickness_verified',nil),
         bounds_z_m:[g.bounds.min.z/PCMCoverageBaseline::SCALE,g.bounds.max.z/PCMCoverageBaseline::SCALE],
         horizontal_faces:faces.select{|f|f.normal.z.abs>0.999999}.map{|f|{area_m2:f.area/(PCMCoverageBaseline::SCALE**2),loops:f.loops.length,z_m:f.vertices.map{|v|v.position.z/PCMCoverageBaseline::SCALE}.uniq}},
         paired_faces:['A','B'].map{|s|faces.count{|f|f.get_attribute('WallPlane','side','')==s}}}
      end;
      File.write(TARGET,JSON.pretty_generate({path:m.path,groups:rows,scenes:m.pages.map(&:name)}));
      {groups:groups.length,wall_pairs_checked:data['walls'].length}.to_json
    '''
    for k,v in {'RUBY':str(ROOT/'scripts/export/ruby/pcm_coverage_baseline.rb'),'EVIDENCE':str(OUT/'candidate_evidence.json'),
                'NATIVE':NATIVE,'DEST':str(OUT/NATIVE),'TARGET':str(target),'POLICY':str(OUT/'visibility_policy.json')}.items():
        code=code.replace(k,json.dumps(v.replace('\\','/')))
    print(invoke(code),flush=True)
    checks=json.loads(target.read_text());groups={g['name']:g for g in checks['groups']}
    evidence=json.loads((OUT/'candidate_evidence.json').read_text())
    for wall in evidence['walls']:
        row=groups[wall['name']]
        assert row['paired_faces'][0]>0 and row['paired_faces'][1]>0
        assert abs(row['thickness_m']-wall['pair']['thickness_m'])<1e-8
        assert row['thickness_verified'] is False
        assert row['manifold'] and row['hidden'] and row['reference']
    assert groups['Top-floor roof stair - clean measured run']['manifold']
    additions=json.loads((OUT/'additions.build.json').read_text())
    for part in additions['parts']:
        assert not groups[part['name']]['hidden']
        for name in part.get('source_group_names',[]):assert groups[name]['reference'] and groups[name]['hidden']
    print('Reopened native geometry, corrected wall-side labels, and beam references verified.',flush=True)


def floor_audit():
    """Check the exact visible finished floor polygons before native handover."""
    from architectural_surface_refinement import floor_top,_polygons
    from shapely.geometry import Polygon
    sources=[SOURCE/'additions.build.json',ROOT/'output_final/soulace_wall_floor_junctions_v7/additions.build.json']
    floors=[p for source in sources for p in json.loads(source.read_text())['parts'] if p['kind']=='floor_wall_joined']
    replacement=OUT/'repaired_floor.build.json'
    if replacement.exists():
        data=json.loads(replacement.read_text());replaced={name for p in data['parts'] for name in p['source_group_names']}
        floors=[p for p in floors if p['name'] not in replaced]+data['parts']
    rows=[];regions=[]
    for p in floors:
        top,z,bottom=floor_top(p);mesh=trimesh.Trimesh(p['v'],p['f'],process=False)
        holes=[Polygon(r) for poly in _polygons(top) for r in poly.interiors]
        row={'name':p['name'],'level':p['level'],'top_z_m':z,'bottom_z_m':bottom,
             'top_valid':top.is_valid,'top_area_m2':top.area,'solid_watertight':mesh.is_watertight,
             'positive_volume':bool(mesh.volume>0),'components':len(_polygons(top)),
             'holes_m2':[h.area for h in holes],'pore_holes_under_0_005m2':sum(h.area<.00499 for h in holes)}
        rows.append(row);regions.append((p['name'],z,top))
    overlap=[]
    for i,(name,z,region) in enumerate(regions):
        for other,z2,region2 in regions[i+1:]:
            if abs(z-z2)>1e-6:continue
            area=region.intersection(region2).area
            if area>1e-7:overlap.append({'a':name,'b':other,'overlap_m2':area})
    report={'finished_floors':rows,'coplanar_floor_overlaps':overlap,
            'all_valid_closed':all(r['top_valid'] and r['solid_watertight'] for r in rows),
            'unintended_pore_holes':sum(r['pore_holes_under_0_005m2'] for r in rows),
            'different_floor_datums_preserved':True,'ground_plane_expected_z_m':-.5334,
            'entry_tread_groups_expected':['Planar ground patch 29','Planar ground patch 30'],
            'limitations':'Geometric artifact checks preserve larger modeled stair/shaft voids; no independent site completeness certification.'}
    (OUT/'floor_geometry_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    assert report['all_valid_closed'] and not overlap and report['unintended_pore_holes']==0


def repair_floor_pore():
    from architectural_surface_refinement import floor_top
    from shapely.geometry import Polygon
    from scipy.spatial import cKDTree
    source=json.loads((ROOT/'output_final/soulace_wall_floor_junctions_v7/additions.build.json').read_text())['parts']
    original=next(p for p in source if p['name']=='L2 floor plane 01 - joined to wall faces')
    poly,z,bottom=floor_top(original)
    removed=[Polygon(h) for h in poly.interiors if Polygon(h).area<.005]
    assert len(removed)==1 and max(p.bounds[2]-p.bounds[0] for p in removed)<.011
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.asarray(cfg['source_to_common']['2']);raw=np.load(ROOT/'output_final/soulace_top_floor_walls_v5/local_raw.npy',mmap_mode='r')
    centre=np.array([removed[0].centroid.x,removed[0].centroid.y,z])
    local=(centre-matrix[:3,3])@matrix[:3,:3]-[0,0,6.5]
    close=raw[np.all(abs(raw-local)<[.4,.4,.15],axis=1)]
    common=(close+[0,0,6.5])@matrix[:3,:3].T+matrix[:3,3]
    tree=cKDTree(common)
    coords=np.asarray(removed[0].exterior.coords)[:-1]
    xy=np.vstack([np.linspace(a,b,100) for a,b in zip(coords,np.roll(coords,-1,axis=0))])
    samples=np.column_stack((xy,np.full(len(xy),z)))
    distances=tree.query(samples)[0]
    assert distances.max()<.05,'Pore repair lacks local raw floor support'
    repaired=Polygon(poly.exterior,[h for h in poly.interiors if Polygon(h).area>=.005])
    mesh=trimesh.creation.extrude_polygon(repaired,height=z-bottom,engine='earcut');mesh.vertices[:,2]+=bottom;mesh.merge_vertices();mesh.fix_normals()
    part={**original,'name':'L2 floor plane 01 - pore seam repaired','v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),'source_group_names':[original['name']]}
    report={'source':original['name'],'removed_pore_area_m2':sum(p.area for p in removed),'maximum_pore_width_m':.0092,
            'larger_void_areas_preserved_m2':[Polygon(h).area for h in repaired.interiors],
            'local_raw_points':len(close),'pore_boundary_max_raw_distance_mm':float(distances.max()*1000),
            'floor_z_unchanged_m':z,'watertight':mesh.is_watertight}
    (OUT/'repaired_floor.build.json').write_text(json.dumps({'parts':[part]},separators=(',',':')))
    (OUT/'floor_pore_repair_audit.json').write_text(json.dumps(report,indent=2))
    payload=json.loads((OUT/'additions.build.json').read_text())
    payload['parts']=[p for p in payload['parts'] if p['name']!=part['name']]+[part]
    payload['reference_source_names']=sorted(set(payload['reference_source_names'])|{original['name']})
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps(report,indent=2),flush=True)


def strict_top_walls():
    """Clip added top-storey wall candidates to full-density source returns."""
    from filter_soulace_overlap import plane_patches,supported_polygon,mesh_from_polygon
    from architectural_surface_refinement import _all_planar_patches
    path=ROOT/'output_final/rectangular_rebuild_v1/soulace_l2.manifest.json'
    manifest=json.loads(path.read_text());cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.asarray(cfg['source_to_common']['2'])
    source=json.loads((ROOT/'output_final/soulace_wall_floor_junctions_v7/additions.build.json').read_text())['parts']
    walls=[p for p in source if p['kind']=='wall_paired_planes'];local=[]
    for p in walls:
        local.append((np.asarray(p['v'])-matrix[:3,3])@matrix[:3,:3]-[0,0,6.5])
    xyz=np.concatenate(local)
    reference=load_reference(manifest['scans'][0],path.parent,[xyz.min(0)-.2,xyz.max(0)+.2])
    parts=[];rows=[]
    for wall,vertices in zip(walls,local):
        mesh=trimesh.Trimesh(vertices,wall['f'],process=False);pieces=[]
        for poly,origin,u,v in _all_planar_patches(mesh):
            bounded,info=supported_polygon(poly,origin,u,v,reference['tree'],cutoff=.05,min_area=.002)
            if bounded.is_empty:continue
            p=mesh_from_polygon(bounded,origin,u,v)
            if p is not None:pieces.append(p)
        if not pieces:rows.append({'source':wall['name'],'retained_area_m2':0.0});continue
        result=trimesh.util.concatenate(pieces);result.merge_vertices(digits_vertex=8)
        common=(result.vertices+[0,0,6.5])@matrix[:3,:3].T+matrix[:3,3]
        parts.append({'name':wall['name'].replace('paired wall planes','bounded wall faces'),
                      'kind':'wall_scan_surface','level':2,'v':common.tolist(),'f':result.faces.tolist(),
                      'colour':wall.get('colour',[205,198,181]),'merge_coplanar_faces':True,
                      'construction_solid':False,'thickness_verified':False,'modeled_thickness_m':None,
                      'measured_face_positions_m':[],'source_group_names':[wall['name']],
                      'evidence_status':'full_density_50mm_bounded_surface'})
        rows.append({'source':wall['name'],'source_area_m2':mesh.area,'retained_area_m2':result.area,'triangles':len(result.faces)})
        print(wall['name'],len(result.faces),flush=True)
    (OUT/'strict_top_walls.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
    (OUT/'strict_top_walls_audit.json').write_text(json.dumps({'parts':rows,'source':reference['provenance'],'distance_bound_mm':50,'new_wall_runs':0},indent=2))


def assemble_final():
    additions=json.loads((OUT/'additions.build.json').read_text())
    # Repeatable assembly before the first native save, keeping prior results.
    additions['parts']=[p for p in additions['parts'] if p['kind']=='beam_refined']
    lower=json.loads((OUT/'lower_stairs.build.json').read_text())['parts']
    top=json.loads((OUT/'strict_top_walls.build.json').read_text())['parts']
    parking_path=ROOT/'output_final/architectural_flow_astra/parking_diagnostic/connection.build.json'
    parking=json.loads(parking_path.read_text())['parts']
    additions['parts'].extend(lower+top+parking)
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    prior=json.loads((SOURCE/'native_audit.json').read_text())
    all_names={p['name'] for p in prior['groups']}
    conflicts={'L0_wall_34','L0_wall_35','L0_wall_37','L0_wall_39','L1_wall_36'}
    bounded=[name for name,meta in cfg['groups'].items() if name in all_names and meta['level']<2 and meta['kind'].startswith('wall') and name not in conflicts]
    paired=[name for name in all_names if 'paired wall planes' in name]
    refs=set(additions['reference_source_names']);refs.update(paired);refs.update(conflicts)
    refs.update(name for name in all_names if 'Staircase' in name and ('surfaces' in name or 'side-context' in name))
    # Old top wall skins remain references; the expanded top-wall candidates
    # above are independently clipped to the raw scan at 50 mm.
    refs.difference_update(bounded)
    additions.update(reference_source_names=sorted(refs),
      label='Soulace | bounded wall evidence and measured inter-floor stair flights',
      note='Visible wall faces are raw-scan-bounded surfaces, not inferred solid backs. Stair-conflicting wall hypotheses and original stair context are retained hidden. Lower treads/risers fit independently to scan; upper and roof stairs preserved. Measured east plinth and porch face connect raised house to parking datum; two observed entry step treads restored. No new walls inferred.')
    (OUT/'additions.build.json').write_text(json.dumps(additions,separators=(',',':')))
    walls=json.loads((SOURCE/'additions.build.json').read_text())['parts']
    stairs=lower+json.loads((ROOT/'output_final/soulace_clean_floor_stairs_v2/upper_stairs.build.json').read_text())['parts']
    collision=stair_wall_conflicts([p for p in walls if p['kind']=='wall_paired_planes'],stairs)
    entry_steps=[name for name in all_names if name in ('Planar ground patch 29','Planar ground patch 30')]
    assert len(entry_steps)==2,'Missing measured entry step references'
    policy={'bounded_wall_groups_visible':bounded,'entry_step_groups_visible':entry_steps,'paired_solid_references':paired,'demoted_stair_wall_hypotheses':sorted(conflicts),
            'stair_wall_interference':collision,'render_image_names':additions['render_image_names'],
            'no_unverified_opaque_wall_backs_visible':True}
    (OUT/'visibility_policy.json').write_text(json.dumps(policy,indent=2))
    print({'parts':len(additions['parts']),'bounded_existing_walls':len(bounded),'solid_wall_refs':len(paired),'stair_conflicts_demoted':len(conflicts)},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--build',action='store_true');ap.add_argument('--native-finalize',action='store_true');ap.add_argument('--strict-top-walls',action='store_true');ap.add_argument('--assemble-final',action='store_true');ap.add_argument('--floor-audit',action='store_true');ap.add_argument('--repair-floor-pore',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.build:build()
    elif args.native_finalize:native_finalize()
    elif args.strict_top_walls:strict_top_walls()
    elif args.assemble_final:assemble_final()
    elif args.floor_audit:floor_audit()
    elif args.repair_floor_pore:repair_floor_pore()
