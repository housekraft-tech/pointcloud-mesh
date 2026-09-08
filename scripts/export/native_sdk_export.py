"""Native SketchUp export/read-back using the installed official C API.

No UI, guessed file format, or remote upload. All SDK calls run on the main
thread. Existing groups are retained; visibility and new geometry are saved to
a new destination. The old native file is never overwritten.
API: https://extensions.sketchup.com/developers/sketchup_c_api/sketchup/
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np
import trimesh

SCALE=1/.0254
ROOT=Path(__file__).resolve().parents[2]

class Ref(C.Structure):
    _fields_=[('ptr',C.c_void_p)]
class Point(C.Structure):
    _fields_=[('x',C.c_double),('y',C.c_double),('z',C.c_double)]
class Transform(C.Structure):
    _fields_=[('values',C.c_double*16)]
class Color(C.Structure):
    _fields_=[('red',C.c_ubyte),('green',C.c_ubyte),('blue',C.c_ubyte),('alpha',C.c_ubyte)]


def snapshots_match(before,after):
    """Serialization may renormalize rotations at floating-point precision."""
    a={p['name']:p for p in before};b={p['name']:p for p in after}
    if set(a)!=set(b):return False
    for name,row in a.items():
        other=b[name]
        if any(row[k]!=other[k] for k in ('faces','hidden','kind','level','reference_only')):return False
        if abs(row['area_m2']-other['area_m2'])>1e-9:return False
        if not np.allclose(row['transform'],other['transform'],atol=1e-12,rtol=0):return False
    return True


class SDK:
    def __init__(self):
        folder=Path('C:/Program Files/SketchUp/SketchUp 2025/SketchUp')
        self.dll_directory=os.add_dll_directory(str(folder))
        self.api=C.CDLL(str(folder/'SketchUpAPI.dll'))
        self.api.SUInitialize()
        for name in ('SUGroupToEntity','SUGroupToDrawingElement','SUFaceToEntity','SUFaceToDrawingElement'):
            getattr(self.api,name).restype=Ref
    def call(self,name,*args):
        status=getattr(self.api,name)(*args)
        if status!=0:raise RuntimeError(f'{name}: SDK error {status}')
    def new(self,kind,*args):
        value=Ref();self.call('SU'+kind+'Create',C.byref(value),*args);return value
    def getref(self,name,obj):
        value=Ref();self.call(name,obj,C.byref(value));return value
    def sequence(self,obj,prefix,suffix):
        count=C.c_size_t();self.call(prefix+'GetNum'+suffix,obj,C.byref(count))
        values=(Ref*count.value)();used=C.c_size_t()
        if count.value:self.call(prefix+'Get'+suffix,obj,C.c_size_t(count.value),values,C.byref(used))
        return list(values)[:used.value]
    def text(self,ref):
        n=C.c_size_t();self.call('SUStringGetUTF8Length',ref,C.byref(n))
        buf=C.create_string_buffer(n.value+1);used=C.c_size_t()
        self.call('SUStringGetUTF8',ref,C.c_size_t(len(buf)),buf,C.byref(used))
        return buf.value.decode('utf-8')
    def name(self,group):
        value=self.new('String');self.call('SUGroupGetName',group,C.byref(value))
        result=self.text(value);self.call('SUStringRelease',C.byref(value));return result
    def attribute(self,group,dictionary,key,default=None):
        entity=self.api.SUGroupToEntity(group);d=Ref()
        if self.api.SUEntityGetAttributeDictionary(entity,dictionary.encode(),C.byref(d))!=0:return default
        value=self.new('TypedValue')
        try:
            if self.api.SUAttributeDictionaryGetValue(d,key.encode(),C.byref(value))!=0:return default
            if isinstance(default,bool):
                out=C.c_bool();self.call('SUTypedValueGetBool',value,C.byref(out));return out.value
            if isinstance(default,int):
                out=C.c_int32();self.call('SUTypedValueGetInt32',value,C.byref(out));return out.value
            out=self.new('String')
            self.call('SUTypedValueGetString',value,C.byref(out));result=self.text(out)
            self.call('SUStringRelease',C.byref(out));return result
        finally:self.call('SUTypedValueRelease',C.byref(value))
    def set_attribute(self,group,dictionary,key,data):
        entity=self.api.SUGroupToEntity(group);d=Ref()
        if self.api.SUEntityGetAttributeDictionary(entity,dictionary.encode(),C.byref(d))!=0:
            self.call('SUAttributeDictionaryCreate',C.byref(d),dictionary.encode())
            self.call('SUEntityAddAttributeDictionary',entity,d)
        value=self.new('TypedValue')
        if isinstance(data,bool):self.call('SUTypedValueSetBool',value,C.c_bool(data))
        elif isinstance(data,int):self.call('SUTypedValueSetInt32',value,C.c_int32(data))
        elif isinstance(data,float):self.call('SUTypedValueSetDouble',value,C.c_double(data))
        else:self.call('SUTypedValueSetString',value,str(data).encode())
        self.call('SUAttributeDictionarySetValue',d,key.encode(),value)
        self.call('SUTypedValueRelease',C.byref(value))
    def load(self,path):
        model=Ref();status=C.c_int()
        self.call('SUModelCreateFromFileWithStatus',C.byref(model),str(Path(path).resolve()).encode(),C.byref(status))
        if status.value!=0:raise ValueError('Model version is newer than installed SDK; refusing lossy write')
        return model
    def groups(self,model):
        return self.sequence(self.getref('SUModelGetEntities',model),'SUEntities','Groups')
    def snapshot(self,model):
        rows=[]
        for group in self.groups(model):
            faces=self.sequence(self.getref('SUGroupGetEntities',group),'SUEntities','Faces')
            area=0
            for face in faces:
                a=C.c_double();self.call('SUFaceGetArea',face,C.byref(a));area+=a.value/SCALE**2
            transform=Transform();self.call('SUGroupGetTransform',group,C.byref(transform))
            hidden=C.c_bool();self.call('SUDrawingElementGetHidden',self.api.SUGroupToDrawingElement(group),C.byref(hidden))
            rows.append({'name':self.name(group),'faces':len(faces),'area_m2':area,
                'transform':list(transform.values),'hidden':hidden.value,
                'kind':self.attribute(group,'CoverageBaseline','kind',''),
                'level':self.attribute(group,'CoverageBaseline','level',0),
                'reference_only':self.attribute(group,'CoverageBaseline','reference_only',False)})
        return rows
    def loop(self,ids):
        result=self.new('LoopInput')
        for index in ids:self.call('SULoopInputAddVertexIndex',result,C.c_size_t(index))
        return result
    def add_part(self,model,part):
        from architectural_surface_refinement import _all_planar_patches,_polygons
        from shapely.geometry.polygon import orient
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
        vertices=[];index_by_xyz={};faces=[]
        def ring_ids(xyz):
            ids=[]
            for p in xyz:
                key=tuple(np.round(p,9))
                if key not in index_by_xyz:index_by_xyz[key]=len(vertices);vertices.append(p)
                index=index_by_xyz[key]
                if not ids or ids[-1]!=index:ids.append(index)
            if len(ids)>1 and ids[-1]==ids[0]:ids.pop()
            return ids
        if part.get('merge_coplanar_faces'):
            for poly,origin,u,v in _all_planar_patches(mesh):
                for polygon in _polygons(poly):
                    polygon=orient(polygon,sign=1.0)
                    rings=[]
                    for ring in [polygon.exterior,*polygon.interiors]:
                        xy=np.asarray(ring.coords)[:-1]
                        ids=ring_ids(origin+xy[:,:1]*u+xy[:,1:]*v)
                        if len(ids)>=3:rings.append(ids)
                    if rings:faces.append(rings)
        else:
            vertices=mesh.vertices.tolist();faces=[[face.tolist()] for face in mesh.faces]
        group=self.new('Group');entities=self.getref('SUModelGetEntities',model)
        self.call('SUEntitiesAddGroup',entities,group)
        self.call('SUGroupSetName',group,part['name'].encode())
        group_entities=self.getref('SUGroupGetEntities',group)
        geometry=self.new('GeometryInput')
        xyz=(Point*len(vertices))(*(Point(*(np.asarray(p)*SCALE)) for p in vertices))
        self.call('SUGeometryInputSetVertices',geometry,C.c_size_t(len(vertices)),xyz)
        for rings in faces:
            outer=self.loop(rings[0]);face_index=C.c_size_t()
            self.call('SUGeometryInputAddFace',geometry,C.byref(outer),C.byref(face_index))
            for ring in rings[1:]:
                inner=self.loop(ring);self.call('SUGeometryInputFaceAddInnerLoop',geometry,face_index,C.byref(inner))
        self.call('SUEntitiesFill',group_entities,geometry,C.c_bool(True))
        self.call('SUGeometryInputRelease',C.byref(geometry))
        material=self.new('Material');self.call('SUMaterialSetName',material,('Observed '+part['name']).encode())
        colour=Color(*part.get('colour',[195,188,166]),255)
        self.call('SUMaterialSetColor',material,C.byref(colour))
        self.call('SUModelAddMaterials',model,C.c_size_t(1),(Ref*1)(material))
        self.call('SUDrawingElementSetMaterial',self.api.SUGroupToDrawingElement(group),material)
        for face in self.sequence(group_entities,'SUEntities','Faces'):
            self.call('SUFaceSetFrontMaterial',face,material);self.call('SUFaceSetBackMaterial',face,material)
        for key,data in {'kind':part['kind'],'level':part.get('level',0),'reference_only':part.get('reference_only',False)}.items():
            self.set_attribute(group,'CoverageBaseline',key,data)
        for key,data in {'thickness_verified':False,'semantic_classification_verified':False,
                         'source':part.get('evidence_status','registered observed geometry'),
                         'source_group_names':json.dumps(part.get('source_group_names',[]))}.items():
            self.set_attribute(group,'ObservedEvidence',key,data)
        if part.get('wall_plane_pair'):
            self.set_attribute(group,'ObservedEvidence','wall_plane_pair_json',json.dumps(part['wall_plane_pair']))
        return group
    def metric_display(self,model):
        manager=self.getref('SUModelGetOptionsManager',model);provider=Ref()
        self.call('SUOptionsManagerGetOptionsProviderByName',manager,b'UnitsOptions',C.byref(provider))
        value=self.new('TypedValue')
        for key,number in [('LengthUnit',2),('LengthPrecision',0)]:
            self.call('SUTypedValueSetInt32',value,C.c_int32(number))
            self.call('SUOptionsProviderSetValue',provider,key.encode(),value)
        self.call('SUTypedValueRelease',C.byref(value))
    def visible_geometry(self,model):
        parts=[]
        for group in self.groups(model):
            hidden=C.c_bool();drawing=self.api.SUGroupToDrawingElement(group)
            self.call('SUDrawingElementGetHidden',drawing,C.byref(hidden))
            if hidden.value:continue
            vs=[];fs=[];offset=0
            transform=Transform();self.call('SUGroupGetTransform',group,C.byref(transform))
            matrix=np.asarray(transform.values).reshape(4,4).T
            for face in self.sequence(self.getref('SUGroupGetEntities',group),'SUEntities','Faces'):
                helper=self.new('MeshHelper',face)
                nv=C.c_size_t();nt=C.c_size_t()
                self.call('SUMeshHelperGetNumVertices',helper,C.byref(nv));self.call('SUMeshHelperGetNumTriangles',helper,C.byref(nt))
                if not nv.value or not nt.value:
                    area=C.c_double();self.call('SUFaceGetArea',face,C.byref(area))
                    self.call('SUMeshHelperRelease',C.byref(helper))
                    if area.value/SCALE**2>1e-10:
                        raise ValueError(f'Native face cannot triangulate: {self.name(group)} area={area.value/SCALE**2}')
                    continue
                xyz=(Point*nv.value)();ids=(C.c_size_t*(nt.value*3))();used=C.c_size_t()
                self.call('SUMeshHelperGetVertices',helper,nv,xyz,C.byref(used))
                self.call('SUMeshHelperGetVertexIndices',helper,C.c_size_t(len(ids)),ids,C.byref(used))
                v=np.array([(p.x,p.y,p.z) for p in xyz])
                v=(v@matrix[:3,:3].T+matrix[:3,3])/SCALE
                vs.extend(v.tolist());fs.extend((np.asarray(ids).reshape(-1,3)+offset).tolist());offset+=len(v)
                self.call('SUMeshHelperRelease',C.byref(helper))
            colour=[195,188,166];mat=Ref();c=Color()
            if self.api.SUDrawingElementGetMaterial(drawing,C.byref(mat))==0 and self.api.SUMaterialGetColor(mat,C.byref(c))==0:
                colour=[c.red,c.green,c.blue]
            if fs:parts.append({'name':self.name(group),'kind':self.attribute(group,'CoverageBaseline','kind',''),
                'level':self.attribute(group,'CoverageBaseline','level',0),'v':vs,'f':fs,'colour':colour})
        return parts
    def scenes(self,model,parts):
        old=self.sequence(model,'SUModel','Scenes')
        if old:self.call('SUModelRemoveScenes',model,C.c_size_t(len(old)),(Ref*len(old))(*old))
        xyz=np.vstack([p['v'] for p in parts]);lo,hi=xyz.min(0),xyz.max(0);centre=(lo+hi)/2;s=float(max(hi-lo))
        groups=self.groups(model)
        for index,(name,offset,up) in enumerate([('3D',[1.3,-1.5,1.25],[0,0,1]),('Top',[0,0,2],[0,1,0]),
                                                ('Front',[0,-2,0],[0,0,1]),('Side',[2,0,0],[0,0,1])]):
            camera=self.new('Camera');eye=Point(*((centre+np.asarray(offset)*s)*SCALE));target=Point(*(centre*SCALE));up=Point(*up)
            self.call('SUCameraSetOrientation',camera,C.byref(eye),C.byref(target),C.byref(up))
            self.call('SUCameraSetPerspective',camera,C.c_bool(False))
            self.call('SUCameraSetAspectRatio',camera,C.c_double(1800/1300))
            self.call('SUCameraSetOrthographicFrustumHeight',camera,C.c_double(s*1.12*SCALE))
            scene=self.new('Scene');self.call('SUSceneSetName',scene,name.encode())
            self.call('SUSceneSetCamera',scene,camera);self.call('SUSceneSetUseCamera',scene,C.c_bool(True))
            self.call('SUSceneSetUseHiddenObjects',scene,C.c_bool(True))
            for group in groups:
                drawing=self.api.SUGroupToDrawingElement(group);hidden=C.c_bool()
                self.call('SUDrawingElementGetHidden',drawing,C.byref(hidden))
                self.call('SUSceneSetDrawingElementHidden',scene,drawing,hidden)
            self.call('SUModelAddScenes',model,C.c_size_t(1),(Ref*1)(scene))
            if index==0:self.call('SUModelSetCamera',model,C.byref(camera))


def export(payload_path,destination,source_override=None,policy_path=None):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Refusing to overwrite existing native output')
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload=json.loads(Path(payload_path).read_text());sdk=SDK()
    source=source_override or payload.get('source_native')
    model=sdk.load(source) if source else sdk.new('Model')
    before=sdk.snapshot(model);by_name={r['name']:r for r in before}
    if len(by_name)!=len(before):raise ValueError('Duplicate native group identities')
    expected=payload.get('expected_base_groups')
    if expected is not None and not source_override and expected!=len(before):raise ValueError('Wrong source native')
    added=[]
    for index,part in enumerate(payload['parts'],1):
        if part['name'] in by_name:
            expected_area=trimesh.Trimesh(part['v'],part['f'],process=False).area
            if abs(by_name[part['name']]['area_m2']-expected_area)>max(.000025,expected_area*1e-5):
                raise ValueError('Recovered existing addition does not match payload: '+part['name'])
            continue
        print(f'Native SDK {index}/{len(payload["parts"])} {part["name"]}',flush=True)
        sdk.add_part(model,part);added.append(part['name'])
    policy=json.loads(Path(policy_path).read_text()) if policy_path else {}
    refs=set(payload.get('reference_source_names',[]))
    refs.update(policy.get('paired_solid_references',[]));refs.update(policy.get('demoted_stair_wall_hypotheses',[]))
    visible=set(policy.get('bounded_wall_groups_visible',[]))|set(policy.get('entry_step_groups_visible',[]))
    for part in payload['parts']:refs.update(part.get('source_group_names',[]))
    refs.difference_update(visible)
    for group in sdk.groups(model):
        name=sdk.name(group);kind=sdk.attribute(group,'CoverageBaseline','kind','')
        reference=sdk.attribute(group,'CoverageBaseline','reference_only',False)
        if name in refs:reference=True
        if name in visible:reference=False
        sdk.set_attribute(group,'CoverageBaseline','reference_only',reference)
        sdk.call('SUDrawingElementSetHidden',sdk.api.SUGroupToDrawingElement(group),C.c_bool(reference or 'ceiling' in kind))
        if kind.startswith('wall'):sdk.set_attribute(group,'ObservedEvidence','thickness_verified',False)
    after=sdk.snapshot(model);after_by_name={r['name']:r for r in after}
    for old in before:
        new=after_by_name[old['name']]
        if old['faces']!=new['faces'] or abs(old['area_m2']-new['area_m2'])>1e-9 or old['transform']!=new['transform']:
            raise ValueError('Existing source geometry or transform changed')
    for part in payload['parts']:
        expected_area=trimesh.Trimesh(part['v'],part['f'],process=False).area
        actual=after_by_name[part['name']]['area_m2']
        if abs(actual-expected_area)>max(.000025,expected_area*1e-5):
            raise ValueError(f'Native area mismatch {part["name"]}: {actual-expected_area}')
    parts=sdk.visible_geometry(model);sdk.metric_display(model);sdk.scenes(model,parts)
    sdk.call('SUModelSaveToFile',model,str(destination).encode())
    sdk.call('SUModelRelease',C.byref(model))
    reopened=sdk.load(destination);checks=sdk.snapshot(reopened)
    if not snapshots_match(after,checks):
        (destination.parent/'native_sdk_roundtrip_failure.json').write_text(json.dumps({'before':after,'after':checks}))
        raise ValueError('Saved native round-trip changed geometry or visibility')
    (destination.parent/'native_sdk_audit.json').write_text(json.dumps({'path':str(destination),'source_native':source,
        'native_library':'Installed SketchUp 2025 SketchUpAPI.dll','load_status':'success_same_version',
        'existing_geometry_and_transforms_preserved':True,'native_area_pass':True,
        'reopened_geometry_and_visibility_pass':True,'groups':checks,'new_groups':added,
        'site_accuracy_certified':False},indent=2))
    (destination.parent/'reopened_visible.build.json').write_text(json.dumps({'label':payload['label'],'parts':parts}))
    sdk.call('SUModelRelease',C.byref(reopened));sdk.api.SUTerminate()
    print(json.dumps({'path':str(destination),'groups':len(checks),'visible_groups':len(parts),'bytes':destination.stat().st_size}),flush=True)


def verify_saved(payload_path,destination,source_override=None,policy_path=None):
    """Read-only recovery of an exported file whose final audit was interrupted."""
    destination=Path(destination).resolve();payload=json.loads(Path(payload_path).read_text())
    source=source_override or payload.get('source_native');sdk=SDK()
    original=sdk.load(source);before=sdk.snapshot(original);sdk.call('SUModelRelease',C.byref(original))
    saved=sdk.load(destination);after=sdk.snapshot(saved);by_name={r['name']:r for r in after}
    max_transform_delta=0
    for old in before:
        new=by_name[old['name']]
        delta=float(np.max(np.abs(np.asarray(old['transform'])-new['transform'])))
        max_transform_delta=max(max_transform_delta,delta)
        if old['faces']!=new['faces'] or abs(old['area_m2']-new['area_m2'])>1e-9 or delta>1e-12:
            raise ValueError(f'Saved source geometry changed beyond round-off: {old["name"]}, transform delta {delta}')
    policy=json.loads(Path(policy_path).read_text()) if policy_path else {}
    refs=set(payload.get('reference_source_names',[]))|set(policy.get('paired_solid_references',[]))|set(policy.get('demoted_stair_wall_hypotheses',[]))
    visible=set(policy.get('bounded_wall_groups_visible',[]))|set(policy.get('entry_step_groups_visible',[]))
    for part in payload['parts']:
        refs.update(part.get('source_group_names',[]))
        expected=trimesh.Trimesh(part['v'],part['f'],process=False).area
        actual=by_name[part['name']]['area_m2']
        if abs(expected-actual)>max(.000025,expected*1e-5):raise ValueError('Saved addition area mismatch: '+part['name'])
    refs.difference_update(visible)
    for name in refs:
        if not by_name[name]['reference_only'] or not by_name[name]['hidden']:raise ValueError('Unverified reference visible: '+name)
    for name in visible:
        if by_name[name]['reference_only'] or by_name[name]['hidden']:raise ValueError('Observed geometry hidden: '+name)
    for part in payload['parts']:
        if 'ceiling' not in part['kind'] and by_name[part['name']]['hidden']:raise ValueError('New supported part hidden')
    parts=sdk.visible_geometry(saved)
    (destination.parent/'reopened_visible.build.json').write_text(json.dumps({'label':payload['label'],'parts':parts}))
    report={'path':str(destination),'source_native':source,'native_library':'Installed SketchUp 2025 SketchUpAPI.dll',
        'load_status':'success_same_version','existing_geometry_and_transforms_preserved':True,
        'native_area_pass':True,'reopened_geometry_and_visibility_pass':True,
        'maximum_transform_serialization_roundoff':max_transform_delta,'transform_roundoff_tolerance':1e-12,
        'groups':after,'site_accuracy_certified':False}
    (destination.parent/'native_sdk_audit.json').write_text(json.dumps(report,indent=2))
    sdk.call('SUModelRelease',C.byref(saved));sdk.api.SUTerminate()
    print(json.dumps({'verified_native':str(destination),'groups':len(after),'visible_groups':len(parts),
                     'max_transform_roundoff':max_transform_delta}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload',required=True);parser.add_argument('--destination',required=True)
    parser.add_argument('--source');parser.add_argument('--policy')
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    (verify_saved if args.verify_existing else export)(args.payload,args.destination,args.source,args.policy)
