"""Minimal glTF 2.0 / GLB writer -- no dependencies.

Matches the target file's conventions: Y-up, metres, one named node per
architectural element, flat-shaded low-poly boxes so every edge stays crisp."""
import json, struct, numpy as np
class GLB:
    def __init__(self):
        self.parts=[]              # (name, vertices Nx3, normals, indices)
        self.groups=[]             # (name, [part indices]) -> a parent node
    def add_box(self,name,lo,hi,acc=None):
        """Axis-aligned box. Vertices are NOT shared between faces, so normals are
        flat and edges read sharp rather than smoothed."""
        x0,y0,z0=lo; x1,y1,z1=hi
        F=[([(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)],(0,0,-1)),
           ([(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),(x0,y0,z1)],(0,0,1)),
           ([(x0,y0,z1),(x0,y1,z1),(x0,y1,z0),(x0,y0,z0)],(-1,0,0)),
           ([(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],(1,0,0)),
           ([(x0,y0,z1),(x0,y0,z0),(x1,y0,z0),(x1,y0,z1)],(0,-1,0)),
           ([(x0,y1,z0),(x0,y1,z1),(x1,y1,z1),(x1,y1,z0)],(0,1,0))]
        V=[];N=[];I=[]
        for quad,nrm in F:
            b=len(V)
            V+=list(quad); N+=[nrm]*4
            I+=[b,b+1,b+2,b,b+2,b+3]
        if acc is None:
            self.parts.append([name,V,N,I])
        else:
            b=len(acc[0]); acc[0].extend(V); acc[1].extend(N)
            acc[2].extend([q+b for q in I])
    def parent(self, name, first_part):
        """A node with no mesh of its own, owning every part added from
        `first_part` onward. This is what makes the file modular rather than a
        flat pile: a wall owns its panels, its arches, its columns and its
        niches, so a consumer can take one wall and get everything on it."""
        self.groups.append((name, first_part, len(self.parts)))

    def new_group(self): return [[],[],[]]
    def add_group(self,name,acc):
        if acc[2]: self.parts.append([name,acc[0],acc[1],acc[2]])
    def write(self,path,generator="rscene lidar->modular"):
        buf=bytearray(); bvs=[]; accs=[]; meshes=[]; nodes=[]
        def push(data,target):
            while len(buf)%4: buf.append(0)
            o=len(buf); buf.extend(data)
            bvs.append(dict(buffer=0,byteOffset=o,byteLength=len(data),target=target))
            return len(bvs)-1
        for name,V,N,I in self.parts:
            Va=np.asarray(V,dtype=np.float32); Na=np.asarray(N,dtype=np.float32)
            Ia=np.asarray(I,dtype=np.uint32)
            bv=push(Va.tobytes(),34962)
            accs.append(dict(bufferView=bv,componentType=5126,count=len(Va),type="VEC3",
                             min=Va.min(axis=0).tolist(),max=Va.max(axis=0).tolist()))
            ap=len(accs)-1
            bv=push(Na.tobytes(),34962)
            accs.append(dict(bufferView=bv,componentType=5126,count=len(Na),type="VEC3"))
            an=len(accs)-1
            bv=push(Ia.tobytes(),34963)
            accs.append(dict(bufferView=bv,componentType=5125,count=len(Ia),type="SCALAR"))
            ai=len(accs)-1
            meshes.append(dict(name=name,primitives=[dict(
                attributes=dict(POSITION=ap,NORMAL=an),indices=ai,material=0)]))
            nodes.append(dict(name=name,mesh=len(meshes)-1))
        # parent nodes: children are the part-nodes already emitted above
        owned=set(); roots=[]
        for gname,a,b in self.groups:
            kids=[i for i in range(a,b)]
            if not kids: continue
            owned.update(kids)
            nodes.append(dict(name=gname,children=kids))
            roots.append(len(nodes)-1)
        roots += [i for i in range(len(self.parts)) if i not in owned]
        g=dict(asset=dict(version="2.0",generator=generator),
               scene=0,scenes=[dict(nodes=sorted(roots))],
               nodes=nodes,meshes=meshes,accessors=accs,bufferViews=bvs,
               materials=[dict(name="concrete",pbrMetallicRoughness=dict(
                   baseColorFactor=[0.78,0.78,0.76,1.0],metallicFactor=0.0,
                   roughnessFactor=0.9))],
               buffers=[dict(byteLength=len(buf))])
        js=json.dumps(g,separators=(',',':')).encode()
        while len(js)%4: js+=b' '
        while len(buf)%4: buf.append(0)
        total=12+8+len(js)+8+len(buf)
        with open(path,'wb') as f:
            f.write(struct.pack('<III',0x46546C67,2,total))
            f.write(struct.pack('<II',len(js),0x4E4F534A)); f.write(js)
            f.write(struct.pack('<II',len(buf),0x004E4942)); f.write(bytes(buf))
        return total,len(self.parts),sum(len(p[3])//3 for p in self.parts)
    def add_quads(self,name,quads,normal):
        """A batch of coplanar quads sharing one normal, as its own part.
        Used by the voxel surface, where a Poisson-style mesh is thousands of
        axis-aligned faces rather than a handful of boxes."""
        V=[];N=[];I=[]
        for q in quads:
            b=len(V); V+=list(q); N+=[normal]*4
            I+=[b,b+1,b+2,b,b+2,b+3]
        if V: self.parts.append([name,V,N,I])
