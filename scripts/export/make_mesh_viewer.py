"""A page showing ONLY the meshed scan -- no model, nothing inferred.

Everything here is the LiDAR and nothing else: an occupancy volume at 40 mm and
the isosurface between filled and empty. Poisson proper needs open3d, which
cannot be installed in this environment (no pip), so this is the other standard
route to the same thing. It is blockier than Poisson at the voxel size, but it
is not an interpretation of the scan -- it IS the scan, which is the point when
you want to judge the scan on its own.

The point cloud is available underneath as the reference the mesh came from,
and a height clip lets you cut down through the flat.
"""
import base64, json, os

CM = json.load(open("output/cloud_meta.json"))

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Meshed scan</title>
<style>
 html,body{margin:0;height:100%;background:#0f1216;color:#dfe6ee;
   font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
 #app{display:flex;height:100%}
 #view{flex:1;position:relative}
 #side{width:300px;overflow:auto;background:#161b22;border-left:1px solid #263041;padding:14px}
 h1{font-size:15px;margin:0 0 4px}
 h2{font-size:12px;margin:16px 0 6px;color:#8fa3bb;text-transform:uppercase;letter-spacing:.06em}
 .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2732}
 .k{color:#8fa3bb} .v{font-variant-numeric:tabular-nums}
 label{display:flex;align-items:center;gap:7px;padding:2px 0;cursor:pointer}
 button{background:#22303f;color:#dfe6ee;border:1px solid #35465c;border-radius:5px;
   padding:5px 9px;cursor:pointer;margin:2px 3px 2px 0;font-size:12px}
 button:hover{background:#2c3d50} button.on{background:#2f5d8a;border-color:#4a86c4}
 input[type=range]{width:100%}
 #hud{position:absolute;left:12px;top:12px;background:#0d1117cc;padding:8px 11px;
   border-radius:6px;border:1px solid #263041;pointer-events:none}
</style>
<div id="app">
  <div id="view"><div id="hud">drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div></div>
  <div id="side">
    <h1>Meshed scan</h1>
    <div style="color:#8fa3bb">The LiDAR only. No walls, no model, nothing
    inferred &mdash; a 30&nbsp;mm isosurface of the scan's own occupancy.</div>
    <h2>Stats</h2><div id="stats"></div>
    <h2>Show</h2>
    <label><input type="checkbox" id="mOn" checked><span>Meshed scan</span></label>
    <label><input type="checkbox" id="pcOn"><span>Point cloud (<span id="pcN"></span> pts)</span></label>
    <div class="k">point size</div><input type="range" id="pcSize" min="1" max="10" value="3">
    <h2>Cut</h2>
    <div class="k">clip everything above <span id="clipV"></span> mm</div>
    <input type="range" id="clip" min="200" max="2750" value="2750">
    <h2>Depth</h2>
    <div class="k">shading &mdash; how depth is read off the surface</div>
    <div><button id="mPlain" class="on">Plain</button><button id="mAO">Cavity</button>
         <button id="mHeight">Height</button><button id="mNormal">Facing</button></div>
    <div class="k">cavity strength</div>
    <input type="range" id="aoK" min="0" max="100" value="60">
    <div class="k">haze with distance</div>
    <input type="range" id="fog" min="0" max="100" value="0">
    <h2>Look</h2>
    <div><button id="fit">Fit view</button><button id="wire">Wireframe</button>
         <button id="flat" class="on">Flat shading</button></div>
    <div><button id="top">Top</button><button id="iso">Iso</button></div>
  </div>
</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const CM=__CM__;
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio);
renderer.localClippingEnabled=true;
view.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0f1216);
const camera=new THREE.PerspectiveCamera(50,1,0.05,500);
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xe6eef8,0x303a46,2.0));
const dl=new THREE.DirectionalLight(0xffffff,1.6); dl.position.set(6,12,8); scene.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,0.7); dl2.position.set(-7,6,-5); scene.add(dl2);
scene.add(new THREE.GridHelper(30,60,0x2b3949,0x1b2430));
function b64ToBuf(b){const s=atob(b),u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}
const clipPlane=new THREE.Plane(new THREE.Vector3(0,-1,0),2.75);
let mesh=null;
fetch('model/scan_surface.glb').then(r=>r.arrayBuffer()).then(buf=>
new GLTFLoader().parse(buf,"",g=>{
  mesh=g.scene;
  let tris=0;
  mesh.traverse(o=>{ if(!o.isMesh) return;
    o.material=new THREE.MeshStandardMaterial({color:0xc3cbd4,roughness:0.95,
      metalness:0.0,side:THREE.DoubleSide,clippingPlanes:[clipPlane],flatShading:true});
    const gg=o.geometry; tris+=(gg.index?gg.index.count:gg.attributes.position.count)/3;
  });
  scene.add(mesh);
  buildDepthCues(mesh);
  const b=new THREE.Box3().setFromObject(mesh), s=b.getSize(new THREE.Vector3());
  document.getElementById('stats').innerHTML=
    rowh('Voxel','30 mm')+rowh('Triangles',Math.round(tris).toLocaleString())+
    rowh('Extent X',(s.x*1000).toFixed(0)+' mm')+
    rowh('Extent Y',(s.z*1000).toFixed(0)+' mm')+
    rowh('Height',(s.y*1000).toFixed(0)+' mm');
  fit();
}));
function rowh(k,v){return `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;}

// ---- depth cues -------------------------------------------------------
// A flat grey solid hides exactly what you want to see: how far a niche is set
// back, how far a column stands proud. Three cheap cues, no post-processing:
//   Cavity  darken by how enclosed a vertex is, measured against its own
//           neighbours in a voxel grid -- recesses go dark, faces stay light
//   Height  colour ramp up the wall, so a step reads as a colour break
//   Facing  colour by surface direction, so every change of plane is a
//           different colour and a 40 mm step is unmissable
let AOcol=null, Hcol=null, Ncol=null;
function buildDepthCues(root){
  const CELL=0.16, occ=new Set(), key=(a,b,c)=>a+'|'+b+'|'+c;
  const parts=[];
  root.traverse(o=>{if(o.isMesh)parts.push(o);});
  parts.forEach(o=>{const P=o.geometry.attributes.position;
    for(let i=0;i<P.count;i++)
      occ.add(key(Math.round(P.getX(i)/CELL),Math.round(P.getY(i)/CELL),
                  Math.round(P.getZ(i)/CELL)));});
  let ymin=1e9,ymax=-1e9;
  parts.forEach(o=>{const P=o.geometry.attributes.position;
    for(let i=0;i<P.count;i++){const y=P.getY(i); if(y<ymin)ymin=y; if(y>ymax)ymax=y;}});
  parts.forEach(o=>{
    const P=o.geometry.attributes.position, n=P.count;
    const ao=new Float32Array(n*3), hc=new Float32Array(n*3), nc=new Float32Array(n*3);
    const N=o.geometry.attributes.normal;
    for(let i=0;i<n;i++){
      const x=P.getX(i),y=P.getY(i),zz=P.getZ(i);
      const gx=Math.round(x/CELL),gy=Math.round(y/CELL),gz=Math.round(zz/CELL);
      let hit=0,tot=0;
      for(let a=-1;a<=1;a++)for(let b=-1;b<=1;b++)for(let c=-1;c<=1;c++){
        if(!a&&!b&&!c) continue; tot++;
        if(occ.has(key(gx+a,gy+b,gz+c))) hit++;
      }
      const v=1-(hit/tot);                       // 1 = exposed, 0 = deep in a cavity
      ao[i*3]=ao[i*3+1]=ao[i*3+2]=v;
      const t=(y-ymin)/Math.max(ymax-ymin,1e-6);
      hc[i*3]=0.25+0.75*t; hc[i*3+1]=0.45+0.35*Math.sin(t*3.14); hc[i*3+2]=1.0-0.7*t;
      nc[i*3]=Math.abs(N.getX(i))*0.9+0.1;
      nc[i*3+1]=Math.abs(N.getY(i))*0.9+0.1;
      nc[i*3+2]=Math.abs(N.getZ(i))*0.9+0.1;
    }
    AOcol=AOcol||[]; Hcol=Hcol||[]; Ncol=Ncol||[];
    o.userData.ao=new THREE.BufferAttribute(ao,3);
    o.userData.hc=new THREE.BufferAttribute(hc,3);
    o.userData.nc=new THREE.BufferAttribute(nc,3);
  });
  setMode('plain');
}
let aoK=0.6, mode='plain';
function setMode(m){
  mode=m;
  ['mPlain','mAO','mHeight','mNormal'].forEach(id=>
    document.getElementById(id).classList.remove('on'));
  document.getElementById({plain:'mPlain',ao:'mAO',height:'mHeight',
    normal:'mNormal'}[m]).classList.add('on');
  if(!mesh) return;
  mesh.traverse(o=>{ if(!o.isMesh) return;
    const g=o.geometry;
    if(m==='plain'){ g.deleteAttribute('color'); o.material.vertexColors=false;
      o.material.color.setHex(0xc3cbd4); }
    else{
      let src = m==='ao'?o.userData.ao : m==='height'?o.userData.hc : o.userData.nc;
      if(m==='ao'){
        const a=o.userData.ao, out=new Float32Array(a.count*3);
        for(let i=0;i<a.count;i++){
          const v=1-(1-a.getX(i))*aoK;
          out[i*3]=out[i*3+1]=out[i*3+2]=v;
        }
        src=new THREE.BufferAttribute(out,3);
      }
      g.setAttribute('color',src);
      o.material.vertexColors=true; o.material.color.setHex(0xffffff);
    }
    o.material.needsUpdate=true;
  });
}
document.getElementById('mPlain').onclick=()=>setMode('plain');
document.getElementById('mAO').onclick=()=>setMode('ao');
document.getElementById('mHeight').onclick=()=>setMode('height');
document.getElementById('mNormal').onclick=()=>setMode('normal');
document.getElementById('aoK').oninput=e=>{aoK=e.target.value/100;
  if(mode==='ao') setMode('ao');};
document.getElementById('fog').oninput=e=>{
  const f=e.target.value/100;
  scene.fog = f>0 ? new THREE.FogExp2(0x0f1216, f*0.10) : null;
  if(mesh) mesh.traverse(o=>{if(o.isMesh)o.material.needsUpdate=true;});
};
// point cloud
let cloud=null;
fetch('cloud.bin').then(r=>r.arrayBuffer()).then(buf=>{
const cb=new Uint8Array(buf);
const N=CM.n, q=new Int16Array(cb.buffer,0,N*3), rgb=cb.subarray(N*6,N*9);
const pos=new Float32Array(N*3), col=new Float32Array(N*3);
for(let i=0;i<N*3;i++){pos[i]=CM.lo[i%3]+(q[i]+32767)*CM.scale[i%3]; col[i]=rgb[i]/255;}
const pg=new THREE.BufferGeometry();
pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
pg.setAttribute('color',new THREE.BufferAttribute(col,3));
const pm=new THREE.PointsMaterial({size:0.012,vertexColors:true,sizeAttenuation:true,
  clippingPlanes:[clipPlane]});
cloud=new THREE.Points(pg,pm); cloud.visible=document.getElementById('pcOn').checked;
scene.add(cloud);
document.getElementById('pcN').textContent=(N/1000).toFixed(0)+'k';
});
document.getElementById('pcOn').onchange=e=>{if(cloud)cloud.visible=e.target.checked;};
document.getElementById('mOn').onchange=e=>{if(mesh)mesh.visible=e.target.checked;};
document.getElementById('pcSize').oninput=e=>pm.size=e.target.value*0.004;
const clipEl=document.getElementById('clip'), clipV=document.getElementById('clipV');
function setClip(){clipPlane.constant=clipEl.value/1000; clipV.textContent=clipEl.value;}
clipEl.oninput=setClip; setClip();
let wire=false;
document.getElementById('wire').onclick=e=>{wire=!wire; e.target.classList.toggle('on',wire);
  if(mesh)mesh.traverse(o=>{if(o.isMesh)o.material.wireframe=wire;});};
let flat=true;
document.getElementById('flat').onclick=e=>{flat=!flat; e.target.classList.toggle('on',flat);
  if(mesh)mesh.traverse(o=>{if(o.isMesh){o.material.flatShading=flat;
    o.material.needsUpdate=true;}});};
function fit(){
  if(!mesh) return;
  const b=new THREE.Box3().setFromObject(mesh),c=b.getCenter(new THREE.Vector3());
  const r=b.getSize(new THREE.Vector3()).length()/2;
  camera.position.set(c.x+r*1.1,c.y+r*0.8,c.z+r*1.1);
  camera.near=r/150; camera.far=r*30; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}
document.getElementById('fit').onclick=fit;
document.getElementById('iso').onclick=fit;
document.getElementById('top').onclick=()=>{
  if(!mesh) return;
  const b=new THREE.Box3().setFromObject(mesh),c=b.getCenter(new THREE.Vector3());
  const r=b.getSize(new THREE.Vector3()).length()/2;
  camera.position.set(c.x,c.y+r*1.8,c.z+0.001); controls.target.copy(c); controls.update();
};
function resize(){const w=view.clientWidth,h=view.clientHeight;
  renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);})();
</script>
"""
HTML = HTML.replace("__CM__", json.dumps(CM))
open("output/scan_mesh.html", "w").write(HTML)
print(f"wrote output/scan_mesh.html  {os.path.getsize('output/scan_mesh.html')/1e6:.1f} MB")
print(f"  fetches model/scan_surface.glb "
      f"({os.path.getsize('output/model/scan_surface.glb')/1e6:.1f} MB) and cloud.bin "
      f"({CM['n']:,} points) from the same directory -- serve, do not open from disk")
