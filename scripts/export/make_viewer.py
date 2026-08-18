"""Single self-contained HTML viewer: the modular model and the LiDAR it was
measured from, in one frame, so the two can be overlaid and checked against
each other. Opens from disk, no server."""
import base64, json, os

GLB = "output/model/shell_fp.glb"
b64 = base64.b64encode(open(GLB, 'rb').read()).decode()
cloud = base64.b64encode(open("/tmp/cloud.bin", 'rb').read()).decode()
surf = base64.b64encode(open("output/model/scan_surface.glb", 'rb').read()).decode()
CM = json.load(open("/tmp/cloud_meta.json"))
meta = json.load(open("output/model/shell_fp.json"))
rows = [dict(id=w["id"], length=w["length_mm"], thick=w["thickness_mm"],
             ops=[dict(k=o["kind"], w=o["width_mm"], s=o["sill_mm"],
                       h=o["head_mm"], a=o["arch_mm"]) for o in w.get("openings", [])])
        for w in meta.get("walls", [])]
FEAT = meta.get('features', [])
_fc = {}
for r in FEAT: _fc[r['kind']] = _fc.get(r['kind'], 0)+1
SUM = dict(walls=len(rows), openings=sum(len(r['ops']) for r in rows),
           beams=_fc.get('BEAM', 0), columns=_fc.get('COLUMN', 0),
           pilasters=_fc.get('PILASTER', 0), steps=_fc.get('STEP', 0),
           niches=_fc.get('NICHE', 0),
           ceiling=meta.get('clear_height_mm'), pts=CM['n'], cut=CM['cut_mm'])

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Scan viewer</title>
<style>
 html,body{margin:0;height:100%;background:#11151a;color:#dfe6ee;
   font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
 #app{display:flex;height:100%}
 #view{flex:1;position:relative}
 #side{width:340px;overflow:auto;background:#161b22;border-left:1px solid #263041;padding:14px}
 h1{font-size:15px;margin:0 0 4px} h2{font-size:12px;margin:16px 0 6px;color:#8fa3bb;
   text-transform:uppercase;letter-spacing:.06em}
 .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2732}
 .k{color:#8fa3bb} .v{font-variant-numeric:tabular-nums}
 label{display:flex;align-items:center;gap:7px;padding:2px 0;cursor:pointer}
 button{background:#22303f;color:#dfe6ee;border:1px solid #35465c;border-radius:5px;
   padding:5px 9px;cursor:pointer;margin:2px 3px 2px 0;font-size:12px}
 button:hover{background:#2c3d50} button.on{background:#2f5d8a;border-color:#4a86c4}
 input[type=range]{width:100%}
 #hud{position:absolute;left:12px;top:12px;background:#0d1117cc;padding:8px 11px;
   border-radius:6px;border:1px solid #263041;pointer-events:none}
 table{width:100%;border-collapse:collapse;font-size:12px}
 td,th{text-align:left;padding:2px 4px;border-bottom:1px solid #1e2732}
 th{color:#8fa3bb;font-weight:500}
</style>
<div id="app">
  <div id="view"><div id="hud">drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div></div>
  <div id="side">
    <h1>Bare-shell scan</h1>
    <div style="color:#8fa3bb">Model and LiDAR in one frame. Toggle the cloud to
    check the model against the scan it was measured from.</div>
    <h2>Summary</h2><div id="summary"></div>
    <h2>LiDAR</h2>
    <label><input type="checkbox" id="pcOn"><span>Show point cloud
      (<span id="pcN"></span> pts, top <span id="pcCut"></span> mm of ceiling cut)</span></label>
    <div class="k">point size</div><input type="range" id="pcSize" min="1" max="10" value="3">
    <div class="k">brightness</div><input type="range" id="pcDim" min="10" max="100" value="70">
    <div><button id="pcOnly">Cloud only</button><button id="modelOnly">Model only</button>
         <button id="both">Both</button></div>
    <h2>Scan surface</h2>
    <label><input type="checkbox" id="sfOn"><span>Show meshed scan (60 mm voxel
      isosurface of the LiDAR)</span></label>
    <div class="k">opacity</div><input type="range" id="sfOp" min="10" max="100" value="100">
    <h2>Model layers</h2><div id="toggles"></div>
    <div><button id="fit">Fit view</button><button id="wire">Wireframe</button>
         <button id="edges" class="on">Edges</button><button id="xray">X-ray walls</button></div>
    <h2>Walls</h2>
    <table id="walls"><thead><tr><th>#</th><th>Length</th><th>Thk</th><th>Openings</th></tr>
    </thead><tbody></tbody></table>
    <h2>Openings</h2>
    <table id="ops"><thead><tr><th>Wall</th><th>Type</th><th>W</th><th>Sill</th><th>Head</th>
    <th>Arch</th></tr></thead><tbody></tbody></table>
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
const WALLS=__WALLS__, SUM=__SUM__, CM=__CM__;
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); view.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x11151a);
const camera=new THREE.PerspectiveCamera(50,1,0.05,500);
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xdfe9f5,0x2b3440,2.1));
const dl=new THREE.DirectionalLight(0xffffff,1.5); dl.position.set(6,12,8); scene.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,0.6); dl2.position.set(-7,6,-5); scene.add(dl2);
scene.add(new THREE.GridHelper(30,60,0x2b3949,0x1b2430));
let root=null, groups={}, wire=false, edgesOn=true, xray=false;
const COL={WALL:0xb9c6d4,ARCH:0x8f6fc4,BEAM:0xb0577f,COLUMN:0xcc7a3d,
           PILASTER:0xd9a441,STEP:0x9c8ab0,NICHE:0x3d7fc1,DOOR:0x2f9e6e,WINDOW:0x4fc3d9,
           FLOOR:0x6f7b88,CEILING:0x8a97a6};
function kindOf(n){for(const k in COL) if(n.startsWith(k)) return k; return 'OTHER';}
function b64ToBuf(b){const s=atob(b),u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}
function addObject(obj){
  if(root) scene.remove(root);
  root=obj; groups={}; scene.add(obj);
  obj.traverse(o=>{ if(!o.isMesh) return;
    const k=kindOf(o.name||'');
    o.material=new THREE.MeshStandardMaterial({color:COL[k]??0xb0b8c2,
      roughness:0.92,metalness:0.0,side:THREE.DoubleSide});
    (groups[k]=groups[k]||[]).push(o);
    const e=new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry,25),
      new THREE.LineBasicMaterial({color:0x22303f}));
    o.add(e); o.userData.edges=e;
  });
  buildToggles(); fit();
}
function buildToggles(){
  const el=document.getElementById('toggles'); el.innerHTML='';
  Object.keys(groups).sort().forEach(k=>{
    const c=document.createElement('label');
    c.innerHTML=`<input type="checkbox" checked><span style="width:11px;height:11px;
      border-radius:2px;display:inline-block;background:#${
      (COL[k]??0xb0b8c2).toString(16).padStart(6,'0')}"></span><span>${k} (${
      groups[k].length})</span>`;
    c.querySelector('input').onchange=e=>groups[k].forEach(o=>o.visible=e.target.checked);
    el.appendChild(c);
  });
}
function fit(){
  const b=new THREE.Box3().setFromObject(root),c=b.getCenter(new THREE.Vector3());
  const r=b.getSize(new THREE.Vector3()).length()/2;
  camera.position.set(c.x+r*1.15,c.y+r*0.85,c.z+r*1.15);
  camera.near=r/120; camera.far=r*30; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}
new GLTFLoader().parse(b64ToBuf("__B64__"),"",g=>addObject(g.scene));

// ---- the scan meshed: the LiDAR's own surface, to overlay on the model ----
let surf=null;
new GLTFLoader().parse(b64ToBuf("__SURF__"),"",g=>{
  surf=g.scene; surf.visible=false;
  surf.traverse(o=>{ if(o.isMesh) o.material=new THREE.MeshStandardMaterial(
    {color:0x4fa88a,roughness:0.95,metalness:0.0,side:THREE.DoubleSide}); });
  scene.add(surf);
});
document.getElementById('sfOn').onchange=e=>{if(surf)surf.visible=e.target.checked;};
document.getElementById('sfOp').oninput=e=>{const f=e.target.value/100;
  if(surf)surf.traverse(o=>{if(o.isMesh){o.material.transparent=f<1;
    o.material.opacity=f; o.material.depthWrite=f>=1;}});};

// ---- LiDAR: Int16 positions dequantised against the cloud's own bbox ----
const cb=new Uint8Array(b64ToBuf("__CLOUD__"));
const N=CM.n, q=new Int16Array(cb.buffer,0,N*3), rgb=cb.subarray(N*6,N*9);
const pos=new Float32Array(N*3), col=new Float32Array(N*3);
for(let i=0;i<N*3;i++){
  pos[i]=CM.lo[i%3]+(q[i]+32767)*CM.scale[i%3];
  col[i]=rgb[i]/255;
}
const pg=new THREE.BufferGeometry();
pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
pg.setAttribute('color',new THREE.BufferAttribute(col,3));
const pm=new THREE.PointsMaterial({size:0.012,vertexColors:true,sizeAttenuation:true});
const cloud=new THREE.Points(pg,pm); cloud.visible=false; scene.add(cloud);
const pcOn=document.getElementById('pcOn');
pcOn.onchange=e=>cloud.visible=e.target.checked;
document.getElementById('pcSize').oninput=e=>pm.size=e.target.value*0.004;
document.getElementById('pcDim').oninput=e=>{
  const f=e.target.value/70; pm.color.setScalar(Math.min(f,1));
  pm.opacity=Math.min(f,1); pm.transparent=f<1;
};
function setBoth(c,m){cloud.visible=c; pcOn.checked=c; if(root) root.visible=m;}
document.getElementById('pcOnly').onclick=()=>setBoth(true,false);
document.getElementById('modelOnly').onclick=()=>setBoth(false,true);
document.getElementById('both').onclick=()=>setBoth(true,true);
document.getElementById('fit').onclick=()=>fit();
document.getElementById('wire').onclick=e=>{wire=!wire; e.target.classList.toggle('on',wire);
  root.traverse(o=>{if(o.isMesh)o.material.wireframe=wire;});};
document.getElementById('edges').onclick=e=>{edgesOn=!edgesOn;
  e.target.classList.toggle('on',edgesOn);
  root.traverse(o=>{if(o.userData.edges)o.userData.edges.visible=edgesOn;});};
document.getElementById('xray').onclick=e=>{xray=!xray; e.target.classList.toggle('on',xray);
  root.traverse(o=>{if(o.isMesh){o.material.transparent=xray;
    o.material.opacity=xray?0.35:1; o.material.depthWrite=!xray;}});};

// ---- panel ----
const rowh=(k,v)=>`<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;
document.getElementById('summary').innerHTML=
  rowh('Walls',SUM.walls)+rowh('Openings',SUM.openings)+rowh('Beams',SUM.beams)+
  rowh('Columns',SUM.columns)+rowh('Pilasters',SUM.pilasters)+rowh('Thickness steps',SUM.steps)+
  rowh('Niches',SUM.niches)+rowh('Clear height',SUM.ceiling+' mm');
document.getElementById('pcN').textContent=(SUM.pts/1000).toFixed(0)+'k';
document.getElementById('pcCut').textContent=SUM.cut;
const wb=document.querySelector('#walls tbody'), ob=document.querySelector('#ops tbody');
WALLS.forEach(w=>{
  wb.insertAdjacentHTML('beforeend',
    `<tr><td>${w.id}</td><td>${w.length}</td><td>${w.thick}</td><td>${w.ops.length||''}</td></tr>`);
  w.ops.forEach(o=>ob.insertAdjacentHTML('beforeend',
    `<tr><td>${w.id}</td><td>${o.k}</td><td>${o.w}</td><td>${o.s}</td><td>${o.h}</td>
     <td>${o.a}</td></tr>`));
});
if(!ob.children.length) ob.insertAdjacentHTML('beforeend',
  '<tr><td colspan="6" style="color:#c98">none detected</td></tr>');
function resize(){const w=view.clientWidth,h=view.clientHeight;
  renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);})();
</script>
"""
HTML = (HTML.replace("__B64__", b64).replace("__CLOUD__", cloud)
            .replace("__WALLS__", json.dumps(rows)).replace("__SUM__", json.dumps(SUM))
            .replace("__CM__", json.dumps(CM)).replace("__SURF__", surf))
open("output/viewer.html", "w").write(HTML)
print(f"wrote output/viewer.html  {os.path.getsize('output/viewer.html')/1e6:.1f} MB")
print(f"  model {os.path.getsize(GLB)/1024:.0f} KB, {SUM['walls']} walls, "
      f"{SUM['openings']} openings, {SUM['beams']} beams, {SUM['columns']} columns, "
      f"{SUM['pilasters']} pilasters, {SUM['niches']} niches")
print(f"  LiDAR {SUM['pts']:,} points, top {SUM['cut']} mm of ceiling cut off")
print(f"  scan surface {os.path.getsize('output/model/scan_surface.glb')/1e6:.1f} MB")
