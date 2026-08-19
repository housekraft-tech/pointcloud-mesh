"""Viewer for the cell-based model and the LiDAR it was measured from.

The old fitted model (shell_fp.glb) is gone from here. It looked the cleanest of
everything we built and was the least true: 30.2% of the surface it drew had no
scanned point within 100 mm, and half its wall thicknesses were the median
rather than a measurement. Keeping it alongside invited exactly the confusion it
caused -- two models in one viewer under one summary.

What is here is measured: walls whose thickness is the gap between two detected
planes, openings whose sill and head come off the planes bounding the void, and
parts that partition the masonry without overlap or gap.
"""
import base64, json, os
from collections import Counter

GLB = "output/model/modular_cells.glb"
b64 = base64.b64encode(open(GLB, 'rb').read()).decode()
CM = json.load(open("output/cloud_meta.json"))
M = json.load(open("output/model/modular_cells.json"))
P = M['parts']; OPS = M.get('openings', [])
kinds = Counter(p['kind'] for p in P)
walls = sorted([p for p in P if p['kind'] == 'WALL'], key=lambda q: -q['volume_l'])
vol = sum(p['volume_l'] for p in P)/1000.0
tw = [(p['thickness_mm'], p['volume_l']) for p in walls]
tw.sort()
cum = 0; half = sum(v for _, v in tw)/2; tmed = tw[0][0] if tw else 0
for t, v in tw:
    cum += v
    if cum >= half: tmed = t; break
SUM = dict(parts=len(P), walls=kinds.get('WALL', 0), slabs=kinds.get('SLAB', 0),
           beams=kinds.get('BEAM', 0), columns=kinds.get('COLUMN', 0),
           openings=len(OPS), doors=sum(1 for o in OPS if o['kind'] == 'door'),
           wins=sum(1 for o in OPS if o['kind'] == 'window'),
           voids=sum(1 for o in OPS if o['kind'] == 'opening'),
           tmed=tmed, vol=round(vol, 1),
           ceiling=M.get('clear_height_mm'), pts=CM['n'], cut=CM['cut_mm'])
WROWS = [dict(n=p['name'], t=p['thickness_mm'], L=p['b_mm']-p['a_mm'],
              h=p['z1_mm']-p['z0_mm'], v=round(p['volume_l'])) for p in walls]
OROWS = [dict(k=o['kind'], w=o['width_mm'], s=o['sill_mm'], h=o['head_mm'],
              wall=o['wall']) for o in sorted(OPS, key=lambda o: (o['kind'], -o['width_mm']))]

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Bare-shell model</title>
<style>
 html,body{margin:0;height:100%;background:#11151a;color:#dfe6ee;
   font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
 #app{display:flex;height:100%}
 #view{flex:1;position:relative}
 #side{width:350px;overflow:auto;background:#161b22;border-left:1px solid #263041;padding:14px}
 h1{font-size:15px;margin:0 0 4px} h2{font-size:12px;margin:16px 0 6px;color:#8fa3bb;
   text-transform:uppercase;letter-spacing:.06em}
 .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2732}
 .k{color:#8fa3bb} .v{font-variant-numeric:tabular-nums}
 label{display:flex;align-items:center;gap:7px;padding:2px 0;cursor:pointer}
 button{background:#22303f;color:#dfe6ee;border:1px solid #35465c;border-radius:5px;
   padding:5px 9px;cursor:pointer;margin:2px 3px 2px 0;font-size:12px}
 button:hover{background:#2c3d50} button.on{background:#2f5d8a;border-color:#4a86c4}
 input[type=range]{width:100%}
 table{width:100%;border-collapse:collapse;font-size:11.5px}
 th{text-align:right;color:#8fa3bb;font-weight:500;border-bottom:1px solid #263041;padding:3px 4px}
 th:first-child,td:first-child{text-align:left}
 td{text-align:right;padding:2px 4px;border-bottom:1px solid #1a222c;
    font-variant-numeric:tabular-nums}
 .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
 #hud{position:absolute;left:12px;top:12px;background:#0d1117cc;padding:8px 11px;
   border-radius:6px;border:1px solid #263041;pointer-events:none}
 .note{color:#7d8a99;font-size:11.5px;line-height:1.45}
</style>
<div id="app">
  <div id="view"><div id="hud">drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div></div>
  <div id="side">
    <h1>Bare-shell model</h1>
    <div class="note">Built from the scan by space partition: walls are the
    masonry between two detected planes, so every thickness is measured and no
    two parts can occupy the same space.</div>
    <h2>Summary</h2><div id="sum"></div>
    <h2>Show</h2>
    <label><input type="checkbox" id="mOn" checked><span>Model</span></label>
    <label><input type="checkbox" id="byPart"><span>Colour each part separately</span></label>
    <label><input type="checkbox" id="ccOn"><span>Cell-complex shell</span></label>
    <label><input type="checkbox" id="sfOn"><span>Meshed scan (30 mm voxels)</span></label>
    <label><input type="checkbox" id="pcOn"><span>Point cloud (<span id="pcN"></span>)</span></label>
    <div class="k">point size</div><input type="range" id="pcSize" min="1" max="10" value="3">
    <h2>Kinds</h2><div id="legend"></div>
    <h2>Cut</h2>
    <div class="k">hide everything above <span id="clipV"></span> mm</div>
    <input type="range" id="clip" min="200" max="3000" value="3000">
    <h2>Look</h2>
    <div><button id="fit">Fit</button><button id="top">Top</button></div>
    <h2>Walls</h2>
    <table><thead><tr><th>part</th><th>thk</th><th>length</th><th>height</th><th>vol</th></tr>
      </thead><tbody id="wt"></tbody></table>
    <h2>Openings</h2>
    <table><thead><tr><th>kind</th><th>width</th><th>sill</th><th>head</th><th>in</th></tr>
      </thead><tbody id="ot"></tbody></table>
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
const SUM=__SUM__, WROWS=__WROWS__, OROWS=__OROWS__, CM=__CM__;
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); renderer.localClippingEnabled=true;
view.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x11151a);
const camera=new THREE.PerspectiveCamera(50,1,0.05,500);
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xdfe9f5,0x2b3440,2.1));
const dl=new THREE.DirectionalLight(0xffffff,1.5); dl.position.set(6,12,8); scene.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,0.6); dl2.position.set(-7,6,-5); scene.add(dl2);
scene.add(new THREE.GridHelper(30,60,0x2b3949,0x1b2430));
const COL={WALL:0xb9c6d4,SLAB:0x6f7b88,BEAM:0xb0577f,COLUMN:0xcc7a3d,
           DOOR:0x2f9e6e,WINDOW:0x4fc3d9,OPENING:0x8f6fc4};
const rowh=(k,v)=>`<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;
document.getElementById('sum').innerHTML=
  rowh('Parts',SUM.parts)+rowh('Walls',SUM.walls)+rowh('Slabs',SUM.slabs)+
  rowh('Beams',SUM.beams)+rowh('Columns',SUM.columns)+
  rowh('Doors',SUM.doors)+rowh('Openings',SUM.voids)+rowh('Windows',SUM.wins)+
  rowh('Wall thickness',SUM.tmed+' mm')+rowh('Masonry',SUM.vol+' m&sup3;')+
  rowh('Clear height',SUM.ceiling+' mm');
document.getElementById('wt').innerHTML=WROWS.map(r=>
  `<tr><td>${r.n}</td><td>${r.t}</td><td>${r.L}</td><td>${r.h}</td><td>${r.v}</td></tr>`).join('');
document.getElementById('ot').innerHTML=OROWS.map(r=>
  `<tr><td>${r.k}</td><td>${r.w}</td><td>${r.s}</td><td>${r.h}</td><td>${r.wall}</td></tr>`).join('');
function b64ToBuf(b){const s=atob(b),u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}
const clip=new THREE.Plane(new THREE.Vector3(0,-1,0),3.0);
let root=null; const seen={};
new GLTFLoader().parse(b64ToBuf("__B64__"),"",g=>{
  root=g.scene;
  let np=0;
  root.children.forEach(part=>{
    const k=(part.name||'').split('_')[0];
    seen[k]=(seen[k]||0)+1;
    const base=new THREE.MeshStandardMaterial({color:COL[k]??0xb0b8c2,
      roughness:0.92,metalness:0.0,side:THREE.DoubleSide,clippingPlanes:[clip]});
    const h=(np*0.381)%1;
    const solo=new THREE.MeshStandardMaterial({
      color:new THREE.Color().setHSL(h,0.55,0.55),roughness:0.92,metalness:0.0,
      side:THREE.DoubleSide,clippingPlanes:[clip]});
    part.traverse(o=>{ if(o.isMesh){o.userData.byKind=base; o.userData.byPart=solo;
      o.material=base;} });
    np++;
  });
  scene.add(root);
  document.getElementById('legend').innerHTML=Object.keys(seen).sort().map(k=>
    `<div class="row"><span><span class="sw" style="background:#${
      (COL[k]??0xb0b8c2).toString(16).padStart(6,'0')}"></span>${k}</span>`+
    `<span class="v">${seen[k]}</span></div>`).join('');
  fit();
});
document.getElementById('mOn').onchange=e=>{if(root)root.visible=e.target.checked;};
document.getElementById('byPart').onchange=e=>{
  if(!root) return; const b=e.target.checked;
  root.traverse(o=>{if(o.isMesh)o.material=b?o.userData.byPart:o.userData.byKind;});
};
// the cell complex it was carved from
let cc=null;
fetch('model/cellcomplex.glb').then(r=>r.arrayBuffer()).then(buf=>
new GLTFLoader().parse(buf,"",g=>{
  cc=g.scene; cc.visible=false;
  cc.traverse(o=>{ if(o.isMesh) o.material=new THREE.MeshStandardMaterial(
    {color:0xd98a4f,roughness:0.9,side:THREE.DoubleSide,clippingPlanes:[clip]}); });
  scene.add(cc);
}));
document.getElementById('ccOn').onchange=e=>{if(cc)cc.visible=e.target.checked;};
// the scan itself
let surf=null;
fetch('model/scan_surface.glb').then(r=>r.arrayBuffer()).then(buf=>
new GLTFLoader().parse(buf,"",g=>{
  surf=g.scene; surf.visible=false;
  surf.traverse(o=>{ if(o.isMesh) o.material=new THREE.MeshStandardMaterial(
    {color:0x4fa88a,roughness:0.95,side:THREE.DoubleSide,clippingPlanes:[clip]}); });
  scene.add(surf);
}));
document.getElementById('sfOn').onchange=e=>{if(surf)surf.visible=e.target.checked;};
let cloud=null;
fetch('cloud.bin').then(r=>r.arrayBuffer()).then(cbuf=>{
  const cb=new Uint8Array(cbuf);
  const N=CM.n, q=new Int16Array(cb.buffer,0,N*3), rgb=cb.subarray(N*6,N*9);
  const pos=new Float32Array(N*3), col=new Float32Array(N*3);
  for(let i=0;i<N*3;i++){pos[i]=CM.lo[i%3]+(q[i]+32767)*CM.scale[i%3]; col[i]=rgb[i]/255;}
  const pg=new THREE.BufferGeometry();
  pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
  pg.setAttribute('color',new THREE.BufferAttribute(col,3));
  const pm=new THREE.PointsMaterial({size:0.012,vertexColors:true,
    sizeAttenuation:true,clippingPlanes:[clip]});
  cloud=new THREE.Points(pg,pm); cloud.visible=false; scene.add(cloud);
  document.getElementById('pcN').textContent=(N/1e6).toFixed(1)+'M pts';
  document.getElementById('pcSize').oninput=e=>pm.size=e.target.value*0.004;
});
document.getElementById('pcOn').onchange=e=>{if(cloud)cloud.visible=e.target.checked;};
const cl=document.getElementById('clip'), cv=document.getElementById('clipV');
function setClip(){clip.constant=cl.value/1000; cv.textContent=cl.value;}
cl.oninput=setClip; setClip();
function fit(){
  if(!root) return;
  const b=new THREE.Box3().setFromObject(root),c=b.getCenter(new THREE.Vector3());
  const r=b.getSize(new THREE.Vector3()).length()/2;
  camera.position.set(c.x+r*1.05,c.y+r*0.75,c.z+r*1.05);
  camera.near=r/200; camera.far=r*30; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}
document.getElementById('fit').onclick=fit;
document.getElementById('top').onclick=()=>{
  if(!root) return;
  const b=new THREE.Box3().setFromObject(root),c=b.getCenter(new THREE.Vector3());
  const r=b.getSize(new THREE.Vector3()).length()/2;
  camera.position.set(c.x,c.y+r*1.7,c.z+0.01); controls.target.copy(c); controls.update();
};
function resize(){const w=view.clientWidth,h=view.clientHeight;
  renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);})();
</script>
"""
HTML = (HTML.replace("__B64__", b64).replace("__SUM__", json.dumps(SUM))
            .replace("__WROWS__", json.dumps(WROWS)).replace("__OROWS__", json.dumps(OROWS))
            .replace("__CM__", json.dumps(CM)))
open("output/viewer.html", "w").write(HTML)
print(f"wrote output/viewer.html  {os.path.getsize('output/viewer.html')/1e6:.1f} MB")
print(f"  {SUM['parts']} parts: {SUM['walls']} walls, {SUM['slabs']} slabs, "
      f"{SUM['beams']} beams, {SUM['columns']} columns")
print(f"  {SUM['openings']} openings: {SUM['doors']} doors, {SUM['voids']} openings, "
      f"{SUM['wins']} windows")
print(f"  wall thickness {SUM['tmed']} mm by volume, masonry {SUM['vol']} m3")
print("  the old fitted model is no longer shown")
