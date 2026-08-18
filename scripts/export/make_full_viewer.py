"""Viewer for a full-resolution cloud -- every point, nothing thinned."""
import json, os
M = json.load(open("output/full_meta.json"))

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Full cloud</title>
<style>
 html,body{margin:0;height:100%;background:#0c0f13;color:#dfe6ee;
   font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
 #app{display:flex;height:100%}
 #view{flex:1;position:relative}
 #side{width:300px;overflow:auto;background:#161b22;border-left:1px solid #263041;padding:14px}
 h1{font-size:15px;margin:0 0 4px}
 h2{font-size:12px;margin:16px 0 6px;color:#8fa3bb;text-transform:uppercase;letter-spacing:.06em}
 .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2732}
 .k{color:#8fa3bb} .v{font-variant-numeric:tabular-nums}
 button{background:#22303f;color:#dfe6ee;border:1px solid #35465c;border-radius:5px;
   padding:5px 9px;cursor:pointer;margin:2px 3px 2px 0;font-size:12px}
 button:hover{background:#2c3d50} button.on{background:#2f5d8a;border-color:#4a86c4}
 input[type=range]{width:100%}
 #hud{position:absolute;left:12px;top:12px;background:#0d1117cc;padding:8px 11px;
   border-radius:6px;border:1px solid #263041;pointer-events:none}
 #load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
   background:#0d1117ee;padding:16px 22px;border-radius:8px;border:1px solid #35465c}
</style>
<div id="app">
  <div id="view">
    <div id="hud">drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div>
    <div id="load">loading 232 MB&hellip; <span id="pct">0%</span></div>
  </div>
  <div id="side">
    <h1>Full cloud</h1>
    <div style="color:#8fa3bb">Every point in the export. Nothing thinned,
    nothing filtered.</div>
    <h2>Stats</h2><div id="stats"></div>
    <h2>Points</h2>
    <div class="k">size</div><input type="range" id="sz" min="1" max="12" value="2">
    <div class="k">brightness</div><input type="range" id="br" min="20" max="200" value="100">
    <h2>Cut</h2>
    <div class="k">show only below <span id="clipV"></span> m</div>
    <input type="range" id="clip" min="0" max="100" value="100">
    <h2>Look</h2>
    <div><button id="fit">Fit</button><button id="top">Top</button>
         <button id="persp" class="on">Perspective</button></div>
  </div>
</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const M=__META__;
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
renderer.localClippingEnabled=true;
view.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0c0f13);
const camera=new THREE.PerspectiveCamera(55,1,0.05,2000);
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
scene.add(new THREE.GridHelper(120,120,0x24313f,0x18212b));
const rowh=(k,v)=>`<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`;

// Positions stay Int16 and colours stay Uint8 all the way to the GPU. Turning
// 27 M points into Float32 on the CPU would cost 324 MB for positions alone and
// as much again for colour; the mapping back to metres is done instead by the
// object's scale and position, which the GPU applies for free.
let done=0;
const pct=document.getElementById('pct');
async function grab(url,tag){
  const r=await fetch(url); const len=+r.headers.get('content-length')||0;
  const chunks=[]; let got=0;
  const rd=r.body.getReader();
  for(;;){const {done:d,value}=await rd.read(); if(d)break;
    chunks.push(value); got+=value.length;
    if(len){pct.textContent=Math.round((done+got)/232e6*100)+'%';}}
  done+=got;
  const out=new Uint8Array(got); let o=0;
  for(const c of chunks){out.set(c,o); o+=c.length;}
  return out;
}
const clipPlane=new THREE.Plane(new THREE.Vector3(0,-1,0),1e6);
Promise.all([grab('full_pos.bin'),grab('full_col.bin')]).then(([pb,cb])=>{
  const N=M.n;
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Int16BufferAttribute(
    new Int16Array(pb.buffer,pb.byteOffset,N*3),3));
  g.setAttribute('color',new THREE.Uint8BufferAttribute(
    new Uint8Array(cb.buffer,cb.byteOffset,N*3),3,true));
  const mat=new THREE.PointsMaterial({size:2,sizeAttenuation:false,
    vertexColors:true,clippingPlanes:[clipPlane]});
  const pts=new THREE.Points(g,mat);
  // world = attr*scale + (lo + 32767*scale)
  pts.scale.set(M.scale[0],M.scale[1],M.scale[2]);
  pts.position.set(M.lo[0]+32767*M.scale[0], M.lo[1]+32767*M.scale[1],
                   M.lo[2]+32767*M.scale[2]);
  g.boundingSphere=new THREE.Sphere(new THREE.Vector3(0,0,0),40000);
  scene.add(pts);
  document.getElementById('load').remove();
  const span=[65534*M.scale[0],65534*M.scale[1],65534*M.scale[2]];
  document.getElementById('stats').innerHTML=
    rowh('Points',N.toLocaleString())+rowh('Source',M.src)+
    rowh('Extent X',span[0].toFixed(1)+' m')+
    rowh('Extent Z',span[2].toFixed(1)+' m')+
    rowh('Height',span[1].toFixed(1)+' m')+
    rowh('Buffers','232 MB');
  const clip=document.getElementById('clip'), cv=document.getElementById('clipV');
  function setClip(){const t=clip.value/100;
    const y=M.lo[1]+t*span[1]; clipPlane.constant=y; cv.textContent=y.toFixed(1);}
  clip.oninput=setClip; setClip();
  document.getElementById('sz').oninput=e=>mat.size=+e.target.value;
  document.getElementById('br').oninput=e=>{const f=e.target.value/100;
    mat.color.setScalar(Math.min(f,2));};
  window._pts=pts; fit();
});
function fit(){
  const p=window._pts; if(!p) return;
  const span=[65534*M.scale[0],65534*M.scale[1],65534*M.scale[2]];
  const c=new THREE.Vector3(M.lo[0]+span[0]/2,M.lo[1]+span[1]/2,M.lo[2]+span[2]/2);
  const r=Math.max(...span)*0.75;
  camera.position.set(c.x+r,c.y+r*0.6,c.z+r);
  camera.near=r/500; camera.far=r*30; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}
document.getElementById('fit').onclick=fit;
document.getElementById('top').onclick=()=>{
  const span=[65534*M.scale[0],65534*M.scale[1],65534*M.scale[2]];
  const c=new THREE.Vector3(M.lo[0]+span[0]/2,M.lo[1]+span[1]/2,M.lo[2]+span[2]/2);
  camera.position.set(c.x,c.y+Math.max(span[0],span[2])*1.2,c.z+0.01);
  controls.target.copy(c); controls.update();
};
function resize(){const w=view.clientWidth,h=view.clientHeight;
  renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);})();
</script>
"""
open("output/full_cloud.html", "w").write(HTML.replace("__META__", json.dumps(M)))
print(f"wrote output/full_cloud.html")
print(f"  streams full_pos.bin ({os.path.getsize('output/full_pos.bin')/1e6:.0f} MB) "
      f"+ full_col.bin ({os.path.getsize('output/full_col.bin')/1e6:.0f} MB)")
print(f"  {M['n']:,} points, no thinning")
