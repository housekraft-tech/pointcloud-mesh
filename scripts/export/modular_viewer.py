"""A viewer for the Poisson-segmented model, at a resolution that shows relief.

The first version inlined a 900k-triangle decimation as base64, because a page
has to load. On a 14 million triangle model that is 6% of the surface, and it
threw away exactly what the model exists for: the niches, the arch soffits, the
boxed conduits. The relief was in the file and not on the screen.

So the geometry is no longer inlined. The page fetches a GLB beside it, which
means it must be opened over the local server rather than off the disk, and in
exchange it can carry millions of triangles instead of hundreds of thousands.
Two are written:

  modular_view.glb    decimated to --target triangles (default 3 M), the one
                      the page loads first
  modular_full.glb    every triangle of the model, loaded on demand by the
                      "full resolution" button

Colour has two modes off one file. COLOR_0 carries the scan's own colour where
the scan has it (bake_rgb.py), so the page can show the flat as it looks; the
part palette is applied as a flat material colour instead, so the same geometry
reads as a modular model. Neither mode reloads anything.
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
import open3d as o3d

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

TEMPLATE = r"""<!doctype html><meta charset=utf-8>
<title>@@TITLE@@</title>
<style>
 body{margin:0;background:#0f1115;color:#dfe3ea;font:13px/1.55 ui-sans-serif,system-ui,sans-serif}
 #c{position:fixed;inset:0}
 #p{position:fixed;top:0;right:0;width:352px;max-height:100vh;overflow:auto;
     background:#161a21f2;padding:14px 16px;border-left:1px solid #2a303a}
 h1{font-size:15px;margin:0 0 2px} h2{font-size:11px;margin:14px 0 4px;color:#8e98a8;
     text-transform:uppercase;letter-spacing:.09em}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
 label{display:block;padding:2px 0;cursor:pointer}
 button{background:#232a35;color:#dfe3ea;border:1px solid #38404d;border-radius:4px;
        padding:4px 9px;margin:2px 3px 2px 0;cursor:pointer;font:inherit}
 button.on{background:#3a5ea8;border-color:#4f79c9}
 button:disabled{opacity:.4;cursor:default}
 table{width:100%;border-collapse:collapse;font-size:12px}
 td{padding:1px 4px 1px 0;border-bottom:1px solid #222831}
 td.n{text-align:right;color:#9fb0c8;font-variant-numeric:tabular-nums}
 .note{color:#7c8695;font-size:11px} #st,#wst{color:#c8b56a}
 input[type=range]{width:118px;vertical-align:middle}
 .lay{border:1px solid #2a303a;border-radius:5px;padding:6px 8px;margin:5px 0}
 .lay b{font-weight:600}
</style>
<canvas id=c></canvas>
<div id=p>
 <h1>@@TITLE@@</h1>
 <div class=note>the scanned surface itself, cropped into named parts</div>

 <h2>Layers &mdash; stack them to see the overlap</h2>
 <div class=lay>
  <label><input type=checkbox id=Lmodel> <b>modular</b> &mdash; the named parts <span class=note>(heavy: it will hitch the tab for a few seconds)</span></label>
  <div class=note>opacity <input type=range id=Omodel min=10 max=100 value=100></div>
 </div>
 <div class=lay>
  <label><input type=checkbox id=Lscan> <b>scan</b> &mdash; the raw Poisson surface</label>
  <div class=note>opacity <input type=range id=Oscan min=10 max=100 value=55>
   <span id=Sscan></span></div>
 </div>
 <div class=lay>
  <label><input type=checkbox id=Rmodel> <b>no ceiling</b> on the modular layer</label>
 </div>
 <div class=lay>
  <label><input type=checkbox id=Lboxes checked> <b>boxes</b> &mdash; the clean model</label>
  <div class=note>opacity <input type=range id=Oboxes min=10 max=100 value=100>
   &nbsp;<label style="display:inline"><input type=checkbox id=Wboxes> wireframe</label>
   &nbsp;<label style="display:inline"><input type=checkbox id=Rboxes> no ceiling</label></div>
 </div>

 <h2>Colour of the modular layer</h2>
 <div id=modes>
  <button data-m=part class=on>by part</button>
  <button data-m=kind>by kind</button>
  @@SCANBTN@@
 </div>

 <h2>Move</h2>
 <div>
  <button id=morbit class=on>orbit</button>
  <button id=mwalk disabled>walk through</button>
  <span id=wst class=note></span>
 </div>
 <div class=note>wheel zooms toward the cursor, 12% of the way to it a notch &middot; double-click
  to orbit that point &middot; right-drag pans &middot; F reframes</div>

 <h2>Resolution</h2>
 <div><button id=hi>load full resolution</button> <span id=st></span></div>

 <h2>Show parts</h2>
 <div id=toggles></div>

 <h2>Measured</h2>
 <table>
  <tr><td>clear height</td><td class=n>@@CLEAR@@ mm</td></tr>
  <tr><td>parts</td><td class=n>@@NPARTS@@</td></tr>
  <tr><td>surface in a named part</td><td class=n>@@KEPT@@%</td></tr>
  <tr><td>gap between parts</td><td class=n>@@CRACKS@@ tri</td></tr>
  @@MASONRY@@
  @@THICK@@
 </table>
 <h2>Parts</h2>
 <table>@@KINDS@@</table>
 <h2>Relief and openings</h2>
 <table>@@FEATS@@</table>
 <h2>Selected</h2>
 <div id=sel class=note>click a part</div>
</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {Octree} from 'three/addons/math/Octree.js';
import {Capsule} from 'three/addons/math/Capsule.js';
const PARTS = @@JPARTS@@;
const KC = @@JKC@@;
const FEATS = @@JFEATS@@;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1115);
const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, .02, 2000);
const rend = new THREE.WebGLRenderer({canvas:document.getElementById('c'), antialias:true});
rend.setSize(innerWidth, innerHeight); rend.setPixelRatio(Math.min(devicePixelRatio,2));
const ctl = new OrbitControls(cam, rend.domElement);
ctl.enableZoom = false;              // replaced below with a fixed step per notch
ctl.enableDamping = true; ctl.dampingFactor = 0.10;
ctl.screenSpacePanning = true; ctl.minDistance = 0.05; ctl.maxDistance = 400;
ctl.panSpeed = 0.9;
scene.add(new THREE.HemisphereLight(0xffffff, 0x555f70, 2.4));
scene.add(new THREE.AmbientLight(0xffffff, .35));
const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(6,-9,12); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0xffffff, .7); dl2.position.set(-8,6,4); scene.add(dl2);
const dl3 = new THREE.DirectionalLight(0xffffff, .5); dl3.position.set(0,0,-10); scene.add(dl3);

// Three layers of the same building, stacked rather than swapped: the named
// parts, the raw Poisson surface they were cut from, and the clean boxes fitted
// to them. Seeing where they disagree is the point, so each has its own opacity
// and they share one coordinate frame.
const LAYERS = {
  model: {url:'modular_view.glb', hi:'modular_full.glb', root:null, op:1.00},
  scan:  {url:'scan_view.glb',    hi:'scan_view.glb',    root:null, op:0.55},
  boxes: {url:'boxes.glb',        hi:'boxes.glb',        root:null, op:1.00},
};
let groups = {}, mode = 'part', framed = false, root = null;
const loader = new GLTFLoader();
const rnd = s => { const x = Math.sin(s*127.1)*43758.5453; return x-Math.floor(x); };

function frame() {
  const r0 = LAYERS.model.root || LAYERS.scan.root || LAYERS.boxes.root;
  if (!r0) return;
  const box = new THREE.Box3().setFromObject(r0);
  const c = box.getCenter(new THREE.Vector3());
  const r = box.getBoundingSphere(new THREE.Sphere()).radius;
  cam.position.set(c.x + r*1.15, c.y + r*0.85, c.z + r*1.15);
  ctl.target.copy(c); ctl.update();
}

function style(key) {
  const L = LAYERS[key]; if (!L.root) return;
  const wire = key === 'boxes' && document.getElementById('Wboxes').checked;
  L.root.traverse(o => {
    if (!o.isMesh) return;
    const m = o.material;
    m.transparent = L.op < 0.999;
    m.opacity = L.op;
    m.depthWrite = L.op > 0.98;
    m.wireframe = wire;
    if (key === 'model') {
      if (mode === 'scan') { m.vertexColors = true; m.color.setHex(0xffffff); }
      else { m.vertexColors = false;
             m.color.set(mode === 'kind' ? (KC[o.userData.kind]||'#8a7fd0') : o.userData.tint); }
    }
    m.needsUpdate = true;
  });
}

function install(key, g) {
  const L = LAYERS[key];
  if (L.root) scene.remove(L.root);
  L.root = g.scene;
  L.root.rotation.x = -Math.PI/2;    // our world is Z-up; glTF and three are Y-up
  scene.add(L.root);
  if (key === 'model') {
    root = L.root; groups = {};
    let n = 0;
    L.root.traverse(o => {
      if (!o.isMesh) return;
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      const p = PARTS[o.name] || PARTS[o.name.replace(/_\d+$/, '')];
      o.userData.part = p; o.userData.kind = p ? p.kind : 'other';
      o.userData.tint = new THREE.Color().setHSL(rnd(n*3+1), .45+.3*rnd(n*7+2), .45+.2*rnd(n*5+3));
      o.material = new THREE.MeshStandardMaterial({roughness:.93, metalness:0,
        side:THREE.DoubleSide});
      (groups[o.userData.kind] = groups[o.userData.kind] || []).push(o); n++;
    });
    const t = document.getElementById('toggles'); t.innerHTML = '';
    Object.keys(groups).sort().forEach(k => {
      const l = document.createElement('label');
      l.innerHTML = `<input type=checkbox checked><span class=sw style="background:${KC[k]||'#8a7fd0'}"></span>${k} (${groups[k].length})`;
      l.querySelector('input').onchange = e => groups[k].forEach(o => o.visible = e.target.checked);
      t.appendChild(l);
    });
  } else {
    L.root.traverse(o => {
      if (!o.isMesh) return;
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      o.material = new THREE.MeshStandardMaterial({
        roughness:.95, metalness:0, side:THREE.DoubleSide,
        color: key === 'boxes' ? new THREE.Color('#8fd6a0') : new THREE.Color('#c9c2b4'),
        vertexColors: !!o.geometry.attributes.color});
    });
  }
  style(key);
  if (!framed) { frame(); framed = true; }
}

const st = document.getElementById('st');
function loadLayer(key, url, note) {
  st.textContent = 'loading ' + note + '…';
  loader.load(url, g => { install(key, g); st.textContent = note + ' loaded'; },
    x => { if (x.total) st.textContent = note + ' ' + Math.round(100*x.loaded/x.total) + '%'; },
    () => { st.textContent = 'could not load ' + url +
      ' — open this page through the local server, not from the file system';
      const cb = document.getElementById('L' + key); if (cb) cb.checked = false; });
}
// The box model is 400 kB and parses instantly; the modular layer is 3 M
// triangles across 80-odd nodes and takes the browser tens of seconds to build,
// during which the canvas is black and the page looks hung. So the boxes are
// shown first and the modular layer arrives behind them.
document.getElementById('Lboxes').checked = true;
loadLayer('boxes', LAYERS.boxes.url, 'the box model');
setTimeout(() => loadLayer('model', LAYERS.model.url, 'the modular parts'), 60);

for (const key of ['model','scan','boxes']) {
  document.getElementById('L'+key).onchange = e => {
    const L = LAYERS[key];
    if (e.target.checked) {
      if (!L.root) loadLayer(key, L.url, key === 'scan' ? 'the Poisson surface'
                                        : key === 'boxes' ? 'the box model' : 'the modular parts');
      else L.root.visible = true;
    } else if (L.root) L.root.visible = false;
  };
  document.getElementById('O'+key).oninput = e => {
    LAYERS[key].op = e.target.value/100; style(key);
  };
}
const OVERHEAD = ['ceiling', 'dropped_ceiling', 'beam'];
function roof(key) {{
  const L = LAYERS[key], off = document.getElementById('R'+key).checked;
  if (!L.root) return;
  L.root.traverse(o => {{
    if (o.isMesh && OVERHEAD.includes(o.userData.kind)) o.visible = !off;
  }});
}}
document.getElementById('Rboxes').onchange = () => roof('boxes');
document.getElementById('Rmodel').onchange = () => roof('model');
document.getElementById('Wboxes').onchange = () => style('boxes');
document.getElementById('hi').onclick = () => loadLayer('model', LAYERS.model.hi, 'full resolution');
document.querySelectorAll('#modes button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#modes button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); mode = b.dataset.m; style('model');
});

// zoom: a fixed 8% of the distance per notch, toward the cursor, clamped, so it
// behaves the same on every mouse. The target travels too, so the orbit centre
// follows you into a room instead of staying in the middle of the building.
const ray = new THREE.Raycaster(), mv = new THREE.Vector2();
const _f = new THREE.Vector3(), _v = new THREE.Vector3();
rend.domElement.addEventListener('wheel', e => {{
  e.preventDefault();
  mv.x = e.clientX/innerWidth*2-1; mv.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mv, cam);
  // measured against the surface under the cursor, not the orbit radius: a
  // dolly moves camera and target together, so the radius never shrinks and a
  // percentage-of-radius step stays a metre a notch right up to the wall
  const hit = ray.intersectObjects(scene.children, true)
                 .filter(i => i.object.visible)[0];
  const reach = hit ? hit.distance : cam.position.distanceTo(ctl.target);
  if (e.deltaY < 0 && reach < 0.25) return;
  const f = (e.deltaY < 0 ? 1 : -1) * 0.12;
  _v.copy(ray.ray.direction).multiplyScalar(reach*f);
  cam.position.add(_v); ctl.target.add(_v); ctl.update();
}}, {{passive:false}});

addEventListener('dblclick', e => {
  mv.x = e.clientX/innerWidth*2-1; mv.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mv, cam);
  const h = ray.intersectObjects(scene.children, true).filter(i => i.object.visible)[0];
  if (h) { ctl.target.copy(h.point); ctl.update(); }
});
addEventListener('click', e => {
  if (walk || e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
  mv.x = e.clientX/innerWidth*2-1; mv.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mv, cam);
  const h = ray.intersectObjects(scene.children, true).filter(i => i.object.visible)[0];
  const d = document.getElementById('sel');
  if (!h) { d.textContent = 'click a part'; return; }
  const nm = h.object.name, p = h.object.userData.part;
  const fs = FEATS.filter(f => f.wall === nm);
  d.innerHTML = `<b>${nm}</b><table>` +
    (p ? Object.entries(p).filter(([k]) => k !== 'name').map(([k,v]) =>
      `<tr><td>${k}</td><td class=n>${Array.isArray(v)?v.join(' , '):v}</td></tr>`).join('') : '') +
    fs.map(f => `<tr><td>${f.kind}</td><td class=n>${f.width_mm||''}&times;${f.height_mm||''} mm</td></tr>`).join('') +
    `</table>`;
});
addEventListener('resize', () => {
  cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
  rend.setSize(innerWidth, innerHeight);
});

// ---- walking through it, colliding with the boxes -----------------------
let walk = false, octree = null, onFloor = false, jumped = 0;
const vel = new THREE.Vector3(), dir = new THREE.Vector3(), keys = {};
const EYE = 1.62, RADIUS = 0.32, SPEED = 3.2, GRAVITY = 22;
const player = new Capsule(new THREE.Vector3(0,0.35,0), new THREE.Vector3(0,EYE,0), RADIUS);
const wst = document.getElementById('wst');
// The collision hull is built on demand. Fetching and octree-ing a mesh on
// page load costs the tab whether or not anyone ever walks, and an octree is
// built on the main thread.
let colliding = false;
document.getElementById('mwalk').disabled = false;
function collision(then) {
  if (octree) { then(); return; }
  if (colliding) return;
  colliding = true; wst.textContent = 'building the collision hull...';
  new GLTFLoader().load('boxes.glb', g => {
    g.scene.rotation.x = -Math.PI/2; g.scene.updateMatrixWorld(true);
    octree = new Octree().fromGraphNode(g.scene);
    wst.textContent = 'ready'; colliding = false; then();
  }, undefined, () => { wst.textContent = 'no boxes.glb to collide with';
                        colliding = false; });
}
function spawn() {
  const r0 = LAYERS.model.root || LAYERS.boxes.root; if (!r0) return;
  const box = new THREE.Box3().setFromObject(r0);
  const c = box.getCenter(new THREE.Vector3()), y = box.min.y + EYE + 0.05;
  player.start.set(c.x, y-EYE+0.35, c.z); player.end.set(c.x, y, c.z);
  cam.position.copy(player.end);
}
function step(dt) {
  if (!onFloor) vel.y -= GRAVITY*dt;
  const damp = Math.exp(-(onFloor ? 12 : 1.5)*dt) - 1;
  vel.x += vel.x*damp; vel.z += vel.z*damp; vel.y += vel.y*damp*0.2;
  const f = new THREE.Vector3(); cam.getWorldDirection(f); f.y = 0; f.normalize();
  const r = new THREE.Vector3().crossVectors(f, cam.up).normalize();
  dir.set(0,0,0);
  if (keys['KeyW']) dir.add(f);
  if (keys['KeyS']) dir.sub(f);
  if (keys['KeyD']) dir.add(r);
  if (keys['KeyA']) dir.sub(r);
  if (dir.lengthSq() > 0) {
    dir.normalize().multiplyScalar(SPEED*(keys['ShiftLeft'] ? 2.2 : 1));
    vel.x = dir.x; vel.z = dir.z;
  }
  if (keys['Space'] && onFloor && performance.now()-jumped > 400) { vel.y = 6.5; jumped = performance.now(); }
  player.translate(vel.clone().multiplyScalar(dt));
  onFloor = false;
  if (octree) {
    const hit = octree.capsuleIntersect(player);
    if (hit) { onFloor = hit.normal.y > 0.3; player.translate(hit.normal.multiplyScalar(hit.depth)); }
  }
  if (player.end.y < -40) spawn();
  cam.position.copy(player.end);
}
addEventListener('keydown', e => {
  if (e.code === 'KeyF' && !walk) frame();
  if (walk) keys[e.code] = true;
});
addEventListener('keyup', e => { keys[e.code] = false; });
rend.domElement.addEventListener('mousemove', e => {
  if (!walk || document.pointerLockElement !== rend.domElement) return;
  cam.rotation.order = 'YXZ';
  cam.rotation.y -= e.movementX/700;
  cam.rotation.x = Math.max(-1.5, Math.min(1.5, cam.rotation.x - e.movementY/700));
});
function setWalk(on) {
  walk = on; ctl.enabled = !on;
  document.getElementById('mwalk').classList.toggle('on', on);
  document.getElementById('morbit').classList.toggle('on', !on);
  if (on) { spawn(); rend.domElement.requestPointerLock();
    wst.textContent = 'W A S D, mouse looks, Space jumps, Shift runs, Esc leaves'; }
  else { document.exitPointerLock(); wst.textContent = 'ready'; frame(); }
}
document.getElementById('mwalk').onclick = () => collision(() => setWalk(true));
document.getElementById('morbit').onclick = () => setWalk(false);
addEventListener('pointerlockchange', () => {
  if (walk && document.pointerLockElement !== rend.domElement) setWalk(false);
});

let prev = performance.now();
(function loop() {
  requestAnimationFrame(loop);
  const now = performance.now(), dt = Math.min((now-prev)/1000, 0.05); prev = now;
  if (walk) { for (let i = 0; i < 4; i++) step(dt/4); } else { ctl.update(); }
  rend.render(scene, cam);
})();
</script>
"""

KIND_COLOR = {"wall": "#5b7fd4", "parapet": "#4aa6c8", "floor": "#b9b2a6",
              "ceiling": "#68c07a", "beam": "#e08a2b", "dropped_ceiling": "#d8c33a",
              "column": "#cf4b4b"}


def build_glb(path, T, V, label, names, rgb, target):
    """One node per part, scan colour in COLOR_0, decimated to `target`."""
    import trimesh
    total = int((label >= 0).sum())
    ratio = 1.0 if target is None else min(1.0, target/total)
    sc = trimesh.Scene()
    kept = 0
    for pid, nm in enumerate(names):
        sel = np.where(label == pid)[0]
        if sel.size == 0:
            continue
        t = T[sel]
        used = np.unique(t)
        rm = np.full(len(V), -1, np.int64); rm[used] = np.arange(len(used))
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(V[used])
        m.triangles = o3d.utility.Vector3iVector(rm[t])
        if rgb is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(rgb[used])
        want = max(60, int(len(t)*ratio))
        if want < len(t):
            m = m.simplify_quadric_decimation(want)
        m.compute_vertex_normals()
        vv = np.asarray(m.vertices); tt = np.asarray(m.triangles)
        if len(tt) == 0:
            continue
        kept += len(tt)
        # glTF without NORMAL renders black under any lit material, and that is
        # exactly how the first build came out: the geometry was all there and
        # none of it could be seen.
        tm = trimesh.Trimesh(vv, tt, vertex_normals=np.asarray(m.vertex_normals),
                             process=False)
        if rgb is not None:
            cc = np.asarray(m.vertex_colors)
            col = np.zeros((len(vv), 4), np.uint8)
            col[:, :3] = np.clip(cc*255, 0, 255).astype(np.uint8); col[:, 3] = 255
            tm.visual.vertex_colors = col
        sc.add_geometry(tm, geom_name=nm, node_name=nm)
    sc.export(str(path))
    log(f"wrote {path.name}: {kept:,} triangles, "
        f"{path.stat().st_size/1e6:.0f} MB")
    return kept


def build_raw(path, T, V, rgb, target):
    """The Poisson mesh as it came out, before anything was named or dropped.

    Worth having beside the model: it is the ground the model is cut from, so
    anything the segmentation lost can be seen against it directly.
    """
    import trimesh
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(T)
    if rgb is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(rgb)
    if target and target < len(T):
        m = m.simplify_quadric_decimation(target)
    m.compute_vertex_normals()
    vv = np.asarray(m.vertices); tt = np.asarray(m.triangles)
    tm = trimesh.Trimesh(vv, tt, vertex_normals=np.asarray(m.vertex_normals),
                         process=False)
    if rgb is not None:
        cc = np.asarray(m.vertex_colors)
        col = np.zeros((len(vv), 4), np.uint8)
        col[:, :3] = np.clip(cc*255, 0, 255).astype(np.uint8); col[:, 3] = 255
        tm.visual.vertex_colors = col
    sc = trimesh.Scene()
    sc.add_geometry(tm, geom_name="poisson_mesh", node_name="poisson_mesh")
    sc.export(str(path))
    log(f"wrote {path.name}: {len(tt):,} triangles of raw mesh, "
        f"{path.stat().st_size/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="output/model/poisson_modular")
    ap.add_argument("cache", nargs="?", default="output/model/poisson_koushik.npz")
    ap.add_argument("--target", type=int, default=3_000_000,
                    help="triangles in the file the page loads first")
    ap.add_argument("--full", action="store_true",
                    help="also write every triangle, for the full-resolution button")
    ap.add_argument("--raw", action="store_true",
                    help="also write the unsegmented Poisson mesh as a layer")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out)

    man = json.load(open(OUT/"manifest.json"))
    T = np.load(a.cache)["T"].astype(np.int64)
    V = np.load(OUT/"verts.npy").astype(np.float64)
    label = np.load(OUT/"labels.npy")
    names = json.load(open(OUT/"names.json"))
    # the mesh's own colour first -- Poisson interpolated it from the scan, so
    # it needs no lookup and has no nearest-neighbour error at all
    cd = np.load(a.cache)
    rgb = None
    if "RGB" in cd.files:
        rgb = cd["RGB"].astype(np.float64)/255.0
        log("colour: the Poisson mesh's own, interpolated from the scan")
    elif (OUT/"vertex_rgb.npy").exists():
        rgb = np.load(OUT/"vertex_rgb.npy").astype(np.float64)/255.0
        log("colour: baked onto the model from the nearest scanned point")
    log(f"{int((label>=0).sum()):,} triangles in {man['n_parts']} parts"
        + ("; scan colour present" if rgb is not None else "; no scan colour"))

    view = OUT/"modular_view.glb"
    if a.force or not view.exists():
        build_glb(view, T, V, label, names, rgb, a.target)
    full = OUT/"modular_full.glb"
    if a.full and (a.force or not full.exists()):
        build_glb(full, T, V, label, names, rgb, None)
    raw = OUT/"scan_view.glb"
    if a.raw and (a.force or not raw.exists()):
        build_raw(raw, T, V, rgb, a.target)

    parts = {p["name"]: p for p in man["parts"]}
    cov = man["coverage"]
    nk, fk = {}, {}
    for p in man["parts"]:
        nk[p["kind"]] = nk.get(p["kind"], 0) + 1
    for f in man["features"]:
        fk[f["kind"]] = fk.get(f["kind"], 0) + 1
    sol = man.get("solid", {})

    kinds_rows = "".join(f"<tr><td>{k}</td><td class=n>{v}</td></tr>"
                         for k, v in sorted(nk.items()))
    feat_rows = "".join(f"<tr><td>{k}</td><td class=n>{v}</td></tr>"
                        for k, v in sorted(fk.items()))
    masonry = (f"<tr><td>masonry</td><td class=n>"
               f"{sol.get('masonry_volume_m3', 0):.1f} m&sup3;</td></tr>") if sol else ""
    thick = (f"<tr><td>thicknesses repeated</td><td class=n>"
             f"{' / '.join(str(int(t)) for t in man.get('thickness_modes_mm', []))}"
             f" mm</td></tr>") if man.get("thickness_modes_mm") else ""

    # A plain template with named placeholders, not an f-string: the page is
    # mostly JavaScript, and every brace in it would otherwise have to be
    # doubled -- which is exactly how the interpolations got eaten last time.
    html = TEMPLATE
    for key, val in (("@@TITLE@@", OUT.name),
                     ("@@SCANBTN@@", '<button data-m=scan>as scanned</button>'
                                     if rgb is not None else ''),
                     ("@@CLEAR@@", f"{man['clear_height_mm']:.0f}"),
                     ("@@NPARTS@@", str(man["n_parts"])),
                     ("@@KEPT@@", f"{cov['kept_frac']*100:.1f}"),
                     ("@@CRACKS@@", str(cov["crack_tris"])),
                     ("@@MASONRY@@", masonry),
                     ("@@THICK@@", thick),
                     ("@@KINDS@@", kinds_rows),
                     ("@@FEATS@@", feat_rows),
                     ("@@JPARTS@@", json.dumps(parts)),
                     ("@@JKC@@", json.dumps(KIND_COLOR)),
                     ("@@JFEATS@@", json.dumps(man["features"]))):
        html = html.replace(key, val)
    (OUT/"viewer.html").write_text(html, encoding="utf-8")
    log(f"wrote {OUT/'viewer.html'} ({(OUT/'viewer.html').stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
