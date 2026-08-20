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

    html = f"""<!doctype html><meta charset=utf-8>
<title>{OUT.name}</title>
<style>
 body{{margin:0;background:#0f1115;color:#dfe3ea;font:13px/1.55 ui-sans-serif,system-ui,sans-serif}}
 #c{{position:fixed;inset:0}}
 #p{{position:fixed;top:0;right:0;width:340px;max-height:100vh;overflow:auto;
     background:#161a21f2;padding:14px 16px;border-left:1px solid #2a303a}}
 h1{{font-size:15px;margin:0 0 2px}} h2{{font-size:11px;margin:14px 0 4px;color:#8e98a8;
     text-transform:uppercase;letter-spacing:.09em}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
 label{{display:block;padding:2px 0;cursor:pointer}}
 button{{background:#232a35;color:#dfe3ea;border:1px solid #38404d;border-radius:4px;
        padding:4px 9px;margin:2px 3px 2px 0;cursor:pointer;font:inherit}}
 button.on{{background:#3a5ea8;border-color:#4f79c9}}
 table{{width:100%;border-collapse:collapse;font-size:12px}}
 td{{padding:1px 4px 1px 0;border-bottom:1px solid #222831}}
 td.n{{text-align:right;color:#9fb0c8;font-variant-numeric:tabular-nums}}
 .note{{color:#7c8695;font-size:11px}} #st{{color:#c8b56a}}
</style>
<canvas id=c></canvas>
<div id=p>
 <h1>{OUT.name}</h1>
 <div class=note>the scanned surface itself, cropped into named parts</div>
 <h2>Colour</h2>
 <div id=modes>
  <button data-m=part class=on>by part</button>
  <button data-m=kind>by kind</button>
  {'<button data-m=scan>as scanned</button>' if rgb is not None else ''}
 </div>
 <div class=note>wheel zooms to the cursor &middot; double-click to orbit that
 point &middot; right-drag pans &middot; W A S D moves &middot; F reframes</div>
 <h2>Layer</h2>
 <div>
  <button id=lmodel class=on>model parts</button>
  <button id=lraw>the Poisson mesh</button>
 </div>
 <h2>Resolution</h2>
 <div><button id=hi>load full resolution</button> <span id=st></span></div>
 <h2>Show</h2>
 <div id=toggles></div>
 <h2>Measured</h2>
 <table>
  <tr><td>clear height</td><td class=n>{man['clear_height_mm']:.0f} mm</td></tr>
  <tr><td>parts</td><td class=n>{man['n_parts']}</td></tr>
  <tr><td>surface in a named part</td><td class=n>{cov['kept_frac']*100:.1f}%</td></tr>
  <tr><td>gap between parts</td><td class=n>{cov['crack_tris']} tri</td></tr>
  {f"<tr><td>masonry</td><td class=n>{sol.get('masonry_volume_m3',0):.1f} m&sup3;</td></tr>" if sol else ""}
  {f"<tr><td>thicknesses repeated</td><td class=n>{' / '.join(str(int(t)) for t in man.get('thickness_modes_mm',[]))} mm</td></tr>" if man.get('thickness_modes_mm') else ""}
 </table>
 <h2>Parts</h2>
 <table>{''.join(f'<tr><td>{k}</td><td class=n>{v}</td></tr>' for k, v in sorted(nk.items()))}</table>
 <h2>Relief and openings</h2>
 <table>{''.join(f'<tr><td>{k}</td><td class=n>{v}</td></tr>' for k, v in sorted(fk.items()))}</table>
 <h2>Selected</h2>
 <div id=sel class=note>click a part</div>
</div>
<script type="importmap">
{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';
const PARTS = {json.dumps(parts)};
const KC = {json.dumps(KIND_COLOR)};
const FEATS = {json.dumps(man['features'])};
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1115);
const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, .02, 2000);
const rend = new THREE.WebGLRenderer({{canvas:document.getElementById('c'), antialias:true}});
rend.setSize(innerWidth, innerHeight); rend.setPixelRatio(Math.min(devicePixelRatio,2));
const ctl = new OrbitControls(cam, rend.domElement);
// Zoom used to dolly toward the middle of the building and stop dead there, so
// you could not get into a room: the target is the centre and OrbitControls
// will not travel past it. Zooming to the cursor instead means the wheel goes
// where you are pointing, and the step scales with how far away you are, so it
// is not glacial across the plan and violent up against a wall.
ctl.zoomToCursor = true;
ctl.enableDamping = true;
ctl.dampingFactor = 0.08;
ctl.screenSpacePanning = true;
ctl.minDistance = 0.05;
ctl.maxDistance = 400;
ctl.zoomSpeed = 1.1;
ctl.panSpeed = 0.9;
ctl.keys = {{LEFT:'KeyA', UP:'KeyW', RIGHT:'KeyD', BOTTOM:'KeyS'}};
ctl.listenToKeyEvents(window);
scene.add(new THREE.HemisphereLight(0xffffff, 0x555f70, 2.4));
scene.add(new THREE.AmbientLight(0xffffff, .35));
const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(6,-9,12); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0xffffff, .7); dl2.position.set(-8,6,4); scene.add(dl2);
const dl3 = new THREE.DirectionalLight(0xffffff, .5); dl3.position.set(0,0,-10); scene.add(dl3);

let groups = {{}}, root = null, mode = 'part', framed = false;
const rnd = (s)=>{{let x=Math.sin(s*127.1)*43758.5453; return x-Math.floor(x);}};

function paint() {{
  Object.values(groups).flat().forEach((o,i)=>{{
    const k = o.userData.kind, m = o.material;
    if (mode === 'scan') {{ m.vertexColors = true; m.color.setHex(0xffffff); }}
    else {{
      m.vertexColors = false;
      m.color.set(mode === 'kind' ? (KC[k]||'#8a7fd0') : o.userData.tint);
    }}
    m.needsUpdate = true;
  }});
}}

function install(g) {{
  if (root) scene.remove(root);
  groups = {{}}; root = g.scene; scene.add(root);
  let n = 0;
  root.traverse(o => {{
    if (!o.isMesh) return;
    // A mesh with no NORMAL attribute is lit as black whatever the material
    // says, which is how the first build came out: every triangle present and
    // none of it visible. Exports carry normals now; this catches the rest.
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    const p = PARTS[o.name] || PARTS[o.name.replace(/_\\d+$/, '')];
    const kind = p ? p.kind : 'other';
    o.userData.part = p; o.userData.kind = kind;
    o.userData.tint = new THREE.Color().setHSL(rnd(n*3+1), .45+.3*rnd(n*7+2), .45+.2*rnd(n*5+3));
    o.material = new THREE.MeshStandardMaterial({{roughness:.93, metalness:0,
      side:THREE.DoubleSide, vertexColors:false}});
    (groups[kind] = groups[kind] || []).push(o); n++;
  }});
  paint();
  if (!framed) {{
    const box = new THREE.Box3().setFromObject(root);
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    cam.position.set(c.x + s.x*.9, c.y - s.y*1.1, c.z + s.z*2.2);
    ctl.target.copy(c); ctl.update(); framed = true;
  }}
  const t = document.getElementById('toggles'); t.innerHTML = '';
  Object.keys(groups).sort().forEach(k => {{
    const l = document.createElement('label');
    l.innerHTML = `<input type=checkbox checked><span class=sw style="background:${{KC[k]||'#8a7fd0'}}"></span>${{k}} (${{groups[k].length}})`;
    l.querySelector('input').onchange = e => groups[k].forEach(o => o.visible = e.target.checked);
    t.appendChild(l);
  }});
}}

const loader = new GLTFLoader();
const st = document.getElementById('st');
function load(url, note) {{
  st.textContent = 'loading ' + note + '…';
  loader.load(url, g => {{ install(g); st.textContent = note + ' loaded'; }},
    x => {{ if (x.total) st.textContent = note + ' ' + Math.round(100*x.loaded/x.total) + '%'; }},
    e => {{ st.textContent = 'could not load ' + url +
      ' — open this page through the local server, not from the file system'; }});
}}
load('modular_view.glb', 'working resolution');
document.getElementById('hi').onclick = () => load(layer === 'raw' ?
  'scan_view.glb' : 'modular_full.glb', 'full resolution');
let layer = 'model';
function setLayer(k, url, note) {{
  layer = k;
  document.getElementById('lmodel').classList.toggle('on', k === 'model');
  document.getElementById('lraw').classList.toggle('on', k === 'raw');
  load(url, note);
}}
document.getElementById('lmodel').onclick = () => setLayer('model', 'modular_view.glb', 'model parts');
document.getElementById('lraw').onclick = () => setLayer('raw', 'scan_view.glb', 'the Poisson mesh');
document.querySelectorAll('#modes button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('#modes button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); mode = b.dataset.m; paint();
}});

const ray = new THREE.Raycaster(), mv = new THREE.Vector2();
addEventListener('click', e => {{
  if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
  mv.x = e.clientX/innerWidth*2-1; mv.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mv, cam);
  const h = ray.intersectObjects(scene.children, true).filter(i => i.object.visible)[0];
  const d = document.getElementById('sel');
  if (!h) {{ d.textContent = 'click a part'; return; }}
  const nm = h.object.name, p = h.object.userData.part;
  const fs = FEATS.filter(f => f.wall === nm);
  d.innerHTML = `<b>${{nm}}</b><table>` +
    (p ? Object.entries(p).filter(([k]) => k !== 'name').map(([k,v]) =>
      `<tr><td>${{k}}</td><td class=n>${{Array.isArray(v)?v.join(' , '):v}}</td></tr>`).join('') : '') +
    fs.map(f => `<tr><td>${{f.kind}}</td><td class=n>${{f.width_mm||''}}&times;${{f.height_mm||''}} mm</td></tr>`).join('') +
    `</table>`;
}});
addEventListener('resize', () => {{
  cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
  rend.setSize(innerWidth, innerHeight);
}});
// double-click puts the orbit centre on what you clicked, so the next zoom
// goes into that room rather than back to the middle of the building
addEventListener('dblclick', e => {{
  mv.x = e.clientX/innerWidth*2-1; mv.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mv, cam);
  const h = ray.intersectObjects(scene.children, true).filter(i => i.object.visible)[0];
  if (!h) return;
  ctl.target.copy(h.point);
  ctl.update();
  document.getElementById('st').textContent =
    'orbiting ' + (h.object.name || 'that point');
}});
addEventListener('keydown', e => {{
  if (e.code === 'KeyF' && root) {{                 // F = frame the whole model
    const box = new THREE.Box3().setFromObject(root);
    const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    cam.position.set(c.x + sz.x*.9, c.y - sz.y*1.1, c.z + sz.z*2.2);
    ctl.target.copy(c); ctl.update();
  }}
}});
(function loop() {{ requestAnimationFrame(loop); ctl.update(); rend.render(scene, cam); }})();
</script>
"""
    (OUT/"viewer.html").write_text(html, encoding="utf-8")
    log(f"wrote {OUT/'viewer.html'} ({(OUT/'viewer.html').stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
