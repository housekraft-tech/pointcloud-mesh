"""A viewer for the Poisson-segmented modular model.

The model itself is 8 million triangles, which no browser will open, so the
viewer carries a decimated copy: enough to read every niche, arch and beam,
about a tenth of the file. The full-resolution model stays in modular.obj /
modular.glb for Blender.

Parts keep their names and their kinds, so the viewer can switch off the
ceiling and look inside, and clicking a part reads back what was measured for
it rather than a colour.
"""
import sys, os, json, base64, time
from pathlib import Path
import numpy as np
import open3d as o3d

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output/model/poisson_modular")
CACHE = sys.argv[2] if len(sys.argv) > 2 else "output/model/poisson_koushik.npz"
TARGET = int(os.environ.get("MV_TARGET", 900_000))     # triangles in the viewer

KIND_COLOR = {"wall": "#5b7fd4", "parapet": "#4aa6c8", "floor": "#b9b2a6",
              "ceiling": "#68c07a", "beam": "#e08a2b", "dropped_ceiling": "#d8c33a",
              "column": "#cf4b4b"}


def decimated_scene(T, V, label, names, man):
    import trimesh
    kinds = {p["name"]: p["kind"] for p in man["parts"]}
    total = int((label >= 0).sum())
    ratio = min(1.0, TARGET/total)
    log(f"decimating {total:,} triangles to about {int(total*ratio):,}")
    sc = trimesh.Scene()
    rng = np.random.default_rng(11)
    kept = 0
    for pid, nm in enumerate(names):
        idx = np.where(label == pid)[0]
        if idx.size == 0:
            continue
        t = T[idx]
        used = np.unique(t)
        rm = np.full(len(V), -1, np.int64); rm[used] = np.arange(len(used))
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(V[used])
        m.triangles = o3d.utility.Vector3iVector(rm[t])
        want = max(200, int(len(t)*ratio))
        if want < len(t):
            m = m.simplify_quadric_decimation(want)
        vv = np.asarray(m.vertices); tt = np.asarray(m.triangles)
        if len(tt) == 0:
            continue
        kept += len(tt)
        tm = trimesh.Trimesh(vv, tt, process=False)
        base = np.array(rng.integers(70, 225, 3))
        col = np.zeros((len(vv), 4), np.uint8)
        col[:, :3] = base; col[:, 3] = 255
        tm.visual.vertex_colors = col
        sc.add_geometry(tm, geom_name=nm, node_name=nm)
    log(f"viewer mesh: {kept:,} triangles")
    return sc


def main():
    man = json.load(open(OUT/"manifest.json"))
    T = np.load(CACHE)["T"].astype(np.int64)
    V = np.load(OUT/"verts.npy").astype(np.float64)
    label = np.load(OUT/"labels.npy")
    names = json.load(open(OUT/"names.json"))

    lite = OUT/"modular_lite.glb"
    if not lite.exists() or os.environ.get("MV_FORCE"):
        decimated_scene(T, V, label, names, man).export(str(lite))
        log(f"wrote {lite} ({lite.stat().st_size/1e6:.1f} MB)")

    b64 = base64.b64encode(open(lite, "rb").read()).decode()
    parts = {p["name"]: p for p in man["parts"]}
    cov = man["coverage"]
    feats = man["features"]
    nk = {}
    for p in man["parts"]:
        nk[p["kind"]] = nk.get(p["kind"], 0) + 1
    fk = {}
    for f in feats:
        fk[f["kind"]] = fk.get(f["kind"], 0) + 1

    html = f"""<!doctype html><meta charset=utf-8>
<title>Modular house from the Poisson mesh</title>
<style>
 body{{margin:0;background:#0f1115;color:#dfe3ea;font:13px/1.5 ui-sans-serif,system-ui,sans-serif}}
 #c{{position:fixed;inset:0}}
 #p{{position:fixed;top:0;right:0;width:330px;max-height:100vh;overflow:auto;
     background:#161a21ee;padding:14px 16px;border-left:1px solid #2a303a}}
 h1{{font-size:15px;margin:0 0 2px}} h2{{font-size:12px;margin:14px 0 4px;color:#8e98a8;
     text-transform:uppercase;letter-spacing:.08em}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
 label{{display:block;padding:2px 0;cursor:pointer}}
 table{{width:100%;border-collapse:collapse;font-size:12px}}
 td{{padding:1px 4px 1px 0;border-bottom:1px solid #222831}}
 td.n{{text-align:right;color:#9fb0c8;font-variant-numeric:tabular-nums}}
 .note{{color:#7c8695;font-size:11px}}
</style>
<canvas id=c></canvas>
<div id=p>
 <h1>Modular house</h1>
 <div class=note>segmented from the Poisson mesh &mdash; every part is the scanned
 surface itself, cropped, not a box fitted to it</div>
 <h2>Show</h2>
 <div id=toggles></div>
 <h2>Measured</h2>
 <table>
  <tr><td>clear height</td><td class=n>{man['clear_height_mm']:.0f} mm</td></tr>
  <tr><td>parts</td><td class=n>{man['n_parts']}</td></tr>
  <tr><td>surface kept</td><td class=n>{cov['kept_frac']*100:.1f}%</td></tr>
  <tr><td>gaps between parts</td><td class=n>{cov['crack_tris']} tri</td></tr>
  <tr><td>dropped as clutter</td><td class=n>{cov['dropped_area_m2']} m&sup2;</td></tr>
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
const FEATS = {json.dumps(feats)};
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1115);
const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, .05, 500);
const rend = new THREE.WebGLRenderer({{canvas:document.getElementById('c'), antialias:true}});
rend.setSize(innerWidth, innerHeight); rend.setPixelRatio(devicePixelRatio);
const ctl = new OrbitControls(cam, rend.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x40485a, 2.1));
const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(6, -9, 12); scene.add(dl);
const groups = {{}};
new GLTFLoader().parse(Uint8Array.from(atob("{b64}"), c=>c.charCodeAt(0)).buffer, '', g => {{
  const box = new THREE.Box3().setFromObject(g.scene);
  const ctr = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
  g.scene.traverse(o => {{
    if (!o.isMesh) return;
    const nm = o.name.replace(/_\\d+$/, m => m), p = PARTS[o.name] || PARTS[nm];
    const kind = p ? p.kind : 'other';
    o.userData.part = p; o.userData.kind = kind;
    o.material = new THREE.MeshStandardMaterial({{
      color: new THREE.Color(KC[kind] || '#8a7fd0'), roughness:.92, metalness:.0,
      flatShading:false, side:THREE.DoubleSide}});
    (groups[kind] = groups[kind] || []).push(o);
  }});
  scene.add(g.scene);
  cam.position.set(ctr.x + sz.x*.9, ctr.y - sz.y*1.1, ctr.z + sz.z*2.4);
  ctl.target.copy(ctr); ctl.update();
  const t = document.getElementById('toggles');
  Object.keys(groups).sort().forEach(k => {{
    const l = document.createElement('label');
    l.innerHTML = `<input type=checkbox checked data-k="${{k}}">` +
      `<span class=sw style="background:${{KC[k]||'#8a7fd0'}}"></span>${{k}} (${{groups[k].length}})`;
    l.querySelector('input').onchange = e => groups[k].forEach(o => o.visible = e.target.checked);
    t.appendChild(l);
  }});
}});
const ray = new THREE.Raycaster(), m = new THREE.Vector2();
addEventListener('click', e => {{
  m.x = e.clientX/innerWidth*2-1; m.y = -(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(m, cam);
  const h = ray.intersectObjects(scene.children, true).filter(i => i.object.visible)[0];
  const d = document.getElementById('sel');
  if (!h) {{ d.textContent = 'click a part'; return; }}
  const p = h.object.userData.part, nm = h.object.name;
  const fs = FEATS.filter(f => f.wall === nm);
  d.innerHTML = `<b>${{nm}}</b><table>` +
    (p ? Object.entries(p).filter(([k]) => k!=='name').map(([k,v]) =>
      `<tr><td>${{k}}</td><td class=n>${{v}}</td></tr>`).join('') : '') +
    fs.map(f => `<tr><td>${{f.kind}}</td><td class=n>${{f.width_mm||''}}&times;${{f.height_mm||''}} mm</td></tr>`).join('') +
    `</table>`;
}});
addEventListener('resize', () => {{
  cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
  rend.setSize(innerWidth, innerHeight);
}});
(function loop() {{ requestAnimationFrame(loop); ctl.update(); rend.render(scene, cam); }})();
</script>
"""
    (OUT/"viewer.html").write_text(html, encoding="utf-8")
    log(f"wrote {OUT/'viewer.html'} ({(OUT/'viewer.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
