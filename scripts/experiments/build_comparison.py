"""build_comparison.py
-------------------
Assemble a single self-contained HTML page comparing the reconstruction outputs
(images embedded as base64 data URIs so the file opens anywhere, offline).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_comparison.py <out_dir> <comparison.html>
"""
import base64
import sys
from pathlib import Path


def data_uri(p):
    b = Path(p).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def card(img, title, sub):
    return f"""<figure class="card">
  <img src="{data_uri(img)}" alt="{title}" loading="lazy">
  <figcaption><h3>{title}</h3><p>{sub}</p></figcaption>
</figure>"""


def main(base, out_html):
    base = Path(base)
    threed = [
        (base / "modular_3d/render/mesh_perspective_1.png", "1 · Modular (footprint extrude)",
         "Clean box walls from the carved footprint. Crisp, but one height profile top-to-bottom."),
        (base / "volumetric/render/mesh_perspective_1.png", "2 · Volumetric (raw points)",
         "112 height slices stacked. True height-varying features, but rougher surfaces."),
        (base / "mesh_volumetric/render/mesh_perspective_1.png", "3 · Mesh-volumetric + rectilinear ★",
         "Complete Poisson mesh source, 1cm slices, gridline-snapped: flat faces, sharp corners, height-varying detail."),
        (base / "mesh/render/mesh_bottom_up.png", "Reference · Complete Poisson mesh",
         "The full gap-filled surface (floor view). Sub-cm fidelity; the source for method 3."),
    ]
    twod = [
        (base / "fusion/fused_floorplan.png", "Fused floorplan",
         "LiDAR walls + RF-DETR windows/balcony-doors, registered from the architect drawing."),
        (base / "mesh_volumetric/mesh_volumetric_dimensioned.png", "Per-room dimensions",
         "Internal clear span per room (mm/ft), measured direct from the LiDAR."),
        (base / "fusion/compare_overlay.png", "As-built vs as-designed",
         "Reconstruction overlaid on the architect drawing — enclosed rooms match 0-2%."),
    ]
    three_html = "\n".join(card(*c) for c in threed if c[0].exists())
    two_html = "\n".join(card(*c) for c in twod if c[0].exists())
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>koushik · reconstruction comparison</title>
<style>
  :root {{ --bg:#0e1013; --panel:#161a20; --line:#262c36; --ink:#e8ebf0; --dim:#9aa4b2; --accent:#5db0ff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 "Segoe UI",system-ui,-apple-system,sans-serif; }}
  header {{ padding:34px 28px 10px; max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:.2px; }}
  header p {{ color:var(--dim); margin:0; }}
  h2 {{ max-width:1200px; margin:34px auto 6px; padding:0 28px; font-size:14px;
    text-transform:uppercase; letter-spacing:1.4px; color:var(--accent); }}
  .grid {{ max-width:1200px; margin:0 auto; padding:12px 20px 8px;
    display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }}
  .card {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; overflow:hidden; }}
  .card img {{ width:100%; display:block; background:#0b0d10; cursor:zoom-in; }}
  figcaption {{ padding:12px 14px 14px; }}
  figcaption h3 {{ margin:0 0 4px; font-size:15px; }}
  figcaption p {{ margin:0; color:var(--dim); font-size:13px; }}
  .card:has(h3:first-line) {{}}
  footer {{ max-width:1200px; margin:20px auto 50px; padding:0 28px; color:var(--dim); font-size:12.5px; }}
  /* lightbox */
  #lb {{ position:fixed; inset:0; background:rgba(6,8,11,.94); display:none;
    align-items:center; justify-content:center; cursor:zoom-out; z-index:9; padding:24px; }}
  #lb img {{ max-width:96vw; max-height:92vh; border-radius:8px; }}
</style></head><body>
<header>
  <h1>Reconstruction comparison — koushik</h1>
  <p>Handheld LiDAR &rarr; walls-only 3D, at increasing fidelity, plus the 2D fusion &amp; deviation.</p>
</header>
<h2>3D reconstruction methods</h2>
<div class="grid">{three_html}</div>
<h2>2D floorplan · semantics · deviation</h2>
<div class="grid">{two_html}</div>
<footer>Click any image to enlarge. Method 3 (★) is the current best: complete Poisson mesh &rarr; 1cm height
slices &rarr; wall-network footprint &rarr; gridline-snapped &amp; plane-flattened. Detail floor is ~1cm (scan noise);
finer slicing does not add real geometry.</footer>
<div id="lb"><img src="" alt=""></div>
<script>
  const lb=document.getElementById('lb'), lbi=lb.querySelector('img');
  document.querySelectorAll('.card img').forEach(i=>i.addEventListener('click',()=>{{lbi.src=i.src;lb.style.display='flex';}}));
  lb.addEventListener('click',()=>lb.style.display='none');
</script>
</body></html>"""
    Path(out_html).write_text(html, encoding="utf-8")
    kb = len(html) / 1024
    print(f"wrote {out_html} ({kb/1024:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
