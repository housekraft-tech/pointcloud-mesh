"""build_gallery.py
----------------
Generate a single index.html that tiles every PNG output in a pipeline
folder, grouped by stage, so you open ONE file in a browser and scroll
through the whole reconstruction (floorplan, mesh renders, Hough, sections,
mining maps). Uses relative paths so it works opened locally.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_gallery.py <folder>   e.g. output/mujammel_all
"""
import sys
from pathlib import Path

GROUPS = [
    ("Deliverable floorplan", ["freespace_floorplan.png", "hough_vector/hough_vector_floorplan.png",
                               "hough_from_mesh/hough_from_mesh.png", "openings/openings_floorplan.png",
                               "tomographic_floorplan.png"]),
    ("3D mesh (Poisson output)", ["mesh_renders/mesh_top_down_plan.png", "mesh_renders/mesh_bottom_up.png",
                                  "mesh_renders/mesh_perspective_1.png", "mesh_renders/mesh_perspective_2.png",
                                  "mesh_renders/mesh_front_elevation.png"]),
    ("Walk path & masks", ["walk_path.png", "mask_walls.png", "mask_beams.png", "mask_railings.png",
                           "hough_from_mesh/mesh_slice_raster.png", "hough_vector/hv_skeleton.png"]),
    ("LAS-signal mining", [f"lasmining/{n}" for n in
                           ["1_coverage_confidence.png", "2_density_quality.png", "3_intensity_material.png",
                            "4_intensity_variance_glass.png", "5_rgb_material.png", "6_colour_saturation.png",
                            "7_fused_overlay.png"]]),
    ("Diagnostic atlas", [f"atlas/{n}" for n in
                          ["03_height_colored.png", "04_trajectory_overlay.png", "05_gpstime_heatmap.png",
                           "06_intensity.png", "07_rgb.png", "02_hough_walls.png",
                           "01_slice_waist.png", "01_slice_mid.png", "08_elevation_profile.png",
                           "09_sections_sweepX.png", "10_sections_sweepY.png"]]),
]


def main(folder):
    folder = Path(folder)
    name = folder.name
    cards = []
    for title, files in GROUPS:
        tiles = []
        for rel in files:
            if (folder / rel).exists():
                label = rel.split("/")[-1].replace(".png", "").replace("_", " ")
                tiles.append(f'<figure><a href="{rel}" target="_blank">'
                             f'<img loading="lazy" src="{rel}"></a>'
                             f'<figcaption>{label}</figcaption></figure>')
        if tiles:
            cards.append(f'<section><h2>{title}</h2><div class="grid">{"".join(tiles)}</div></section>')
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name} — reconstruction gallery</title>
<style>
 body{{margin:0;background:#12140f;color:#e8ebe4;font:15px/1.5 system-ui,sans-serif;}}
 header{{padding:20px 24px;border-bottom:1px solid #2a2e26;}}
 h1{{margin:0;font-size:20px;font-family:ui-monospace,monospace;color:#9fb89f;}}
 header p{{margin:6px 0 0;color:#8a908280;color:#8a9082;font-size:13px;}}
 section{{padding:18px 24px;border-bottom:1px solid #22261f;}}
 h2{{font-size:14px;text-transform:uppercase;letter-spacing:.05em;color:#c98a5a;margin:0 0 14px;
     font-family:ui-monospace,monospace;}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}}
 figure{{margin:0;background:#1a1d16;border:1px solid #2a2e26;border-radius:6px;overflow:hidden;}}
 img{{width:100%;display:block;background:#000;cursor:zoom-in;}}
 figcaption{{padding:8px 10px;font-size:12px;color:#a9b0a2;font-family:ui-monospace,monospace;}}
 a{{color:inherit;text-decoration:none;}}
</style></head><body>
<header><h1>{name} · reconstruction gallery</h1>
<p>Click any image to open full-size. Generated from the pipeline output folder.</p></header>
{"".join(cards)}
</body></html>"""
    out = folder / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({sum(len(f) for _, f in GROUPS)} slots) -- open it in a browser")


if __name__ == "__main__":
    main(sys.argv[1])
