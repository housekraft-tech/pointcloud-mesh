"""Assemble report.html from the measured results and the encoded figures."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
imgs = json.loads((HERE / "images.json").read_text())
summary = json.loads((HERE / "scores/summary.json").read_text())
best = json.loads((HERE / "scores/robust_c95_swin_score.json").read_text())
depth = json.loads((HERE / "depth/depth_scores_mujammelexport.json").read_text())

rows = "".join(
    '<tr class="{cls}">'
    '<td class="mono">{run}</td>'
    '<td class="num">{n}</td>'
    '<td class="num">{a:.1f}</td>'
    '<td class="num {neg}">{e:+.1f}%</td>'
    '<td class="num"><span class="bar" style="--v:{i}"></span>{i:.3f}</td>'
    '<td class="num">{m}</td>'
    '<td class="num">{mi:.3f}</td></tr>'.format(
        cls="is-best" if r["run"] == "robust_c95_swin" else "",
        run=r["run"], n=r["n_pred"], a=r["area_m2"], e=r["area_err_pct"],
        neg="neg" if abs(r["area_err_pct"]) > 15 else "",
        i=r["footprint_iou"], m=r["matched_iou50"], mi=r["mean_iou"])
    for r in summary)

room_rows = "".join(
    '<tr><td>{n}</td><td class="num">{g:.2f}</td><td class="num">{p:.2f}</td>'
    '<td class="num"><span class="pill {c}">{i:.3f}</span></td></tr>'.format(
        n=r["gt_name"], g=r["gt_area_m2"], p=r["pred_area_m2"], i=r["iou"],
        c="ok" if r["iou"] >= 0.5 else ("mid" if r["iou"] >= 0.25 else "bad"))
    for r in best["rooms"])

med = lambda k: st.median([d[k] for d in depth])

CSS = """
:root {
  --paper:#EEF2F3; --surface:#FFFFFF; --surface-2:#F6F9F9; --line:#D3DDE0;
  --ink:#101A1F; --muted:#54676F; --faint:#7C8F97;
  --accent:#0A6E79; --accent-soft:#D7ECEE;
  --ok:#2C7A50; --mid:#9C6716; --bad:#A63A38;
  --ok-bg:#DBEEE3; --mid-bg:#F5E7CE; --bad-bg:#F5DEDC;
  --shadow:0 1px 2px rgba(16,26,31,.05), 0 8px 24px -12px rgba(16,26,31,.18);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0B1114; --surface:#131C21; --surface-2:#182228; --line:#26343B;
    --ink:#E4EDF0; --muted:#93A8B1; --faint:#6E858E;
    --accent:#3FC6D3; --accent-soft:#12333A;
    --ok:#5CC08B; --mid:#DFAA4E; --bad:#E37A78;
    --ok-bg:#14301F; --mid-bg:#33280F; --bad-bg:#341A19;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 28px -14px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --paper:#0B1114; --surface:#131C21; --surface-2:#182228; --line:#26343B;
  --ink:#E4EDF0; --muted:#93A8B1; --faint:#6E858E;
  --accent:#3FC6D3; --accent-soft:#12333A;
  --ok:#5CC08B; --mid:#DFAA4E; --bad:#E37A78;
  --ok-bg:#14301F; --mid-bg:#33280F; --bad-bg:#341A19;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 28px -14px rgba(0,0,0,.7);
}
* { box-sizing:border-box; }
body {
  background:var(--paper); color:var(--ink);
  font-family:"Source Serif 4", Georgia, "Times New Roman", serif;
  font-size:17px; line-height:1.62; margin:0; padding:0 20px 96px;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1180px; margin:0 auto; }
.col { max-width:660px; }
h1, h2, h3, .disp { font-family:"Bricolage Grotesque", "Helvetica Neue", system-ui, sans-serif; text-wrap:balance; }
h1 { font-size:clamp(2.1rem,5.2vw,3.3rem); line-height:1.04; font-weight:700; letter-spacing:-.022em; margin:0 0 .55rem; }
h2 { font-size:1.5rem; font-weight:700; letter-spacing:-.012em; margin:0 0 .35rem; }
h3 { font-size:1.06rem; font-weight:700; margin:0 0 .35rem; }
p { margin:0 0 1.05rem; }
a { color:var(--accent); }
.mono, code, td.num, th, .path { font-family:"IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace; }
.eyebrow {
  font-family:"IBM Plex Mono", monospace; font-size:.7rem; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent); margin:0 0 1.1rem;
}
header { padding:76px 0 44px; border-bottom:1px solid var(--line); }
.standfirst { font-size:1.2rem; color:var(--muted); max-width:62ch; margin:0; }
.verdicts { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin:38px 0 0; }
.verdict {
  background:var(--surface); border:1px solid var(--line); border-radius:3px;
  padding:16px 18px; box-shadow:var(--shadow); display:flex; flex-direction:column; gap:7px;
}
.tag {
  font-family:"IBM Plex Mono", monospace; font-size:.66rem; font-weight:600;
  letter-spacing:.13em; text-transform:uppercase; padding:3px 8px; border-radius:2px; align-self:flex-start;
}
.tag.ok { background:var(--ok-bg); color:var(--ok); }
.tag.mid { background:var(--mid-bg); color:var(--mid); }
.tag.bad { background:var(--bad-bg); color:var(--bad); }
.verdict .what { font-family:"Bricolage Grotesque", system-ui, sans-serif; font-weight:700; font-size:1.02rem; }
.verdict .why { font-size:.93rem; color:var(--muted); line-height:1.5; }
section { padding:54px 0 0; }
.lede { font-size:1.05rem; }
figure { margin:26px 0 8px; }
figure img { width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:3px; background:#05080A; }
figcaption { font-size:.87rem; color:var(--muted); margin-top:10px; max-width:76ch; line-height:1.5; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
@media (max-width:720px) { .two { grid-template-columns:1fr; } }
.tablewrap { overflow-x:auto; margin:24px 0 6px; border:1px solid var(--line); border-radius:3px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:.86rem; }
th {
  text-align:right; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); padding:12px 14px; border-bottom:1px solid var(--line);
  white-space:nowrap; background:var(--surface-2);
}
th:first-child, td:first-child { text-align:left; }
td { padding:10px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; font-size:.83rem; }
td.neg { color:var(--bad); }
tr.is-best td { background:var(--accent-soft); font-weight:600; }
.bar {
  display:inline-block; width:52px; height:5px; border-radius:2px; margin-right:9px;
  background:linear-gradient(to right, var(--accent) calc(var(--v)*100%), var(--line) 0);
  vertical-align:middle;
}
.pill { display:inline-block; min-width:56px; padding:2px 8px; border-radius:2px; font-family:"IBM Plex Mono",monospace; font-size:.78rem; }
.pill.ok { background:var(--ok-bg); color:var(--ok); }
.pill.mid { background:var(--mid-bg); color:var(--mid); }
.pill.bad { background:var(--bad-bg); color:var(--bad); }
.callout {
  border-left:3px solid var(--accent); background:var(--surface); padding:18px 22px;
  margin:28px 0; border-radius:0 3px 3px 0; box-shadow:var(--shadow);
}
.callout p:last-child { margin-bottom:0; }
.stats {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:3px; margin:26px 0; overflow:hidden;
}
.stat { background:var(--surface); padding:16px 18px; }
.stat .v { font-family:"IBM Plex Mono",monospace; font-size:1.22rem; font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.stat .k { font-family:"IBM Plex Mono",monospace; font-size:.66rem; letter-spacing:.11em; text-transform:uppercase; color:var(--faint); margin-top:4px; }
.stat.bad .v { color:var(--bad); }
ul { padding-left:1.1rem; margin:0 0 1.05rem; }
li { margin-bottom:.55rem; }
.path { font-size:.82rem; background:var(--surface-2); border:1px solid var(--line); padding:2px 6px; border-radius:2px; }
footer { margin-top:60px; padding-top:26px; border-top:1px solid var(--line); color:var(--muted); font-size:.9rem; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
"""

BODY = """
<div class="wrap">
<header>
  <p class="eyebrow">Zero-shot trial &middot; Koushik + Mujammel scans &middot; August 2026</p>
  <h1>What the published models<br>can and cannot do for us</h1>
  <p class="standfirst">Three approaches from the BIMScript author's orbit, run for real against our own scans and the architect's plan. No retraining, no cherry-picked scene &mdash; the released checkpoints, our data, scored the same way every time. The one that survives is now wired into the reconstruction, and the flat comes out as a solid model.</p>

  <div class="verdicts">
    <div class="verdict">
      <span class="tag ok">Adopted &mdash; topology</span>
      <span class="what">CAGE</span>
      <span class="why">Straight from synthetic Structured3D it recovers four of our rooms above 0.90 IoU. Its coordinates are useless at 62&nbsp;mm/px &mdash; but snapped onto our measured planes, room dimensions land at 12&nbsp;mm median against the tape.</span>
    </div>
    <div class="verdict">
      <span class="tag bad">Reject &mdash; measurement</span>
      <span class="what">Monocular metric depth</span>
      <span class="why">Median error 1,034&nbsp;mm against LiDAR ground truth. Even fitting a perfect per-image scale after the fact leaves 387&nbsp;mm.</span>
    </div>
    <div class="verdict">
      <span class="tag mid">Adopt &mdash; export only</span>
      <span class="what">BIMScript</span>
      <span class="why">The model is unusable at F1@2cm&nbsp;=&nbsp;0.07 and unreleased anyway. The MIT text&rarr;Revit&rarr;IFC4 exporter is real code and fits our wall table today.</span>
    </div>
  </div>
</header>

<section>
  <div class="col">
    <h2>Getting CAGE to run at all</h2>
    <p class="lede">The repo asks for Linux, CUDA 11.1 and two compiled extensions. None of that is needed to run it forward.</p>
    <p><code>diff_ras</code> is a rasterisation <em>loss</em> &mdash; training only, so it stubs out. The deformable-attention CUDA op ships with a pure-PyTorch reference implementation in the same file, which patches straight in. The denoising path hardcodes <code>.cuda()</code>; that becomes a no-op. Both released checkpoints then load on Windows CPU with <strong>zero missing keys</strong>, and a scene takes a couple of minutes.</p>
    <p>That matters beyond convenience. It means we can evaluate this whole class of model against our own scans without standing up a Linux box first.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>The domain gap is one line of preprocessing</h2>
    <p>CAGE eats a 256&times;256 top-down density map &mdash; a histogram of point counts. Structured3D builds those from panorama depth, so sampling is near-uniform. A handheld scan is nothing like that: dwell time and range make some pixels twenty-five times denser than others, so after normalising by the maximum the whole floorplan sits at about 4% brightness and the network sees almost nothing.</p>
    <p>Clipping the per-pixel counts at their 95th percentile before normalising fixes it completely. That single change moves the output from 8 rooms to a complete 15-room tiling.</p>
  </div>
  <div class="two">
    <figure>
      <img src="__DENSITY_INPUT__" alt="Top-down density map of the flat, bright wall lines against dark room interiors">
      <figcaption><strong>The working input.</strong> 256&times;256, 62&nbsp;mm per pixel, counts clipped at p95. Walls read as bright lines and room interiors as texture &mdash; the distribution the model was trained on.</figcaption>
    </figure>
    <figure>
      <img src="__DENSITY_SKELETON__" alt="Wall-only skeleton density map, thin lines on black">
      <figcaption><strong>The negative result.</strong> Counting only returns 1.0&ndash;2.2&nbsp;m above the floor gives a clean wall skeleton and <em>worse</em> output. The model was trained on filled interiors, so a skeleton is out of distribution. Worth not repeating.</figcaption>
    </figure>
  </div>
</section>

<section>
  <div class="col">
    <h2>Every run, scored the same way</h2>
    <p>Fifteen configurations across two backbones. Scale is never fitted &mdash; only a rigid placement onto the architect plan, pinned to one orientation across all runs, because the scan-to-drawing rotation is a single physical fact rather than a per-run free parameter.</p>
  </div>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Run</th><th>Rooms</th><th>Area m&sup2;</th><th>Area err</th><th>Footprint IoU</th><th>Matched @0.5</th><th>Mean IoU</th>
      </tr></thead>
      <tbody>__ROWS__</tbody>
    </table>
  </div>
  <p class="col" style="font-size:.9rem;color:var(--muted);margin-top:14px;">Ground truth is 15 named spaces totalling 111.9&nbsp;m&sup2;. <span class="mono">robust_c95_swin</span> is the run everything below refers to.</p>
</section>

<section>
  <div class="col">
    <h2>The best run, and why its score understates it</h2>
    <p>Four rooms land above 0.90 IoU with areas within a few percent of the plan: Bedroom&nbsp;3 at 0.955, Bedroom at 0.934, and both small bathrooms above 0.90. The apparent failures are mostly not failures.</p>
  </div>
  <figure>
    <img src="__BEST_OVERLAY__" alt="Fifteen coloured room polygons tiling the density map of the flat">
    <figcaption><strong>robust_c95_swin.</strong> Fifteen watertight, non-overlapping polygons, most with exactly four corners. Room&nbsp;13 is the open-plan centre; 11, 2, 12 and 20 are the bedrooms and bathrooms that score above 0.90.</figcaption>
  </figure>

  <div class="callout col">
    <h3>The open-plan merge</h3>
    <p>CAGE returns the centre of the flat as one 40.37&nbsp;m&sup2; room. That polygon contains 98.4% of the Living Room, 99.4% of the Dining Room, 97.7% of the Foyer and 78% of the Kitchen &mdash; 39.31&nbsp;m&sup2; of plan area, so the merged room is within <strong>+2.7%</strong> of what it actually encloses.</p>
    <p>It is not wrong about the building. The architect's plan names four spaces where there are no separating walls to find. Three of the five apparent misses are that one merge, which is why the matched-at-0.5 column reads 7/15 while the geometry is essentially right.</p>
  </div>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Plan space</th><th>Plan m&sup2;</th><th>Predicted m&sup2;</th><th>IoU</th></tr></thead>
      <tbody>__ROOM_ROWS__</tbody>
    </table>
  </div>
</section>

<section>
  <div class="col">
    <h2>Across the clipping sweep</h2>
    <p>The same pattern holds everywhere: the band of small, fully walled rooms reconstructs reliably; the open-plan centre and the balconies are where agreement breaks down.</p>
  </div>
  <figure>
    <img src="__VS_GT_SWIN__" alt="Five panels comparing swin predictions against the architect plan outline">
    <figcaption><strong>swinv2-L backbone.</strong> Grey outline is the architect plan. Green is 0.5 IoU or better, orange 0.25&ndash;0.5, red below that.</figcaption>
  </figure>
  <figure>
    <img src="__VS_GT_R50__" alt="Five panels comparing resnet50 predictions against the architect plan outline">
    <figcaption><strong>ResNet-50 backbone.</strong> Noisier, and prone to degenerate self-intersecting polygons in the centre &mdash; though its p90 run matches the most rooms outright, 8 of 15.</figcaption>
  </figure>
</section>

<section>
  <div class="col">
    <h2>Wiring it in: CAGE's topology, our planes</h2>
    <p class="lede">The split isn't just an idea now &mdash; it's built, and it works.</p>
    <p>Take the room graph and throw the coordinates away. Every edge gets re-solved onto a wall face measured from the raw points by <code>recon.metrology</code>: <strong>361 faces at 159&nbsp;&micro;m median standard error</strong>. Corners then fall out as exact intersections of two measured planes instead of regressed pixel positions.</p>
    <p>One detail decided the whole result. Snapping to the <em>nearest</em> face lands on the wrong side of a wall about a fifth of the time &mdash; and the errors that produces are exactly one wall thick (194 or 253&nbsp;mm, the scan's own measured modes). The fix is to identify the <em>wall</em> first, as a face pair separated by a plausible thickness, then take the face that looks into this room. That, plus rejecting weakly-supported surfaces like wardrobe fronts, took the median error from 42&nbsp;mm to 12&nbsp;mm.</p>
  </div>
  <figure>
    <img src="__SNAP_SUMMARY__" alt="Three panels: room boundaries before and after snapping, histogram of edge corrections, per-room error bars">
    <figcaption>54 of 64 edges re-solved. The correction histogram straddles one density pixel, which is what you'd expect if CAGE were quantisation-limited rather than wrong.</figcaption>
  </figure>

  <div class="col">
    <p>Scored against the hand tape-measured survey &mdash; the only ground truth in this project fine enough to see the difference &mdash; comparing <strong>the same polygon</strong> before and after:</p>
  </div>
  <div class="stats">
    <div class="stat"><div class="v">90.3 &rarr; 12.0</div><div class="k">Median error, mm</div></div>
    <div class="stat"><div class="v">138.5 &rarr; 44.2</div><div class="k">p90 error, mm</div></div>
    <div class="stat"><div class="v">0 &rarr; 5</div><div class="k">of 14 within 10 mm</div></div>
    <div class="stat"><div class="v">159 &micro;m</div><div class="k">Face measurement stderr</div></div>
  </div>
  <div class="col">
    <p>Utility went from 148.8&nbsp;mm out to <strong>7.8&nbsp;mm</strong>. Common washroom 83.8 to 12.3. Every one of the seven rooms improved, and five of fourteen dimensions now sit inside the survey's own 10&nbsp;mm tolerance, up from none.</p>
    <p>Note that IoU against the architect plan barely moved for all this &mdash; 0.796 to 0.803. That is not a weak result, it is the wrong instrument: shifting a room edge 20&nbsp;mm changes a 3.5&nbsp;m room's area by well under one percent. Room-level IoU is structurally blind to the thing the snap is for, which is worth remembering before trusting it as a pipeline metric.</p>
    <p>The residual is honest and diagnosable. Foyer is still 55.8&nbsp;mm out and KBR 44.6&nbsp;mm, both because CAGE's edge sits far enough from any structural face that the wall-pair test finds nothing to lock onto. Ten of sixty-four edges were left where CAGE put them for that reason. Those are the edges to attack next &mdash; not by improving the model, but by widening the search to the walls the room graph says must be there.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>And then it is a building</h2>
    <p class="lede">Every surface below is measured. Nothing is nominal except the exterior skin, because nothing was scanned outside the flat.</p>
    <p>The room polygons extrude between a floor at <span class="mono">-0.2303&nbsp;m</span> and a ceiling at <span class="mono">2.4894&nbsp;m</span> &mdash; both fitted by the same estimator as the walls, to <strong>7 and 18&nbsp;micrometres</strong> of standard error, giving a clear height of <strong>2,719.6&nbsp;mm</strong>. Interior walls need no nominal thickness either: the gap between two rooms' snapped boundaries <em>is</em> the measured face pair, so each partition comes out at its own true thickness.</p>
  </div>
  <figure>
    <img src="__MODEL3D__" alt="Four renders of the reconstructed flat: cutaway isometric, coloured room volumes, plan view, and closed exterior">
    <figcaption>15 rooms, 123.2 m&sup2; net internal area, 335.05 m&sup3;. Exported as <span class="mono">flat_solid.glb</span>, <span class="mono">flat_rooms.glb</span> and <span class="mono">flat_cutaway.glb</span>.</figcaption>
  </figure>
  <div class="col">
    <p>One artefact is worth pointing at rather than hiding: in the plan view the top band of small rooms floats free of the main body. Where CAGE left a gap wider than a wall between two rooms, there is no measured partition to fill it, so the model has a hole. That is the topology talking, not the geometry &mdash; and it is the same open-plan boundary problem, seen from the other side.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Doors, windows, balcony doors</h2>
    <p class="lede">A top-down density map cannot see an opening at all &mdash; that was your point about beams, and it applies to every hole in a wall.</p>
    <p>So the storey gets sliced. Each wall's points go into a (u,&nbsp;z) grid instead of being flattened, and an opening is a void in it: a door reaches the floor and stops at a header, a window sits in a mid band above a sill, a partition that stops short of the slab leaves a void along the top. CAGE was also re-run on six horizontal bands from 0.15&nbsp;m to 2.70&nbsp;m &mdash; the room count moves between 9 and 10 across them, which is the same signal at plan scale: boundaries that exist at head height and not at knee height are openings, not walls.</p>
    <p>Extents are measured, not binned. The void finder works on a 30&nbsp;mm grid, so every raw width came back a multiple of 30; each of the four edges is then re-solved against the raw point-density half-max crossing.</p>
  </div>
  <figure>
    <img src="__OPENINGS3D__" alt="Four renders of the flat with door, window and balcony-door openings cut into the walls and colour coded">
    <figcaption>18 openings cut into the wall solid, which stays watertight. Blue door, orange balcony door, green cased opening, magenta high-level void.</figcaption>
  </figure>

  <div class="tablewrap">
    <table>
      <thead><tr><th>Type</th><th>Width mm</th><th>Height mm</th><th>Sill mm</th><th>Wall</th></tr></thead>
      <tbody><tr><td>balcony door</td><td class="num">2340</td><td class="num">2411</td><td class="num">-0</td><td class="num">ext</td></tr><tr><td>balcony door</td><td class="num">1460</td><td class="num">2102</td><td class="num">-11</td><td class="num">ext</td></tr><tr><td>balcony door</td><td class="num">1452</td><td class="num">2411</td><td class="num">-0</td><td class="num">ext</td></tr><tr><td>door</td><td class="num">1000</td><td class="num">2240</td><td class="num">10</td><td class="num">int</td></tr><tr><td>door</td><td class="num">931</td><td class="num">2151</td><td class="num">90</td><td class="num">int</td></tr><tr><td>door</td><td class="num">891</td><td class="num">1880</td><td class="num">0</td><td class="num">int</td></tr><tr><td>door</td><td class="num">852</td><td class="num">2031</td><td class="num">-0</td><td class="num">int</td></tr><tr><td>door</td><td class="num">812</td><td class="num">2091</td><td class="num">-10</td><td class="num">int</td></tr><tr><td>door</td><td class="num">740</td><td class="num">2240</td><td class="num">10</td><td class="num">int</td></tr><tr><td>door</td><td class="num">740</td><td class="num">2051</td><td class="num">-0</td><td class="num">int</td></tr><tr><td>door</td><td class="num">740</td><td class="num">2031</td><td class="num">-0</td><td class="num">int</td></tr><tr><td>door</td><td class="num">633</td><td class="num">2160</td><td class="num">-10</td><td class="num">int</td></tr><tr><td>door</td><td class="num">620</td><td class="num">2071</td><td class="num">-0</td><td class="num">int</td></tr><tr><td>door</td><td class="num">620</td><td class="num">2051</td><td class="num">-10</td><td class="num">int</td></tr><tr><td>high level void</td><td class="num">2460</td><td class="num">585</td><td class="num">2136</td><td class="num">int</td></tr><tr><td>wide opening</td><td class="num">2200</td><td class="num">2251</td><td class="num">-10</td><td class="num">int</td></tr><tr><td>wide opening</td><td class="num">1731</td><td class="num">2111</td><td class="num">-10</td><td class="num">int</td></tr><tr><td>wide opening</td><td class="num">1512</td><td class="num">2430</td><td class="num">-10</td><td class="num">int</td></tr></tbody>
    </table>
  </div>

  <div class="col">
    <h3>Checked against the architect's opening schedule</h3>
    <p>Matched by <em>position</em>, using the placement already fitted for the rooms and never re-fitted here. Eight of nineteen paired within 1.2&nbsp;m; four agree on type as well.</p>
  </div>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Ours</th><th>Plan</th><th>Pos mm</th><th>Our w</th><th>Plan w</th><th>&Delta;w</th><th>&Delta;h</th></tr></thead>
      <tbody><tr><td>door</td><td>door</td><td class="num">101</td><td class="num">620</td><td class="num">754</td><td class="num ">-134</td><td class="num">-29</td></tr><tr><td>door</td><td>door</td><td class="num">199</td><td class="num">620</td><td class="num">758</td><td class="num ">-138</td><td class="num">-49</td></tr><tr><td>door</td><td>door</td><td class="num">174</td><td class="num">740</td><td class="num">917</td><td class="num ">-177</td><td class="num">-69</td></tr><tr><td>door</td><td>door</td><td class="num">263</td><td class="num">891</td><td class="num">1113</td><td class="num ">-222</td><td class="num">-220</td></tr><tr><td>door</td><td>balcony door</td><td class="num">781</td><td class="num">812</td><td class="num">1498</td><td class="num neg">-685</td><td class="num">-9</td></tr><tr><td>door</td><td>balcony door</td><td class="num">119</td><td class="num">1000</td><td class="num">1800</td><td class="num neg">-800</td><td class="num">+140</td></tr><tr><td>door</td><td>balcony door</td><td class="num">412</td><td class="num">740</td><td class="num">1800</td><td class="num neg">-1060</td><td class="num">+140</td></tr><tr><td>high level void</td><td>window</td><td class="num">694</td><td class="num">2460</td><td class="num">579</td><td class="num neg">+1881</td><td class="num">-615</td></tr></tbody>
    </table>
  </div>

  <div class="callout col">
    <h3>The honest reading</h3>
    <p>Door heights land within 29&ndash;220&nbsp;mm of the drawing's 2,100&nbsp;mm, and sills within 10&nbsp;mm of the floor. But every matched door comes out <strong>134&ndash;222&nbsp;mm narrower</strong> than the plan, consistently in one direction. That is not noise: the scanner measures the clear gap between the frame reveals, while the drawing gives the structural opening. One frame either side accounts for it almost exactly.</p>
    <p>Which is the right answer depends on the question. For as-built clear widths, ours is correct and the drawing is not. For checking against a drawing, the frame allowance has to be added first &mdash; and doing that silently is how a deviation-detection tool ends up reporting a fault that is really a definitional mismatch.</p>
    <p>Two further gaps, stated rather than smoothed over: five of the plan's six windows were not detected, because glazing returns nothing and the wall above the sill often went unscanned; and one 7.97&nbsp;m &ldquo;window&rdquo; spanning a whole exterior wall is flagged as unscanned surface and excluded from the model rather than built as glass.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Three corrections from looking at the model</h2>
    <p class="lede">Every one of these came from spotting something wrong in the render, not from a metric.</p>

    <h3>1. Not every wall reaches the ceiling</h3>
    <p>The model extruded all 64 wall segments floor-to-slab. Reading each wall's occupancy row by row in z instead gives its real top: <strong>42 full-height, 22 stopping short</strong>. Those 22 top out at 2110, 2121, 2125, 2144, 2149, 2151, 2156, 2157&nbsp;mm &mdash; and the tape survey's dropped ceilings are Utility 2122, Common washroom 2150, MBR washroom 2150, KBR washroom 2161. The wet rooms' false ceilings, recovered to a few millimetres, from a measurement made for a different reason.</p>

    <h3>2. There was a hole in the floor</h3>
    <p>The floor slab was the room union buffered outward, so wherever CAGE left a gap between rooms the slab inherited it. A floor is continuous whatever the room graph says; interior rings are now filled before extruding.</p>

    <h3>3. An open door leaf splits its own opening</h3>
    <p>This is the balcony-door problem. One leaf standing open is a solid vertical panel across the doorway, so the void finder sees the clear side and stops at the leaf; the other leaf, being glass, returns almost nothing and reads as void. An 1800&nbsp;mm balcony door therefore comes back as a ~900&nbsp;mm "door" with an obstruction beside it. Voids on the same wall that share a sill and a head and sit about a leaf-width apart are now rejoined into one opening.</p>
    <p>The visibility gate helps here too, and its limit is worth stating: it asks whether the scanner ever had clear line of sight through a void, which separates a real opening from a furniture shadow &mdash; it removed two &mdash; but an unscanned surface is also "clear", so it cannot tell glass from missing data.</p>
  </div>
  <figure>
    <img src="__V2MODEL__" alt="Four renders of the corrected model with per-wall heights, cut openings and a continuous floor">
    <figcaption>42 full-height and 22 partial-height wall segments, 15 openings cut, floor continuous. Exported as <span class="mono">v2_solid.glb</span> and <span class="mono">v2_marked.glb</span>.</figcaption>
  </figure>
  <div class="col">
    <p>Still open, and stated rather than buried: five of the plan's six windows are undetected, because glazing returns nothing and the wall above the sill often went unscanned &mdash; the two cases are indistinguishable from geometry alone. Balcony railings remain the next thing to separate from structure; nothing in this scan topped out at parapet height, so the balcony boundaries are still being read as full walls.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>One wall per wall</h2>
    <p class="lede">The missing doors, the hole in the floor and the broken connectivity were one bug, not three.</p>
    <p>CAGE returns each room as an independent polygon, so two rooms either side of a partition never share a boundary &mdash; their edges sit 100&ndash;250&nbsp;mm apart with the wall in between. Everything downstream treated those as two separate walls. Nothing needed moving: both edges were already measured correctly, and the gap between them <em>is</em> the wall. What was missing was a structure saying so.</p>
    <p>Pairing edges on the same axis, a plausible thickness apart, with overlapping spans, gives 62 walls from 64 room edges &mdash; 20 interior, joining 19 room pairs. Two mistakes had to be corrected along the way: pairing in index order let an early mediocre match consume an edge a better pair needed, and treating an edge as consumed by its first wall lost the rest, because one long bedroom wall faces a bathroom, then a corridor, then another bedroom over disjoint spans.</p>
  </div>
  <div class="stats">
    <div class="stat"><div class="v">193.4 mm</div><div class="k">Median interior wall</div></div>
    <div class="stat"><div class="v">193.7 mm</div><div class="k">Pipeline's own mode</div></div>
    <div class="stat"><div class="v">19</div><div class="k">Room pairs joined</div></div>
    <div class="stat"><div class="v">21</div><div class="k">Openings on walls</div></div>
  </div>
  <div class="col">
    <p>That thickness agreement is worth pausing on. The wall graph never sees the thickness statistics &mdash; it derives 193.4&nbsp;mm from the gap between two independently snapped room boundaries, and the reconstruction pipeline had separately measured the flat's dominant partition at 193.7&nbsp;mm. Two different routes, 0.3&nbsp;mm apart.</p>
    <p>Detecting openings on the walls instead of on room edges then measures each wall once, on its own centre-line, with its own thickness &mdash; and every opening knows which room lies on each side by construction. Doors went from 11 to 16, a real 2,930&nbsp;mm window appeared at a 1,040&nbsp;mm sill and 1,220&nbsp;mm height (the drawing says 1,200), and the bedroom-to-balcony opening that was missing came back at 1,452&nbsp;mm.</p>
    <p>Still open: 42 of 62 wall segments remain unpaired and are treated as exterior, so only 6 openings currently resolve to two named rooms. Those unpaired segments are where the remaining connectivity gaps and the last floor fragments live &mdash; the slab is now one 74.6&nbsp;m&sup2; piece plus five rooms still adrift.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Where it currently stands</h2>
    <p class="lede">Two defects sit upstream of everything else, and they are worth fixing before anything further is built on top.</p>

    <h3>Two of the fifteen rooms are not rooms</h3>
    <p>Polygons 9 and 10 have an IoU of <strong>0.93</strong> and identical centroids &mdash; the same room emitted twice, which is why one label prints on top of another in the overlay. Polygon 8 sits <strong>95.7%</strong> inside polygon 1. CAGE's own duplicate filter should have removed both.</p>
    <p>So the real room count is <strong>13, not 15</strong>. The earlier headline that fifteen predicted rooms met fifteen named spaces was partly coincidence: two predictions are junk and two real spaces are missing. It also means the adjacency check was reporting 9&harr;10 as a room pair sharing 6&nbsp;m&sup2; when that is one room counted twice.</p>

    <h3>Thirteen percent of the floor belongs to no room</h3>
    <p><strong>14.1&nbsp;m&sup2;</strong> of free floor is unassigned, in two pieces: a <strong>9.1&nbsp;m&sup2;</strong> strip down the left side, which is an entire balcony never detected &mdash; the plan calls it Balcony&nbsp;2 at 7.87&nbsp;m&sup2; &mdash; and a <strong>2.3&nbsp;m&sup2;</strong> corridor between the top row of rooms and the living room.</p>
    <p>To be exact about what is disconnected: no <em>room</em> is. Bridged by 350&nbsp;mm, all fifteen polygons form a single connected group. What is disconnected is that left strip, which is floor with nothing claiming it &mdash; which is why it reads as a hole in the plan and why the floor slab fragmented there.</p>
  </div>
  <figure>
    <img src="__COVERAGE__" alt="Density map with room polygons; duplicates outlined in red and unassigned floor filled magenta">
    <figcaption>Magenta is free floor no room claims; red outlines the two duplicate polygons. Both defects are upstream of the wall graph, so fixing them there corrects the model, the openings and the connectivity at once.</figcaption>
  </figure>
  <div class="col">
    <p>The two fixes are small. Drop any polygon exceeding 0.5 IoU with another or sitting 90% inside one &mdash; that removes the two duplicates for nothing. Then promote the unassigned free-floor regions to rooms, which is exactly the missed balcony and the corridor, taking the count to thirteen real rooms plus two recovered spaces.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Both defects fixed, and the chain re-run</h2>
    <p class="lede">Dropping the duplicates and promoting the unclaimed floor took ten minutes and improved every stage downstream.</p>
    <p>Polygon 10 went (IoU 0.93 against 9) and polygon 8 went (96% inside 1), leaving <strong>13 real rooms</strong>. The unclaimed free floor came back as two rooms: the <strong>8.73&nbsp;m&sup2;</strong> balcony down the left side and the <strong>2.31&nbsp;m&sup2;</strong> corridor above the living room. Fifteen rooms again &mdash; but fifteen real ones this time.</p>
  </div>
  <figure>
    <img src="__CLEANBA__" alt="Before and after: room polygons over the density map, with duplicates and unassigned floor marked">
    <figcaption>Floor coverage <strong>87.0% &rarr; 97.0%</strong>; unassigned floor <strong>14.1&nbsp;m&sup2; &rarr; 1.3&nbsp;m&sup2;</strong>. Recovered rooms are outlined in white.</figcaption>
  </figure>
  <div class="stats">
    <div class="stat"><div class="v">19 &rarr; 24</div><div class="k">Room pairs joined</div></div>
    <div class="stat"><div class="v">6 &rarr; 10</div><div class="k">Openings linking 2 rooms</div></div>
    <div class="stat"><div class="v">12 &rarr; 10</div><div class="k">Floor slab pieces</div></div>
    <div class="stat"><div class="v">4 &rarr; 5</div><div class="k">Balcony doors</div></div>
  </div>
  <div class="col">
    <p>Interior wall thickness held at <strong>193.4&nbsp;mm</strong> against the pipeline's own 193.7&nbsp;mm, from 25 interior walls now rather than 20 &mdash; the agreement is not an artefact of which walls happened to pair.</p>
  </div>
  <figure>
    <img src="__V3MODEL__" alt="Four renders of the rebuilt model from the cleaned rooms">
    <figcaption>Rebuilt from the cleaned rooms: 48 full-height and 17 partial-height wall segments, 20 openings cut, 129.06&nbsp;m&sup2; net internal area. <span class="mono">v2_solid.glb</span>, <span class="mono">v2_marked.glb</span>, renders <span class="mono">v3_*</span>.</figcaption>
  </figure>
  <div class="col">
    <p>What is left is honest and small: 1.3&nbsp;m&sup2; of floor still unassigned in slivers, and the slab in 10 pieces rather than one. Those remaining pieces are rooms whose connecting wall still failed to pair, which is now the single limiting factor rather than one of four.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>Can depth models measure? No.</h2>
    <p>You asked whether depth is what we're missing, so I tested it rather than argued it. The Mujammel scan carries 16-bit RGB per point, so rendering it from a virtual camera gives a photo-like image <em>and</em> the exact metric depth that produced it &mdash; ground truth with no registration, no scale ambiguity and no annotation.</p>
    <p>Depth-Anything-V2 Metric-Indoor-Large, across 24 views:</p>
  </div>
  <div class="stats">
    <div class="stat bad"><div class="v">0.266</div><div class="k">Median AbsRel</div></div>
    <div class="stat bad"><div class="v">1,034 mm</div><div class="k">Median MAE</div></div>
    <div class="stat bad"><div class="v">1,249 mm</div><div class="k">Median RMSE</div></div>
    <div class="stat bad"><div class="v">387 mm</div><div class="k">MAE, scale fitted</div></div>
    <div class="stat"><div class="v">0.31</div><div class="k">Median &delta; &lt; 1.25</div></div>
  </div>
  <figure>
    <img src="__DEPTH_CMP__" alt="Four panels: colour render, ground truth depth, predicted depth, error map">
    <figcaption>Colour render &middot; LiDAR ground-truth depth &middot; predicted depth &middot; absolute error clipped at 1&nbsp;m. The model reads the <em>shape</em> of the room correctly and the <em>distance</em> wrongly.</figcaption>
  </figure>
  <figure>
    <img src="__DEPTH_CMP2__" alt="A second set of four depth comparison panels">
    <figcaption>A second view. The error map is close to uniform rather than structured, which is the tell: the failure is global scale, not local geometry.</figcaption>
  </figure>
  <div class="col">
    <p>Two honest caveats, both pointing the same way. Point-cloud renders are not photographs &mdash; speckle and missing exposure make this pessimistic against published benchmarks. And that last figure is the generous case: fitting an optimal scale and shift per image, which you could never do at inference without already knowing the answer, still leaves 387&nbsp;mm.</p>
    <p>Take the best published indoor figure instead of ours &mdash; roughly 5% AbsRel, so about 150&nbsp;mm at 3&nbsp;m &mdash; and it is still three orders of magnitude above the sub-millimetre wall placement the metrology stage already achieves. Depth models cannot contribute to measurement. What they can contribute is semantics and completion, in the regions LiDAR never saw.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>One more thing about our data</h2>
    <p>The Koushik export contains <strong>no colour at all</strong> &mdash; all three RGB channels are zero across 23.2 million points, leaving only intensity. Every RGB-conditioned method is therefore inapplicable to it as exported: BIMScript's lifted-feature encoder, LiteReality, MultiFloor3D and monocular depth alike.</p>
    <p>Mujammel has full 16-bit colour. Soulace has fisheye video with calibration. If image-conditioned methods are going to be part of the pipeline, the Koushik scan needs re-exporting with colour, or it stays a geometry-only dataset.</p>
  </div>
</section>

<section>
  <div class="col">
    <h2>What I'd actually build</h2>
    <p>The result confirms the split I proposed before running any of it, and now there are numbers behind it: <strong>learned models decide topology, our metrology decides geometry.</strong></p>
    <ul>
      <li><strong>Take CAGE's room graph and discard its coordinates &mdash; done, and it holds up.</strong> Median room-dimension error against the tape survey drops from 90 mm to 12 mm on the same polygons. The wall-pair face selection is the part that matters; nearest-face snapping is worse than useless because it fails by exactly one wall thickness.</li>
      <li><strong>Treat the open-plan merge as a signal, not a bug.</strong> CAGE returns the rooms the building actually has. Reconciling that against the four names on the plan is a deviation-detection output in its own right.</li>
      <li><strong>Add the p95 count clip to the density path</strong> wherever we rasterise a scan for any learned model. It is one line, and it decided every result on this page.</li>
      <li><strong>Skip height banding</strong> for pretrained models. Tested, worse, documented.</li>
      <li><strong>Stop using room IoU as the pipeline metric.</strong> It moved 0.007 across a change that cut dimension error sevenfold. Score against the survey in millimetres instead.</li>
      <li><strong>Keep depth models away from measurement.</strong> Their place is occlusion filling and semantics, on the scans that have colour.</li>
    </ul>
  </div>
</section>

<footer class="col">
  <p><strong>Everything is in <span class="path">C:&bsol;Users&bsol;PC&bsol;Documents&bsol;pointcloud-mesh&bsol;cage_output</span></strong> (22&nbsp;MB).</p>
  <p><span class="mono">density/</span> and <span class="mono">slices/</span> the input maps, flat and per height band &middot;
     <span class="mono">overlays/</span> predictions on their own density map &middot;
     <span class="mono">compare/</span> against the architect plan, plus <span class="mono">coverage_check.png</span> &middot;
     <span class="mono">scores/</span> per-run JSON and the ranking table &middot;
     <span class="mono">polygons/</span> raw predicted polygons &middot;
     <span class="mono">snapped/</span> metric rooms with per-edge provenance and the survey comparison &middot;
     <span class="mono">openings/</span> wall graph, wall tops, leaves, openings and their validation &middot;
     <span class="mono">model3d/</span> the GLB/OBJ models and renders &mdash; <span class="mono">v2_*</span> are current &middot;
     <span class="mono">depth/</span> depth scores and comparison panels &middot;
     <span class="mono">scripts/</span> 26 files, everything needed to reproduce.</p>
  <p>The CAGE checkout and its 3&nbsp;GB of checkpoints stay in the session scratchpad; paths are at the top of <span class="mono">scripts/infer.py</span>.</p>
</footer>
</div>
"""

body = BODY
for key, token in [("density_input", "__DENSITY_INPUT__"),
                   ("density_skeleton", "__DENSITY_SKELETON__"),
                   ("best_overlay", "__BEST_OVERLAY__"),
                   ("vs_gt_swin", "__VS_GT_SWIN__"),
                   ("vs_gt_r50", "__VS_GT_R50__"),
                   ("depth_cmp", "__DEPTH_CMP__"),
                   ("depth_cmp2", "__DEPTH_CMP2__"),
                   ("snap_summary", "__SNAP_SUMMARY__"),
                   ("model3d", "__MODEL3D__"),
                   ("openings3d", "__OPENINGS3D__"),
                   ("v2model", "__V2MODEL__"),
                   ("coverage", "__COVERAGE__"),
                   ("clean_ba", "__CLEANBA__"),
                   ("v3model", "__V3MODEL__")]:
    body = body.replace(token, imgs[key])
body = body.replace("__ROWS__", rows).replace("__ROOM_ROWS__", room_rows)

head = (
    "<title>Scan to Floorplan Trials</title>\n"
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&"
    "family=IBM+Plex+Mono:wght@400;500;600&"
    'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">\n'
    "<style>" + CSS + "</style>\n")

out = HERE / "report.html"
out.write_text(head + body, encoding="utf-8")
print("wrote {}  {:.0f} KB".format(out, len(head + body) / 1024))
print("depth medians: AbsRel {:.3f}  MAE {:.0f}mm  RMSE {:.0f}mm  aligned {:.0f}mm  d1 {:.2f}".format(
    med("absrel"), med("mae_mm"), med("rmse_mm"), med("mae_aligned_mm"), med("delta1")))
