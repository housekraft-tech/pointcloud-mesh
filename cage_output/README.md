# CAGE + monocular-depth evaluation on our scans

Folder: `C:\Users\PC\Documents\pointcloud-mesh\cage_output`

Zero-shot test of **CAGE** (NeurIPS 2025, Structured3D-pretrained) on the Koushik
handheld-LiDAR scan, plus a metric-depth feasibility test on Mujammel.
Nothing here was retrained -- these are the released checkpoints, run as-is.

## What was run

* `scripts/make_density*.py` -- Structured3D-convention 256x256 top-down density
  maps from `koushikexport.las`, via the existing `recon.isolate` storey
  isolation. Variants differ only in footprint robustness, point-count clipping
  and height banding.
* `scripts/infer.py` -- CAGE forward pass on **CPU on Windows**. The repo wants
  Linux + CUDA 11.1 + two compiled extensions; neither is needed for inference
  (`diff_ras` is a training loss, and the deformable-attention op has a
  pure-PyTorch reference implementation in-tree). Both released checkpoints
  load with **0 missing keys**.
* `scripts/eval_compare.py` -- scores predictions against `data/bbox_data.json`
  (the architect plan, 15 named spaces). Scale is never fitted; only a rigid
  placement is searched, pinned to one orientation across all variants.
* `scripts/render_views.py` + `scripts/depth_test.py` -- Depth-Anything-V2
  Metric-Indoor-Large against exact LiDAR depth on 24 rendered views.

## Headline numbers

| run | rooms | area err | footprint IoU | matched @IoU0.5 |
|---|---|---|---|---|
| `robust_c95_swin` | 15 | +9.1% | **0.796** | 7/15 |
| `robust_c90_r50`  | 19 | -2.2% | 0.593 | **8/15** |

Best run (`robust_c95_swin`) per-room IoU: Bedroom 3 **0.955**, Bedroom
**0.934**, Bathroom 2 **0.916**, Bathroom 3 **0.904**.

The apparent misses are mostly not errors. CAGE returns the open-plan centre as
one 40.37 m2 room; that polygon contains 98.4% of Living, 99.4% of Dining,
97.7% of Foyer and 78% of Kitchen -- 39.31 m2 of plan area, so it is within
**+2.7%** of what it actually encloses. The architect plan names four spaces
where the building has no separating walls.

## Findings

1. **Count clipping is the whole domain gap.** Structured3D density maps come
   from panorama depth (near-uniform sampling); handheld LiDAR dwell time makes
   some pixels 25x denser, leaving the floorplan at ~4% brightness. Clipping
   counts at the 95th percentile moves the result from 8 rooms to a complete
   15-room tiling. This is one line of preprocessing.
2. **Height banding does not help.** Counting only points 1.0-2.2 m above the
   floor gives a beautiful wall skeleton and *worse* results (best 5/15) -- the
   model was trained on filled-interior maps, so a skeleton is out of
   distribution. Negative result, worth not repeating.
3. **Resolution is the accuracy ceiling.** The density map is 62 mm/px. CAGE
   cannot express a wall position more precisely than that, which is ~2 orders
   of magnitude coarser than our metrology stage.
4. **Monocular metric depth cannot measure.** On real-colour renders with exact
   LiDAR ground truth: median AbsRel **0.266**, MAE **1034 mm**, and even after
   fitting an optimal per-image scale and shift the residual is **387 mm**.
   Renders are not photographs, so this is pessimistic -- but published
   best-case indoor AbsRel (~5%, ~150 mm at 3 m) is still ~1000x our bar.
5. **The Koushik export has no RGB at all** (all three channels zero; intensity
   only). Every RGB-conditioned method is inapplicable to it as exported.
   Mujammel has 16-bit RGB; Soulace has fisheye video + calibration.

## Contents

* `density/` -- the 256x256 input maps, one per variant
* `overlays/` -- predicted polygons drawn on the density map they came from
* `compare/` -- aligned prediction vs architect plan, `vs_gt_r50.png`, `vs_gt_swin.png`
* `scores/` -- per-run JSON + `summary.json` ranking table + `variants.json`
* `polygons/` -- raw predicted polygons (density-image pixels)
* `depth/` -- depth scores and rgb|gt|pred|error comparison panels
* `scripts/` -- everything needed to reproduce, self-contained

## Reproducing

CAGE checkout, checkpoints (Google Drive, ~3 GB) and the Swin ImageNet weight
live in the session scratchpad, not here. `scripts/infer.py` has the paths at
the top.
