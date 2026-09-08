# Evidence-constrained LiDAR to SketchUp

This flow separates architectural hypotheses from supported surfaces. It is
property-independent, but **not a certified automatic as-built survey**. Nearness
to a return cannot prove that the return belongs to a wall instead of a door,
furniture, a stair riser, or clutter.

## Stages

1. `scripts/isolidarflow.py`: isolate the unit/storey, determine its rigid frame,
   detect wall/column/beam/opening hypotheses, and make room-owned geometry.
2. `scripts/export/run_architectural_flow.py`: clip fresh hypotheses to registered
   LAS evidence, regularize small supported planar irregularities, preserve level
   datums, and audit both scan coverage and model support.
3. Native SketchUp export: import faces, remove internal coplanar triangulation,
   retain source references when updating a model, save a new `.skp`, and verify
   native group identities and surface areas. Capture top/front/side/3D views.

The upstream GLB is a candidate model, **not** the final evidence-constrained model.

## New isolated unit or storey

Run from the repository root in PowerShell, using new output folders:

```powershell
.\venv311\Scripts\python.exe scripts\isolidarflow.py input.las output_final\example_detection --debug-png --plane-max-points 4000000
.\venv311\Scripts\python.exe scripts\export\run_architectural_flow.py --manifest output_final\example_detection\architectural.manifest.json --out output_final\example_supported
.\venv311\Scripts\python.exe scripts\export\run_architectural_flow.py --out output_final\example_supported --export-only --native-name Example_supported.skp
```

The last command defaults to the installed official SketchUp 2025 native C API
(`native_sdk_export.py`). It reads/writes `.skp` directly without controlling the
open application, verifies saved group geometry and reference visibility, and
creates a fresh document if there is no source-native entry in the manifest.
Use `--native-backend bridge` for the older interactive Ruby export. That route
preserves a modified active model with `save_copy` first, and may require the
purge-on-save dialog dismissed. Neither route overwrites a prior handover.

The raw detection entry handles an isolated unit/primary storey. A complete
multi-storey scan must first have its levels identified and registered. Repeat
`--manifest` for those levels; this command does not silently assume floor heights
or discover every staircase. An explicitly declared `model_to_building` rigid
matrix controls placement. Parking must have its own actual datum; do not lift
the ground to eliminate a genuine level difference.

## Existing registered reconstruction

The manifest declares:

- `label`, `units: "metres"`, and a unique integer `level`.
- `candidate_model`: JSON with named `parts`, each containing `kind`, metre-space
  `v` coordinates, and zero-based triangular `f` indices.
- `scans`: uniquely named raw LAS sources, each with `scan_to_model` 4×4 matrix.
- Optional `current_model`: an audited `bounded_lidar_baseline` surface product.
  Arbitrary wall solids cannot use this bypass; omit it to clip candidates first.
- Optional `model_to_building`: rigid placement; defaults to identity.
- Optional `wall_metadata`: thickness priors for matching face labels, never proof
  of an unobserved back face or verified wall thickness.
- Optional `source_native` and `source_native_audit`: one common prior building
  model, for append-and-hide-reference updates instead of a new document.

Relative file paths resolve against the manifest directory. Transformations may
not scale, shear or reflect the scan. Multiple scans of the same property belong
in the same level manifest, not duplicate level entries.

The checked flat configuration is
`configs/architectural_flow/koushik_mujammel.json`; Koushik/Mujammel are the flat
also referred to as Engrance, not separate properties in this example.

## Anti-invention rules

- No complete candidate wall is restored just because most of it has overlap.
  Fresh wall faces are clipped to a conservative continuous 50 mm raw-return
  envelope; missing backs remain open surfaces.
- Local cleanup is limited to 15 mm boundary displacement, with independent
  surface and fine boundary checks at 50 mm. Larger voids remain. These cleanup
  checks are sampled, not the continuous bound used by initial clipping.
- Exact coplanar floor union preserves existing footprints and holes. It does
  not unite floors at different heights or add an unobserved foundation.
- The floor cleanup stage removes duplicate coplanar faces and zero-area
  triangles. Small enclosed pores up to 0.005 m² are filled only when the entire
  pore passes the conservative 50 mm raw-return envelope; unsupported holes stay.
- A wall/stair interference diagnostic flags candidates occupying the walking
  area. Collision alone is not permission to remove a real wall: check its scan
  support and semantic identity first.
- New missing rooms, walls and openings are not generated by this refinement
  stage. Geometry not identified upstream remains unresolved, with uncovered
  scan points shown for review.

Soulace's lower stair was fitted to independent measured tread/riser bands by
`refine_measured_stair_flights.py`; its lane and endpoint selection is still an
explicit case configuration. The fitting primitive is reusable, but automatic
stair-lane discovery is not yet implemented. Parking junction recovery similarly
uses declared inspected regions, not a building-wide guessed plinth.

## Reading the QA

Each `L*/audit.json` records input registration, source identity, processed return
counts, local decisions and bidirectional distance statistics. Refinement streams
all LAS returns and retains one original point per 20 mm occupied voxel for its
working evidence set. This is **not** a full-density distance test: thinning can
increase reported distances. Initial clipping uses full-density cropped LAS.

- Model → scan measures geometric support for proposed surfaces.
- Scan → model reports missing coverage; unmatched returns can include furniture,
  door leaves and noise, not only missing architecture.
- A separately reported count identifies previously represented scan samples
  lost after cleanup. Zero loss is not proof the original model was complete.
- The review contains top/front/side scan diagnostics and a 3D geometry preview.
  `native_*.png` files are actual SketchUp renders, not generated illustrations.
  With the SDK route, `checked_*.png` previews are CPU depth-buffer renders of
  geometry read through the native API, **not screenshots of the SketchUp UI**.

10 mm remains the target and 20–50 mm the agreed fallback, but neither SLAM
registration nor room dimensions are independently certified by these checks.
Do not label nominal thickness or point-cloud fit as ±10 mm site accuracy.
