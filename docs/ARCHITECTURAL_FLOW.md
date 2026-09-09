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

## Revising a checked native model

The checked `.skp` is never edited in place. Bounded fixes are computed against
`reopened_visible.build.json` (what the native file actually shows), written as
fragments, and appended by `assemble_revision.py` as new groups while every
replaced group is hidden as a reference. Each step keeps its own audit.

```powershell
# raw returns, one per 10 mm voxel, cached once per property (manifest or a
# {"scans": [...]} support list in the common frame)
.\venv311\Scripts\python.exe scripts\export\raw_evidence.py --spec <manifest> --model <reopened_visible.build.json> --cache <work>\evidence_10mm.npz

# read-only: per-plane islands, pores, duplicate and cross-group overlapping
# faces; independently detected vertical scan planes >50 mm from every visible
# wall, compared level-aware with the nearest parallel wall plane
.\venv311\Scripts\python.exe scripts\export\wall_surface_review.py --model <reopened> --manifest <manifest> --out <work>\review --label "..."

# floor-to-wall junctions: seat floating bases (<=15 mm) and extend floors to
# wall bases (<=150 mm) only where the strip is within 50 mm of raw returns
.\venv311\Scripts\python.exe scripts\export\floor_wall_junctions.py --model <reopened> --out <work>\junctions --fix --evidence-spec <manifest> --evidence-cache <work>\evidence_10mm.npz --working-out <work>\working_after_junctions.build.json

# wall faces: exact plane union, cross-group overlap assigned to the larger
# plane, islands <0.02 m2 removed only when no previously represented scan
# sample is lost, pores <0.005 m2 filled only inside the 50 mm raw envelope
.\venv311\Scripts\python.exe scripts\export\wall_surface_cleanup.py --model <work>\working_after_junctions.build.json --out <work>\cleanup --evidence-spec <manifest> --evidence-cache <work>\evidence_10mm.npz

# run the junction pass once more on the cleaned model: removing a sub-0.02 m2
# base strip can re-open a seated base, so seating is always the last edit
.env311\Scripts\python.exe scripts\exportloor_wall_junctions.py --model <work>\working_after_cleanup.build.json --out <work>\junctions2 --fix --evidence-spec <manifest> --evidence-cache <work>\evidence_10mm.npz

# observed faces: audit regions that are probable opposite faces or have no
# parallel visible plane, >=60 % unmatched, >=1 m2, planar within 15 mm and
# inside the unit footprint, bounded to the 50 mm raw envelope
.\venv311\Scripts\python.exe scripts\export\observed_wall_faces.py --model <work>\working_after_junctions.build.json --audit <work>\review\wall_completeness_audit.json --out <work>\observed --evidence-spec <manifest> --evidence-cache <work>\evidence_10mm.npz

.\venv311\Scripts\python.exe scripts\export\assemble_revision.py --previous <checked folder> --fragment <work>\junctions\junction_fix.build.json --fragment <work>\cleanup\wall_cleanup.build.json --fragment <work>\junctions2\junction_fix.build.json --fragment <work>\observed\observed_faces.build.json --out <work> --native-name <name>.skp --label "..." --note "..."
```

`polygon_hygiene.py` prepares every exported loop: a 0.15 mm morphological
opening with a slightly smaller dilation, so no loop repeats a vertex or comes
within SketchUp's 0.025 mm merge tolerance of another loop, and the exact
cleaned rings travel to the exporter as `planar_loops` instead of being
re-derived from a triangulation. Observed faces (`wall_face_observed`, amber)
are measured surfaces with unverified identity: they add no back face,
thickness or opening. Rerun `wall_surface_review.py` and
`floor_wall_junctions.py` (without `--fix`) on the new
`reopened_visible.build.json` to confirm the saved file.

## Exterior face refinement

Outer faces are the ones a viewer sees from the street, so they are held to a
different standard from interior faces: scan shadows from cars, plants and
railings are closed and outlines are straightened, while interior faces stay
exactly as measured. Which faces are exterior is decided in the raw returns,
not in the reconstruction, so the same command runs on any property:

```powershell
.\venv311\Scripts\python.exe scripts\export\run_exterior_refinement.py --previous <checked folder> --evidence-cache <work>\evidence_10mm.npz --out <work> --native-name <name>.skp --label "..."
```

- `select_exterior_walls.py`: for every near-exact vertical plane of every
  wall or parapet part, a grid of in-plane samples marches away from the plane
  on both sides through a 50 mm occupancy grid of the raw returns (solid at
  >= 3 returns). Along the free path, from 0.35 m (past any wall thickness) to
  4 m, a second march goes straight up; leaving the scanned volume without a
  return means the sample looks at open air. Covered rooms always carry their
  ceiling in the scan, terraces, compounds and streets never do, and neither
  floor completeness nor the model's ceilings enter the decision. A plane with
  >= 60 % open samples is an `exterior_face`; 20-60 % is a
  `partly_exterior_face` (a wall that leaves the house and continues as a
  garden wall) and carries the in-plane mask of its open samples; the rest are
  `interior_face` and are never modified. `selection_*.png` colour every wall
  part by its strongest class from four outside viewpoints.
- `regularize_wall_surfaces.py`: only the selected planes (only inside the
  mask, plus 0.3 m, for partly exterior planes) are regularized, and every
  plane reports what was measured and what was inferred:
  - recovery (measured): inside the rectangle of the plane's main pieces every
    25 mm cell with a raw return within 50 mm of the plane is restored
    (`recovered_measured_area_m2`), surface that earlier cleanup thresholds
    dropped, never a bridge across nothing;
  - outline notches up to 0.15 m deep are closed (`inferred_outline_area_m2`);
  - a hole completely surrounded by the plane is filled when it is not
    rectangular (area / bounding box < 0.85) and no larger than a scan shadow
    (1 m2, 1.5 m) (`filled_gap_area_m2`); rectangular holes wider than 40 mm
    are doors, windows, vents and service openings, kept open and re-cut as
    straight rectangles; larger irregular voids are not invented;
  - islands under 0.02 m2 are dropped (`dropped_islands_m2`);
  - every ring is snapped to axis-aligned edges when no vertex moves more than
    50 mm (`snapped_rings`); rings that would need more stay as measured
    (`unsnapped_rings`).
- `recover_recessed_faces.py` (`--recover-recessed`): behind every void in
  an exterior plane's rectangle where the returns show a surface 50-300 mm
  off the plane and the model has no surface within 30 mm of those returns,
  the dominant offset is fitted, the covered 25 mm cells are meshed and the
  face is added as an amber observed vertical face with unverified identity:
  measured surface, no thickness, no opening, no back face. Voids whose
  surface the model already carries are listed as `already_in_model`.
- `complete_exterior_rectangles.py` (`--complete-rectangles`, declared
  inference for a drawing-clean exterior): every whole-plane exterior face is
  completed to its rectangle, horizontal extent by the level datums (floor
  below, floor above, own top on the last level). Cells with returns on the
  plane are measured fill; cells where the returns or a parallel model face
  show another surface 50-350 mm away stay open (recesses, sunshades, the face
  behind an opening); rectangular openings stay open; cells where the scan
  shows nothing are filled and reported as `inferred_fill_m2`. The evidence
  status of these parts ends in `INFERRED`.
- `rectilinear_exterior.py` (`--rectilinear`, declared inference): the
  user's "only sharp 90-degree corners" standard. Every exterior ring is
  snapped to axis-aligned edges with a rising tolerance (50, 100, 200,
  350 mm) or replaced by its bounding rectangle; holes survive only as
  rectangles (straightened openings, boxed recess voids); pieces under 0.1 m2
  are dropped; parallel planes within 15 mm on one level are merged into one
  block on their mean plane, so one wall run is one face. Blocks are named
  `L<n> exterior block <k> - rectilinear - INFERRED outline` and list every
  part they merged; `rectilinear/audit.json` records boxed rings, dropped
  area and the offset spread absorbed by each merge.
- `hide_duplicate_faces.py` (`--hide-duplicates`): native faces lying within
  30 mm of a rebuilt plane over 70 % of their area are hidden as references,
  which removes the doubled outlines in previews and in SketchUp.
- `assemble_revision.py` appends the replacements and hides the replaced
  groups; `scope_verification.json` proves every unselected native group read
  back with the same face count, area and visibility, and `review_exterior_*.png`
  are rendered from the reopened geometry.
