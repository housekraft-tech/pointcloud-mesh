# Rectilinear Scene Rebuild — Design

**Date:** 2026-08-11
**Branch:** `rectilinear-scene-rebuild`
**Status:** Approved design, pending implementation plan

## 1. Purpose

Rebuild the SLAM2000 LAS → floorplan + modular 3D pipeline from scratch. The
consumer is an **interior designer** working on a **bare-shell** (unfurnished,
un-plastered) apartment. The deliverable is a modular model in which every wall,
opening, column, beam and wall feature is its own named, dimensioned object, and
in which the building's real imperfections are preserved rather than idealised
away.

The architecture is adapted from
[LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent/): a
deterministic scene-init followed by a critique loop, with the scene held as a
legible program rather than as an opaque mesh. Two of LiteReality's inputs are
unavailable here (Apple RoomPlan's layout and posed RGB frames), so both roles
are replaced — the layout is solved from the cloud, and the critique compares
against the point cloud itself rather than against photographs.

## 2. What the current pipeline gets wrong

Confirmed with the user against the existing `isolidarflow` output:

1. **Regularisation flattens the truth.** Walls are snapped to a global Manhattan
   grid, runs are merged, corners resolved and endpoints snapped. Genuine
   rectangular steps, L-cut returns, grooves and electrical boxes are absorbed
   into their parent wall before they are ever measured. The result models an
   idealised apartment, not this one.
2. **Wrong things are detected as features.** Irregular concrete masses, formwork
   ridges and scan artefacts are promoted into walls, columns and steps, while
   real elements are missed or merged.
3. **Not actually modular.** Walls are fused, openings are not first-class
   objects, and beams/columns are not separable, so a designer cannot take one
   wall and work with it.

Dimensional accuracy of the *measurement* primitives was **not** reported as a
failure. The measurement layer is roughly sound; interpretation and
representation are what is broken. The rebuild therefore targets extraction,
representation and validation, and does not assume the raw metrology is wrong.

## 3. Governing insight

A bare-shell concrete building is **rectilinear by construction**. Formwork
produces flat faces meeting at sharp edges. A 75 mm bump is a rectangular prism
with square edges, not an organic blob.

Two consequences drive the whole design:

- **"Clean" and "truthful" are not in tension.** Clean rectilinear geometry *is*
  the truth. The existing pipeline's error is not that it idealises to flat
  planes — it is that it snaps each surface to a *global* frame shared with other
  surfaces, which destroys genuine local steps.
- **Rectangularity is a free, deterministic junk filter.** A real formwork
  feature fits a rectangular prism with planar faces. An irregular concrete mass
  cannot. This is the discriminator between "extrusion I must design around" and
  "lump of leftover concrete", with no heuristics about size or position.

Therefore the two invariants of the rebuild:

> **Fit every surface only to its own points. Never snap one surface to another.**
>
> **Compute edges by intersecting fitted planes. Never mesh them from points.**

## 4. Architecture

Extraction by **plane arrangement**, representation as a **declarative CSG scene
document**, with **free-space** used in one strictly bounded role.

### 4.1 Source of truth: the scene document

The scene is a **JSON document, schema-validated** — not executable code.
LiteReality uses Python-as-truth because an LLM edits it; here the LLM never
touches geometry, so code-as-truth would cost determinism and diffability
without buying anything. A readable `room.py` is *emitted from* the document for
humans and Blender. The document is what is versioned, diffed and validated.

```
Scene
├─ provenance   scan path + hash, pipeline version, config echo, timestamp
├─ frame        gravity-derived Z, reported XY rotation, storey levels
├─ parts        Slab | Wall | Column | Beam | Opening | Room
├─ unmodeled    quarantined point clusters that failed rectangularity
└─ diagnostics  residual field statistics
```

Parts are declared as **intersections of half-spaces** derived from locally
fitted planes, plus a list of subtracted boxes. Meshing is purely an export
concern, so the core never needs a CSG library.

### 4.2 Part model

- **Wall** — two independently fitted faces. Thickness is the *measured*
  face-to-face distance, never a centreline plus an assumed thickness. Each face
  owns its child features.
- **Feature** (child of a wall face) — `Extrusion`, `Intrusion`, `Groove`,
  `LCut`, `ElectricalPoint`. Each is a rectangular prism with measured `u`/`z`
  extent and depth, relative to its parent face.
- **Opening** — belongs to a wall; typed `door`, `window`, `balcony_door`,
  `pass_through`. Head, sill and jamb positions are measured from the reveal
  surfaces themselves.
- **Column** / **Beam** — prismatic parts assembled from their own face sets.
- **Slab** — floor and ceiling planes with boundary polygons.
- **Room** — a bounded volume with its polygon, height and bounding-face
  references.

Illustrative wall record:

```
Wall_03
  plane: fitted to own points, tilt 0.4° (measured, not zeroed)
  length 4.182 m   height 2.748 m   thickness 0.2031 m
  └ features
     Groove_01     u=1.20→1.26  z=0.00→2.75  depth −0.012
     Extrusion_01  u=2.80→3.15  z=0.00→2.75  depth +0.075
     ElecBox_01    u=3.60→3.68  z=1.20→1.28  depth −0.045
     LCut_01       corner return, 0.35 × 0.20
  residual after features: p95 = 3 mm
```

### 4.3 Two properties built in from the start

**Every dimension carries provenance and uncertainty.** Not `thickness: 0.203`
but:

```json
{"value": 0.2031, "method": "face-to-face raw points",
 "n_points": 18422, "p95_residual": 0.0021}
```

A number a designer cannot audit is a number they will eventually stop trusting,
and provenance makes regressions visible instead of silent.

**Manhattan is measured, never enforced.** No snapping stage exists in this
architecture. The scene reports each wall's deviation from the dominant frame as
data. This makes the `ff03819` philosophy structural rather than one stage's good
behaviour.

### 4.4 Bounded role for free-space

Free-space carving contributes exactly two things:

1. **Interior direction** — which side of a surface is inside. Geometry alone is
   unreliable for balconies, external walls and any surface scanned from one
   side only.
2. **Opening validation** — confirming a void is a real hole rather than a data
   gap.

It does **not** define geometry. Its voxel resolution would cap accuracy and its
edges are stair-stepped; both are unacceptable for the primary model.

#### 4.4.1 Amendment (2026-08-13): the mechanism is flood-fill, not trajectory

This section originally specified the **walk trajectory** as the mechanism.
Measurement on real scans retired that choice:

- Reconstructing the path by binning points into `gps_time` slices and taking
  each slice's centroid does not recover the walk. A 360° scanner sees the whole
  room at once, so a time-slice centroid is the centre of what was *seen*, not
  where the scanner *stood*. The recovered path is a zigzag with no physical
  meaning.
- No sensor geometry is available to do better: `scan_angle_rank`,
  `point_source_id`, `return_number` and `number_of_returns` are all identically
  zero in both reference exports. Per-point data is XYZ, `gps_time`, intensity
  and (on one scan) RGB.

Neither of the two jobs above actually needs a path. Both are satisfied by
**occupancy flood-fill**: voxelise the cloud, seed the fill in air above the
floor patch's centroid, and interior is whatever the fill reaches. A face's
inside is the side adjacent to filled voxels; an opening is a void the fill
passes through.

A trajectory exported from the vendor's post-processing software remains a
welcome *upgrade* — it is independent evidence of which openings were walked
through, which is how a door is distinguished from a window — but it is no
longer a dependency. `trajectory.load_trajectory` already parses such a file.

#### 4.4.2 Second amendment (2026-08-13): enclosure, not flood-fill

§4.4.1 replaced the trajectory with an occupancy flood-fill seeded in air above
the floor. Measurement retired that too.

A flood-fill defines interior as "reachable from an interior seed without
crossing occupancy". That requires a **watertight** occupancy envelope, and a
real scan never has one: a single missing voxel — from a scan hole, an
occlusion, or glazing that returns nothing — lets the fill escape into the
grid's outer padding, which wraps the entire model. Measured on
`isolated_structural_v2.las` at 50 mm cells:

| | flood-fill | enclosure |
|---|---|---|
| one-room crop | 100.00% of free cells filled | 12.3% |
| full isolated scan | 100.00% filled | 40.8% |
| fill/enclosure touching a grid boundary face | all six | none |

100% is not an interior; it is the whole bounding box. The full scan leaks as
readily as the crop, so this is not an artefact of cropping.

The mechanism is therefore **horizontal enclosure**: a free cell is interior
when rays along −x, +x, −y and +y all strike occupancy within its own Z slice.
This is a *local* test. It needs no watertight envelope, is unaffected by any
number of leaks, costs one cumulative sum per axis, and is deterministic. A
cell inside a room is walled on all four sides; a cell outside the building
escapes in at least one direction; a cell on a balcony correctly reads as
outside.

`build_occupancy` is unchanged and still correct. `flood_interior` is retained
because it is the right tool for a genuinely closed envelope (a synthetic
sealed room, or a scan whose holes have been repaired), but it is **not** the
default and must not be used to decide interior direction on real data.

The wider lesson, recorded because it has now happened twice: a mechanism that
is obviously correct on synthetic geometry can be unusable on a real scan, and
only the real scan will say so. Both retirements here — the trajectory and the
flood-fill — were caught by measuring, not by reasoning.

## 5. Pipeline

Thirteen stages, each a pure function with typed input and output.

| # | Stage | Responsibility |
|---|---|---|
| 1 | `ingest` | LAS → PointSet. Retains **all** available attributes (xyz, gps_time, intensity, rgb, and returns/scan-angle where present). Adapter boundary. |
| 2 | `level` | Gravity-derived Z; dominant XY direction **reported** as a frame, not applied as a snap. |
| 3 | `freespace` | Trajectory recovery and free-space carving, for §4.4 roles only. |
| 4 | `patches` | Normal-based region growing at native resolution with strict planarity **and** connectivity. |
| 5 | `graph` | Patch adjacency and intersection lines; coplanarity classes recorded, never merged. |
| 6 | `classify` | floor / ceiling / wall face / step face / opening reveal / beam soffit, from orientation, free-space side and Z band. |
| 7 | `assemble` | Group patches into Wall / Column / Beam; pair opposite faces for thickness; features as offsets from a parent face. |
| 8 | `rect_gate` | Fit each candidate feature to a rectangular prism; pass → `Feature`, fail → `unmodeled`. |
| 9 | `openings` | Voids in wall faces, validated against free-space passage; reveals measured. |
| 10 | `rooms` | Free-space connected components bounded by classified faces. |
| 11 | `solidify` | Declarative half-space CSG per part. Watertight by construction. |
| 12 | `critique` | Residual field; iterate 4–11 on flagged regions. |
| 13 | `emit` | `scene.json`, then all exporters. |

**Stage 4 is the heart of the rebuild.** Whole-cloud RANSAC is precisely what
fuses a 75 mm step into its parent wall — one dominant plane wins and swallows
its neighbours. Region growing with a connectivity requirement makes the step its
own patch by construction.

## 6. Critique loop

The residual field is the signed distance from every point to the nearest model
surface, checked in **both** directions:

- **Unexplained points** (`|d| > τ_fit`, clustered): the model owes an object
  here. Rectangular → add the part. Irregular and large → `unmodeled`. Small →
  noise.
- **Unsupported surfaces**: a model face whose point-support density is below
  threshold is a hallucination — flagged and removed.

Iteration tightens parameters on flagged regions only. Seeded and bounded by a
maximum round count, so a given scan and config converge to an identical scene
every run.

### 6.1 Tolerances

All thresholds live in one config, one commented line each.

| Symbol | Meaning | Initial |
|---|---|---|
| `τ_fit` | plane inlier distance | ≈3 mm |
| `τ_feature` | minimum depth to count as a feature; below this it is surface roughness, not an L-cut | ≈8 mm |
| `τ_actionable` | smallest unexplained cluster worth acting on | tuned in implementation |
| `max_rounds` | critique iteration bound | tuned in implementation |

Initial values are starting points to be calibrated against the golden synthetic
scenes (§8) and the reference scans; they are not claims of achieved accuracy.

### 6.2 Where the LLM is allowed

Geometry is entirely deterministic. The LLM has one optional, clearly bounded
job: **advisory labelling of things already measured.** Given a rendered view of
a part plus its measured dimensions, it may propose a semantic label — is this
prism a structural pillar or an unmodelled concrete mass, is this opening a
balcony door or a window.

Constraints, enforced structurally rather than by convention:

- It runs **after** `solidify`, on parts that already exist with fixed
  dimensions.
- Its output writes only to a part's `label` and `label_confidence` fields,
  tagged `source: "llm"`. The schema gives it no path to any coordinate.
- A run with the LLM disabled produces identical geometry and identical
  dimensions; only labels differ.

This keeps the deliverable reproducible while still getting help on the calls
where rules are genuinely brittle.

## 7. Failure policy

**Never silently drop, never silently invent.** Every input point ends in exactly
one of three buckets:

- `modeled` — explained by a part.
- `unmodeled` — measured but not classified. **Exported on its own layer** so the
  designer sees that something is physically there, even though the pipeline
  would not name it.
- `noise` — below the actionable threshold; counted in the report.

A stage that cannot do its job raises, and the run writes a partial scene plus a
report. A partial answer that says so is worth more than a confident wrong one.

## 8. Testing

**Golden synthetic scenes are the primary accuracy test.** Construct a known
apartment in code — including a 75 mm rectangular extrusion, a groove, an L-cut
return and a switch box — sample points on it with realistic scanner noise, run
the full pipeline, and assert recovered dimensions to tolerance. This makes
"accurate" a CI-checkable property rather than an opinion, and being pure numpy
it runs in the current local environment, which lacks `trimesh` and `laspy`.

Supporting layers:

- Unit tests per stage against hand-built inputs.
- Regression runs on real scans asserting scene invariants: part counts, room
  closure, residual p95, bucket totals.
- Determinism test: the same scan and config produce a byte-identical
  `scene.json`.

## 9. Exports

All derived from `scene.json`; none is a separate source of truth.

| Artefact | Consumer | Notes |
|---|---|---|
| `plan.dxf` | AutoCAD | Layered: walls, openings, columns, beams, electrical, features, dimensions, unmodeled. |
| `elev_<wall>.dxf` | AutoCAD | Per-wall elevations. High value for a bare shell — every electrical point, groove and step dimensioned on the wall the trade actually works on. |
| `model.obj` / `.fbx` | SketchUp, Rhino, 3ds Max | One named object per part, strict naming convention. |
| `model.glb` | Blender | Collections mirroring the part hierarchy. |
| `room.py` | humans, Blender | Readable emitted scene program. |
| `report.md` + PNGs | review | Residual diagnostics, bucket totals, per-stage counts. |

Naming convention: `W03`, `W03_EXT01`, `OP_D02`, `COL01`, `BM01`, `RM02`.

## 10. Repository layout

```
src/rscene/
  core/    pure numpy — patches, graph, fitting, features, residual, scene model
  io/      LAS adapter, scene JSON read/write
  export/  dxf, obj/fbx, glb, room.py
  cli.py
tests/
  golden/  synthetic scene builders and accuracy assertions
```

`core` stays dependency-light (numpy/scipy only) so the layer that needs the most
iteration remains runnable and unit-testable locally. LAS I/O and meshing sit
behind thin adapters.

The existing `scripts/` tree stays in place for reference and is deleted once the
new pipeline reaches parity. Git history preserves it either way.

## 11. Explicitly out of scope

- Furniture detection and removal — the subject is a bare shell.
- Materials, textures, PBR and articulation — LiteReality's focus, not this one.
- Learned 3D asset retrieval (TRELLIS, GroundingDINO) — no imagery available.
- Apple RoomPlan integration — evaluated separately; it is cm-grade and cannot
  touch geometry. May later contribute semantics only.
- IFC / Revit export — not among the designer's target tools.
- Any LLM in the geometry path. LLM use is confined to advisory semantic
  labelling, flagged as such in the manifest, and can never move a surface.
