# Rectangular reconstruction stage v1

Engrance, Koushik and Mujammel refer to the same example flat in this task.
The two LAS files are separate registered scans of it. Soulace is the other,
three-level property. Names are labels in input manifests, not algorithm rules.

## What is implemented

`scripts/export/rectangular_rebuild.py` accepts a candidate `.build.json` model
and one or more LAS/LAZ scans with explicit rigid scan-to-model transforms.
Coordinates must be metres in a common, gravity/rectilinear-aligned frame.
Oblique candidate faces are reported and excluded, not silently flattened.
This is a candidate-guided reconstruction/validation stage, not a new raw-scan
room detector. Existing candidate openings and extents remain hypotheses.

1. Associate area-spaced probes with full-resolution scan neighborhoods.
2. Gate local neighborhoods by surface-normal agreement and planarity.
3. Fit each face's dominant near-candidate depth mode with a robust median.
   Opposite wall faces are not averaged together. Multiple scans receive equal
   weight at the per-patch estimate stage; source disagreement is recorded.
4. Reconcile candidate-shared plane coordinates, with disagreement and
   coordinate-inversion vetoes. No new scan registration or scale adjustment.
5. Select large rectangular regions. Candidate openings veto rectangle growth.
   L-shaped polygons remain unions of rectangles, not filled bounding boxes.
6. Recheck area-uniform samples and boundary probes against all LAS returns
   retained in a conservatively padded candidate ROI. Reference points are not
   downsampled. Display clouds alone are sampled.
7. Export real quadrilateral SketchUp faces grouped by source object; hide
   coplanar internal seams, without generating hidden solid thickness.
8. Reopen the native file and check object identities and per-object areas.

The manifest adapters are in `prepare_rectangular_examples.py`. They contain
only input paths, recorded coordinate transformations and output labels.
There are no room IDs, manually selected walls, expected wall counts or custom
per-building fitting thresholds in the engine.

## Two explicit modes

**Continuity-assisted rectangles:** 50 mm support grid; bounded one-cell
closing; at least 80% sampled support per retained rectangle. Small unknown
areas may be bridged. Amber faces identify rectangles containing unobserved
area. Such area can lie more than 50 mm from a return. The complete rectangle
is not certified as measured merely because most of it has support.

**Strict rectangles:** 25 mm support grid; no closing or hole filling. A cell
is admitted only when its centre-to-return distance plus its half diagonal is
at most 50 mm minus a 2 micrometre numerical margin. Nearest-set distance is
1-Lipschitz, so the whole cell is bounded. A rectangle consists only of admitted
cells and stays inside its candidate polygon. Export rounding is covered by
the margin; boundary and fresh area samples are also checked. This mode makes
no inferred continuity fills but can remove a substantial amount of geometry.

Both modes use the same fitting logic. The tolerance is a scan-proximity limit,
not a statement that every room dimension is accurate to that tolerance.

## Limits and no-hallucination policy

- A nearby return might be a door, wardrobe, furniture, glass artifact or other
  non-wall surface. Point proximity and a rectangular shape do not prove wall
  identity. The stage does not yet classify doors from RGB or evaluate sensor
  rays through candidate walls.
- A lack of returns means unknown, not necessarily a doorway or window.
- No new wall profiles, hidden backs, wall thickness or missing rooms are
  inferred here. Existing source profiles may remain as supported surfaces.
- Do not label the strict file "hallucination-free" or "survey certified".
  Its defensible guarantee is no geometric continuity fill outside the stated
  scan-proximity envelope, not perfect semantics or survey accuracy.
- Inspect the source disagreements and unknown areas. Independent dimensions
  or calibrated control are needed for a room-accuracy claim.

## Reproduce

From the workspace, using `venv311/Scripts/python.exe`:

```powershell
python scripts/export/prepare_rectangular_examples.py
python scripts/export/rectangular_rebuild.py --manifest output_final/rectangular_rebuild_v1/mujammel.manifest.json --out NEW_OUTPUT_FOLDER
python scripts/export/export_rectangular_skp.py --folder NEW_OUTPUT_FOLDER --name Engrance_rectangular.skp
```

Use `mujammel_strict.manifest.json` for no-fill rectangles. Use the corresponding
`soulace_l0_strict`, `soulace_l1_strict` and `soulace_l2_strict` manifests for the
whole house. Output generation refuses to overwrite an existing handover.

`assemble_rectangular_levels.py` maps each level through its recorded raw frame
into the L0 coordinate system. It uses local-zero elevations 0, 3.2041 and
6.5544 m, not the previous nominal display offsets 0, 3.2 and 6.5 m. The recorded
small inter-level yaw differences are preserved. These are coordinate-frame
offsets, not independently surveyed floor-to-floor heights.

Native files have top, front, side and 3D scenes. The whole-house model also has
those four views for each of its three levels. Ceiling surfaces are retained
on separate tags and initially hidden. Models remain incomplete open surfaces.

## Scanning doors

For the main walkthrough, keep doors open and stationary where access/fire
rules permit. Avoid pressing a leaf against the wall if the hidden wall is
needed. A fixed, roughly right-angle position may allow observation from both
sides. If not, capture a separately identified supplementary scan with the
door in another fixed position and ample unchanged building overlap. Register
using the building; do not combine both door positions into a wall fit.

Opening doors before capture and keeping them still follows general mobile
SLAM guidance: https://knowledge.navvis.com/docs/cop-960-best-practices-for-scanning-while-opening-doors-with-navvis-vlx
The supplementary-scan suggestion is this project's capture recommendation,
not a claim that SLAM GO automatically removes changed door states.
