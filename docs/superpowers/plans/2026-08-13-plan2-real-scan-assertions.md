# Plan 2 — Real-Scan Behavioural Assertions for Tasks 5–12

**Date:** 2026-08-14
**Scope:** Specifies the `real_scan`-marked test each remaining Plan 2 task must carry.
Synthetic tests answer *is the number right*; these answer *does the stage do anything at
all on real data*. No assertion below claims accuracy — on the real scan nobody knows
the true dimensions.

## Ground rules (apply to every task)

- Fixture: `data/isolated_structural_v2.las`, cropped spatially to `x ∈ (−3.2, 1.0)`,
  `y ∈ (−8.0, −3.0)` (454,708 points), `patch_neighbor_k=64`. Use the shared
  `_crop_real_scan` helper (Task 10 Step 1 of the plan); never `--max-points`.
- Mark `@pytest.mark.real_scan`, skip when the LAS is absent — same pattern as
  `tests/rscene/test_occupancy.py::test_real_scan_occupancy_and_flood_fill`.
- **Never assert on a literal `face_id`.** Face ids re-order whenever upstream code
  changes. Select faces by role, size rank, or geometry.
- **Run-to-run drift is real.** Two measurement sessions of the same code produced
  88 faces / 86 kept / 32,994 unassigned (7.26%) and 90 / 89 / 34,444 (7.57%).
  All thresholds below carry margin for that drift; do not tighten them to the
  single number you happen to measure.
- Each stage's test should build its input once via a module-scoped fixture chain
  (crop → normals → patches → merge → gate → recruit) so eight tests don't redo a
  ~30 s pipeline eight times.

## Measured this session (fresh run, current working tree)

| quantity | value |
|---|---|
| patches / merged faces / kept / rejected | 196 / 90 / 89 / 1 |
| unassigned after recruit | 34,444 (7.57%) |
| frame `floor_z` / `ceiling_z` | −0.276 / 2.519 |
| classify (plan's Task 6 algorithm, band 0.30) | floor 5, ceiling 11, wall 56, unknown 10, oblique 7 |
| points by role | floor 70,964 · ceiling 139,317 · wall 170,871 · unknown 10,334 · oblique 862 |
| largest floor / ceiling face | 63,602 pts at z=−0.24 / 76,684 pts at z=2.50 |
| occupancy grid | (87, 103, 60), 35,117 occupied cells |
| flood fill from floor+1.2 m | **reaches 100.0% of free cells, touches all grid boundaries — the crop leaks** |
| interior sides (plan's Task 5 probe, naive) | 54/89 faces decided, 68.4% of face points; **slab signs come out wrong** (largest floor and ceiling both decide interior on the far side) |
| wall pairing (plan's Task 7 algorithm) | 56 wall faces → 24 candidate pairs → **5 mutual pairs**; thicknesses 62.1, 60.3, 64.7, 80.8, 206.6 mm; 42% of wall-face points paired |
| feature candidates (plan's Task 8 algorithm) | 26 candidates: 11 pass gate, 15 quarantined (7,798 pts); `median_spacing` = 17.26 mm on the crop |
| double-claim defect | pair partners (e.g. the 62.1 mm pair's smaller face) **also** emerge as "features" of their own wall partner |

---

## Task 5 — Interior side per face

**1. What the real scan demonstrates.** Two things. (a) The stage decides a sign for a
substantial share of faces — a no-op leaves every `interior_sign` as `None`. (b) A
*topological* correctness property that needs no ground truth: the scanner stood inside
the room, so interior air is above the floor and below the ceiling. That is knowable
without knowing a single dimension.

**Measured status — read before implementing.** On the cropped scan the flood fill
**leaks**: it reaches 100% of free cells and touches every grid boundary, because the
crop cuts the envelope and occlusion holes puncture it. Under the plan's naive
±1.5-cell probe, 54/89 faces get a sign, but the **largest floor and ceiling faces both
decide the wrong side** (the thin flooded sheet beyond the slab wins the probe count).
The plan's Step 3 code as written fails the assertion below, and that is the assertion
doing its job. The implementation must be robust to a leaky crop — e.g. decide by
comparing *occupied solid* behind each side instead of interior air, or dilate
occupancy one cell before filling, or treat boundary-clipped probes as non-votes. Do
not weaken the assertion to make the naive probe pass.

**2. Precondition.**
```python
assert len(faces) >= 60                       # merge+gate produced a real population
assert interior.sum() > 100_000               # fill actually ran (measured 502,538)
```

**3. Assertions.**
```python
# (a) not a no-op — thresholds to RE-MEASURE after the leak fix (see 5 below)
decided = [f for f in faces if f.interior_sign is not None]
assert len(decided) >= 30
assert sum(f.n_points for f in decided) / sum(f.n_points for f in faces) >= 0.50

# (b) topology, not accuracy: air is above the floor, below the ceiling
floor = max((f for f in faces if f.role == "floor"), key=lambda f: f.n_points)
ceiling = max((f for f in faces if f.role == "ceiling"), key=lambda f: f.n_points)
if floor.interior_sign is not None:
    assert floor.interior_sign * floor.normal[2] > 0, "floor's interior must be above it"
if ceiling.interior_sign is not None:
    assert ceiling.interior_sign * ceiling.normal[2] < 0, "ceiling's interior must be below it"
assert floor.interior_sign is not None or ceiling.interior_sign is not None, \
    "both slabs undecidable: the stage is not reaching the largest faces"
```
Margins: (a) naive probe measured 54 faces / 68.4% of points; 30 / 50% leaves room for
a stricter, leak-robust rule to decide fewer faces. (b) has no margin — it is a sign.

**4. What it catches.** (a): `assign_interior_sides` never called, probe step of 0,
probing the un-flooded array. (b): the exact failure measured today — leaked fill
electing the far side of a slab; also an inverted sign convention (`+1`/`−1` swapped),
which every synthetic test would pass symmetrically if the test itself were written
with the same swapped convention.

**5. Measure first?** Yes, for (a) only: after the leak-robust implementation exists,
run once, report decided-count and decided-point-fraction, then set thresholds at
roughly 60–75% of measured. (b) needs no measurement.

---

## Task 6 — Face classification

**1. What the real scan demonstrates.** The role partition on real geometry: one
dominant floor and ceiling exist near the frame's storey levels, the bulk of vertical
area is labelled `wall`, and nothing is left `None`. Synthetic rooms cannot show the
`unknown`/`oblique` populations at all — the real crop has 10 unknown-horizontal and 7
oblique faces, which is where a mis-ordered threshold hides.

**2. Precondition.**
```python
assert len(faces) >= 60
assert frame.floor_z is not None and frame.ceiling_z is not None
assert frame.ceiling_z - frame.floor_z > 2.0   # a real storey (measured 2.795)
```

**3. Assertions.**
```python
assert all(f.role is not None for f in faces)
assert set(f.role for f in faces) <= {"floor", "ceiling", "wall", "oblique", "unknown"}

pts = lambda role: sum(f.n_points for f in faces if f.role == role)
# a dominant slab face on each level (measured 63,602 and 76,684)
assert max((f.n_points for f in faces if f.role == "floor"), default=0) > 30_000
assert max((f.n_points for f in faces if f.role == "ceiling"), default=0) > 30_000
# floors sit at floor level, ceilings at ceiling level — role vs frame consistency
for f in faces:
    if f.role == "floor":
        assert abs(f.centroid[2] - frame.floor_z) <= cfg["classify_slab_band_m"]
    if f.role == "ceiling":
        assert abs(f.centroid[2] - frame.ceiling_z) <= cfg["classify_slab_band_m"]
# walls are the largest vertical population (measured 56 faces, 170,871 pts)
assert sum(1 for f in faces if f.role == "wall") >= 40
assert pts("wall") > 100_000
# the residual buckets stay residual (measured unknown 10,334 pts, oblique 862 pts)
assert pts("unknown") + pts("oblique") < 0.10 * sum(f.n_points for f in faces)
```

**4. What it catches.** Swapped `floor_z`/`ceiling_z` (floors classified at ceiling
level — the centroid-band check fails); a `vertical_normal_max_z` comparison written
`>` instead of `<` (walls collapse into oblique, the ≥40 fails); a band so wide every
horizontal face becomes a slab (`unknown` count drops to 0 — acceptable — but the
centroid-band check then catches mid-height sills labelled floor); classify silently
skipping gated faces (the `all(role is not None)` fails).

**5. Measure first?** No — all numbers above were measured this session and carry
30–50% headroom. If the merge or gate changes materially, re-run the measurement
script before adjusting.

---

## Task 7 — Wall assembly with measured thickness

**1. What the real scan demonstrates.** That pairing finds real opposing-face pairs in
a scan where most walls are seen from one side only — and does not invent pairs. The
crop yields 5 mutual pairs out of 56 wall faces (24 candidate pairs before the
mutual-best filter), with thicknesses 60–207 mm. Synthetic fixtures always pair
cleanly; only real data exercises the unpaired-majority path.

**2. Precondition.**
```python
wall_faces = [f for f in faces if f.role == "wall"]
assert len(wall_faces) >= 40                  # measured 56
```

**3. Assertions.**
```python
walls, unpaired = assemble_walls(faces, xyz, cfg)

# accounting: every wall face is in exactly one Wall, paired or not
assert len(walls) + sum(1 for w in walls if w.face_b is not None) == len(wall_faces)
# pairing is neither dead nor promiscuous (measured 5 paired, 51 single)
paired = [w for w in walls if w.face_b is not None]
assert 2 <= len(paired) <= 15
assert len(walls) - len(paired) >= 25, "almost everything paired: overlap/mutual gate broken"
# every measured thickness is in the physical band and carries provenance
for w in paired:
    assert 0.05 <= w.thickness.value <= 0.45
    assert w.thickness.n_points > 0 and w.thickness.method
# unpaired walls are honest: no invented thickness
for w in walls:
    if w.face_b is None:
        assert w.thickness is None
```

**4. What it catches.** A no-op (`paired == 0` fails the lower bound); a broken
overlap or mutual-best filter (dozens of spurious pairs fail the upper bound and the
≥25-unpaired check — 24 candidate pairs exist before mutuality, so this is a live
failure mode, not hypothetical); a fallback that fabricates a default thickness for
single-sided walls (the `is None` check); `d`-differencing regressions large enough to
throw a thickness out of [0.05, 0.45] (the precise origin-shift guard stays synthetic).

**5. Measure first?** No. Bounds derive from this session's run (5 pairs, 46+
unpaired). One caution: the 206.6 mm pair joins a 4,169-pt face to a 108-pt face —
if the implementer adds a minimum-support gate to pairing, re-measure; the pair count
could drop to 4 and the lower bound of 2 still holds.

---

## Task 8 — Rectangular features and the junk gate

**1. What the real scan demonstrates.** Both sides of the gate fire on real data —
which no synthetic fixture achieved (Task 3's lesson). Measured: 26 candidate faces
(wall faces with a larger parallel neighbour within 0.30 m), of which 11 pass the
rectangularity gate and 15 are quarantined (fills 0.18–0.70, 7,798 pts). The
quarantine path finally has real traffic.

**Measured defect — the assertion below is designed to catch it.** Run on today's
faces, the plan's `extract_features` promotes the *smaller face of each wall pair* as
a "feature" of its partner (the 62.1 mm pair's small face returns as a 62.1 mm
"extrusion", likewise 60.3, 64.7, 80.8 mm). One physical surface would be claimed by
both a Wall and a Feature. Feature extraction must exclude faces already consumed as
wall-pair partners (run after `assemble_walls` and skip both faces of every paired
wall), or the double-claim assertion fails — as it should.

**2. Precondition.**
```python
# candidates exist on both sides of the gate's input
assert len(features) + len(quarantined) >= 10   # measured 26 pre-exclusion
```

**3. Assertions.**
```python
features, quarantined = extract_features(faces, xyz, cfg)   # after assemble_walls

# both branches fire on real data
assert len(features) >= 3                        # measured 11 incl. pair-partners; ≥3 survives exclusion
assert len(quarantined) >= 3                     # measured 15
# gate coherence
for f in features:
    assert f.rect_fit >= cfg["feature_min_rect_fit"]
    assert 0.0 < f.depth.value <= cfg["feature_max_depth_m"]
# no double-claim: a face is a wall-pair member or a feature, never both
pair_members = {w.face_a for w in walls if w.face_b is not None} | \
               {w.face_b for w in walls if w.face_b is not None}
claimed_by_features = {getattr(f, "source_face", f.parent_face) for f in features}
assert not (pair_members & claimed_by_features), "a wall face was re-claimed as a feature"
```
(The `Feature` record must expose which candidate face it consumed — `source_face` or
equivalent — for the last assertion to be writable; add it if the dataclass lacks it.
`parent_face` alone identifies the parent, not the claimed face.)

**4. What it catches.** A dead gate (`quarantined == 0` while 15 sparse candidate
faces exist); an extractor that never finds parents (`features == 0`); the measured
double-claim of pair partners; a `rect_fit` computed as fill of the wrong box (values
over 1.0 or accepted values below the gate fail the coherence loop).

**5. Measure first?** Partially. The 3/3 lower bounds are safe against today's
numbers, but the exclusion of pair partners removes 4 of the 11 passing candidates —
after implementing the exclusion, run once and report the surviving counts before
committing the thresholds. Also note `median_spacing` on the crop is **17.26 mm**, not
the 6.1 mm nominal density — the fill denominator is nearly 3× more forgiving than the
plan assumed; report the measured `rect_fit` distribution alongside.

---

## Task 9 — Parts in the scene document

**1. What the real scan demonstrates.** Nothing meaningful beyond what synthetic
round-trip tests already prove. Serialization is structural: if a hand-built
`Wall`/`Feature`/`Face` with a `None` thickness round-trips, a real one does too —
there is no "awkwardness of real data" a JSON encoder can be defeated by that the
synthetic test doesn't cover. **Do not write a dedicated real-scan test for Task 9.**

The one invariant that *is* real-data-specific — every input point ends in exactly one
bucket — belongs to the pipeline, not the serializer, and is specified as part of Task
10's assertion below.

**2–4.** Not applicable. **5.** Nothing to measure.

---

## Task 10 — `rscene parts` CLI

**1. What the real scan demonstrates.** The whole chain runs end-to-end on real data
and its output honours the global accounting contract. This is the integration point,
so it carries the strongest version of the bucket invariant.

**2. Precondition.**
```python
assert main(["parts", crop, str(out), "--set", "patch_neighbor_k=64"]) == 0
payload = json.loads((out / "scene.json").read_text())
total = payload["diagnostics"]["n_input_points"]
assert total > 400_000                          # the crop, not a subsample
```

**3. Assertions** (replaces the draft in the plan's Task 10 Step 1 with measured
numbers):
```python
# merging did real work (measured 196 → 89/86)
assert len(payload["faces"]) < 0.65 * len(payload["patches"])
# recruitment did real work (measured 7.3–7.6%; Plan 1 figure was 13.69%)
assert payload["unassigned_points"] / total < 0.10
# every paired wall thickness physical (measured 60.3–206.6 mm)
for w in payload["walls"]:
    if w["thickness"] is not None:
        assert 0.05 <= w["thickness"]["value"] <= 0.45
# at least one wall paired, most not (measured 5 of 56)
n_paired = sum(1 for w in payload["walls"] if w["face_b"] is not None)
assert 2 <= n_paired < len(payload["walls"])
# THE accounting invariant: every point in exactly one bucket, no duplicates
face_pts = np.concatenate([np.asarray(f["point_idx"] + f["loose_idx"]) for f in payload["faces"]])
assert len(face_pts) == len(np.unique(face_pts)), "a point is claimed by two faces"
assert len(face_pts) + payload["unassigned_points"] + unmodeled_point_total == total
# the report tells the recruitment story
report = (out / "report.md").read_text().lower()
assert "before recruitment" in report and "after recruitment" in report
```
`unmodeled_point_total` is however the scene records points carried by
density-gate-rejected and quarantined faces — the implementer defines the field, the
test forces it to exist and to sum.

**4. What it catches.** Any stage silently dropped from the chain (unassigned jumps
back toward 13.7% or faces ≈ patches); recruitment claiming a point two faces already
own (the uniqueness check — this is the class of bug that only appears when faces
genuinely abut, i.e. on real data); the density gate's rejects vanishing instead of
landing in `unmodeled` (the sum fails); a report that stops stating the recruitment
delta.

**5. Measure first?** No for the ratios (measured with margin). Yes for one number:
after Task 10 exists, run once and record `n_paired`, unmodeled totals, and runtime in
the test's docstring so the next reviewer can compare against the shipped baseline.

---

## Task 11 — Golden part-level truth

**1. What the real scan demonstrates.** Nothing. This task *is* the synthetic side of
the division of labour — its entire point is asserting recovered dimensions against
known truth, which the real scan cannot supply by definition. Bolting a real-scan
assertion onto it would duplicate Task 10's integration test under a different name.
**No real-scan test for Task 11; state that in the task's commit message** so the
absence reads as a decision, not an omission.

**2–4.** Not applicable. **5.** Nothing to measure.

---

## Task 12 — `room.py` scene program

**1. What the real scan demonstrates.** The emitted program survives real-shaped
data — which is where pretty-printers die. The real scene contains everything the
synthetic room lacks: ~50 unpaired walls whose `thickness` is `None`, quarantined
faces, oblique faces, measurement values at full float precision, and dozens of parts
whose ids must stay unique. The behavioural property: **executing the emitted
`room.py` reproduces the scene it came from.**

**2. Precondition.**
```python
walls_json = payload["walls"]
assert any(w["thickness"] is None for w in walls_json), "no unpaired wall: trivial input"
assert any(w["thickness"] is not None for w in walls_json)
assert len(payload["features"]) >= 1
```

**3. Assertions.**
```python
ns = {}
exec((out / "room.py").read_text(), ns)         # emitted file executes stand-alone
program_walls = ns["SCENE"].walls               # whatever the emitted API exposes

# one object per part, same ids
assert sorted(w.wall_id for w in program_walls) == sorted(w["wall_id"] for w in walls_json)
# dimensions are the measured values, bit-exact — not rounded for prettiness
for pw, jw in zip(sorted(program_walls, key=lambda w: w.wall_id),
                  sorted(walls_json, key=lambda w: w["wall_id"])):
    if jw["thickness"] is None:
        assert pw.thickness is None
    else:
        assert pw.thickness == jw["thickness"]["value"]     # ==, not approx
    assert pw.length == jw["length"]["value"]
# emission is deterministic
assert emit_room_py(scene) == emit_room_py(scene)
```

**4. What it catches.** A formatter that rounds (`f"{v:.3f}"` fails bit-exact equality
on real measured floats — on synthetic boxes the values are often round numbers and the
bug hides); an emitter that skips `thickness=None` walls or crashes formatting them
(~50 such walls on the crop, zero in most synthetic fixtures); duplicate or renumbered
part ids; an emitted file that imports something unavailable at exec time.

**5. Measure first?** No numbers needed — every assertion is an equality against the
scene the program was generated from. The precondition guarantees the input is
non-trivial using structure, not magnitudes.

---

## Re-measurement protocol

Where a task above says "measure first", the implementer must, before writing the
threshold: run the stage on the standard crop, paste the measured figures into the
test's docstring, set the threshold at 60–75% of measured (lower bounds) or ~150%
(upper bounds), and state the measured value in the assertion's failure message. A
threshold that cannot cite its measurement gets rejected in review — that rule is what
this document exists to enforce.
