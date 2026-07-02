# isolidarflow — validated 2D wall-flow experiments (2026-07-02)

Three diagnostic scripts that drove the `recon` pipeline stages end-to-end on a
real scan for the first time and iterated the wall-run logic until the output
reads as an architectural floorplan. Each writes a PNG + JSON into
`output/koushik_iso/`.

**Prerequisite:** `output/koushik_iso/isolated.las` — produce it with the
Phase A isolation CLI:

```
venv311\Scripts\python.exe scripts\reconstruct.py <scan.las> output\koushik_iso
```

Then run any experiment:

```
venv311\Scripts\python.exe scripts\experiments\diag_floorplan2d_v3.py
```

## What each version showed (real koushik scan, 3.76M isolated points)

| | walls | rooms >=1 m² closed | notes |
|---|---|---|---|
| **v1** — committed `group_wall_runs` defaults | 10 | 5 (wrong shapes) | runs overshoot the whole plan; bathroom block got ZERO walls; 103/191 vertical planes unclaimed |
| **v2** — u-gap split (1.2 m) + relief 1.0→0.35 + rescue (`full_height_frac` 0.4, `min_run_length` 0.5, ≥1200 inliers) | 21 | 8 (mostly correct) | bathroom block recovered with measured 100–474 mm thicknesses; living room still open at T-junctions |
| **v3** — v2 + u-gap 2.8 + **midline recentring** + **endpoint-to-line (T-junction) snap** + hole classification | 19 | **7 incl. the 45.2 m² living space** | solid-wall architectural render; walls centered between their two faces (fixes "rooms sharing wall internals"); 3 in-run holes flagged as opening candidates |

## Key findings baked into these scripts

1. `group_wall_runs` chains planes by perpendicular offset only — collinear
   planes from *different* walls merge into one apartment-spanning run.
   Fix: split runs at along-wall coverage gaps (2.8 m threshold so
   doorway/balcony-scale holes stay inside their run as opening candidates).
2. `pair_thickness` measures thickness but never moves the centerline off the
   detected face — the whole wall body gets attributed to one room. Fix:
   `recenter_to_midline` (shift by ±t/2 toward the back face).
3. `resolve_corners` is endpoint-to-endpoint only and cannot close a
   T-junction. Fix: `snap_endpoints_to_lines` (extend a dangling endpoint onto
   a perpendicular run's interior, 0.7 m reach).
4. Open3D RANSAC is unseeded — the same cloud gave 191/238/244 vertical planes
   across runs. Seed with `o3d.utility.random.seed(0)` for reproducibility.
5. Poisson on the isolated cloud (`output/koushik_iso/mesh_isolated_v1.obj`)
   is a good *reference* surface but skins over real doors/windows (watertight
   by construction) — do NOT use it as the source for opening detection.

## Known gaps (see the implementation plan)

- Opening detection here is only u-coverage holes; real per-wall UV occupancy +
  scanner-trajectory visibility gating is the proper method
  (doors under continuous headers are invisible to u-coverage).
- Furniture in room interiors is untouched; strip it as
  "points not explained by any structural plane" once planes are final.
- Corner precision closes rooms at polygonize epsilon 0.30 but not yet 0.05.

## 3D preview iterations (same day, user-reviewed in Blender)

- `sharp_preview.py` → v1: stepped boolean wall solids + columns/beams/slabs;
  proved sharp L-cuts and watertightness (19/19), but no openings.
- `sharp_preview_v2.py`: **gps_time walkthrough door cuts validated** (53
  crossings → 12 doors), beam boxes removed, phantom-coverage filter.
- `sharp_preview_v3.py`: per-stretch z-evidence for beam soffits (found the
  stretch-merge flaw: fused wall+header evidence → no hollow gaps).
- `sharp_preview_v4_rooms.py` / `sharp_preview_v5_rooms.py`: room-wise
  interior panels, one per detected plane at its own offset (RELIEF preserved
  -- user's top priority), doors cut; v5 unions per wall + renders unassigned
  walls + dedupes shared walls.
- `strip_furniture.py` (plane-membership -- TORE WALLS, kept as a negative
  result) vs `strip_furniture_v2.py` (geometric rule: wall corridors
  untouchable, constant floor band, interiors cleared -- correct).
- `clean_mesh.py`: removes only DISCONNECTED mesh islands; attached wall
  relief cannot be affected.

Architecture conclusion from these iterations (user-approved): walls own a
feature graph (grooves, L-extrusions, beam soffits, openings) and split ONCE
into room-owned WallSegments; one mesh per segment via manifold3d CSG. See
"Architecture Revision R1" in the plan.

Productionizing all of this into `scripts/recon/` is specified in
`docs/superpowers/plans/2026-07-02-isolidarflow-sharp-3d.md`.
