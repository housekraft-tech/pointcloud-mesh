# Pointcloud → modular floorplan & 3D reconstruction (isolidarflow)

Indoor SLAM LAS scan → isolated unit → 2D architectural floorplan + room-wise
sharp 3D model (GLB) + cleaned Poisson reference mesh.

Everything runs locally, CPU-only, from this folder on the `isolidarflow`
branch. The production implementation plan is
`docs/superpowers/plans/2026-07-02-isolidarflow-sharp-3d.md`; the runnable
flow below is the validated experiment chain it productionizes
(see `scripts/experiments/README.md` for what each script proved).

## Setup (once)

```powershell
# Python 3.11 64-bit required (32-bit cannot load open3d)
C:\Users\PC\AppData\Local\Programs\Python\Python311\python.exe -m venv venv311
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Scans live in this folder (not in git): `koushikexport.las`, `mujammelexport.las`.

## The flow — run in this order

All commands from the repo root. `<name>` is `koushik` or `mujammel`.

```powershell
# 1) ISOLATE the apartment (Z-band + trajectory connectivity; drops the
#    neighbouring building seen through the balcony and SLAM drift).
#    Writes output/<name>_iso/isolated.las + report.txt
.\venv311\Scripts\python.exe scripts\reconstruct.py koushikexport.las output\koushik_iso --max-points 4000000

# 2) 2D FLOORPLAN diagnostic (solid measured-thickness walls, closed rooms,
#    T-junction closure). Writes output/koushik_iso/floorplan2d_diag_v3.png + .json
.\venv311\Scripts\python.exe scripts\experiments\diag_floorplan2d_v3.py

# 3) FURNITURE STRIP (geometric rule: wall corridors untouchable, constant
#    floor band, room interiors cleared).
#    Writes output/<name>_iso/isolated_structural_v2.las
.\venv311\Scripts\python.exe scripts\experiments\strip_furniture_v2.py koushik

# 4) POISSON reference mesh on the furniture-free cloud.
#    Writes output/koushik_iso/mesh_isolated_v3_structural.obj
.\venv311\Scripts\python.exe scripts\reconstruct_mesh.py output\koushik_iso\isolated_structural_v2.las output\koushik_iso\mesh_isolated_v3_structural.obj

# 5) (optional) drop tiny DISCONNECTED mesh islands (attached wall relief is
#    never touched). Writes mesh_isolated_v4_clean.obj
.\venv311\Scripts\python.exe scripts\experiments\clean_mesh.py output\koushik_iso\mesh_isolated_v3_structural.obj output\koushik_iso\mesh_isolated_v4_clean.obj 0.15

# 6) ROOM-WISE SHARP 3D MODEL: interior wall panels per detected plane
#    (relief preserved), gps_time walkthrough door cuts, one object per wall,
#    rooms as collections. Writes output/koushik_iso/model_rooms_v5.glb
.\venv311\Scripts\python.exe scripts\experiments\sharp_preview_v5_rooms.py
```

Open the results in Blender: `model_rooms_v5.glb` (modular deliverable
preview) next to `mesh_isolated_v4_clean.obj` (Poisson reference).

**Other scan:** steps 2 and 6 currently hard-code `output\koushik_iso` in
their `OUT_DIR` constant — change that one line (e.g. to `mujammel_iso`)
after running steps 1 and 3 with the other scan's name. Fixed properly by the
plan's Task 10 CLI.

### Earlier iterations kept for comparison

`scripts/experiments/` also contains the earlier versions (diag v1/v2,
sharp_preview v1–v4, strip_furniture v1) with docstrings explaining exactly
which failure mode each iteration exposed and fixed — useful when tuning, and
they are the reference implementations cited by the implementation plan.

## Production pipeline (in progress)

`scripts/recon/` is the tested production package (isolate, planes, structure,
regularize, solids, floorplan2d, assemble). The implementation plan wires the
experiment learnings into it as 10 TDD tasks, including Architecture Revision
R1: walls own a feature graph (grooves, L-extrusions, beam soffits, openings)
and split once into room-owned WallSegments — one watertight mesh per segment.

## Legacy

- `scripts/reconstruct_mesh.py` — the original LAS → Poisson OBJ path (used
  as step 4 above; koushik full-cloud ≈ 25–45 min, isolated ≈ 4 min at
  voxel 15 mm / depth 10).
- `scripts/floorplan_reconstruct.py` + `floorplan_geometry.py` — the
  top-down density-image floorplan pipeline (superseded by the flow above;
  scheduled for salvage-and-delete in the plan).
- `scripts/analyze_las.py` — inspect LAS header/attributes/point spacing.
- `scripts/obj_to_fbx.py` — optional Blender headless OBJ → FBX.
