# Pointcloud mesh reconstruction

LAS point cloud → cleaned mesh (OBJ) via Open3D.

## Pipeline

1. `analyze_las.py` — inspect LAS header, attributes, point spacing
2. `reconstruct_mesh.py` — voxel downsample → outlier removal → normals → Poisson → OBJ
3. `obj_to_fbx.py` — optional Blender headless OBJ → FBX conversion
4. `floorplan_geometry.py` + `floorplan_reconstruct.py` — Phase 0 (bounding-box auto-crop) + Phase 1 (density-image wall/opening detection) → `manifest.json`, `floorplan.png`, `reconstructed.obj`. Replaces `segment_walls_and_grooves.py`.
5. `find_z_band.py` — inspect a scan's Z-density histogram to pick its primary story's Z-band by eye. **Run this before `floorplan_reconstruct.py` on any real scan** — automatic Z-band detection has not proven reliable across different real scans/sampling densities (see `floorplan_geometry.find_dense_z_band`'s docstring); `floorplan_reconstruct.py` warns loudly if run without `--z-band`.
6. `floorplan_reconstruct_test.py` — fast test-patch smoke test (small crop + point cap), same pattern as `reconstruct_mesh_test.py`
7. `validate_measurements.py` — diff hand tape-measured ground truth against `manifest.json`, report per-measurement mm error

### Real-scan usage

```bash
python scripts/find_z_band.py data/koushikexport.las
# inspect the printed histogram, pick z_min/z_max bracketing one story's floor/ceiling spikes
python scripts/floorplan_reconstruct.py data/koushikexport.las output/koushik --z-band -1.63,1.09
```

## Local setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p data output
# copy your scan to data/koushikexport.las (753MB — not in git)

python scripts/analyze_las.py data/koushikexport.las
python scripts/reconstruct_mesh.py data/koushikexport.las output/mesh_v1.obj
```

## Run on Jarvis Labs

Open3D meshing is CPU-bound (GPU is not used), but Jarvis VMs have many fast CPU cores.

### One-time setup

```bash
pip install jarvislabs
jl setup
jl gpus
```

### Recommended workflow

```bash
# 1) Create instance (L4 is enough — meshing is CPU-bound)
jl create --gpu L4 --storage 50 --name pointcloud-mesh
jl list   # note machine_id

# 2) Upload repo (excludes venv/output via .gitignore when cloning; upload skips big files)
jl upload <machine_id> . /home/pointcloud -r

# 3) Upload LAS separately (753MB — excluded from git)
jl upload <machine_id> ./data/koushikexport.las /home/pointcloud/data/koushikexport.las

# 4) SSH and run
jl ssh <machine_id>
```

On the instance:

```bash
cd /home/pointcloud
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p output
python scripts/reconstruct_mesh.py data/koushikexport.las output/mesh_v1.obj
exit
```

Download results and pause:

```bash
jl download <machine_id> /home/pointcloud/output ./output -r
jl pause <machine_id>
```

### One-shot managed run (after LAS is on the instance)

```bash
jl run scripts/reconstruct_mesh.py --on <machine_id> -- \
  data/koushikexport.las output/mesh_v1.obj
```

### Faster settings

In `scripts/reconstruct_mesh.py`: `VOXEL_SIZE = 0.02`, `POISSON_DEPTH = 9`.

| Machine | ~4.4M points |
|---------|----------------|
| Mac M1 | 45–90 min |
| Jarvis L4 VM | 10–25 min |
