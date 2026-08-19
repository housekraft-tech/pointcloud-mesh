"""Screened Poisson reconstruction of the scan -- the real thing, via open3d.

This replaces the voxel isosurface, which could never have crisp edges: a voxel
isosurface puts every face on the lattice, so an edge is always a stair-step at
the voxel pitch. Poisson fits a continuous indicator function to oriented points
instead, so a groove, a reveal and an arch come out as actual surface.

Parameters follow the isolidarflow branch's recreate_mesh.py, which produced a
complete surface where an earlier attempt was patchy:

  12 mm input voxel      fine enough to keep reveals and grooves
  oriented normals       consistent orientation, or Poisson turns inside out
  depth 11               detail; the cost that makes this slower than a raster
  trim 1.5% density      keep sparse-but-real walls rather than 5% which ate them
  drop small clusters    Poisson inflates balloons around stray points; those
                         are the floaters, and they are removed by size

Run with .venv311 -- open3d has no wheel for the 3.13 in .venv.
"""
import sys, os, json, time
import numpy as np
import laspy
import open3d as o3d

SRC = sys.argv[1] if len(sys.argv) > 1 else "output/mujammel_aligned_z0.las"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/model/poisson.ply"
VOXEL = float(os.environ.get("PM_VOXEL", 0.012))
DEPTH = int(os.environ.get("PM_DEPTH", 11))
TRIM = float(os.environ.get("PM_TRIM", 1.5))
MIN_TRI = int(os.environ.get("PM_MINTRI", 4000))
CUT = float(os.environ.get("PM_CUT", 0.12))       # ceiling slab off the top

t0 = time.time()
def stage(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

stage(f"reading {SRC}")
with laspy.open(SRC) as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
try:
    C = np.column_stack([p.red, p.green, p.blue]).astype(np.float64)
    C = C/(65535.0 if C.max() > 255 else 255.0)
except Exception:
    C = None
if os.path.exists("output/keep_mask.npy") and os.environ.get("PM_MASK", "1") == "1":
    k = np.load("output/keep_mask.npy")
    if len(k) == len(P):
        P = P[k]; C = C[k] if C is not None else None
        stage(f"declutter: dropped {(~k).sum():,} free-standing points")
if os.path.exists("output/fp_walls.json"):
    H = json.load(open("output/fp_walls.json"))['clear_height']
else:
    # No measured shell yet: take the ceiling off the height histogram itself.
    # The 99.5th percentile sits on the ceiling slab; anything above it is the
    # neighbouring building seen through a balcony, and is not this flat.
    H = float(np.percentile(P[:, 2], 99.5)) - float(np.percentile(P[:, 2], 0.5))
    P[:, 2] -= float(np.percentile(P[:, 2], 0.5))
    stage(f"no fp_walls.json: clear height taken as {H*1000:.0f} mm from the scan")
m = P[:, 2] < H-CUT
P = P[m]; C = C[m] if C is not None else None
stage(f"{len(P):,} points below the ceiling cut")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P)
if C is not None: pcd.colors = o3d.utility.Vector3dVector(C)
stage(f"voxel downsample {VOXEL*1000:.0f} mm")
pcd = pcd.voxel_down_sample(VOXEL)
stage(f"  {len(pcd.points):,} points")
stage("statistical outlier removal")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
stage(f"  {len(pcd.points):,} points")
stage("normals + consistent orientation")
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*4, max_nn=30))
try:
    pcd.orient_normals_consistent_tangent_plane(30)
except Exception as e:
    stage(f"  orientation skipped ({e})")
stage(f"poisson depth={DEPTH}  (this is the slow part)")
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=DEPTH, n_threads=-1)
dens = np.asarray(dens)
stage(f"  {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")
stage(f"trim lowest {TRIM}% density")
mesh.remove_vertices_by_mask(dens < np.percentile(dens, TRIM))
stage("clean: dedup, then drop floating clusters")
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
tc, ntri, _ = mesh.cluster_connected_triangles()
tc = np.asarray(tc); ntri = np.asarray(ntri)
small = ntri[tc] < MIN_TRI
stage(f"  {len(ntri):,} clusters, dropping {int(small.sum()):,} triangles "
      f"in clusters under {MIN_TRI:,} -- these are the floaters")
mesh.remove_triangles_by_mask(small)
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
o3d.io.write_triangle_mesh(OUT, mesh)
stage(f"wrote {OUT}: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris, "
      f"{os.path.getsize(OUT)/1e6:.1f} MB")
