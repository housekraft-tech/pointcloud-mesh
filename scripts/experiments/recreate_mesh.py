"""recreate_mesh.py
----------------
Recreate a MORE COMPLETE Poisson mesh from the isolated cloud. The old mesh was
patchy because it trimmed 5% of Poisson vertices and used depth 10. Here:
finer input voxel, oriented normals (cleaner Poisson), higher depth for detail,
minimal density trim (keep sparse-but-real walls), and remove only the small
floating balloon components -> a complete, watertight-ish surface.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\recreate_mesh.py <isolated.las> <out.obj>
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reconstruct_mesh import load_las_as_o3d, log_stage, log

VOXEL = 0.012           # 12mm input voxel (finer than 15mm)
DEPTH = 11              # higher Poisson depth -> more detail (arch/reveals)
TRIM_PCT = 1.5         # keep almost everything (was 5%)
MIN_CLUSTER_TRI = 4000  # drop floating balloon bits below this many triangles


def main(inp, outp):
    outp = Path(outp); outp.parent.mkdir(parents=True, exist_ok=True)
    pcd = load_las_as_o3d(inp)
    with log_stage(f"Voxel downsample {VOXEL*1000:.0f}mm"):
        pcd = pcd.voxel_down_sample(VOXEL)
    with log_stage("Outlier removal"):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    with log_stage("Normal estimation + orientation"):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 4, max_nn=30))
        try:
            pcd.orient_normals_consistent_tangent_plane(30)   # consistent -> cleaner Poisson
        except Exception as e:
            log(f"   normal orientation skipped ({e})")
    with log_stage(f"Poisson depth={DEPTH}"):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=DEPTH, n_threads=-1)
    dens = np.asarray(dens)
    with log_stage(f"Trim lowest {TRIM_PCT}% density (keep sparse real walls)"):
        mesh.remove_vertices_by_mask(dens < np.percentile(dens, TRIM_PCT))
    with log_stage("Clean: dedup + drop small floating components"):
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        tc, ntri, _ = mesh.cluster_connected_triangles()
        tc = np.asarray(tc); ntri = np.asarray(ntri)
        mesh.remove_triangles_by_mask(ntri[tc] < MIN_CLUSTER_TRI)
        mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(outp), mesh)
    log(f"wrote {outp}: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
