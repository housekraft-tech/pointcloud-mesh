"""Remove tiny floating intrusions from a Poisson mesh: cluster connected
triangles, drop every component below an area threshold (detached membrane
flakes, SLAM ghost shards hovering off walls / behind beams). The main
apartment shell is orders of magnitude larger and is untouched.

Usage: python clean_mesh.py <in.obj> <out.obj> [min_area_m2]
"""
import sys
import time

import numpy as np
import open3d as o3d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(in_path, out_path, min_area_m2=0.15):
    mesh = o3d.io.read_triangle_mesh(in_path)
    log(f"loaded {len(mesh.triangles):,} triangles")

    cluster_idx, tri_counts, areas = mesh.cluster_connected_triangles()
    cluster_idx = np.asarray(cluster_idx)
    areas = np.asarray(areas)
    small = areas < float(min_area_m2)
    n_small = int(small.sum())
    remove = small[cluster_idx]
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    log(f"removed {n_small:,} of {len(areas):,} components "
        f"({int(remove.sum()):,} triangles) below {min_area_m2} m2")

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, mesh)
    log(f"wrote {out_path}: {len(mesh.triangles):,} triangles")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
