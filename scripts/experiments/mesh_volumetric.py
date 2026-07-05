"""mesh_volumetric.py
------------------
Volumetric 3D driven by the COMPLETE Poisson MESH instead of raw scan points.
The mesh is denoised + gap-filled, so the per-height occupancy has far fewer
holes and cleaner surfaces than the raw cloud. The mesh and the processed cloud
share the same frame (both from the aligned isolated.las -- verified by matching
bounds), so no alignment is needed.

  LAS  -> free-space carve -> footprint / wall-network / rooms   (structure)
  MESH -> dense surface vertices -> per-height wall occupancy      (clean detail)
  stack -> marching cubes -> smooth -> one accurate 3D + dims

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\mesh_volumetric.py <isolated.las> <mesh.obj> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct
from scripts.experiments.volumetric_3d import (wall_footprint, build_volume, volume_to_mesh,
                                               dimensioned_plan)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las, mesh_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"reconstructing structure from {las} ...")
    R = reconstruct(las)
    ws = wall_footprint(R)

    log(f"loading mesh {mesh_path} ...")
    m = o3d.io.read_triangle_mesh(mesh_path)
    mv = np.asarray(m.vertices)
    if mv.shape[0] < 500_000:                      # sparse mesh -> sample its surface
        mv = np.asarray(m.sample_points_uniformly(number_of_points=4_000_000).points)
    log(f"mesh occupancy source: {mv.shape[0]:,} surface points")

    # ICP-align the mesh onto the processed cloud (bounds match but a few-cm
    # residual from the re-alignment would clip walls against the LAS footprint).
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(mv)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(np.column_stack([R["x"], R["y"], R["z"]]))
    sd = src.voxel_down_sample(0.05); dd = dst.voxel_down_sample(0.05)
    reg = o3d.pipelines.registration.registration_icp(
        sd, dd, 0.30, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = reg.transformation
    mv = (mv @ T[:3, :3].T) + T[:3, 3]
    log(f"ICP aligned mesh -> cloud (fitness {reg.fitness:.3f}, rmse {reg.inlier_rmse*1000:.0f}mm)")

    # use mesh points as the occupancy source, keep LAS structure
    Rm = dict(R)
    Rm["x"], Rm["y"], Rm["z"] = mv[:, 0], mv[:, 1], mv[:, 2]
    vol, levels = build_volume(Rm, ws)
    log("marching cubes ...")
    walls = volume_to_mesh(vol, R, R["z_floor"], ws)

    zf = R["z_floor"]
    fx = R["x"].max() - R["x"].min(); fy = R["y"].max() - R["y"].min()
    floor = trimesh.creation.box(extents=(fx, fy, 0.08))
    floor.apply_translation(((R["x"].min() + R["x"].max()) / 2, (R["y"].min() + R["y"].max()) / 2, zf - 0.05))
    scene = trimesh.Scene()
    walls.visual.face_colors = [205, 205, 210, 255]
    floor.visual.face_colors = [150, 130, 110, 255]
    scene.add_geometry(walls, geom_name="walls")
    scene.add_geometry(floor, geom_name="floor")
    scene.export(str(out_dir / "mesh_volumetric_model.glb"))
    scene.export(str(out_dir / "mesh_volumetric_model.obj"))
    log(f"exported mesh_volumetric_model: {len(walls.vertices):,}v / {len(walls.faces):,}f")
    dimensioned_plan(R, ws, out_dir)               # dims measured from LAS points (accurate)
    import os
    os.replace(str(out_dir / "volumetric_dimensioned.png"), str(out_dir / "mesh_volumetric_dimensioned.png"))
    log(f"wrote mesh_volumetric_dimensioned.png -> {out_dir}")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
