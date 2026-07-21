"""register_scans.py
-----------------
Rigid-register the RGB scan (mujammel) onto the geometry scan (koushik).

koushik carries the model but its RGB channel is ALL ZEROS (geometry-only
export); mujammel has real colour but sits in its own frame -- their bboxes
differ by metres, so colour cannot be sampled across without this transform.
Same flat, so a rigid 6-DoF fit is the right model.

Coarse FPFH+RANSAC to get in the basin, then point-to-plane ICP. Both clouds
are cropped to the interior height band first: floor and ceiling returns are
near-degenerate for orientation, and outdoor/balcony points differ between the
two scans and would drag the fit.

Writes transform.json (4x4, mujammel -> koushik) with the fitness and RMSE so
downstream code can refuse to use a bad fit.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\register_scans.py \\
      <source_rgb.las> <target.las> <target_measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import open3d as o3d

VOX = 0.06           # m: registration voxel
VOX_COARSE = 0.18    # m: voxel for the global stage
Z_LO, Z_HI = 0.60, 0.60   # m: trim above floor / below ceiling


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(las_path, z_lo, z_hi, want_rgb=False):
    import laspy
    las = laspy.read(las_path)
    P = np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)].astype(float)
    keep = (P[:, 2] > z_lo) & (P[:, 2] < z_hi)
    P = P[keep]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    if want_rgb:
        C = np.c_[np.asarray(las.red), np.asarray(las.green),
                  np.asarray(las.blue)].astype(float)[keep]
        pc.colors = o3d.utility.Vector3dVector(np.clip(C / 65535.0, 0, 1))
    log(f"{Path(las_path).name}: {len(P):,} pts in band z {z_lo:.2f}..{z_hi:.2f}")
    return pc


def prep(pc, voxel):
    d = pc.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=120))
    return d, f


def main(src_las, tgt_las, tgt_mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(tgt_mj))
    zf = float(np.median([r["z_floor"] for r in d["rooms"]]))
    zc = float(np.median([r["z_ceiling"] for r in d["rooms"]]))
    log(f"target band from measurements: {zf+Z_LO:.2f} .. {zc-Z_HI:.2f}")

    tgt = load(tgt_las, zf + Z_LO, zc - Z_HI)
    # the source scan's own floor level is unknown here; use the same absolute
    # band -- both exports share a vertical datum even though XY differ
    src = load(src_las, zf + Z_LO, zc - Z_HI)

    sc, sf = prep(src, VOX_COARSE)
    tc, tf = prep(tgt, VOX_COARSE)
    log(f"coarse: {len(sc.points):,} src / {len(tc.points):,} tgt pts")

    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        sc, tc, sf, tf, True, VOX_COARSE * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX_COARSE * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
    log(f"RANSAC fitness={res.fitness:.3f} rmse={res.inlier_rmse:.4f}")

    sf2 = src.voxel_down_sample(VOX)
    tf2 = tgt.voxel_down_sample(VOX)
    for p in (sf2, tf2):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX * 3, max_nn=40))
    T = res.transformation
    for thr in (VOX * 6, VOX * 3, VOX * 1.5):
        icp = o3d.pipelines.registration.registration_icp(
            sf2, tf2, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        T = icp.transformation
        log(f"ICP thr={thr:.3f} fitness={icp.fitness:.3f} rmse={icp.inlier_rmse*1000:.1f} mm")

    ok = icp.fitness > 0.60 and icp.inlier_rmse < 0.08
    json.dump(dict(transform=np.asarray(T).tolist(),
                   fitness=float(icp.fitness),
                   rmse_mm=float(icp.inlier_rmse * 1000),
                   usable=bool(ok),
                   source=str(src_las), target=str(tgt_las)),
              open(out / "scan_transform.json", "w"), indent=1)
    log(f"USABLE={ok}  (fitness {icp.fitness:.3f}, rmse {icp.inlier_rmse*1000:.1f} mm)")
    log(f"wrote {out/'scan_transform.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
