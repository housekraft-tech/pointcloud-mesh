"""pipeline.py
------------
The end-to-end pipeline as an explicit two-stage flow with a saved
intermediate, exactly:

    raw LAS  ->  [Stage A: isolate + axis-align]  ->  isolated.las (saved)
             ->  [Stage B: everything else]        ->  images / floorplan / atlas

Stage A runs the expensive isolation ONCE on the full raw cloud (~27M pts)
and writes the axis-aligned unit cloud to <out>/isolated.las (~3.7M pts) plus
a report.txt. Stage B then reads that small isolated cloud for every
downstream renderer, so nothing re-isolates the raw file and the whole suite
runs fast.

Everything lands in one folder: output/<name>_all/
    isolated.las               the Stage-A artifact (axis-aligned unit)
    report.txt                 point counts, z-band, footprint
    walk_path.png              operator route (time-colored)
    freespace_floorplan.png    deliverable floorplan (carved walls + doorways)
    walls_mask.png
    tomographic_floorplan.png  wall / beam / railing height-signature map
    atlas/                     full diagnostic suite (slices, hough, gps-time
                               heatmap, intensity, rgb, vertical sections)
    lasmining/                 mined raw-signal maps (coverage confidence,
                               density, intensity/glass, rgb material)

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\pipeline.py <raw_scan.las> <scan_name>
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan, save_scan_las
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime
from scripts.experiments import (walk_path_viz, freespace_floorplan,
                                 tomographic_floorplan, diagnostic_atlas,
                                 las_information_mining, openings_from_sections)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def isolate_and_save(raw_las, out_dir, cfg):
    """Stage A: raw LAS -> axis-aligned isolated unit -> isolated.las."""
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass
    log(f"Stage A: isolating {raw_las} ...")
    scan = load_scan(str(raw_las), max_points=cfg["max_points"], rng_seed=seed)
    n_raw = scan.n
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (sensor_trajectory_from_gpstime(scan.xyz, scan.gps_time)
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, iso_stats = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                                   max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)  # save the ALIGNED unit so Stage B needn't re-align

    iso_path = out_dir / "isolated.las"
    save_scan_las(scan, str(iso_path))
    z_floor = float(np.percentile(scan.xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(scan.xyz[:, 2], cfg["z_ceiling_pct"]))
    fp = scan.xyz.max(axis=0) - scan.xyz.min(axis=0)
    with open(out_dir / "report.txt", "w") as f:
        f.write(f"raw points        : {n_raw:,}\n")
        f.write(f"isolated points   : {scan.n:,}\n")
        f.write(f"dropped (drift/neighbour): {iso_stats.get('dropped', 0):,}\n")
        f.write(f"z-band (floor,ceil): ({z_floor:.3f}, {z_ceiling:.3f}) m  height {z_ceiling-z_floor:.3f} m\n")
        f.write(f"footprint (aligned): {fp[0]:.1f} x {fp[1]:.1f} m\n")
        f.write(f"gps_time / rgb / intensity present: "
                f"{scan.gps_time is not None} / {scan.rgb is not None} / {scan.intensity is not None}\n")
    log(f"Stage A done: {n_raw:,} -> {scan.n:,} pts, wrote {iso_path}")
    return str(iso_path)


def build_poisson_mesh(iso_path, out_dir):
    """Stage C: Poisson surface mesh from isolated.las, then drop tiny
    floating islands. The organic complete surface -- a visual QC reference
    beside the clean carved floorplan, and a denoised/gap-filled base that
    slices more cleanly than raw points.

    REUSES an existing clean mesh if one is already present (Poisson is slow,
    ~4 min; no point regenerating what's already done)."""
    from scripts.reconstruct_mesh import main as recon_mesh
    from scripts.experiments.clean_mesh import main as clean_mesh
    raw_obj = str(out_dir / "poisson_mesh.obj")
    clean_obj = str(out_dir / "poisson_mesh_clean.obj")
    if Path(clean_obj).exists():
        log(f"   reusing existing {clean_obj} (skip regeneration)")
        return
    if not Path(raw_obj).exists():
        recon_mesh(iso_path, raw_obj)
    clean_mesh(raw_obj, clean_obj, 0.15)
    # Our data is Z-up (LAS convention); OBJ viewers (Blender default) assume
    # Y-up, so the model shows up rotated +90 on its side. Convert Z-up -> Y-up
    # (rotate -90 about X: (x,y,z)->(x,z,-y)) so both meshes open upright.
    _to_yup(raw_obj)
    _to_yup(clean_obj)


def _to_yup(obj_path):
    import trimesh
    import numpy as np
    m = trimesh.load(obj_path, process=False)
    v = np.asarray(m.vertices)
    m.vertices = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])  # Z-up -> Y-up
    m.export(obj_path)


def build_mesh_floorplan(out_dir):
    """Stage D: the ACCURATE mesh-based floorplan. Load the Poisson mesh,
    rotate 180 about the up axis, project the wall-height vertex band top-down
    (dense + denoised), then make_floorplan (skeletonize -> Hough -> Manhattan
    -> merge parallels -> close junctions -> stub removal) -> clean CAD plan."""
    import trimesh
    import numpy as np
    import cv2
    from scripts.experiments import make_floorplan
    clean_obj = out_dir / "poisson_mesh_clean.obj"
    if not clean_obj.exists():
        log("   no poisson mesh -> skip mesh floorplan"); return
    mesh = trimesh.load(str(clean_obj), process=False)
    v = np.asarray(mesh.vertices)
    spans = v.max(axis=0) - v.min(axis=0)
    up = int(np.argmin(spans)); plane = [a for a in (0, 1, 2) if a != up]
    c = v.mean(axis=0); vr = v - c
    for a in plane:
        vr[:, a] = -vr[:, a]                       # 180 about up axis
    vr = vr + c
    up_lo, up_hi = vr[:, up].min(), vr[:, up].max()
    floor = up_lo + 0.10 * (up_hi - up_lo)
    band = (vr[:, up] >= floor + 0.5) & (vr[:, up] <= floor + 2.0)
    vb = vr[band][:, plane]
    ppm = 80
    minx, miny = vb.min(axis=0) - 0.4
    maxx, maxy = vb.max(axis=0) + 0.4
    W = int((maxx - minx) * ppm) + 1; H = int((maxy - miny) * ppm) + 1
    raster = np.zeros((H, W), np.uint8)
    cc = np.clip(((vb[:, 0] - minx) * ppm).astype(int), 0, W - 1)
    rr = np.clip(((maxy - vb[:, 1]) * ppm).astype(int), 0, H - 1)
    raster[rr, cc] = 255
    raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    fp_dir = out_dir / "mesh_floorplan"
    fp_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(fp_dir / "mesh_slice_raster.png"), raster)
    make_floorplan.main(str(fp_dir / "mesh_slice_raster.png"), str(fp_dir))


def main(raw_las, scan_name, poisson=True):
    out_dir = ROOT / "output" / f"{scan_name}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    log(f"=== pipeline {scan_name} -> {out_dir} ===")

    iso_path = isolate_and_save(raw_las, out_dir, cfg)

    # Stage B: every renderer consumes the small isolated.las (fast)
    stages = [
        ("walk path", lambda: walk_path_viz.main(iso_path, scan_name, str(out_dir))),
        ("free-space floorplan", lambda: freespace_floorplan.main(iso_path, scan_name, str(out_dir))),
        ("tomographic", lambda: tomographic_floorplan.main(iso_path, scan_name, str(out_dir))),
        ("diagnostic atlas", lambda: diagnostic_atlas.main(iso_path, scan_name, str(out_dir / "atlas"))),
        ("las information mining", lambda: las_information_mining.main(iso_path, scan_name, str(out_dir / "lasmining"))),
        ("openings from sections", lambda: openings_from_sections.main(iso_path, scan_name, str(out_dir / "openings"))),
    ]
    if poisson:
        # Stage C+D last -- Poisson is the slow one (~3-5 min); the accurate
        # mesh floorplan (D) reuses the mesh right after it's built.
        stages.append(("poisson mesh (slow)", lambda: build_poisson_mesh(iso_path, out_dir)))
        stages.append(("mesh floorplan (accurate, rot180)", lambda: build_mesh_floorplan(out_dir)))
    ok, failed = [], []
    for label, fn in stages:
        t0 = time.time()
        try:
            log(f"Stage B: {label} ...")
            fn()
            ok.append(label)
            log(f"   done ({time.time()-t0:.0f}s)")
        except Exception as exc:
            failed.append(label)
            log(f"   FAILED: {exc}")
            traceback.print_exc()

    log(f"=== {scan_name}: Stage A + {len(ok)}/{len(stages)} Stage-B stages ok "
        f"({len(failed)} failed) ===")
    log(f"all outputs (incl. isolated.las) in: {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
