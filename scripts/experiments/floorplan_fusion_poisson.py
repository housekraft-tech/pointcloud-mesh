"""Poisson-mesh cross-section measurement: the next refinement on top of
floorplan_fusion.py's raw-point wall measurement.

Reuses fusion3's already-registered wall positions (fused_walls.json) --
no need to recompute registration. For each registered wall, instead of a
raw-point perpendicular-offset histogram (which fails outright when a
railing's real points are too sparse/fragmented -- confirmed on the
"['Balcony', 'Living Room']" wall in fusion3, thickness=None), take a true
geometric cross-section of the Poisson mesh (already denoised/gap-filled)
at the wall's own height band and measure endpoints/thickness from that
clean, continuous line instead.

Usage: venv311\\Scripts\\python.exe scripts\\experiments\\floorplan_fusion_poisson.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.recon.trajectory import approx_trajectory
from scripts.isolidarflow import DEFAULT_CONFIG

FUSED_WALLS_JSON = ROOT / "output/koushik_fusion_rectified/fused_walls.json"
MESH_PATH = ROOT / "output/koushik_iso/mesh_isolated_v4_clean.obj"
LAS_PATH = ROOT / "koushikexport.las"
OUT_DIR = ROOT / "output/koushik_fusion_poisson"

SLICE_HEIGHTS_REL = [1.3, 1.6, 2.0]  # mid/upper-wall band -- avoid floor (0.15) and railing
                                     # zone (0.9) per user: search where solid wall is most
                                     # reliably present, not at floor/ceiling transition zones
BAND_M = 0.45  # widened -- registration is a good but imperfect footprint-IoU fit (0.896),
              # not sub-cm line-to-line; 0.20m was too tight against the real residual offset


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def recompute_alignment_and_zband():
    """Re-derive the SAME axis-alignment rotation R and z_floor/z_ceiling
    used when fused_walls.json was built, so the Poisson mesh (built in the
    UNALIGNED frame by strip_furniture_v2.py) can be rotated into the same
    aligned frame the fused walls already live in. dominant_axes returns a
    pure Z-rotation, so Z values themselves are unaffected -- only X/Y need
    the rotation applied.
    """
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass
    scan = load_scan(str(LAS_PATH), max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (approx_trajectory(scan.gps_time, scan.xyz, dt_s=cfg["traj_dt_s"])
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, _ = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                           max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    aligned = frame.axis_align(scan, R)
    z_floor = float(np.percentile(aligned.xyz[:, 2], cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(aligned.xyz[:, 2], cfg["z_ceiling_pct"]))
    return np.asarray(R, dtype=float), z_floor, z_ceiling


def mesh_cross_section_2d(mesh, z):
    """Horizontal plane cut -> list of 2D (x,y) line segments at height z."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return np.empty((0, 2, 2))
    planar, _tf = section.to_planar()
    out = []
    for poly in planar.discrete:
        for i in range(len(poly) - 1):
            out.append([poly[i], poly[i + 1]])
    return np.array(out) if out else np.empty((0, 2, 2))


def measure_wall_from_mesh_sections(p0, p1, mesh, z_floor, slice_heights_rel, band_m=BAND_M):
    """For each slice height, take the mesh cross-section and find segments
    close to this wall's expected line; measure the true endpoints/offset
    from those clean geometric lines. Returns per-height results plus an
    aggregate thickness (spread of the segments' own perpendicular offsets
    across heights, i.e. does the wall's mesh surface appear at a
    consistent offset -> thin single surface, or two distinct offsets ->
    front+back face)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    d = d / L
    n = np.array([-d[1], d[0]])

    per_height = []
    all_offsets = []
    for rel_h in slice_heights_rel:
        z = z_floor + rel_h
        segs = mesh_cross_section_2d(mesh, z)
        if segs.shape[0] == 0:
            per_height.append(dict(z=z, n_segs=0))
            continue
        mids = segs.mean(axis=1)
        u_mid = (mids - p0) @ d
        perp_mid = (mids - p0) @ n
        near = (np.abs(perp_mid) <= band_m) & (u_mid >= -0.3) & (u_mid <= L + 0.3)
        n_near = int(np.count_nonzero(near))
        if n_near == 0:
            per_height.append(dict(z=z, n_segs=int(segs.shape[0]), n_near=0))
            continue
        near_perp = perp_mid[near]
        near_u = u_mid[near]
        per_height.append(dict(z=z, n_segs=int(segs.shape[0]), n_near=n_near,
                               mean_offset=float(near_perp.mean()),
                               u_span=(float(near_u.min()), float(near_u.max()))))
        all_offsets.extend(near_perp.tolist())

    n_confirmed_heights = sum(1 for h in per_height if h.get("n_near", 0) > 0)
    thickness_m = None
    if len(all_offsets) >= 4:
        off = np.array(all_offsets)
        p_lo, p_hi = np.percentile(off, [5, 95])
        thickness_m = float(max(p_hi - p_lo, 0.02))
    return dict(per_height=per_height, n_confirmed_heights=n_confirmed_heights,
               n_total_heights=len(slice_heights_rel), thickness_m=thickness_m)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("recomputing axis-alignment R and z-band (must match fusion3 exactly)...")
    R, z_floor, z_ceiling = recompute_alignment_and_zband()
    log(f"z_floor={z_floor:.3f} z_ceiling={z_ceiling:.3f}")

    log(f"loading Poisson mesh {MESH_PATH.name}...")
    mesh = trimesh.load(str(MESH_PATH), process=False)
    log(f"mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces (unaligned frame)")
    mesh.vertices = mesh.vertices @ R.T  # rotate into the SAME aligned frame as fused_walls.json
    log("rotated mesh into the aligned frame")

    with open(FUSED_WALLS_JSON) as f:
        fused_walls = json.load(f)
    log(f"loaded {len(fused_walls)} fused walls from {FUSED_WALLS_JSON.name}")

    results = []
    for w in fused_walls:
        is_balcony = any("Balcony" in a for a in w["area_names"])
        raw_thickness = w["measured"]["thickness_m"] if w["measured"] else None
        p0 = w["measured"]["p0"] if w["measured"] else w["floorplan_p0"]
        p1 = w["measured"]["p1"] if w["measured"] else w["floorplan_p1"]
        mesh_result = measure_wall_from_mesh_sections(p0, p1, mesh, z_floor, SLICE_HEIGHTS_REL)
        results.append(dict(area_names=w["area_names"], is_balcony=is_balcony,
                            raw_point_thickness_m=raw_thickness, p0=p0, p1=p1,
                            mesh_section=mesh_result))
        tag = " [BALCONY]" if is_balcony else ""
        raw_str = f"{raw_thickness:.3f}m" if raw_thickness is not None else "UNMEASURED"
        mesh_str = (f"{mesh_result['thickness_m']:.3f}m" if mesh_result and mesh_result["thickness_m"] is not None
                   else "unmeasured")
        log(f"{tag} raw-point thickness={raw_str}  mesh-section thickness={mesh_str}  "
            f"confirmed at {mesh_result['n_confirmed_heights'] if mesh_result else 0}/"
            f"{len(SLICE_HEIGHTS_REL)} heights  areas={w['area_names']}")

    with open(OUT_DIR / "mesh_section_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT_DIR / 'mesh_section_results.json'}")

    # ---------- render: each slice height's mesh cross-section, all fused
    # walls overlaid, with the balcony wall(s) highlighted ----------
    all_xy = []
    for w in fused_walls:
        all_xy.append(w["floorplan_p0"])
        all_xy.append(w["floorplan_p1"])
    all_xy = np.array(all_xy)
    xmin, ymin = all_xy.min(axis=0) - 1.0
    xmax, ymax = all_xy.max(axis=0) + 1.0
    ppm = 70
    W, H = int((xmax - xmin) * ppm), int((ymax - ymin) * ppm)

    def to_px(pt):
        return (int((pt[0] - xmin) * ppm), int((ymax - pt[1]) * ppm))

    for rel_h in SLICE_HEIGHTS_REL:
        z = z_floor + rel_h
        segs = mesh_cross_section_2d(mesh, z)
        img = np.full((H, W, 3), 20, dtype=np.uint8)
        for seg in segs:
            cv2.line(img, to_px(seg[0]), to_px(seg[1]), (200, 200, 200), 2)
        for w in fused_walls:
            is_balcony = any("Balcony" in a for a in w["area_names"])
            color = (0, 220, 255) if is_balcony else (0, 140, 0)
            cv2.line(img, to_px(w["floorplan_p0"]), to_px(w["floorplan_p1"]), color, 1)
        cv2.putText(img, f"Poisson mesh cross-section at z=floor+{rel_h:.2f}m ({z:.2f}m abs)  "
                         f"gray=mesh-section  green=ML wall  cyan=BALCONY wall",
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = OUT_DIR / f"mesh_slice_h{rel_h:.2f}.png"
        cv2.imwrite(str(out_path), img)
        log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
