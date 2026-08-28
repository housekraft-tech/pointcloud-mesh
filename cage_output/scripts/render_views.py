"""Render colour + exact-depth views from the LiDAR point cloud.

The LAS carries per-point RGB, so a plain pinhole projection with a z-buffer
gives a photo-like image AND the metric depth that produced it. That depth is
the ground truth a monocular depth model has to reproduce -- no registration,
no scale ambiguity, no annotation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
from recon.io_las import load_scan
from recon.isolate import select_z_band, isolate_unit

LAS_NAME = sys.argv[1] if len(sys.argv) > 1 else "koushikexport.las"
OUT = Path(__file__).resolve().parent / ("views_" + Path(LAS_NAME).stem)
OUT.mkdir(exist_ok=True)
YAW_DEG = 0.0  # per-scan yaw is irrelevant for a depth test
W, H, FOV = 640, 480, 70.0
SPLAT = 2  # px radius, fills the gaps between projected points


def look_at(eye, target, up=(0, 0, 1)):
    f = np.array(target) - np.array(eye)
    f = f / np.linalg.norm(f)
    up = np.array(up, dtype=float)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])  # world -> camera (x right, y down, z forward)
    return R


def render(xyz, rgb, eye, target):
    R = look_at(eye, target)
    cam = (xyz - eye) @ R.T
    z = cam[:, 2]
    keep = z > 0.15
    cam, col, z = cam[keep], rgb[keep], z[keep]

    fpx = (W / 2) / np.tan(np.deg2rad(FOV) / 2)
    u = (cam[:, 0] * fpx / z + W / 2).astype(np.int32)
    v = (cam[:, 1] * fpx / z + H / 2).astype(np.int32)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[m], v[m], z[m], col[m]

    depth = np.full((H, W), np.inf, dtype=np.float32)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    order = np.argsort(-z)            # painter's algorithm, near last
    u, v, z, col = u[order], v[order], z[order], col[order]
    for dy in range(-SPLAT, SPLAT + 1):
        for dx in range(-SPLAT, SPLAT + 1):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            depth[vv[ok], uu[ok]] = z[ok]
            image[vv[ok], uu[ok]] = col[ok]
    valid = np.isfinite(depth)
    return image, depth, valid


def main():
    scan = load_scan(str(ROOT / LAS_NAME), max_points=8_000_000)
    z_band = select_z_band(scan.xyz[:, 2])
    unit, _ = isolate_unit(scan, np.zeros((0, 3)), z_band)

    xyz = unit.xyz.copy()
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T

    # The Koushik export carries no colour (all RGB channels are 0), so shade
    # from LiDAR return intensity. A depth model sees a greyscale interior
    # rather than a photo -- a handicap to note when reading the numbers.
    rgb = getattr(unit, "rgb", None)
    if rgb is not None and np.asarray(rgb).any():
        rgb = np.asarray(rgb)[:, ::-1].copy()
    else:
        inten = np.asarray(unit.intensity, dtype=np.float32)
        lo, hi = np.percentile(inten, [2, 98])
        g = np.clip((inten - lo) / max(hi - lo, 1e-6), 0, 1)
        g = (g * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=1)
        print("no RGB in LAS -- shading from intensity", flush=True)

    z_floor = z_band[0]
    eye_h = z_floor + 1.5

    # free-floor candidates: cells with floor return but little occupancy above
    band = (xyz[:, 2] > z_floor + 1.0) & (xyz[:, 2] < z_floor + 2.2)
    cell = 0.25
    mn = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - mn) / cell).astype(int)
    nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    occ = np.zeros((nx, ny), np.int32)
    np.add.at(occ, (ij[band, 0], ij[band, 1]), 1)
    allc = np.zeros((nx, ny), np.int32)
    np.add.at(allc, (ij[:, 0], ij[:, 1]), 1)
    free = (occ < 20) & (allc > 50)
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(free)
    cand = np.argwhere(dist > 6)       # >1.5 m from any obstruction
    print(f"{len(cand)} candidate viewpoints", flush=True)

    rng = np.random.default_rng(0)
    picks = cand[rng.choice(len(cand), size=min(6, len(cand)), replace=False)]

    meta = []
    for n, (ci, cj) in enumerate(picks):
        eye = np.array([mn[0] + (ci + .5) * cell, mn[1] + (cj + .5) * cell, eye_h])
        for k, yaw in enumerate([0, 90, 180, 270]):
            a = np.deg2rad(yaw)
            tgt = eye + np.array([np.cos(a), np.sin(a), -0.05])
            img, dep, val = render(xyz, rgb, eye, tgt)
            cov = val.mean()
            if cov < 0.75:
                continue
            name = f"view_{n}_{yaw}"
            cv2.imwrite(str(OUT / f"{name}_rgb.png"), img)
            np.save(OUT / f"{name}_depth.npy", np.where(val, dep, np.nan))
            dv = dep[val]
            meta.append({"name": name, "eye": eye.tolist(), "yaw": yaw,
                         "coverage": round(float(cov), 3),
                         "depth_min": round(float(dv.min()), 3),
                         "depth_max": round(float(dv.max()), 3),
                         "depth_median": round(float(np.median(dv)), 3)})
            print(f"  {name}: cov={cov:.2f} depth {dv.min():.2f}-{dv.max():.2f} m",
                  flush=True)
    (OUT / "views.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} views to {OUT}")


if __name__ == "__main__":
    main()
