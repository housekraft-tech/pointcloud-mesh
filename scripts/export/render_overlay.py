"""Two models in one image, so their disagreement is visible rather than inferred.

Side by side, two plans look similar and the eye cannot tell whether a wall
moved 5 mm or 50. Overlaid and tinted, a wall that agrees goes one colour and a
wall that has moved shows both colours side by side, with the offset legible
against the wall's own thickness.

Each model is ray-cast separately and the nearer hit wins the pixel, so this is
a real depth composite: geometry that is genuinely behind stays behind.
"""
import argparse

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

TINTS = [(0.20, 0.45, 0.95), (0.95, 0.35, 0.15), (0.15, 0.70, 0.35)]


def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                  o3d.utility.Vector3iVector(m.faces))))
    return s


def cast(m, eye, target, up, w, h, fov):
    s = scene_of(m)
    rays = s.create_rays_pinhole(fov_deg=fov, center=target, eye=eye, up=up,
                                 width_px=w, height_px=h)
    r = s.cast_rays(rays)
    z = r["t_hit"].numpy()
    n = r["primitive_normals"].numpy()
    d = rays.numpy()[:, :, 3:6]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    key = np.array([0.4, 0.25, 0.88])
    key /= np.linalg.norm(key)
    lam = 0.45*np.abs((n * -d).sum(2)) + 0.55*np.clip(np.abs(n @ key), 0, 1)
    return z, np.clip(0.25 + 0.75*lam, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1200)
    ap.add_argument("--h", type=int, default=900)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--azim", type=float, default=270.0)
    ap.add_argument("--elev", type=float, default=89.0)
    ap.add_argument("--dist", type=float, default=2.3)
    ap.add_argument("--at", nargs=3, type=float, default=None)
    ap.add_argument("--radius", type=float, default=None)
    a = ap.parse_args()

    ms = [trimesh.load(p, force="mesh") for p in a.meshes]
    lo = np.min([m.bounds[0] for m in ms], axis=0)
    hi = np.max([m.bounds[1] for m in ms], axis=0)
    c = np.array(a.at, float) if a.at else (lo + hi) / 2
    R = a.radius if a.radius else float(np.linalg.norm(hi - lo)) * a.dist / 2
    az, el = np.radians(a.azim), np.radians(a.elev)
    eye = c + R*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    eye = eye.astype(np.float32)
    tgt = c.astype(np.float32)
    up = np.array([0, 0, 1], np.float32)

    img = np.ones((a.h, a.w, 3), np.float32)
    best = np.full((a.h, a.w), np.inf, np.float32)
    for k, m in enumerate(ms):
        z, sh = cast(m, eye, tgt, up, a.w, a.h, a.fov)
        hit = np.isfinite(z) & (z < best)
        best = np.where(hit, z, best)
        tint = np.array(TINTS[k % len(TINTS)], np.float32)
        col = tint[None, None, :] * sh[:, :, None]
        img[hit] = col[hit]
    Image.fromarray((img*255).astype(np.uint8)).save(a.out)
    print(f"{a.out}  " + ", ".join(
        f"{p.split('/')[-1]} = "
        f"{'blue' if i%3==0 else 'orange' if i%3==1 else 'green'}"
        for i, p in enumerate(a.meshes)))


if __name__ == "__main__":
    main()
