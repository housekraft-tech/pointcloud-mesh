"""Shade meshes side by side by ray casting, so an edge looks like an edge.

There is no GPU here -- Filament refuses to run headless on this box -- but
none is needed. A camera grid of rays into open3d's RaycastingScene returns
the hit triangle's normal per pixel, and shading THAT is exactly the picture
that answers the question: a rolled fillet becomes a smooth ramp of shading
several pixels wide, a real corner becomes a one-pixel step. Comparing the
same view of the scan and the rebuild puts the two next to each other.
"""
import argparse

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


def scene_of(m):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                  o3d.utility.Vector3iVector(m.faces))))
    return s


def shade(m, eye, target, up, w, h, fov):
    s = scene_of(m)
    rays = s.create_rays_pinhole(fov_deg=fov,
                                 center=target, eye=eye, up=up,
                                 width_px=w, height_px=h)
    r = s.cast_rays(rays)
    hit = r["t_hit"].numpy()
    nrm = r["primitive_normals"].numpy()
    d = rays.numpy()[:, :, 3:6]
    d = d / np.linalg.norm(d, axis=2, keepdims=True)
    lam = np.abs((nrm * -d).sum(axis=2))            # headlight
    key = np.array([0.4, 0.25, 0.88])
    key = key / np.linalg.norm(key)
    lam = 0.45*lam + 0.55*np.clip(np.abs(nrm @ key), 0, 1)
    img = np.clip(0.12 + 0.88*lam, 0, 1)
    img[~np.isfinite(hit)] = 1.0                    # background
    return (img*255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=900)
    ap.add_argument("--h", type=int, default=700)
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--azim", type=float, default=35.0, help="deg")
    ap.add_argument("--elev", type=float, default=28.0, help="deg")
    ap.add_argument("--dist", type=float, default=1.9, help="x the diagonal")
    ap.add_argument("--radius", type=float, default=None,
                    help="m: put the camera exactly this far from --at, instead "
                         "of scaling off the whole model's diagonal")
    ap.add_argument("--at", nargs=3, type=float, default=None,
                    help="look at this point instead of the centre")
    a = ap.parse_args()

    ms = [trimesh.load(p, force="mesh") for p in a.meshes]
    lo = np.min([m.bounds[0] for m in ms], axis=0)
    hi = np.max([m.bounds[1] for m in ms], axis=0)
    c = np.array(a.at, float) if a.at else (lo + hi) / 2
    R = a.radius if a.radius else float(np.linalg.norm(hi - lo)) * a.dist / 2
    az, el = np.radians(a.azim), np.radians(a.elev)
    eye = c + R*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    panels = [shade(m, eye.astype(np.float32), c.astype(np.float32),
                    np.array([0, 0, 1], np.float32), a.w, a.h, a.fov) for m in ms]
    gap = np.full((a.h, 8), 210, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.hstack([strip, gap, p])
    Image.fromarray(strip).save(a.out)
    print(f"{a.out}  ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
