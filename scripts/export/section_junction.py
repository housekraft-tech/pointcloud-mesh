"""Slice a junction and plot it, at one scale, from whichever meshes you name.

An earlier version of this figure showed the Poisson wall wandering about
20 mm as it came down to the floor, which is not what a screened Poisson
reconstruction of a plastered wall does at a 4 mm octree cell. The suspicion
worth testing is that the plot was made from a DECIMATED mesh -- 60k triangles
over 1110 m2 is a 13 cm triangle, and a 13 cm triangle cannot represent a
straight line to better than its own size.

So this takes the mesh as an argument and says how many triangles it has, and
it fixes the aspect ratio to 1:1 -- a section plotted on unequal axes makes a
straight wall look drunk. It also reports the RMS deviation of each limb from
its own fitted line, which is the number the picture is trying to convey.
"""
import argparse

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def section(mesh, origin, normal):
    """Segments of the slice, in a frame CENTRED ON `origin`.

    trimesh's to_planar() picks its own arbitrary 2-D frame, so the requested
    point lands wherever it likes and a window around it is empty -- which is
    how the first attempt produced a blank plot. Projecting the segments onto
    axes built from the slicing normal, with the origin at the point asked for,
    means the window means what it says.
    """
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=normal,
                                            plane_origin=origin)
    if segs is None or not len(segs):
        return np.zeros((0, 2, 2))
    t = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, t)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    q = segs - origin
    return np.stack([q @ u, q @ v], axis=-1)


def straightness(xy, axis):
    """RMS of the limb about its own fitted line, in mm."""
    if len(xy) < 3:
        return float("nan")
    t = xy[:, axis]
    o = xy[:, 1 - axis]
    A = np.vstack([t, np.ones_like(t)]).T
    m, c = np.linalg.lstsq(A, o, rcond=None)[0]
    return float(np.sqrt(np.mean((o - (m * t + c)) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--at", nargs=3, type=float, required=True,
                    help="a point on the junction")
    ap.add_argument("--normal", nargs=3, type=float, default=[0, 1, 0],
                    help="the slicing plane's normal")
    ap.add_argument("--half", type=float, default=0.30, help="m either side")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    at = np.array(a.at, float)
    nrm = np.array(a.normal, float)
    nrm /= np.linalg.norm(nrm)
    fig, axes = plt.subplots(1, len(a.meshes), figsize=(5.2*len(a.meshes), 5.0))
    axes = np.atleast_1d(axes)
    for ax, path, lab in zip(axes, a.meshes, a.labels):
        m = trimesh.load(path, force="mesh")
        S = section(m, at, nrm)
        n = 0
        for seg in S:
            if np.abs(seg).max() > 3*a.half:
                continue
            ax.plot(seg[:, 0], seg[:, 1], lw=1.2, color="#1f4e9c")
            n += 1
        ax.set_aspect("equal")
        ax.set_xlim(-a.half, a.half)
        ax.set_ylim(-a.half, a.half)
        ax.grid(alpha=0.3)
        ax.set_title(f"{lab}\n{len(m.faces):,} tris, {n} segments in view",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
