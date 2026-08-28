"""Turn the building onto the axes, once, at the start.

The scanner's frame has this building sitting about 14 degrees off X and Y.
Nothing downstream is wrong because of that, but everything is worse for it:
bounding boxes are oversized, plan rasters carry the whole footprint diagonally,
axis tests have to be written as tolerances, and every picture comes out
skewed. It is also simply not what a designer expects to receive.

The yaw comes from the walls themselves, not from a bounding box. Wall normals
are pooled as exp(4*i*theta) -- the quadruple angle, because a 90 degree grid
maps onto one full turn -- and weighted by how many points each wall carries.
That makes the estimate a vote of the actual measured surface: the long facade
counts for more than a cupboard side, and two walls at right angles reinforce
each other instead of cancelling.

Only a rotation about Z and a translation are applied. No scaling, no tilt --
the floor is already level and the walls are already plumb, and a similarity
transform that "improves" either would be inventing accuracy.
"""
import argparse

import numpy as np


def yaw_of(N, W):
    """The building's grid direction, as a weighted vote of its wall normals."""
    th = np.arctan2(N[:, 1], N[:, 0])
    z = np.sum(W * np.exp(4j * th))
    return float(np.angle(z) / 4.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--las", default=None, help="also write an aligned LAS")
    ap.add_argument("--las-out", default=None)
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D, pts, lab = z["n"], z["d"], z["pts"], z["lab"]
    cnt = z["count"]
    vert = np.abs(N[:, 2]) < 0.2
    yaw = yaw_of(N[vert], cnt[vert].astype(float))
    print(f"walls sit {np.degrees(yaw):+.3f} deg off the axes "
          f"({vert.sum()} wall surfaces voting)")

    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # origin at the footprint's corner, so the model starts at 0,0
    P2 = pts @ R.T
    org = np.array([P2[:, 0].min(), P2[:, 1].min(), 0.0])
    P2 -= org
    N2 = N @ R.T
    D2 = D + (N2 @ (-org))          # n.(x) = d  ->  n'.(Rx - org) = d - n'.org

    th = np.degrees(np.arctan2(N2[vert, 1], N2[vert, 0])) % 90
    off = np.minimum(th, 90 - th)
    w = cnt[vert] / cnt[vert].sum()
    print(f"after: wall azimuths sit {np.sum(off*w):.3f} deg from the grid "
          f"(weighted), {100*(off < 2).mean():.0f}% within 2 deg")
    print(f"footprint now {P2[:,0].max():.2f} x {P2[:,1].max():.2f} m, "
          f"z {P2[:,2].min():.2f}..{P2[:,2].max():.2f}")

    np.savez_compressed(a.out, n=N2, d=D2, rms=z["rms"], count=cnt,
                        pts=P2.astype(np.float32), lab=lab,
                        all_pts=(z["all_pts"] @ R.T - org).astype(np.float32),
                        yaw=yaw, origin=org, R=R)
    print(f"-> {a.out}")

    if a.las and a.las_out:
        import laspy
        with laspy.open(a.las) as r:
            hdr = r.header
            with laspy.open(a.las_out, mode="w", header=hdr) as w2:
                for ch in r.chunk_iterator(6_000_000):
                    p = np.column_stack([ch.x, ch.y, ch.z]) @ R.T - org
                    ch.x, ch.y, ch.z = p[:, 0], p[:, 1], p[:, 2]
                    w2.write_points(ch)
        print(f"-> {a.las_out}")


if __name__ == "__main__":
    main()
