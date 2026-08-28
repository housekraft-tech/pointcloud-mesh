"""Poisson's completeness, the points' accuracy.

Two routes have been built and each is better than the other at one thing. The
Poisson route gives complete walls, few pieces and a surface that reads as a
building -- because a screened Poisson reconstruction closes over gaps, which
is exactly the property the point route lacks. The point route gives plane
positions determined to hundredths of a millimetre and edges that are 98% flat,
because nothing was smoothed on the way in.

Neither has to be given up. The Poisson mesh decides SHAPE -- which surface
exists, how far it runs, what it meets. The points decide WHERE each of those
surfaces is. So every plane in the Poisson segmentation is matched to the plane
fitted to the actual points, and its equation is replaced. The topology, the
extent and the coverage all survive; the geometry moves onto the measurement.

A Poisson plane with no point-fitted match keeps its own equation and is
flagged, because that is the honest outcome: it is surface the reconstruction
invented, and 30% of the mesh is exactly that.
"""
import argparse

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="segmentation of the Poisson mesh")
    ap.add_argument("--points", required=True, help="planes fitted to the LAS")
    ap.add_argument("--transform", required=True,
                    help="npy 4x4 taking the Poisson frame to the scan frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ang", type=float, default=4.0, help="deg")
    ap.add_argument("--off", type=float, default=0.05,
                    help="m: how far a Poisson plane may be from its match")
    ap.add_argument("--max-move", type=float, default=0.015,
                    help="m: refuse a correction bigger than this. Moving a "
                         "plane 49 mm after its neighbours were fitted tears "
                         "every junction it had -- the first attempt went from "
                         "206 pieces to 5,272 that way. A genuine smoothing "
                         "bias is millimetres; anything larger is a mismatch.")
    ap.add_argument("--near", type=float, default=1.5,
                    help="m: the point surface must be near the Poisson patch")
    a = ap.parse_args()

    z = np.load(a.labels)
    V, F, flab, PN, PD = z["V"], z["F"], z["flab"], z["plane_n"], z["plane_d"]
    p = np.load(a.points)
    QN, QD, qpts, qlab = p["n"], p["d"], p["pts"], p["lab"]

    T = np.load(a.transform)
    R, t = T[:3, :3], T[:3, 3]
    # a plane in the scan frame, expressed in the Poisson frame
    QN2 = QN @ R
    QD2 = QD - QN @ t
    qc = np.array([(qpts[qlab == i].mean(0) if (qlab == i).any() else
                    np.full(3, 1e9)) for i in range(len(QN))])
    qc2 = (qc - t) @ R

    A = trimesh.Trimesh(V, F, process=False).area_faces
    cos = np.cos(np.radians(a.ang))
    swapped = 0
    moved = []
    src = np.zeros(len(PN), np.int32) - 1
    for i in range(len(PN)):
        sel = np.flatnonzero(flab == i)
        if not len(sel):
            continue
        c = V[np.unique(F[sel])].mean(0)
        dot = QN2 @ PN[i]
        sgn = np.sign(dot)
        cand = np.flatnonzero(np.abs(dot) >= cos)
        if not len(cand):
            continue
        dd = np.abs(sgn[cand]*QD2[cand] - PD[i])
        near = np.linalg.norm(qc2[cand] - c, axis=1)
        ok = (dd <= a.off) & (near <= a.near)
        if not ok.any():
            continue
        k = cand[ok][np.argmin(dd[ok])]
        if float(dd[ok].min()) > a.max_move:
            continue
        s = sgn[k]
        moved.append(abs(float(s*QD2[k] - PD[i])))
        PN[i] = s * QN2[k]
        PD[i] = s * QD2[k]
        src[i] = k
        swapped += 1

    tot = float(A.sum())
    got = float(A[np.isin(flab, np.flatnonzero(src >= 0))].sum())
    moved = np.array(moved) if moved else np.zeros(1)
    print(f"{swapped:,} of {len(PN):,} Poisson planes moved onto a measured plane")
    print(f"  they carry {100*got/tot:.1f}% of the mesh area")
    print(f"  they moved: median {np.median(moved)*1000:.2f} mm, "
          f"90th {np.percentile(moved,90)*1000:.2f} mm, "
          f"max {moved.max()*1000:.1f} mm")
    print(f"  {len(PN)-swapped:,} planes had no measured match and keep their own "
          f"equation (invented or unscanned)")
    np.savez_compressed(a.out, V=V, F=F, flab=flab, plane_n=PN, plane_d=PD,
                        measured=(src >= 0))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
