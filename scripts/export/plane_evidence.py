"""Decide, with evidence, whether two nearly-coincident planes are one surface.

Half the wall surfaces in this storey have a near-twin: 49 pairs within 3 degrees
and 50 mm of each other, overlapping for up to 9.6 m. Each twin generates its own
cutting line, which is where the slivers come from -- 701 of 1,133 plan cells end up
narrower than 60 mm, and the built region falls into 43 disconnected pieces held
apart by hairline cracks.

But merging on proximity would be worse than not merging. Three different things
look identical in plan:

  * one surface observed twice and over-segmented   -> merge
  * a genuine step or recess of a few centimetres   -> keep, it is content
  * the two faces of a thin wall                    -> keep, they are different walls

Proximity cannot tell them apart. Three pieces of evidence can:

  SIDE      every LAS return knows when it was measured, so the sensor's position at
            that moment is a lookup. Two patches seen from OPPOSITE sides are two
            faces of a solid and must never merge, however close they are.
  JOINT FIT refit both patches as one plane, on full-resolution points, weighted by
            area rather than by point count -- otherwise the bigger patch simply
            wins -- and require EVERY member to still fit. No patch may be discarded
            as an outlier to make the merge look good.
  STEP      bin the signed depth over the shared support. A connected region that
            differs by more than the noise allows is a step, and a step is a feature.

The thresholds scale with each patch's own measured RMS, because a 12 mm residual
means something different on a 3 mm patch than on a 9 mm one.
"""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree


def frame_of(n):
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def bin_weights(P, u, v, cell):
    """One vote per occupied cell, not per point: a densely-scanned metre of wall
    must not outvote a sparsely-scanned one when the plane is refitted."""
    ij = np.floor(np.c_[P @ u, P @ v] / cell).astype(np.int64)
    _, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def fit(P, w, n0):
    c = (P * w[:, None]).sum(0) / w.sum()
    Q = (P - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    n = vt[-1]
    if n @ n0 < 0:
        n = -n
    return n, float(np.average(P @ n, weights=w))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--las", required=True)
    ap.add_argument("--sensor", required=True, help="npz with traj, fidx, ts")
    ap.add_argument("--pairs", default=None,
                    help="explicit i,j pairs to test, e.g. 24:49,16:38,24:91,87:24")
    ap.add_argument("--ang", type=float, default=3.0, help="deg")
    ap.add_argument("--gap", type=float, default=0.05, help="m")
    ap.add_argument("--overlap", type=float, default=0.30, help="m")
    ap.add_argument("--cell", type=float, default=0.025, help="m: support bin")
    ap.add_argument("--side", type=float, default=0.90,
                    help="fraction of returns that must agree on the sensor side")
    ap.add_argument("--step-area", type=float, default=0.10, help="m2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    z = np.load(a.planes)
    N, D, RMS, pts, lab = z["n"], z["d"], z["rms"], z["pts"], z["lab"]
    s = np.load(a.sensor)
    traj, fidx = s["traj"], s["fidx"]

    import laspy
    keep = []
    with laspy.open(a.las) as r:
        for ch in r.chunk_iterator(8_000_000):
            keep.append(np.column_stack([ch.x, ch.y, ch.z]).astype(np.float32))
    P = np.vstack(keep)
    print(f"{len(P):,} full-resolution points, {len(traj):,} sensor positions")
    # the sensor stream must be in the same frame as the planes, or every side
    # test is meaningless. Check before trusting it.
    lo, hi = P.min(0), P.max(0)
    inside = np.all((traj > lo - 2) & (traj < hi + 2), axis=1).mean()
    print(f"sensor positions inside the cloud's bounds: {100*inside:.0f}%"
          + ("" if inside > 0.8 else "   <-- FRAME MISMATCH, side test unusable"))
    tree = cKDTree(P)

    def evidence(i, j):
        Pi, Pj = pts[lab == i], pts[lab == j]
        if len(Pi) < 50 or len(Pj) < 50:
            return None
        n_i, n_j = N[i], N[j]
        dot = float(n_i @ n_j)
        sgn = 1.0 if dot > 0 else -1.0
        u, v = frame_of(n_i)
        out = {"a": int(i), "b": int(j),
               "angle_deg": float(np.degrees(np.arccos(min(1.0, abs(dot))))),
               "offset_mm": float(abs(D[i] - sgn*D[j]) * 1000),
               "rms_mm": [float(RMS[i]*1000), float(RMS[j]*1000)]}

        # SIDE: which side of the plane was the sensor on, per return
        sides = []
        for Pk in (Pi, Pj):
            q = Pk[::max(1, len(Pk)//4000)]
            _, ii = tree.query(q, workers=-1)
            sp = traj[np.clip(fidx[ii].astype(np.int64), 0, len(traj)-1)]
            sd = np.sign((sp - q) @ n_i)
            sides.append(float(max((sd > 0).mean(), (sd < 0).mean())))
            out.setdefault("side_sign", []).append(float(np.sign(sd.sum())))
        out["side_agreement"] = sides
        out["opposite_faces"] = bool(out["side_sign"][0] * out["side_sign"][1] < 0)

        # JOINT FIT on full-resolution points, area-weighted, no outlier trimming
        merged = []
        for Pk in (Pi, Pj):
            q = Pk[::max(1, len(Pk)//20000)]
            _, ii = tree.query(q, workers=-1)
            merged.append(P[np.unique(ii)])
        A, B = merged
        allp = np.vstack([A, B])
        w = np.concatenate([bin_weights(A, u, v, a.cell), bin_weights(B, u, v, a.cell)])
        nn, dd = fit(allp, w, n_i)
        res = []
        for Pk in (A, B):
            r = np.abs(Pk @ nn - dd)
            res.append({"median_mm": float(np.median(r)*1000),
                        "p95_mm": float(np.percentile(r, 95)*1000)})
        out["joint"] = res
        ok = []
        for k, (Pk, rr) in enumerate(zip((A, B), res)):
            rms_k = float(RMS[[i, j][k]])
            ok.append(rr["median_mm"] <= max(5.0, 1500*rms_k)
                      and rr["p95_mm"] <= max(12.0, 3000*rms_k))
        out["joint_ok"] = ok

        # STEP: signed depth binned over the shared support
        du = np.concatenate([A @ u, B @ u])
        dv = np.concatenate([A @ v, B @ v])
        dz = allp @ nn - dd
        gi = np.floor(np.c_[du, dv] / 0.05).astype(np.int64)
        key, inv = np.unique(gi, axis=0, return_inverse=True)
        med = np.zeros(len(key))
        np.maximum.at(med, inv, 0)
        sums = np.bincount(inv, weights=dz, minlength=len(key))
        cnts = np.bincount(inv, minlength=len(key))
        mean = sums / np.maximum(cnts, 1)
        tol = max(0.010, 3*np.sqrt(RMS[i]**2 + RMS[j]**2))
        stepped = (np.abs(mean) > tol) & (cnts >= 4)
        out["step_area_m2"] = float(stepped.sum() * 0.05 * 0.05)
        out["step_tol_mm"] = float(tol*1000)

        if out["opposite_faces"]:
            out["verdict"] = "two faces of a solid - never merge"
        elif min(sides) < a.side:
            out["verdict"] = "ambiguous - seen from both sides, not a wall face"
        elif out["step_area_m2"] > a.step_area:
            out["verdict"] = "step or recess - keep separate"
        elif all(ok):
            out["verdict"] = "same face - MERGE"
        else:
            out["verdict"] = "does not fit one plane - keep separate"
        return out

    if a.pairs:
        want = [tuple(int(x) for x in p.split(":")) for p in a.pairs.split(",")]
    else:
        vert = [i for i in range(len(N)) if abs(N[i, 2]) < 0.2 and (lab == i).sum() > 50]
        want = []
        cos = np.cos(np.radians(a.ang))
        for k, i in enumerate(vert):
            for j in vert[k+1:]:
                dot = float(N[i] @ N[j])
                if abs(dot) < cos:
                    continue
                sgn = 1.0 if dot > 0 else -1.0
                if abs(D[i] - sgn*D[j]) > a.gap:
                    continue
                want.append((i, j))
        print(f"{len(want)} candidate pairs")

    rows = []
    for i, j in want:
        e = evidence(i, j)
        if e is None:
            continue
        rows.append(e)
        print(f"{i:4d}/{j:<4d} {e['angle_deg']:4.2f} deg {e['offset_mm']:6.1f} mm  "
              f"side {e['side_agreement'][0]:.2f}/{e['side_agreement'][1]:.2f}"
              f"{' OPPOSED' if e['opposite_faces'] else '        '}  "
              f"joint p95 {e['joint'][0]['p95_mm']:5.1f}/{e['joint'][1]['p95_mm']:5.1f}  "
              f"step {e['step_area_m2']:5.2f} m2  ->  {e['verdict']}")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
