"""Fold the over-segmented patches back into surfaces, but only where it is free.

Lowering the minimum plane size found the small walls -- 82 patches became 475,
and the vertical/horizontal split went from absurd to plausible -- but it also
cut single surfaces into several patches, and the fit RMS went from 3.8 mm to
7.6 mm because the new patches are small and noisy. Most of that is not real:
one floor reported as five patches is five chances to be wrong about the same
plane.

Merging is therefore done with a veto rather than a tolerance. Two patches join
only if, AFTER refitting on their combined points, the residual is no worse
than the better of the two was to begin with, give or take --slack. That is the
rule that distinguishes a surface split in two from a genuine 20 mm step: the
step cannot be fitted by one plane without the residual blowing up, so it is
not merged, and the recess survives.

The merged plane is refitted by PCA for orientation and by MEDIAN for offset,
on every point of every member -- so a wall assembled from six patches ends up
better determined than any of them, which is the point of doing it at all.
"""
import argparse

import numpy as np


def refit(pts, n0):
    """Orientation by PCA, offset by median, then one robust pass."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    if n @ n0 < 0:
        n = -n
    d = float(np.median(pts @ n))
    r = pts @ n - d
    keep = np.abs(r) < max(3 * np.std(r), 0.004)
    if keep.sum() > 50:
        q = pts[keep]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[-1]
        if n @ n0 < 0:
            n = -n
        d = float(np.median(q @ n))
        r = q @ n - d
    return n, d, float(r.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ang", type=float, default=3.0, help="deg")
    ap.add_argument("--off", type=float, default=0.020,
                    help="m: how far apart two candidates may sit")
    ap.add_argument("--slack", type=float, default=0.0015,
                    help="m: how much worse the merged residual may be")
    a = ap.parse_args()

    z = np.load(a.planes)
    n, d, rms, cnt = z["n"], z["d"], z["rms"], z["count"]
    pts, lab = z["pts"], z["lab"]
    allp = z["all_pts"]
    member = {i: pts[lab == i] for i in range(len(n))}
    print(f"in: {len(n):,} patches, median RMS {np.median(rms)*1000:.2f} mm")

    cos = np.cos(np.radians(a.ang))
    order = np.argsort(-cnt)
    groups = []                      # each: [normal, offset, rms, [members]]
    for i in order:
        P = member[i]
        if len(P) < 3:
            continue
        placed = False
        for g in groups:
            s = 1.0 if n[i] @ g[0] > 0 else -1.0
            if abs(n[i] @ g[0]) < cos:
                continue
            if abs(s * d[i] - g[1]) > a.off:
                continue
            cand = np.vstack([P] + [member[k] for k in g[3]])
            nn, dd, rr = refit(cand, g[0])
            # the veto: merging must not make the fit worse
            if rr > min(g[2], rms[i]) + a.slack:
                continue
            g[0], g[1], g[2] = nn, dd, rr
            g[3].append(i)
            placed = True
            break
        if not placed:
            nn, dd, rr = refit(P, n[i])
            groups.append([nn, dd, rr, [i]])

    N = np.array([g[0] for g in groups])
    D = np.array([g[1] for g in groups])
    R = np.array([g[2] for g in groups])
    C = np.array([sum(cnt[k] for k in g[3]) for g in groups])
    parts = np.array([len(g[3]) for g in groups])
    nz = np.abs(N[:, 2])
    print(f"out: {len(N):,} surfaces "
          f"({(nz<0.2).sum()} vertical, {(nz>0.9).sum()} horizontal, "
          f"{((nz>=0.2)&(nz<=0.9)).sum()} sloped)")
    print(f"  median RMS {np.median(R)*1000:.2f} mm, "
          f"point-weighted {np.sum(R*C/C.sum())*1000:.2f} mm, "
          f"under 5 mm: {100*(R<0.005).mean():.0f}%")
    print(f"  patches per surface: median {np.median(parts):.0f}, "
          f"max {parts.max()}; {int((parts>1).sum())} surfaces were assembled "
          f"from more than one")
    se = R / np.sqrt(np.maximum(C, 1))
    print(f"  standard error of the offsets: median {np.median(se)*1000:.3f} mm")

    newlab = np.full(len(lab), -1, np.int32)
    for gi, g in enumerate(groups):
        for k in g[3]:
            newlab[lab == k] = gi
    np.savez_compressed(a.out, n=N, d=D, rms=R, count=C, parts=parts,
                        pts=pts, lab=newlab, all_pts=allp)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
