"""Fit the planes to the POINTS, not to a surface someone reconstructed for us.

Measured on this scan: 30% of the Poisson surface has no LAS point within
200 mm of it -- that is the reconstruction closing over what was never seen --
and where the scan does support it, the surface still sits about 9 mm off the
points. Every plane fitted from that mesh inherits both: a smoothing error we
cannot undo, and no way to tell an observed surface from an invented one.

So the planes come from the points. For each one:

  * the offset is the MEDIAN of its own points along its normal, not a
    least-squares fit. A least-squares plane is dragged by whatever is stuck to
    the wall -- a pipe, a bracket, the operator -- and by the long tail of
    stragglers behind it; the median is not.
  * a SUPPORT mask records where points actually are, at --support resolution.
    A plane is only allowed to exist where it was measured. This is the part
    that makes deviation detection honest: you can never report a deviation
    against a face nobody scanned.

Segmentation is RANSAC for the plane, then a spatial clustering of its inliers,
because one plane equation usually describes several separate pieces of a
building -- every window reveal on a facade is coplanar and none of them is the
same surface.
"""
import argparse
import time

import laspy
import numpy as np
import open3d as o3d

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load(path, lo=None, hi=None, cap=None):
    keep = []
    with laspy.open(path) as r:
        log(f"{path.split('/')[-1]}: {r.header.point_count:,} points")
        for chunk in r.chunk_iterator(6_000_000):
            p = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float32)
            if lo is not None:
                m = np.all((p >= lo) & (p <= hi), axis=1)
                p = p[m]
            if len(p):
                keep.append(p)
    P = np.vstack(keep) if keep else np.zeros((0, 3), np.float32)
    if cap and len(P) > cap:
        P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--las", required=True)
    ap.add_argument("--bounds", default=None,
                    help="npz whose V gives the volume to work in")
    ap.add_argument("--voxel", type=float, default=0.015,
                    help="m: downsample for segmentation only; the fits use "
                         "every point that lands on the plane")
    ap.add_argument("--dist", type=float, default=0.012,
                    help="m: RANSAC inlier distance")
    ap.add_argument("--min-pts", type=int, default=4000,
                    help="stop when the best plane is smaller than this")
    ap.add_argument("--cluster", type=float, default=0.20,
                    help="m: inliers further apart than this are separate faces")
    ap.add_argument("--min-cluster", type=int, default=800)
    ap.add_argument("--out", required=True, help="npz")
    a = ap.parse_args()

    lo = hi = None
    if a.bounds:
        Vb = np.load(a.bounds)["V"]
        lo, hi = Vb.min(0) - 0.02, Vb.max(0) + 0.02
        log(f"volume {np.round(lo,2)} .. {np.round(hi,2)}")
    P = load(a.las, lo, hi)
    log(f"{len(P):,} points in the volume")

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    ds = pc.voxel_down_sample(a.voxel)
    Q = np.asarray(ds.points, np.float32)
    log(f"{len(Q):,} points after a {a.voxel*1000:.0f} mm voxel downsample")

    # RANSAC on everything at once is greedy on AREA, so it eats the floor, the
    # ceiling and every slab first and runs out of budget before it reaches the
    # walls: the first run returned 103 horizontal planes and 7 vertical ones
    # for a storey with forty-odd walls. Binning by orientation first makes
    # walls compete with walls instead of against a 200 m2 floor.
    ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=40))
    nz = np.abs(np.asarray(ds.normals)[:, 2])
    bins = {"horizontal": nz > 0.90, "vertical": nz < 0.20,
            "sloped": (nz >= 0.20) & (nz <= 0.90)}
    planes, seg_pts, seg_bin = [], [], []
    for name, mask in bins.items():
        idx0 = np.flatnonzero(mask)
        if len(idx0) < a.min_pts:
            log(f"{name}: {len(idx0):,} points -- skipped")
            continue
        rest = ds.select_by_index(idx0)
        log(f"{name}: {len(idx0):,} points")
        k = 0
        while len(rest.points) >= a.min_pts:
            model, idx = rest.segment_plane(a.dist, 3, 400)
            if len(idx) < a.min_pts:
                break
            inl = rest.select_by_index(idx)
            lab = np.array(inl.cluster_dbscan(a.cluster, a.min_cluster))
            for c in range(lab.max() + 1):
                sel = np.asarray(inl.points)[lab == c]
                if len(sel) < a.min_cluster:
                    continue
                n = np.array(model[:3], float)
                n /= np.linalg.norm(n)
                planes.append((n, sel))
                seg_pts.append(sel)
            while len(seg_bin) < len(seg_pts):
                seg_bin.append(name)
            rest = rest.select_by_index(idx, invert=True)
            k += 1
            if k % 40 == 0:
                log(f"  {name}: {k} rounds, {len(rest.points):,} left")
        log(f"  {name}: {k} rounds, {len(rest.points):,} points unclaimed")
    log(f"{len(planes):,} plane patches found")

    # Robust offset from the FULL-resolution points that land on each plane
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64))))
    N, D, RMS, CNT = [], [], [], []
    for n, sel in planes:
        d0 = float(np.median(sel @ n))
        c = sel.mean(0)
        # every full-resolution point near this patch, then trim and re-median
        want = []
        for p in sel[::max(1, len(sel)//400)]:
            _, ii, _ = tree.search_radius_vector_3d(p.astype(np.float64), 0.05)
            want.append(np.asarray(ii))
        if not want:
            continue
        ii = np.unique(np.concatenate(want))
        pts = P[ii]
        r = pts @ n - d0
        keep = np.abs(r) < 3 * a.dist
        if keep.sum() < 200:
            continue
        d1 = float(np.median(pts[keep] @ n))
        rr = pts[keep] @ n - d1
        N.append(n)
        D.append(d1)
        RMS.append(float(rr.std()))
        CNT.append(int(keep.sum()))
    N, D = np.array(N), np.array(D)
    RMS, CNT = np.array(RMS), np.array(CNT)
    log(f"{len(N):,} planes fitted on full-resolution points")
    print(f"\nplane-fit RMS against the POINTS (mm):")
    w = CNT / CNT.sum()
    print(f"  median {np.median(RMS)*1000:5.2f}   "
          f"point-weighted mean {np.sum(RMS*w)*1000:5.2f}   "
          f"90th {np.percentile(RMS, 90)*1000:5.2f}")
    print(f"  planes under 5 mm RMS: {100*(RMS < 0.005).mean():.0f}%")
    # the membership too: the next stage needs it, and so does any picture of
    # what was actually segmented
    pts = np.vstack(seg_pts).astype(np.float32)
    lab = np.concatenate([np.full(len(s_), i, np.int32)
                          for i, s_ in enumerate(seg_pts)])
    np.savez_compressed(a.out, n=N, d=D, rms=RMS, count=CNT,
                        pts=pts, lab=lab,
                        orient=np.array(seg_bin[:len(seg_pts)]),
                        all_pts=Q[::3].astype(np.float32))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
