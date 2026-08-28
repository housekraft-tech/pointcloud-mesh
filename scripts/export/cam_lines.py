"""Align the camera by matching LINES, and prove the score can tell right from wrong.

The first attempt scored a pose by sampling blurred Canny edges wherever the
model said an edge should be. Swept over a twelve-second range of clock offset
that score was flat noise -- 0.03 to 0.07, no peak, six seconds out scoring as
well as zero -- so the optimiser was fitting nothing and the corrections it
reported meant nothing. An objective that rewards landing on ANY texture cannot
distinguish a pose, and a construction photograph is nothing but texture.

A line has an ORIENTATION, and that is what makes the difference. A wall/floor
junction projected to the wrong place lands across a photographed line at an
angle, and an angle is expensive. So both sides are reduced to straight
segments -- the model's from its depth and normal discontinuities, the photo's
from a line-segment detector -- and a model segment only scores where a photo
segment agrees with it in DIRECTION as well as position.

The mesh matters as much as the metric: this aligns against building fabric
only, the large axis-aligned surfaces, because a scaffold pole generates model
edges that are real and useless.

Before believing any correction, --landscape sweeps one parameter and prints
the curve. A metric worth optimising has a peak. If the curve is flat, the
answer is that the alignment cannot be measured this way -- which is a result,
not a failure, and better found here than three steps downstream.
"""
import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from cam_align import (ODO, TS, T_IC, decode, undistort, pose_at, depth_edges)


def segments(mask, min_len):
    """Straight runs in an edge mask, as (x1, y1, x2, y2)."""
    m = (mask > 0.25).astype(np.uint8) * 255
    ls = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=30,
                         minLineLength=min_len, maxLineGap=4)
    return np.zeros((0, 4)) if ls is None else ls.reshape(-1, 4).astype(float)


def photo_segments(img, min_len):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 50, 150)
    ls = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=40,
                         minLineLength=min_len, maxLineGap=4)
    return np.zeros((0, 4)) if ls is None else ls.reshape(-1, 4).astype(float)


def parts(seg):
    p0 = seg[:, :2]
    p1 = seg[:, 2:]
    d = p1 - p0
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    ang = np.arctan2(d[:, 1], d[:, 0]) % np.pi
    return p0[ok], p1[ok], (p0[ok] + p1[ok]) / 2, L[ok], ang[ok]


def line_score(mseg, pseg, ang_tol, dist_tol):
    """Model line length that a photo line agrees with, in place AND direction."""
    if not len(mseg) or not len(pseg):
        return 0.0
    mp0, mp1, mc, mL, ma = parts(mseg)
    pp0, pp1, pc, pL, pa = parts(pseg)
    if not len(mL) or not len(pL):
        return 0.0
    da = np.abs(ma[:, None] - pa[None, :])
    da = np.minimum(da, np.pi - da)
    d = pp1 - pp0
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    off = np.abs((mc[:, None, :] - pp0[None, :, :]) * nrm[None, :, :]).sum(-1)
    good = (da < ang_tol) & (off < dist_tol)
    w = np.where(good, np.exp(-off / dist_tol) * np.exp(-da / ang_tol), 0.0)
    return float((mL * w.max(1)).sum() / max(mL.sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", default="C:/Users/PC/AppData/Local/Temp/L0_fabric.stl")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--min-len", type=int, default=40, help="px")
    ap.add_argument("--ang-tol", type=float, default=0.10, help="rad")
    ap.add_argument("--dist-tol", type=float, default=12.0, help="px")
    ap.add_argument("--cand", type=int, default=600,
                    help="how many frames to screen before picking")
    ap.add_argument("--seen", type=float, default=0.6,
                    help="fraction of rays that must land on the fabric")
    ap.add_argument("--near", type=float, default=0.03,
                    help="s: a frame further than this from a trajectory sample "
                         "has an interpolated orientation, not a measured one")
    ap.add_argument("--offset", type=int, default=0,
                    help="take a different, disjoint frame set -- the test of "
                         "whether a peak is in the rig or in the sample")
    ap.add_argument("--min-seg", type=int, default=15,
                    help="a frame with fewer photo lines than this is noise")
    ap.add_argument("--landscape", choices=["dt", "rx", "ry", "rz"], default=None,
                    help="sweep one parameter and print the curve instead of fitting")
    ap.add_argument("--out", default="soulace_output/video/lines")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m = o3d.io.read_triangle_mesh(a.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    lo = np.asarray(m.vertices).min(0)
    hi = np.asarray(m.vertices).max(0)
    print(f"mesh {a.mesh}: {len(m.triangles):,} tris")

    odo = np.loadtxt(ODO)
    cam_t = np.array([float(x) for x in io.open(TS).read().split() if x.strip()])
    ok = np.where((cam_t >= odo[0, 1] + 1) & (cam_t <= odo[-1, 1] - 1))[0]
    # The trajectory is 1.41 Hz and the device turns a MEDIAN of 10.6 degrees
    # between consecutive samples -- 33.9 at the 90th percentile, 82.9 at worst.
    # Slerp across a gap like that is a straight line drawn through a head turn,
    # so a frame in the middle of one can be several degrees out on its own,
    # which is far larger than the degree of mounting error being measured. That
    # error is per-frame and changes sign, so fitting a single rotation to it is
    # fitting noise -- and it is why the peak moved from -1.2 s to +1.6 s when
    # the frame set changed. Only frames that sit ON a trajectory sample are
    # admissible. 3,607 of the 27,924 do at 50 ms, which is plenty.
    near = np.abs(cam_t[:, None] - odo[None, :, 1]).min(1)
    ok = ok[near[ok] < a.near]
    print(f"{len(ok):,} frames sit within {a.near*1000:.0f} ms of a trajectory sample")
    good = []
    for idx in ok[::max(1, len(ok)//a.cand)]:
        Rwb, pos = pose_at(cam_t[idx], odo)
        if not (lo[2] - 0.5 < pos[2] < hi[2] + 0.5):
            continue
        _, z = depth_edges(scene, Rwb, pos, np.eye(3), 80, 45, a.fov)
        seen = (z > 0.3) & (z < 10.0)
        if seen.mean() > a.seen:
            good.append(int(idx))
    # spread across the walk, not clustered in whichever room scored best: the
    # last run picked six frames inside 47 seconds of each other, which is one
    # view sampled six times, not six views
    good.sort()
    picked = sorted(good[a.offset::max(1, len(good)//a.frames)][:a.frames])
    print(f"{len(good):,} frames see the fabric from a sampled pose; using {picked}")

    raw = decode(picked, out / "raw")
    shots = []
    for idx in picked:
        if raw.get(idx) is None:
            continue
        img = undistort(raw[idx], a.w, a.h, a.fov)
        ps = photo_segments(img, a.min_len)
        # A frame with three detected lines contributes noise and a vote. Two of
        # the first six were like that -- one segment, three segments -- and with
        # only four frames carrying the score the sweep came out multi-modal.
        if len(ps) < a.min_seg:
            print(f"  frame {idx}: {len(ps)} photo segments -- dropped")
            continue
        shots.append((idx, img, ps))
        print(f"  frame {idx}: {len(ps)} photo segments")

    def score(p):
        dR = R.from_rotvec(p[:3]).as_matrix()
        dt = p[3]
        tot = 0.0
        for idx, img, ps in shots:
            Rwb, pos = pose_at(cam_t[idx] + dt, odo)
            me, _ = depth_edges(scene, Rwb, pos, dR, a.w, a.h, a.fov)
            ms = segments(me, a.min_len)
            tot += line_score(ms, ps, a.ang_tol, a.dist_tol)
        return -tot / max(len(shots), 1)

    if a.landscape:
        print(f"\nsweeping {a.landscape} -- a usable metric has a peak here")
        rng = (np.arange(-2.0, 2.01, 0.2) if a.landscape == "dt"
               else np.radians(np.arange(-6, 6.1, 0.5)))
        k = {"dt": 3, "rx": 0, "ry": 1, "rz": 2}[a.landscape]
        vals = []
        for v in rng:
            p = np.zeros(4)
            p[k] = v
            s = -score(p)
            vals.append(s)
            unit = "s" if a.landscape == "dt" else "deg"
            shown = v if a.landscape == "dt" else np.degrees(v)
            print(f"  {shown:+6.2f} {unit} -> {s:.4f}  " + "#" * int(s * 300))
        vals = np.array(vals)
        peak = vals.max()
        floor_ = np.median(vals)
        print(f"peak {peak:.4f}, median {floor_:.4f}, "
              f"contrast {peak/max(floor_,1e-9):.2f}x "
              f"({'usable' if peak > 1.5*floor_ else 'FLAT -- not usable'})")
        return

    base = -score(np.zeros(4))
    print(f"score as calibrated: {base:.4f}")
    best, bp = base, np.zeros(4)
    for dt in np.arange(-1.0, 1.01, 0.1):
        p = np.zeros(4)
        p[3] = dt
        s = -score(p)
        if s > best:
            best, bp = s, p.copy()
    for ax in range(3):
        for dd in np.radians([-4, -3, -2, -1, 1, 2, 3, 4]):
            p = bp.copy()
            p[ax] = dd
            s = -score(p)
            if s > best:
                best, bp = s, p.copy()
    print(f"coarse: {np.degrees(bp[:3]).round(2)} deg, {bp[3]:+.3f} s -> {best:.4f}")
    r = minimize(score, bp, method="Nelder-Mead",
                 options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
    fin = -r.fun
    print(f"refined: {np.degrees(r.x[:3]).round(3)} deg, {r.x[3]:+.3f} s -> {fin:.4f} "
          f"({100*(fin-base)/max(base,1e-9):+.1f}%)")

    dR = R.from_rotvec(r.x[:3]).as_matrix()
    for idx, img, ps in shots:
        for tag, mat, tt in (("before", np.eye(3), 0.0), ("after", dR, r.x[3])):
            Rw, po = pose_at(cam_t[idx] + tt, odo)
            me, _ = depth_edges(scene, Rw, po, mat, a.w, a.h, a.fov)
            ov = img.copy()
            for x1, y1, x2, y2 in segments(me, a.min_len).astype(int):
                cv2.line(ov, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for x1, y1, x2, y2 in ps.astype(int):
                cv2.line(ov, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.imwrite(str(out / f"{idx:05d}_{tag}.jpg"), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(out / "correction.npy", r.x)
    print(f"overlays -> {out}  (yellow = model lines, magenta = photo lines)")


if __name__ == "__main__":
    main()
