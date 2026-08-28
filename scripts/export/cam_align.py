"""Line the camera up with the model, and measure how well it lines up.

The calibration has everything the projection needs -- T_imu2clrcam_refine for
the extrinsic, a Kannala-Brandt fisheye for the intrinsic, and odometerdata.txt
for the trajectory on the same clock as the frames. Putting a sparse point
cloud through that chain LOOKS about right, which is exactly the problem: about
right is not a number, and every later step (projecting semantics onto the
walls, texturing) inherits whatever the error is.

So this measures it. The model is rendered from the frame's pose as a depth
image, and the edges of that render -- depth steps and creases, which is where
a wall meets a floor or an opening starts -- are compared against the edges in
the photograph. The score is the photo's edge strength sampled where the model
says an edge should be. A correct pose puts model edges on top of photo edges
and scores high; a pose a few degrees out puts them on blank plaster.

Then it searches: three small rotations and a time offset, coarse grid first,
then Nelder-Mead. What comes out is the correction, the score before and after,
and an overlay per frame so the number can be checked by eye.
"""
import argparse
import io
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R, Slerp

VIDEO = "data/Soulace/corcam_1.h265"
TS = "data/Soulace/corcam_1.ts"
ODO = "data/Soulace/odometerdata.txt"
FPS = 30.0

CX, CY, FX, FY = 1996.644928, 1537.618281, 806.839343, 805.774503
KB = np.array([1.641566e-02, -2.061062e-02, 1.042082e-02, -2.588998e-03])
T_IC = np.array([[0.003678, 0.005680, 0.999977, 0.066647],
                 [-0.999991, -0.001935, 0.003689, 0.000457],
                 [0.001955, -0.999982, 0.005673, -0.025186],
                 [0.0, 0.0, 0.0, 1.0]])


def decode(indices, out_dir, scale=0.25):
    """Every frame we want, in ONE pass.

    The .h265 is a raw elementary stream with no timestamps, so -ss cannot seek
    it -- it silently returns nothing, which is what made the first run decode
    zero frames. Selecting the frame numbers in a single decode is the only way
    that works, and it is also the cheapest: one pass instead of one per frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("f_*.png"):
        f.unlink()
    sel = "+".join(f"eq(n\,{i})" for i in indices)
    w, h = int(4000*scale), int(3000*scale)
    cmd = ["ffmpeg", "-v", "error", "-r", str(FPS), "-i", VIDEO,
           "-vf", f"select='{sel}',scale={w}:{h}", "-vsync", "0",
           "-frames:v", str(len(indices)), str(out_dir / "f_%03d.png")]
    subprocess.run(cmd, check=False)
    got = sorted(out_dir.glob("f_*.png"))
    return {i: cv2.imread(str(p)) for i, p in zip(indices, got)}


def undistort(img, w, h, fov, scale=0.25):
    """Fisheye to a plain pinhole, with the intrinsics scaled to match."""
    K = np.array([[FX*scale, 0, CX*scale], [0, FY*scale, CY*scale], [0, 0, 1]])
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    Knew = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, KB, np.eye(3), Knew, (w, h),
                                                 cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def pose_at(t, odo):
    tt = odo[:, 1]
    i = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
    q = R.from_quat(np.column_stack([odo[:, 9], odo[:, 10], odo[:, 11], odo[:, 8]]))
    Rwb = Slerp(tt[i:i+2], q[i:i+2])([np.clip(t, tt[i], tt[i+1])])[0].as_matrix()
    pos = np.array([np.interp(t, tt, odo[:, 2+k]) for k in range(3)])
    return Rwb, pos


def depth_edges(scene, Rwb, pos, dR, w, h, fov):
    """What the model says the picture should look like, as an edge map."""
    Rbc = dR @ T_IC[:3, :3]
    tbc = T_IC[:3, 3]
    # camera -> world:  x_w = Rwb @ (Rbc^T (x_c - tbc)) + pos
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    d = np.stack([(j - w/2) / f, (i - h/2) / f, np.ones_like(j, float)], -1)
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    dirs = (d.reshape(-1, 3) @ Rbc) @ Rwb.T
    eye = Rwb @ (-Rbc.T @ tbc) + pos
    rays = np.concatenate([np.broadcast_to(eye, dirs.shape), dirs], 1)
    res = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    z = res["t_hit"].numpy().reshape(h, w)
    n = res["primitive_normals"].numpy().reshape(h, w, 3)
    z[~np.isfinite(z)] = 0.0
    g = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, 3))
    ng = sum(np.abs(cv2.Sobel(n[:, :, k], cv2.CV_32F, 1, 0, 3)) +
             np.abs(cv2.Sobel(n[:, :, k], cv2.CV_32F, 0, 1, 3)) for k in range(3))
    e = np.clip(g / max(g.max(), 1e-6), 0, 1) + np.clip(ng / max(ng.max(), 1e-6), 0, 1)
    return e / max(e.max(), 1e-6), z


def photo_edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 40, 120).astype(np.float32) / 255.0
    return cv2.GaussianBlur(e, (0, 0), 3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", default="Soulace Sketchup/Soulace_L0_ground_edges.stl")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--out", default="soulace_output/video/align")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m = o3d.io.read_triangle_mesh(a.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    lo = np.asarray(m.vertices).min(0)
    hi = np.asarray(m.vertices).max(0)
    print(f"mesh {a.mesh}: {len(m.triangles):,} tris, z {lo[2]:.2f}..{hi[2]:.2f} m")

    odo = np.loadtxt(ODO)
    cam_t = np.array([float(x) for x in io.open(TS).read().split() if x.strip()])
    ok = np.where((cam_t >= odo[0, 1]) & (cam_t <= odo[-1, 1]))[0]
    # Standing inside the bounding box is not the same as standing inside the
    # building: the first attempt picked four frames in the alley outside, where
    # the model projects almost nothing and the score therefore measures
    # nothing. A candidate has to actually SEE the model -- most of its rays
    # landing on it, from close enough that the edges are sharp.
    good = []
    for idx in ok[::17]:
        Rwb, pos = pose_at(cam_t[idx], odo)
        if not (lo[2] - 0.5 < pos[2] < hi[2] + 0.5):
            continue
        _, zz = depth_edges(scene, Rwb, pos, np.eye(3), 80, 45, a.fov)
        seen = (zz > 0.3) & (zz < 12.0)
        if seen.mean() > 0.6:
            good.append((float(seen.mean()), int(idx)))
    if not good:
        raise SystemExit("no frame looks at this storey")
    good.sort(reverse=True)
    good = [i for _, i in good]
    step = max(1, len(good) // a.frames)
    picked = sorted(good[::step][:a.frames])
    print(f"{len(good):,} frames see the model; using {picked}")

    raw = decode(picked, Path(a.out) / "raw")
    shots = []
    for idx in picked:
        if raw.get(idx) is None:
            continue
        img = undistort(raw[idx], a.w, a.h, a.fov)
        Rwb, pos = pose_at(cam_t[idx], odo)
        shots.append((idx, img, photo_edges(img), Rwb, pos))
    print(f"{len(shots)} frames decoded")

    def score(p):
        # The fourth parameter is a CLOCK offset, and on a handheld scanner it
        # matters more than the extrinsic: the operator turns their head at a
        # radian a second, so 100 ms of skew is 6 degrees of pointing error --
        # far more than the degree of mounting error being searched for. Fitting
        # the rotation without it just absorbs the skew of whichever frames were
        # picked.
        dR = R.from_rotvec(p[:3]).as_matrix()
        dt = p[3] if len(p) > 3 else 0.0
        tot = 0.0
        for idx, img, pe, _, _ in shots:
            Rwb, pos = pose_at(cam_t[idx] + dt, odo)
            me, z = depth_edges(scene, Rwb, pos, dR, a.w, a.h, a.fov)
            w8 = me > 0.25
            if w8.sum() < 200:
                continue
            tot += float(pe[w8].mean())
        return -tot / max(len(shots), 1)

    base = -score(np.zeros(4))
    print(f"score as calibrated: {base:.4f}")
    best, bp = base, np.zeros(4)
    for dt in np.arange(-0.40, 0.401, 0.04):
        p = np.zeros(4)
        p[3] = dt
        v = -score(p)
        if v > best:
            best, bp = v, p.copy()
    print(f"best clock offset alone: {bp[3]:+.3f} s -> {best:.4f}")
    for ax in range(3):
        for dd in np.radians([-3, -2, -1, -0.5, 0.5, 1, 2, 3]):
            p = bp.copy()
            p[ax] = dd
            v = -score(p)
            if v > best:
                best, bp = v, p.copy()
    print(f"plus a nudge: {np.degrees(bp[:3]).round(2)} deg, "
          f"{bp[3]:+.3f} s -> {best:.4f}")
    r = minimize(score, bp, method="Nelder-Mead",
                 options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-5})
    fin = -r.fun
    print(f"refined: {np.degrees(r.x[:3]).round(3)} deg, {r.x[3]:+.3f} s "
          f"-> {fin:.4f} ({100*(fin-base)/max(base,1e-9):+.1f}% on the "
          f"calibrated pose)")

    dR = R.from_rotvec(r.x[:3]).as_matrix()
    for idx, img, pe, _, _ in shots:
        Rwb, pos = pose_at(cam_t[idx] + r.x[3], odo)
        for tag, mat, tt in (("as_calibrated", np.eye(3), 0.0),
                             ("refined", dR, r.x[3])):
            Rw, po = pose_at(cam_t[idx] + tt, odo)
            me, z = depth_edges(scene, Rw, po, mat, a.w, a.h, a.fov)
            ov = img.copy()
            ov[me > 0.25] = (0, 255, 255)
            cv2.imwrite(str(out / f"{idx:05d}_{tag}.jpg"), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(out / "dR.npy", dR)
    print(f"overlays -> {out}")


if __name__ == "__main__":
    main()
