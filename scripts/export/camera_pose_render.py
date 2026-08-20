"""Render the scan from exactly where the camera was, and compare.

With odometerdata.txt there is no pose to solve for. The trajectory is 6-DoF at
1.415 Hz on the same clock as the LAS and the camera timestamps, and
slam_calib.yaml gives the rigid transform from the body frame to the colour
camera. So a frame's pose is a lookup and two matrix multiplies:

    world -> body     from the trajectory, interpolated at the frame's time
    body  -> camera   T_imu2clrcam_refine, from the calibration
    camera -> image   the pinhole the flat video already uses

Putting the coloured point cloud through that and setting it beside the
photograph is the test of the whole chain at once: clock alignment, trajectory,
extrinsics and intrinsics. If the two line up, everything that needs a camera
pose -- projecting door and window masks onto walls, texturing the model -- is
unblocked. If they do not, this shows which way they are out.
"""
import sys, io, time, argparse, subprocess
from pathlib import Path
import numpy as np, cv2, laspy

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

LAS = "data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las"
VIDEO = "data/Soulace/corcam_1.h265"
TS = "data/Soulace/corcam_1.ts"
ODO = "data/Soulace/odometerdata.txt"

# slam_calib.yaml: camera_instrinsic_parameters_clr_cam (Kannala-Brandt fisheye)
CX, CY, FX, FY = 1996.644928, 1537.618281, 806.839343, 805.774503
KB = np.array([1.641566e-02, -2.061062e-02, 1.042082e-02, -2.588998e-03])
# slam_calib.yaml: T_imu2clrcam_refine, row major
T_IC = np.array([[0.003678, 0.005680, 0.999977, 0.066647],
                 [-0.999991, -0.001935, 0.003689, 0.000457],
                 [0.001955, -0.999982, 0.005673, -0.025186],
                 [0.0, 0.0, 0.0, 1.0]])


def frame_image(idx, w, h, fov):
    """Frame `idx` of the fisheye, undistorted to a plain pinhole camera."""
    cmd = ["ffmpeg", "-v", "error", "-r", "30", "-i", VIDEO,
           "-vf", f"select=eq(n\\,{idx})", "-vframes", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    need = 4000*3000*3
    if len(raw) < need:
        raise SystemExit(f"could not decode frame {idx}")
    img = np.frombuffer(raw[:need], np.uint8).reshape(3000, 4000, 3)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    f = (w/2)/np.tan(np.radians(fov)/2)
    Knew = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, KB, np.eye(3), Knew, (w, h),
                                                 cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def pose_at(t, odo):
    """World->body rotation and body position, interpolated to time t."""
    from scipy.spatial.transform import Rotation as R, Slerp
    tt = odo[:, 1]
    i = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt)-2))
    q = R.from_quat(np.column_stack([odo[:, 9], odo[:, 10], odo[:, 11], odo[:, 8]]))
    sl = Slerp(tt[i:i+2], q[i:i+2])
    Rwb = sl([np.clip(t, tt[i], tt[i+1])])[0].as_matrix()
    pos = np.array([np.interp(t, tt, odo[:, 2+k]) for k in range(3)])
    return Rwb, pos


def render(P, C, Rwb, pos, w, h, fov, splat=2):
    """The coloured cloud through the camera at that pose."""
    Rbc, tbc = T_IC[:3, :3], T_IC[:3, 3]
    Q = (P - pos) @ Rwb                 # world -> body
    Q = Q @ Rbc.T + tbc                 # body  -> camera
    z = Q[:, 2]
    m = z > 0.2
    if not m.any():
        return np.zeros((h, w, 3), np.uint8), 0
    f = (w/2)/np.tan(np.radians(fov)/2)
    u = (Q[m, 0]/z[m])*f + w/2
    v = (Q[m, 1]/z[m])*f + h/2
    zz = z[m]; cc = C[m]
    k = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zz, cc = u[k].astype(np.int32), v[k].astype(np.int32), zz[k], cc[k]
    order = np.argsort(-zz)
    img = np.zeros((h, w, 3), np.uint8)
    u, v, cc = u[order], v[order], cc[order]
    for dx in range(splat):
        for dy in range(splat):
            uu = np.clip(u+dx, 0, w-1); vv = np.clip(v+dy, 0, h-1)
            img[vv, uu] = cc
    return img, int(k.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--radius", type=float, default=18.0)
    ap.add_argument("--points", type=int, default=6_000_000)
    ap.add_argument("--out", default="soulace_output/video/pose_check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    odo = np.loadtxt(ODO)
    cam_t = np.array([float(x) for x in io.open(TS).read().split() if x.strip()])
    lo, hi = odo[0, 1], odo[-1, 1]
    have = np.where((cam_t >= lo) & (cam_t <= hi))[0]
    idx = a.frame if a.frame is not None else int(have[len(have)//3])
    if cam_t[idx] < lo or cam_t[idx] > hi:
        near = int(have[np.argmin(np.abs(have-idx))])
        log(f"frame {idx} is {lo-cam_t[idx]:.0f} s before the trajectory starts; "
            f"using frame {near}")
        idx = near
    t = cam_t[idx]
    Rwb, pos = pose_at(t, odo)
    log(f"frame {idx}  t={t:.3f}s  camera at [{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}] m")

    log("reading the coloured cloud")
    with laspy.open(LAS) as r:
        p = r.read()
    P = np.column_stack([p.x, p.y, p.z]).astype(np.float32)
    C = np.column_stack([p.blue, p.green, p.red]).astype(np.float32)
    C = (C/(65535.0 if C.max() > 255 else 255.0)*255).astype(np.uint8)
    near = np.linalg.norm(P - pos, axis=1) < a.radius
    P, C = P[near], C[near]
    if len(P) > a.points:
        s = np.random.default_rng(0).choice(len(P), a.points, replace=False)
        P, C = P[s], C[s]
    log(f"{len(P):,} points within {a.radius:.0f} m")

    photo = frame_image(idx, a.width, a.height, a.fov)
    shot, n = render(P, C, Rwb, pos, a.width, a.height, a.fov)
    log(f"{n:,} points landed in the image")
    cv2.imwrite(str(out/f"photo_{idx:05d}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out/f"lidar_{idx:05d}.jpg"), shot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    pair = np.hstack([photo, shot])
    cv2.putText(pair, f"camera frame {idx}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    cv2.putText(pair, "LiDAR from the same pose", (a.width+16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(out/f"compare_{idx:05d}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out/f"blend_{idx:05d}.jpg"),
                cv2.addWeighted(photo, 0.5, shot, 0.5, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"wrote {out}/compare_{idx:05d}.jpg")


if __name__ == "__main__":
    main()
