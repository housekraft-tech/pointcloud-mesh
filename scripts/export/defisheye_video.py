"""Turn the scanner's fisheye video into a flat, watchable MP4.

corcam_1.h265 is a 4000x3000 circular fisheye at 30 Hz -- 27,924 frames, one per
line of corcam_1.ts, on the same clock as the LiDAR. Straight out of the device
it is a bulging disc in the middle of a black frame, which is unwatchable and
also unusable for anything that assumes a normal camera.

slam_calib.yaml gives the lens as a Kannala-Brandt fisheye:

    camera_instrinsic_parameters_clr_cam:
      4000 3000 2.0  cx 1996.64  cy 1537.62  fx 806.84  fy 805.77
      k1 1.6416e-02  k2 -2.0611e-02  k3 1.0421e-02  k4 -2.5890e-03

so the picture can be undistorted properly rather than merely cropped. Two ways
out, because "flat" means different things:

  rect       a normal rectilinear camera, straight lines straight. Cannot cover
             the whole fisheye -- a 180 degree view needs infinite width -- so
             it is cut to --fov, 110 degrees by default.
  equirect   the full hemisphere unrolled into a 2:1 panorama. Keeps everything
             the lens saw; verticals stay straight, horizontals bow.

Frames are decoded by ffmpeg, remapped by OpenCV, and encoded back by ffmpeg.
Nothing is held in memory beyond a frame at a time, so the whole 15 minutes runs
in bounded RAM.
"""
import sys, json, argparse, subprocess, time, shutil
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# from slam_calib.yaml, camera_instrinsic_parameters_clr_cam
W_SRC, H_SRC = 4000, 3000
CX, CY = 1996.644928, 1537.618281
FX, FY = 806.839343, 805.774503
K1, K2, K3, K4 = 1.641566e-02, -2.061062e-02, 1.042082e-02, -2.588998e-03


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries",
                          "stream=codec_name,width,height,avg_frame_rate,nb_frames",
                          "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0] if out.strip() else {}


def maps(mode, out_w, out_h, fov_deg, scale):
    """Pixel lookup from the output image back into the fisheye."""
    K = np.array([[FX*scale, 0, CX*scale], [0, FY*scale, CY*scale], [0, 0, 1]])
    D = np.array([K1, K2, K3, K4])
    if mode == "rect":
        f = (out_w/2) / np.tan(np.radians(fov_deg)/2)
        Knew = np.array([[f, 0, out_w/2], [0, f, out_h/2], [0, 0, 1]])
        import cv2
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), Knew, (out_w, out_h), cv2.CV_16SC2)
        return m1, m2
    # equirectangular: longitude across, latitude down, then through the lens
    lon = (np.arange(out_w) + 0.5)/out_w*np.pi - np.pi/2      # -90..90
    lat = (np.arange(out_h) + 0.5)/out_h*np.pi - np.pi/2
    LON, LAT = np.meshgrid(lon, lat)
    x = np.cos(LAT)*np.sin(LON); y = np.sin(LAT); z = np.cos(LAT)*np.cos(LON)
    th = np.arctan2(np.hypot(x, y), z)
    r = th*(1 + K1*th**2 + K2*th**4 + K3*th**6 + K4*th**8)
    phi = np.arctan2(y, x)
    mx = (FX*scale*r*np.cos(phi) + CX*scale).astype(np.float32)
    my = (FY*scale*r*np.sin(phi) + CY*scale).astype(np.float32)
    return mx, my


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="data/Soulace/corcam_1.h265")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rect", "equirect"], default="rect")
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="decode the fisheye at this fraction of 4000x3000")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = all of it")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()
    import cv2
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")

    info = probe(a.video)
    log(f"source: {info.get('codec_name')} {info.get('width')}x{info.get('height')} "
        f"{info.get('avg_frame_rate')}")
    sw, sh = int(W_SRC*a.scale), int(H_SRC*a.scale)
    m1, m2 = maps(a.mode, a.width, a.height, a.fov, a.scale)
    log(f"{a.mode}: {sw}x{sh} fisheye -> {a.width}x{a.height}"
        + (f" at {a.fov:.0f} deg" if a.mode == "rect" else " panorama"))

    dec = ["ffmpeg", "-v", "error"]
    if a.start:
        dec += ["-ss", str(a.start)]
    dec += ["-r", str(a.fps), "-i", str(a.video)]
    if a.duration:
        dec += ["-t", str(a.duration)]
    dec += ["-vf", f"scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", str(a.out)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p_in = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10**8)
    p_out = subprocess.Popen(enc, stdin=subprocess.PIPE)
    nbytes = sw*sh*3
    n = 0
    try:
        while True:
            buf = p_in.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
            flat = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
            p_out.stdin.write(flat.tobytes())
            n += 1
            if n % 300 == 0:
                log(f"  {n:,} frames ({n/a.fps:.0f} s of video)")
    finally:
        p_in.stdout.close(); p_out.stdin.close()
        p_in.wait(); p_out.wait()
    sz = Path(a.out).stat().st_size/1e6 if Path(a.out).exists() else 0
    log(f"wrote {a.out}: {n:,} frames, {n/a.fps:.1f} s, {sz:.0f} MB")


if __name__ == "__main__":
    main()
