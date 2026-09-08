"""Measure the fisheye image circle in an extracted corcam frame and compare it
with the two calibrated camera models in slam_calib.yaml."""
import sys, numpy as np, cv2
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f"image {w}x{h}")
mask = (img > 18).astype(np.uint8)
ys, xs = np.nonzero(mask)
cx, cy = xs.mean(), ys.mean()
r = np.hypot(xs - cx, ys - cy)
r95 = np.percentile(r, 99.5)
print(f"measured circle: centre=({cx:.1f},{cy:.1f})  radius(99.5pct)={r95:.1f} px")
print(f"  cols with signal: {xs.min()}..{xs.max()} (width {xs.max()-xs.min()})")
print(f"  rows with signal: {ys.min()}..{ys.max()} (height {ys.max()-ys.min()})")
# clr_cam: equidistant (Kannala-Brandt) fisheye, r = f*theta (+ poly)
cxc, cyc, fx, fy = 1996.644928, 1537.618281, 806.839343, 805.774503
k = [1.641566e-02, -2.061062e-02, 1.042082e-02, -2.588998e-03]
print(f"\nclr_cam model: cx={cxc} cy={cyc} fx={fx} fy={fy}")
print(f"  distance from measured centre: {np.hypot(cx-cxc, cy-cyc):.1f} px")
for deg in (80, 90, 95, 100):
    th = np.deg2rad(deg)
    rr = fx*(th + k[0]*th**3 + k[1]*th**5 + k[2]*th**7 + k[3]*th**9)
    print(f"  theta={deg:3d} deg -> r={rr:7.1f} px")
th = np.deg2rad(np.arange(60, 130, 0.5))
rr = fx*(th + k[0]*th**3 + k[1]*th**5 + k[2]*th**7 + k[3]*th**9)
print(f"  => measured radius {r95:.0f} px corresponds to theta = "
      f"{np.interp(r95, rr, np.rad2deg(th)):.1f} deg (half-FOV)")
