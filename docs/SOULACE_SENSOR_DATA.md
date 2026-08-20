# Soulace auxiliary sensor data — reverse-engineering report

**Scan:** `data/Soulace/`, SLAM2000 handheld (FeimaRobotics / FJDynamics), SN `SLAM200024330206`,
Sunday 19 April 2026, 10:45:45 – 11:01:18 local.
**Status:** read-only investigation. Every number below was measured; probe scripts are in
`scripts/experiments/soulace_probe/`.

---

## 0. Executive summary

| File | Size | What it actually is | Decoded? |
|---|---|---|---|
| `20260419-104545_Lp_Imu.fmimr` | 26.1 MB | Body IMU (InvenSense **ICM-40609**), 6-axis + temp, **1000.00 Hz**, absolute-timestamped | **Fully** |
| `20260419-104545_Hp_Imu.fmimr` | 30.9 MB | Second body IMU (`feima-i2000-imu`), 6-axis, 24/28-bit, ~999.85 Hz | **Framing + time yes; channel bit-split confirmed by correlation** |
| `20260419-104545_Lidar_Imu.imu` | 9.2 MB | **Livox** LiDAR's built-in ICM-40609, float32 rad/s + g, ~193 Hz, mounted **on the spinning head** | **Fully (values); timestamp partially** |
| `20260419-104545_Ec_Data.fmraster` | 24.7 MB | **Rotary encoder of the spinning LiDAR head** — absolute azimuth, 2048 counts/rev, 1351 Hz | **Fully** |
| `corcam_1.ts` | 356 KB | ASCII list of **27,924 camera frame timestamps**, 30.00 Hz | **Fully** |
| `optcam_1.ts` | 356 KB | ASCII list of 27,926 frame timestamps for the *second* camera (whose video is absent) | **Fully** |
| `corcam_1.h265` | 1.5 GB | Raw HEVC, **4000×3000**, 27,924 frames, keyframe every 30 frames | **Fully (ffprobe + frame extract)** |
| `slam_calib.yaml` | 1.9 MB | Full sensor extrinsics + two camera models. 18 lines; 99.7 % of the bytes are one opaque LUT | **Fully (except 2 vendor blobs)** |

**The single most important finding:** *every* stream — both IMUs, the encoder, both camera
timestamp lists, **and the LAS `gps_time`** — is on **one common clock**. No offset estimation
is needed to fuse them. See §1.

**The second most important finding:** the scanner is a **Livox head spinning at 39.5 rpm on a
2048-count absolute encoder**, and we can read that encoder at 1351 Hz. This is what makes true
trajectory recovery tractable (§6.1). It also explains the "~0.7 s frames" our current pipeline
detects in the LAS `gps_time`: one revolution is **1.519 s**, so 0.7 s is a *half*-revolution.

**The main gap:** there is **no pose/trajectory table anywhere** in this export, and the raw
LiDAR file that would make trajectory recovery trivial is **listed in the manifest but missing
from the folder** (§7).

---

## 1. The common clock (evidence first)

The three `FM`-framed binaries encode time as `hour, minute, second, nanoseconds`. The hour
field is only **3 bits (hour mod 8)**:

- Filename says the scan started `10:45:45`; the first Lp IMU record decodes to `h=2, m=45, s=45`.
  `10 mod 8 = 2`. ✔
- `Description_File.txt` says `Time Data: Sun Apr 19 11:01:17 2026`; the last Lp IMU record
  decodes to `h=3, m=1, s=17.891`. `11 mod 8 = 3`. ✔

So `t_stream = (hour mod 8)*3600 + min*60 + sec + ns/1e9`, and `t_wallclock = t_stream + 8*3600`.

Measured ranges, all in the same `t_stream` seconds:

| Stream | first | last | span | wall-clock |
|---|---|---|---|---|
| Lp IMU | 9945.245015 | 10877.891015 | 932.646 s | 10:45:45.245 → 11:01:17.891 |
| `corcam_1.ts` | 9947.105441 | 10877.800741 | 930.695 s | 10:45:47.105 → 11:01:17.801 |
| `optcam_1.ts` | 9947.110581 | 10877.805909 | 930.695 s | 10:45:47.111 → 11:01:17.806 |
| Encoder | 10005.758541 | 10877.794121 | 872.036 s | 10:46:45.759 → 11:01:17.794 |
| **LAS `gps_time`** | **10028.865** | **10878.415** | **849.550 s** | 10:47:08.865 → 11:01:18.415 |

Both LAS files (`optimised_…-001.las`, 73,869,287 pts and the clipped
`clip_texture_optimize_…-003.las`, 68,395,154 pts) have the **identical** `gps_time` range — the
"clip" is spatial only.

**`gps_time` is not GPS time.** It is this device uptime/wall-clock hybrid. Values ~1.0e4 s are
inconsistent with GPS week-seconds for a Sunday 05:15 UTC scan (which would be ~18,900 s), and
the IMU literally spells out the same value as `h:m:s`. This is why the LAS times are directly
comparable to the camera `.ts` files with **zero offset**.

Startup sequence (useful): IMU and cameras start first, the head starts spinning 60 s later, and
LiDAR points begin 23 s after that. **The first ~83 s of camera footage (~2,500 frames) has no
LiDAR coverage.** At the other end, LAS points run 0.52 s *past* the last IMU sample and 0.61 s
past the last camera frame — trim that tail when fusing.

---

## 2. `slam_calib.yaml`

Only **18 lines**. 1,980,660 of the 1,984,398 bytes are the single `correct:` array. Complete
structure:

```
version: 20250318
parameters:
  hardware:
    sn: SLAM200024330206
    deviceType: SLAM2000
    lidar2rasterR: [0.00221543,-0.00117371,0.00000260,0.99999686]   # quaternion (x,y,z,w)
    raster2bodyR:  [0.00001000,0.00001000,0.01053181,0.99994454]    # quaternion (x,y,z,w)
    pcl2imuR:      [0.00108232,0.00658423,0.00248626,0.99997465]    # quaternion (x,y,z,w)
    pcl2imuT:      [0.03702,0.0,0.11866]                            # metres
    lidarCorrect:  [0,0]
    code: <base64, 2440 chars -> 1828 bytes>
  param:
    mountTolaserTransform_opt_cam:        [10.409947,2.143666,1.037134,1.538266,-1.780053,1.779360]
    camera_instrinsic_parameters_opt_cam: [4000,3000,2.000000,2023.390295,1443.489217,2080.171445,
                                           -1.773634e-08,5.929176e-15,-7.509026e-22,3.227641e-29,
                                           -3.284532e-07,1.026261e-07,-1.822558e-03,-3.882564e-04]
    mountTolaserTransform_clr_cam:        [10.406504,2.149720,0.993075,-1.541512,1.173390,1.178273]
    camera_instrinsic_parameters_clr_cam: [4000,3000,2.000000,1996.644928,1537.618281,
                                           806.839343,805.774503,
                                           1.641566e-02,-2.061062e-02,1.042082e-02,-2.588998e-03]
    T_imu2clrcam_refine: [ 0.003678, 0.005680, 0.999977, 0.066647,
                          -0.999991,-0.001935, 0.003689, 0.000457,
                           0.001955,-0.999982, 0.005673,-0.025186,
                           0,0,0,1]                                  # row-major 4x4
    T_imu2optcam_refine: [ 0.008473,-0.001174,-0.999963, 0.060853,
                          -0.999946, 0.006061,-0.008480,-0.002607,
                           0.006071, 0.999981,-0.001123, 0.018944,
                           0,0,0,1]                                  # row-major 4x4
correct: [4,2,691204, 0,0,0,0,0,0,0, ...]   # 691,610 ints
```

### What this gives us

**Complete extrinsic chain, LiDAR → IMU → camera:**
- `pcl2imuR` (quaternion) + `pcl2imuT` = `[0.03702, 0.0, 0.11866] m` — point-cloud frame to IMU.
  The rotation is ~0.8° off identity; the lever arm is 3.7 cm / 11.9 cm.
- `T_imu2clrcam_refine` / `T_imu2optcam_refine` — full rigid 4×4, IMU to each camera. Both are
  clean rotations (axis-permutation-like, ±1 entries) with cm-scale translations
  (6.7 cm, 0.05 cm, −2.5 cm) and (6.1 cm, −0.26 cm, 1.9 cm). The `_refine` suffix indicates these
  are the post-factory-refinement values — use these, not `mountTolaserTransform_*`.
- `mountTolaserTransform_*` are 6-vectors (3 translation + 3 rotation) but their translation
  magnitudes (10.4, 2.1, 1.0) are far too large for metres on a handheld — these look like a
  different/legacy parameterisation or different units. **Prefer `T_imu2*cam_refine`.**
- `lidar2rasterR` and `raster2bodyR` relate the LiDAR to the *encoder* ("raster") frame and the
  encoder to the body — both are near-identity (0.26° and 1.21°). These are exactly the
  transforms needed to turn an encoder count into a physical head azimuth (§6.1).

**Camera intrinsics — two different models, both 4000×3000:**

- `clr_cam` (11 params) = **Kannala-Brandt equidistant fisheye**:
  `width=4000, height=3000, [2.0 = model id?], cx=1996.644928, cy=1537.618281,
  fx=806.839343, fy=805.774503, k1..k4 = 0.01641566, −0.02061062, 0.01042082, −0.00258900`.
- `opt_cam` (14 params) = **Scaramuzza/OCamCalib omnidirectional polynomial**:
  `width=4000, height=3000, [2.0], cx=2023.390295, cy=1443.489217,` then a 4-term
  polynomial `a0=2080.171445, a2=−1.773634e-08, a3=5.929176e-15, a4=−7.509026e-22`
  (plus `3.227641e-29`) and affine terms `c,d,e = −3.284532e-07, 1.026261e-07,
  −1.822558e-03, −3.882564e-04`. (The exact ordering of the last five is inferred from the
  standard OCamCalib layout and is **not** independently verified here.)

**Which camera is `corcam_1.h265`?** Measured directly. I extracted frame 0 and fitted the image
circle: centre **(2010.5, 1534.0)**, i.e. **14.4 px** from the `clr_cam` principal point
(1996.6, 1537.6) but **91 px** from the `opt_cam` one (2023.4, 1443.5). So
**`corcam_1.h265` is the `clr_cam`** and should be undistorted with the Kannala-Brandt model.
The image is a circular fisheye; the 99.5-percentile bright radius is 1449 px, but that includes
a bright lens-barrel annulus — the KB model puts θ=90° at r=1236 px and saturates near
r=1310 px, so the usable circle is ≈1240–1300 px radius (≈180–195° FOV).

**The two opaque blobs (low value, vendor internals):**
- `code`: 1828 bytes = `u32 2` header + **456 float32**, all in ±0.026 (radians), 96 of them
  exactly zero. Almost certainly per-laser-beam angular corrections. Already applied by the
  vendor when producing the LAS.
- `correct`: 691,610 ints, 56.7 % zeros, range −1214…+1214, with a `[4, 2, 691204, 0×7]` header
  (the declared payload 691,204 is 396 short of the 691,600 actually present). Repeating blocks
  of ~88 non-zero values padded with zeros — consistent with per-channel intensity/reflectivity
  correction curves. **Not decoded further; not worth it.**

---

## 3. The `FM` binary container (`.fmimr`, `.fmraster`)

All three share a **1024-byte header** (`feimarobotics-slam-imu` / `-raster` at offset 0, the
specific device string at offset 0x33, rest zero-padded), then a stream of records:

```
offset 0..1   'FM'  (0x46 0x4D)  magic
offset 2      u8    rolling sequence counter (0..255, wraps)
offset 3      u8    message id  (0x3A Lp IMU, 0x39 Hp IMU, 0x3B encoder)
offset 4      u8    payload length  (Lp 22 w/ byte5=0, Hp 27, encoder 15)
offset 5..    payload
last byte     u8    checksum
```

> Caveat: for Lp and the encoder, offsets 4–5 read equally well as a little-endian `u16` length
> (22 and 15). For Hp, offset 5 is timestamp data, so the length must be a `u8` at offset 4
> (=27). Both readings give the correct stride; I did not determine which is the vendor's intent.

### 3.1 `Lp_Imu.fmimr` — body ICM-40609 — **fully decoded**

Device string: `feima-icm40609-imu`. Stride **28 bytes**, `(26,115,140 − 1024) / 28 = 932,647`
records **exactly, zero leftover**, and **100.000 %** of them start with `FM`.

```
[0:2]   'FM'
[2]     u8  sequence
[3]     u8  0x3A
[4:6]   u16 22 (payload length)
[6]     u8  hour (mod 8)
[7]     u8  minute
[8]     u8  second
[9:13]  u32 nanoseconds within the second   (max observed 999,015,460 < 1e9 ✔)
[13:19] 3 x i16  gyro  x,y,z
[19:25] 3 x i16  accel x,y,z
[25:27] i16      temperature
[27]    u8       checksum
```

Evidence:
- Time is **perfectly monotonic** over all 932,647 records; median Δt = **1000.0 µs → exactly
  1000.00 Hz**. The nanosecond field steps by exactly 1,000,000 and resets at 1e9 (the raw
  `u32` diff shows exactly two values: `+1,000,000` × 931,714 and
  `+3,295,967,296` × 932 — the latter being `1e6 − 1e9 + 2^32`, i.e. the once-per-second reset;
  932 resets over 932.6 s ✔).
- `‖accel‖` mean = **4103.26 LSB**, std 423.8. Divided by 4096 LSB/g (the ICM-40609 ±8 g range)
  that is **1.0018 g**. ✔ Gravity sits on accel **X** (mean 4072.65 vs 88.4 and 208.5).
- Temperature field mean 1108.8, range 448…1542 → ≈33 °C via the ICM formula `raw/132.48 + 25`.
  Plausible, not independently verified.
- Gyro means (12.0, 34.4, −9.2 LSB) ≈ 0 — **the body does not rotate continuously**, in
  contrast with the head IMU (§3.3). This is the cleanest proof that Lp is body-mounted.

### 3.2 `Hp_Imu.fmimr` — second body IMU — **framing decoded, scaling inferred**

Device string: `feima-i2000-imu`. Stride **33 bytes**, message id `0x39`.

```
[0:2]  'FM'
[2]    u8  sequence
[3]    u8  0x39
[4]    u8  27 (payload length)
[5:8]  u24 timestamp, microseconds, wraps at 2^24 (16.777 s) — NO h/m/s prefix
[8:32] 6 x u32le, each = [4-bit tag | 28-bit signed value]
[32]   u8  checksum
```

- 929,681 of the FM markers are spaced exactly 33 bytes. That accounts for 30,679,473 of the
  30,914,048 post-header bytes = **99.24 %**. The remaining 0.76 % (9,612 records spaced 22 B,
  930 spaced 16 B, and a scatter of others) are **other message types I did not classify**.
- Timestamp: median step **1000 µs → 999.85 Hz**, total unwrapped span **929.819 s** — matching
  the other streams. Because it has no `h/m/s` field it must be anchored to the global clock
  (trivial: its span and rate pin it against Lp to sub-ms).
- **The channel mapping was confirmed by cross-correlating Hp against the fully-decoded Lp**
  (same rigid body, both ~1000 Hz), which is decisive:

  | Hp slot | Lp field | r | std ratio |
  |---|---|---|---|
  | 4 | gyro X | **+1.00** | 877× |
  | 3 | gyro Y | **+0.99** | 877× |
  | 5 | gyro Z | **−0.97** | 908× |
  | 1 | accel X | **+0.96** | 15.9× |
  | 0 | accel Y | **+0.91** | 15.0× |
  | 2 | accel Z | **−0.95** | 15.2× |

  So Hp slots 0–2 are accel (Y, X, Z) and 3–5 are gyro (Y, X, Z), with Z negated on both —
  a different mounting orientation. Implied scales: accel ≈ 15.5 × 4096 ≈ **6.4e4 LSB/g**
  (≈2^16; `‖accel‖` mean 64,091 → 0.978 g), gyro ≈ 877 × 16.4 ≈ **1.4e4 LSB/(deg/s)**.
  The exact factory scale factors are **not** in `slam_calib.yaml` and are therefore inferred,
  not known.
- **Verdict: Hp is redundant with Lp** for our purposes and is the one stream not worth
  investing in.

### 3.3 `Lidar_Imu.imu` — Livox head IMU — **fully decoded (values)**

Header strings recovered: `feimarobotics-slam-imu`, `feima .imu format`, `feima ICM40609`,
`Sun Apr 19 10:45:45 2026`, `SLAM200024330206`, `livox_tech`, `47MDM4Q0020500`, and the IP
`c0 a8 01 64` = **192.168.1.100**. This is a **raw capture of the Livox sensor's UDP IMU
packets**.

No `FM` framing. Header is **1152 bytes**; records are **51 bytes** with magic `a2 a7 18`:
`(9,180,387 − 1152) / 51 = 179,985` exactly. Walking the magic finds 177,077 hits, of which
**170,896 are spaced exactly 51** and **6,180 are spaced 75** (a second, longer packet type).

```
[0:3]   a2 a7 18   magic
[3]     u8  sequence
[15:39] 6 x float32:  gyro x,y,z (rad/s), accel x,y,z (g)
[40:44] u8 x4  sensor IP 192.168.1.100
[46:51] 5-byte LE  nanosecond timestamp (low bytes of a wider counter)
```

- **100.00 %** of the 170,896 records yield finite, physically sane float32.
- `‖accel‖` mean = **1.00557 g**, std 0.103. ✔ (Units are g, directly.)
- **`gyro_x` mean = +4.1022 rad/s, std 0.74** — a large, sustained rotation about X, while
  `gyro_y`/`gyro_z` mean ≈ 0. **This IMU is on the spinning head.** 4.1022 rad/s = 0.6529 rev/s
  = 39.2 rpm.
- `acc_x` mean +0.992 g with y,z ≈ 0 → the spin axis and gravity coincide, i.e. the head spins
  about a roughly vertical axis. (Consistent with Lp, whose gravity is also on accel X.)
- Timestamp: the 5-byte field at offset 46 is monotonic (frac_increasing = 1.0000) with a
  median step of 5.31e6 ns → **188 Hz** and a total span of **885.4 s**. Nominal Livox IMU rate
  is 200 Hz; the discrepancy is because 6,180 packets (the 75-byte type) are excluded from this
  subset. **The high bytes of this timestamp are truncated in my read, so it is a relative
  clock — I did not establish its absolute offset to the global clock.** That is a loose end,
  but a harmless one: the encoder (§3.4) provides the same head-angle information on the
  *absolute* clock at a higher rate.

### 3.4 `Ec_Data.fmraster` — **the rotary encoder** — fully decoded, and the most valuable stream

Despite the name "raster", this is the **absolute angular encoder of the spinning LiDAR mount**
("Ec" = encoder). Stride **21 bytes**, `(24,747,424 − 1024)/21 = 1,178,400` records exactly,
100.000 % `FM` magic, message id `0x3B`.

```
[0:6]   FM / seq / 0x3B / len=15
[6:13]  hour, minute, second, u32 nanoseconds   (same absolute clock)
[13]    u8  0
[14:16] u16 encoder angle,      0..2047   (2048 counts/rev = 0.1758 deg/count)
[16:18] u16 revolution counter, 6..580
[18:20] u16 0
[20]    u8  checksum
```

The evidence here is about as clean as reverse-engineering gets:

- Angle field takes **exactly 2048 distinct values, 0…2047**.
- Its first difference is **exactly +1 for 1,177,824 records** and **exactly −2047 for 575
  records** — i.e. one record per encoder count, wrapping 575 times, with **no other value at
  all**.
- Revolution counter is **monotonic non-decreasing**, spans 6…580 = **575 distinct values**,
  exactly matching the 575 wraps. ✔
- 574 complete revolutions in 872.036 s = **0.6582 rev/s = 4.1358 rad/s = 39.49 rpm**;
  record rate 1351.3 Hz vs. predicted 0.6582 × 2048 = **1348.1 Hz** ✔.
- **Independent cross-check:** the head IMU's `gyro_x` gives 4.1022 rad/s. The encoder gives
  4.1358 rad/s. **Agreement to 0.81 %**, from two completely independent files decoded by
  completely different means. This is strong mutual confirmation that both decodes are correct.

**Consequence for our pipeline:** one revolution = 1/0.6582 = **1.519 s**; a half-revolution is
**0.760 s**. Our existing code groups LAS returns into "~0.7 s frames" heuristically — those are
half-revolutions of this head, and we can now get their **exact** boundaries from the encoder
instead of guessing.

---

## 4. `corcam_1.ts` / `optcam_1.ts` — plain-text frame timestamps

Not MPEG-TS. Plain ASCII, one `%.6f` value per line, LF-terminated.

| | `corcam_1.ts` | `optcam_1.ts` |
|---|---|---|
| lines | **27,924** | **27,926** |
| first | 9947.105441 | 9947.110581 |
| last | 10877.800741 | 10877.805909 |
| span | 930.695 s | 930.695 s |
| mean Δt | 33.3308 ms | 33.3284 ms |
| median Δt | 33.3260 ms | 33.3260 ms |
| rate | **30.0023 Hz** | **30.0044 Hz** |
| monotonic | yes | yes |
| dropped-frame gaps (>1.5× median) | 4 (max 110.6 ms) | 3 (max 105.6 ms) |

Both are on the same clock as the LAS `gps_time` (§1) — **no offset, no scaling, no alignment
step required**. The two cameras are offset from each other by ~5 ms and are effectively
synchronised.

---

## 5. `corcam_1.h265`

`ffprobe`: **HEVC / H.265, Main profile, level 6.0, 4000×3000, yuv420p, r_frame_rate 30/1**,
82 bytes of extradata (VPS/SPS/PPS), raw Annex-B elementary stream (no container, hence no
duration/PTS metadata).

- **Frame count: 27,924 packets** (`ffprobe -count_packets`, demux only, no decode) — **exactly
  equal to the 27,924 lines in `corcam_1.ts`**. So the mapping is **1:1, frame *i* ↔ line *i***,
  and every frame has an absolute timestamp on the common clock. This is the cleanest possible
  outcome.
- **Keyframe every 30 frames** (verified over the first 1,200 packets: IDR at packet
  1, 31, 61, 91, …), i.e. **one keyframe per second**. Random access is cheap — you never decode
  more than 29 frames to reach an arbitrary one. Extracting a few thousand frames for texturing
  is entirely practical.
- I decoded frame 0 successfully: a sharp, well-exposed **circular fisheye** interior view of
  the property (marble floor, window, the operator's leg at the edge). Image quality looks
  fully adequate for texturing — no visible motion blur at 30 fps handheld walking pace.
- Bitrate ≈ 1.6 GB / 930.7 s ≈ **13.7 Mbit/s** at 12 MP — moderate compression, some
  block/ringing artefacts expected on flat walls but fine for surface texture.
- `optcam_1.h265` **does not exist** — only its timestamp file was exported.

---

## 6. What is actually achievable — ranked by value ÷ effort

### 6.1 ★★★ Recover a true sensor trajectory from the encoder — **the big one**

**Is the data there?** There is **no explicit pose table** — I checked `slam_calib.yaml` (18
lines, no trajectory) and every other file; the trajectory is only implicit in the registered
point cloud. **But it is recoverable far more rigorously than our geometric median**, because of
a constraint we did not previously have:

> For a LiDAR return at time *t*, the ray from the sensor origin to that point must have, in the
> head frame, an **azimuth equal to the encoder angle at time *t***. We now know that angle to
> **0.176°** at **1351 Hz** on the **same clock as the LAS `gps_time`**.

Concretely: with roll/pitch fixed by the IMU gravity vector and yaw as one unknown, each point
contributes a constraint `n(t)·(p_i − s) = 0`, where `n(t)` is the normal of the plane containing
the spin axis at `enc(t) + yaw` (mapped into the cloud frame via `lidar2rasterR`, `raster2bodyR`,
`pcl2imuR/T`). For a short window this is a 1-D search over yaw wrapping a **linear** least
squares for the 3-DoF position `s` — and each window has *thousands* of points, so it is
massively over-determined. Chain the windows (or fit splines over `s(t)`, `yaw(t)`) and you get a
smooth 6-DoF trajectory.

**Why it matters concretely:** our current approximation is a per-0.7 s geometric median, and the
symptom is that **~10 % of rays clip a surface before their endpoint**, which forced
`freespace_carve.py` to demand **8 rays** before declaring a voxel free. With true per-point
origins that error rate should collapse, letting the carve drop to 1–2 rays — which means
**dramatically better free-space carving, thinner spurious geometry, and better opening
detection**. This is the highest-leverage item in the whole report.

*Effort:* medium (a few hundred lines plus tuning). *Risk:* the spin axis direction in the cloud
frame must be pinned down from `lidar2rasterR`/`raster2bodyR`, and my reading of those quaternion
conventions (x,y,z,w order, and which direction each transform maps) is **inferred from naming,
not verified** — expect a short sign/convention hunt at the start.

### 6.2 ★★★ Photo-texturing from `corcam_1.h265`

**What we have — everything except pose:**
- ✔ Intrinsics: `clr_cam` Kannala-Brandt fisheye, and I verified by measurement that this is the
  right model for this video (§2).
- ✔ Extrinsics: complete LiDAR → IMU → camera chain (`pcl2imuR/T` + `T_imu2clrcam_refine`).
- ✔ Frame timestamps: 1:1 with frames, on the LAS clock, zero offset.
- ✔ Images: 27,924 × 12 MP, keyframe every second so extraction is cheap.
- ✘ **Missing: the camera pose in world** — i.e. exactly the trajectory of §6.1.

So **6.1 unlocks 6.2**. They should be planned as one project, not two.

**A strong free validation, and possibly a better solver.** The LAS is named
`texture_optimize_…` and carries RGB — the vendor produced that colour by projecting *these same
frames*. That means (a) the geometric chain provably works, and (b) we can validate our own
reprojection cheaply: project a point into the frame nearest its `gps_time` and compare the
sampled colour to the LAS RGB. Better still, this can be turned into the **solver itself** —
optimise each frame's 6-DoF pose to maximise photometric agreement between the image and the
already-coloured point cloud, initialised from our current geometric-median trajectory. That is
well-conditioned and directly optimises the quantity texturing cares about.

**Genuine gain over what we do now:** we currently bake ~68 M per-vertex colours. A 12 MP image
projected onto a wall at 3 m gives roughly **an order of magnitude more angular detail** than
per-vertex colour on our mesh — this is real added information for the interior-finish fidelity
the product is about, not a re-packaging.

*Effort after 6.1:* medium-high (frame selection, visibility/occlusion, UV atlas, exposure and
seam blending). *Caveats not determined:* whether the sensor is rolling-shutter (would need
per-row time correction at walking speed), and the exposure/white-balance metadata is absent.

### 6.3 ★★ Exact revolution-based frame boundaries — cheap, immediate

Replace the heuristic "~0.7 s frame" grouping in the LAS-trajectory code with **exact
half-revolution boundaries read from the encoder's revolution counter and angle wrap**. Pure win:
a handful of lines, removes a magic number, and makes the existing geometric-median estimate
measurably cleaner even *before* 6.1 lands. Do this first as a warm-up — it also forces the
encoder-to-LAS time join that 6.1 needs anyway.

### 6.4 ★★ Gravity / verticality reference

The Lp IMU gives the gravity direction in the body frame at 1000 Hz over the whole scan, with
`pcl2imuR`/`pcl2imuT` mapping into the cloud frame. Useful for the deviation-detection product:
**is a wall we flagged as "out of plumb" actually out of plumb, or is our world Z tilted?**

Honest caveat: gravity in the *body* frame is only half the answer — to test the *world* Z you
need the body's world orientation at that instant, which is again the trajectory (§6.1). So this
is **coupled to 6.1**, not independent. What *is* independent and immediately usable: detecting
**stationary intervals** (where `‖accel‖ ≈ 1 g` and `‖gyro‖ ≈ 0`), which mark the operator
standing still — useful for picking clean texture frames and for spotting where SLAM had the
least motion-induced error.

### 6.5 ★ Drift and data-quality diagnostics

Integrating gyro between two times gives a relative rotation that can be compared against the
rotation implied by the point cloud, exposing SLAM drift and loop-closure jumps. Also cheap:
the 4 dropped camera frames and the exact end-of-scan trim points (§1) are now known precisely.
Modest value; useful mainly as a confidence signal on a scan.

### 6.6 — Not worth doing

- **Hp IMU**: redundant with Lp, coarser provenance (scale factors inferred, 0.76 % of the stream
  unclassified). No reason to touch it.
- **`correct` / `code` calib blobs**: vendor beam and intensity corrections already baked into
  the exported LAS. Decoding them buys nothing unless we start from raw LiDAR — which we can't
  (§7).
- **`optcam`**: its video was not exported, so its timestamps and intrinsics are unusable.

---

## 7. What is missing, and what to ask for

1. **`20260419-104545_Lidar_Data.fmlidar` (1,820,658,129 bytes) is listed in
   `Description_File.txt` but is NOT in `data/Soulace/`.** This is the raw per-point stream
   (ranges, beam angles, per-point timestamps in the *sensor* frame). With it, trajectory
   recovery becomes near-trivial — you would solve for the pose that maps raw sensor-frame rays
   onto the registered cloud, instead of inferring geometry from the encoder. **This is the
   single highest-value thing to request from the operator**, and it should still exist on the
   device or in the source project folder `SN_00033/SLAM_PRJ_001/`.
2. **`optcam_1.h265`** — the second camera's video. Its timestamps and calibration were exported
   but the imagery was not. A second viewpoint would improve texture coverage and occlusion
   handling.
3. **An exported trajectory.** Most vendor SLAM apps can export the solved pose stream
   (`.traj`, TUM, or CSV). If the SLAM2000 desktop software offers this, it makes §6.1 redundant
   and §6.2 immediate. **Worth checking before investing in 6.1.**

## 8. Things I could not determine (flagged honestly)

- The exact intended width of the `FM` length field (u8 vs u16) — both readings give correct
  strides (§3).
- The remaining 0.76 % of the Hp stream (9,612 records at 22 B, 930 at 16 B, plus stragglers).
- Hp's factory scale factors (accel ≈2^16 LSB/g and gyro ≈1.4e4 LSB/(deg/s) are **inferred by
  correlation against Lp**, not read from calibration).
- The absolute time offset of the Livox head IMU's own timestamp (high bytes truncated). Harmless
  — the encoder supersedes it.
- The parameter ordering within the `opt_cam` OCamCalib polynomial (assumed standard layout).
- The units/convention of `mountTolaserTransform_*` (translations far too large for metres).
- Quaternion component order and transform direction for `lidar2rasterR` / `raster2bodyR` /
  `pcl2imuR` — assumed `(x,y,z,w)` and "A→B" per the name, but not verified against data.
- Whether the camera is global- or rolling-shutter.
- The internal structure of `correct` and `code` beyond their element counts and ranges.

---

## 9. Reproducing this

All probes are read-only and run in seconds to a couple of minutes on
`venv311/Scripts/python.exe`, except the LAS scan (~2 min per file, chunked, low memory).

| Script | Purpose |
|---|---|
| `scripts/experiments/soulace_probe/probe_calib.py` | Summarise `slam_calib.yaml` keys/sizes |
| `scripts/experiments/soulace_probe/probe_calib_blobs.py` | Characterise the `code` / `correct` blobs |
| `scripts/experiments/soulace_probe/probe_ts.py` | Decode the camera `.ts` timestamp lists |
| `scripts/experiments/soulace_probe/probe_stride.py` | Blind record-stride search (FM spacing + constant-column scoring) |
| `scripts/experiments/soulace_probe/probe_records.py` | Generic record dumper + monotonic-field hunter |
| `scripts/experiments/soulace_probe/probe_imu.py` | Full Lp IMU decode; applies the stride hypothesis to all three FM files |
| `scripts/experiments/soulace_probe/probe_walk.py` | Length-prefixed walker (shows why the naive walk fails on Hp) |
| `scripts/experiments/soulace_probe/probe_hp.py` | Hp 33-byte record decode |
| `scripts/experiments/soulace_probe/probe_hp_split.py` | Settles Hp's channel mapping by correlation against Lp |
| `scripts/experiments/soulace_probe/probe_lidarimu.py` | First-pass probe of the Livox `.imu` header / divisor search |
| `scripts/experiments/soulace_probe/probe_livox_imu.py` | Livox `.imu` header/stride derivation |
| `scripts/experiments/soulace_probe/probe_livox_walk.py` | Livox `.imu` magic-based walk and float decode |
| `scripts/experiments/soulace_probe/probe_raster.py` | Encoder payload decode |
| `scripts/experiments/soulace_probe/probe_verify.py` | **Cross-validation: encoder rev/s vs head-IMU gyro (0.81 %)** |
| `scripts/experiments/soulace_probe/probe_fisheye.py` | Measures the image circle; identifies `corcam` as `clr_cam` |
| `scripts/experiments/soulace_probe/probe_las_time.py` | Chunked LAS header + `gps_time` range |

Video probes (ffmpeg at `C:\Program Files (x86)\ffmpeg\bin`):

```sh
ffprobe -v error -show_streams data/Soulace/corcam_1.h265
ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 data/Soulace/corcam_1.h265
ffmpeg  -v error -i data/Soulace/corcam_1.h265 -frames:v 1 -y frame0.jpg
```
