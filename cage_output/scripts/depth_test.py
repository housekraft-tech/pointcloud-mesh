"""Can a monocular metric-depth model measure? Scored against LiDAR depth.

Each view has an exact per-pixel depth from the point cloud that produced the
image, so this is a direct read of what a depth model would contribute to a
measurement pipeline -- in millimetres, on this flat, not on NYU.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
VIEWS = HERE / ("views_" + (sys.argv[1] if len(sys.argv) > 1 else "koushikexport"))
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def main():
    from transformers import pipeline
    print(f"loading {MODEL}", flush=True)
    pipe = pipeline("depth-estimation", model=MODEL, device=-1)

    meta = json.loads((VIEWS / "views.json").read_text())
    rows = []
    for m in meta:
        name = m["name"]
        bgr = cv2.imread(str(VIEWS / f"{name}_rgb.png"))
        # splatting single points leaves speckle; a light median recovers surfaces
        bgr = cv2.medianBlur(bgr, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(VIEWS / f"{name}_clean.png"), bgr)

        from PIL import Image
        out = pipe(Image.fromarray(rgb))
        pred = np.array(out["predicted_depth"], dtype=np.float32)
        if pred.shape != rgb.shape[:2]:
            pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]))

        gt = np.load(VIEWS / f"{name}_depth.npy")
        v = np.isfinite(gt) & (gt > 0.3) & (gt < 12.0)
        if v.sum() < 5000:
            continue
        p, g = pred[v], gt[v]

        absrel = float(np.mean(np.abs(p - g) / g))
        mae_mm = float(np.mean(np.abs(p - g)) * 1000)
        rmse_mm = float(np.sqrt(np.mean((p - g) ** 2)) * 1000)
        bias_mm = float(np.mean(p - g) * 1000)
        # scale-and-shift aligned: the best case if you solved scale externally
        A = np.stack([p, np.ones_like(p)], axis=1)
        s, b = np.linalg.lstsq(A, g, rcond=None)[0]
        pa = s * p + b
        mae_al_mm = float(np.mean(np.abs(pa - g)) * 1000)
        d1 = float(np.mean(np.maximum(p / g, g / p) < 1.25))

        rows.append({"view": name, "n_px": int(v.sum()),
                     "gt_median_m": round(float(np.median(g)), 2),
                     "absrel": round(absrel, 4), "mae_mm": round(mae_mm, 1),
                     "rmse_mm": round(rmse_mm, 1), "bias_mm": round(bias_mm, 1),
                     "mae_aligned_mm": round(mae_al_mm, 1),
                     "scale": round(float(s), 3), "shift_m": round(float(b), 3),
                     "delta1": round(d1, 3)})
        print(f"{name:14s} AbsRel={absrel:.3f} MAE={mae_mm:7.1f}mm "
              f"RMSE={rmse_mm:7.1f}mm bias={bias_mm:+8.1f}mm "
              f"MAE(scale-aligned)={mae_al_mm:6.1f}mm d1={d1:.2f}", flush=True)

        # side-by-side figure
        def colorize(d, lo, hi):
            x = np.clip((d - lo) / (hi - lo), 0, 1)
            return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        lo, hi = float(np.nanpercentile(gt, 2)), float(np.nanpercentile(gt, 98))
        err = np.abs(pred - gt)
        panel = np.hstack([bgr, colorize(np.nan_to_num(gt, nan=hi), lo, hi),
                           colorize(pred, lo, hi),
                           colorize(np.nan_to_num(err, nan=0), 0, 1.0)])
        cv2.imwrite(str(VIEWS / f"{name}_cmp.png"), panel)

    (HERE / "results" / f"depth_scores_{(sys.argv[1] if len(sys.argv) > 1 else chr(107)+chr(111)+chr(117)+chr(115)+chr(104)+chr(105)+chr(107))}.json").write_text(json.dumps(rows, indent=1))
    if rows:
        print("\n=== summary over %d views ===" % len(rows))
        for k in ["absrel", "mae_mm", "rmse_mm", "mae_aligned_mm", "delta1"]:
            vals = [r[k] for r in rows]
            print(f"  {k:16s} median={np.median(vals):9.3f}  "
                  f"min={min(vals):9.3f}  max={max(vals):9.3f}")


if __name__ == "__main__":
    main()
