"""Render predicted room polygons over the density map they came from."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PALETTE = [(66,133,244),(219,68,55),(244,180,0),(15,157,88),(171,71,188),
           (0,172,193),(255,112,67),(158,157,36),(94,53,177),(240,98,146),
           (0,151,167),(124,179,66),(255,167,38),(84,110,122),(233,30,99),
           (26,35,126),(0,188,212),(255,87,34),(76,175,80),(103,58,183)]


def main(result_path, scale=4):
    res = json.loads(Path(result_path).read_text())
    dens = np.load(HERE / res["density"])
    img = cv2.cvtColor((dens * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (256 * scale, 256 * scale), interpolation=cv2.INTER_NEAREST)
    img = (img * 0.55).astype(np.uint8)

    overlay = img.copy()
    for i, p in enumerate(res["polys_refined"]):
        pts = (np.array(p, dtype=np.int32) * scale).reshape(-1, 1, 2)
        col = PALETTE[i % len(PALETTE)]
        cv2.fillPoly(overlay, [pts], col)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    for i, p in enumerate(res["polys_refined"]):
        arr = np.array(p, dtype=np.int32) * scale
        pts = arr.reshape(-1, 1, 2)
        col = PALETTE[i % len(PALETTE)]
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        for c in arr:
            cv2.circle(img, tuple(int(v) for v in c), 4, (255, 255, 255), -1)
            cv2.circle(img, tuple(int(v) for v in c), 4, col, 1, cv2.LINE_AA)
        cx, cy = arr.mean(axis=0).astype(int)
        cv2.putText(img, str(i), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    out = HERE / "results" / f"{Path(result_path).stem}_overlay.png"
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  ({len(res['polys_refined'])} polys)")


if __name__ == "__main__":
    main(sys.argv[1])
