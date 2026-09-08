"""box_ocr_floorplan.py
----------------------
Bounding-box-localized OCR: run the RF-DETR elements detector to find room
regions, then OCR INSIDE each room box (cropped + heavily upscaled + digit
allowlist) to read that one room's printed nominal dimension. Localized OCR is
far more reliable than whole-image OCR on cluttered drawings.

Output: room_schedule_boxed.json -> per detected room: label, box (px), nominal
w x l (mm), confidence.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\box_ocr_floorplan.py <floorplan.png> <out_dir>
"""
import sys, re, json
from pathlib import Path
import numpy as np
import cv2
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer", "Store", "Pooja"}
DIM_RE = re.compile(r"(\d{3,4})\s*[xX×*,.]+\s*(\d{3,4})")


def parse_dims(text):
    """Yield all plausible (w,l) mm pairs from a text blob."""
    t = re.sub(r"[^\dxX×*,.\s]", "", text)
    out = []
    for a, b in DIM_RE.findall(t):
        a, b = int(a), int(b)
        if 200 <= a <= 12000 and 200 <= b <= 12000:
            out.append((min(a, b), max(a, b)))
    for tok in re.findall(r"\b(\d{8})\b", re.sub(r"\D", " ", t)):
        a, b = int(tok[:4]), int(tok[4:])
        if 200 <= a <= 12000 and 200 <= b <= 12000:
            out.append((min(a, b), max(a, b)))
    return out


def main(img_path, out_dir):
    import easyocr
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    sess = load("elements")
    dets = run_model(sess, bgr, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    rooms_det = [d for d in dets if d["name"] in ROOM_CLASSES]
    print(f"elements: {len(dets)} dets, {len(rooms_det)} room regions")

    reader = easyocr.Reader(["en"], gpu=False)
    SCALE = 5.0
    schedule = []
    for d in rooms_det:
        x0, y0, x1, y1 = d["box"]
        # pad the crop so the dimension label just outside the tight box is included
        pad = 0.12 * max(x1 - x0, y1 - y0)
        cx0 = max(0, int(x0 - pad)); cy0 = max(0, int(y0 - pad))
        cx1 = min(W, int(x1 + pad)); cy1 = min(H, int(y1 + pad))
        crop = bgr[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        txt = reader.readtext(big, allowlist="0123456789xX×*", detail=0,
                              text_threshold=0.4, low_text=0.3)
        cands = parse_dims(" ".join(txt))
        best = max(cands, key=lambda wl: wl[0] * wl[1]) if cands else None
        schedule.append(dict(label=d["name"], score=round(d["score"], 2),
                             box=[round(v) for v in d["box"]],
                             w=best[0] if best else None, l=best[1] if best else None,
                             raw=" ".join(txt)[:40]))

    (out / "room_schedule_boxed.json").write_text(json.dumps(schedule, indent=2))
    print(f"\n{'room':14}{'score':7}{'nominal WxL (mm)':20}{'raw ocr'}")
    print("-" * 66)
    got = 0
    for r in sorted(schedule, key=lambda r: r["label"]):
        dim = f"{r['w']}x{r['l']}" if r["w"] else "—"
        if r["w"]:
            got += 1
        print(f"{r['label']:14}{r['score']:<7}{dim:20}{r['raw']}")
    print("-" * 66)
    print(f"dimensions read: {got}/{len(schedule)} rooms")
    print(f"-> {out}/room_schedule_boxed.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
