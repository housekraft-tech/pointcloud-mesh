"""rfdetr_infer.py
---------------
Run the exported RF-DETR ONNX detectors (trained on 2D architectural drawings)
on a floorplan image and return detections in pixel coords. Two models:

  rfdetr_elements.onnx        rooms + furniture + doors  (50 classes)
  rfdetr_walls_windows.onnx   objects / balcony door / wall / window (4)

Both are RF-DETR exports: input [1,3,880,880] RGB, outputs
  dets   [1,N,4]  boxes cxcywh normalised 0..1
  labels [1,N,C]  per-query class logits  ->  SIGMOID (focal loss, not softmax)

Usage (standalone viz):
  venv311\\Scripts\\python.exe scripts\\experiments\\rfdetr_infer.py <image.png> <out_dir> [elements|walls|both]
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
IN_RES = 880
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

ELEMENTS_CLASSES = {
    0: "beds-sink-sofa-Ly3D", 1: "Accent Chair", 2: "Balcony", 3: "Bathroom", 4: "Bed",
    5: "Bedroom", 6: "Chair", 7: "Coffee Table", 8: "Commode", 9: "Crockery Unit",
    10: "Dining Room", 11: "Dining Table", 12: "Dishwasher", 13: "Door", 14: "Duct",
    15: "Foyer", 16: "Foyer Cabinet", 17: "Fridge", 18: "Garage", 19: "Kitchen",
    20: "Kitchen-Slab", 21: "Lift", 22: "Living Room", 23: "Lobby", 24: "Parking",
    25: "Pre-Foyer", 26: "Side Table", 27: "Sink", 28: "Sit-Out", 29: "Sofa", 30: "Stove",
    31: "Study Room", 32: "Study Table", 33: "TV", 34: "Table", 35: "Terrace", 36: "Utility",
    37: "Walkin", 38: "Wardrobe", 39: "Wash", 40: "Washing Machine", 41: "bay-window",
    42: "book cabinet", 43: "breakfast Counter", 44: "flexroom", 45: "puja cabinet",
    46: "pujaroom", 47: "spiral-stairs", 48: "stairs", 49: "store",
}
WALLS_WINDOWS_CLASSES = {0: "objects", 1: "balcony door", 2: "wall", 3: "window"}

ELEMENTS_CONFIDENCE = 0.65
WALLS_WINDOWS_CONFIDENCE = 0.50
WALLS_WINDOWS_CLASS_THRESHOLDS = {0: 0.50, 1: 0.38, 2: 0.52, 3: 0.52}
ELEMENTS_CLASS_THRESHOLDS = {
    1: 0.50, 7: 0.30, 8: 0.50, 10: 0.50, 13: 0.50, 15: 0.30, 16: 0.30, 20: 0.35, 23: 0.40,
    26: 0.60, 27: 0.30, 29: 0.49, 30: 0.30, 32: 0.40, 33: 0.50, 37: 0.40, 38: 0.50, 43: 0.45,
    48: 0.30,
}

# room-type classes in the elements model (what we fuse onto watershed rooms)
ROOM_CLASSES = {2, 3, 5, 10, 15, 18, 19, 22, 23, 24, 25, 28, 31, 35, 36, 37, 44, 46, 49}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _preprocess(bgr):
    """BGR uint8 HxW -> (1,3,880,880) float, and the (origW, origH) for rescale."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (IN_RES, IN_RES), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = (r - IMAGENET_MEAN) / IMAGENET_STD
    return r.transpose(2, 0, 1)[None].astype(np.float32), (w, h)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_model(sess, bgr, class_map, base_thr, class_thr, max_det=300):
    """Return list of dets: {cls, name, score, box=(x0,y0,x1,y1) px in ORIGINAL image}."""
    inp, (W, H) = _preprocess(bgr)
    name = sess.get_inputs()[0].name
    dets, labels = sess.run(None, {name: inp})
    dets = dets[0]                       # (N,4) cxcywh norm
    prob = _sigmoid(labels[0])          # (N,C)
    C = prob.shape[1]
    ncls = max(class_map) + 1
    prob = prob[:, :ncls]               # ignore unused padded class columns
    # flatten top-k over (query x class) -- standard RF-DETR sigmoid postprocess
    flat = prob.reshape(-1)
    k = min(max_det, flat.size)
    top = np.argpartition(flat, -k)[-k:]
    out = []
    for idx in top:
        q, c = idx // ncls, idx % ncls
        s = float(prob[q, c])
        thr = class_thr.get(int(c), base_thr)
        if s < thr:
            continue
        cx, cy, bw, bh = dets[q]
        x0 = (cx - bw / 2) * W; y0 = (cy - bh / 2) * H
        x1 = (cx + bw / 2) * W; y1 = (cy + bh / 2) * H
        out.append(dict(cls=int(c), name=class_map.get(int(c), str(c)), score=s,
                        box=(float(x0), float(y0), float(x1), float(y1))))
    return _nms(out, iou_thr=0.5)


def _nms(dets, iou_thr=0.5):
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(_iou(d["box"], k["box"]) < iou_thr or k["cls"] != d["cls"] for k in keep):
            keep.append(d)
    return keep


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load(kind):
    import onnxruntime as ort
    fn = {"elements": "models/rfdetr_elements.onnx",
          "walls": "models/rfdetr_walls_windows.onnx"}[kind]
    path = ROOT / fn
    if not path.exists():
        return None
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def draw(bgr, dets, title):
    im = bgr.copy()
    rng = np.random.default_rng(7)
    colors = {}
    for d in dets:
        c = colors.setdefault(d["cls"], tuple(int(v) for v in rng.integers(60, 255, 3)))
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        cv2.rectangle(im, (x0, y0), (x1, y1), c, 2)
        t = f"{d['name']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(im, (x0, y0 - th - 3), (x0 + tw + 2, y0), c, -1)
        cv2.putText(im, t, (x0 + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(im, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return im


def main(img_path, out_dir, which="both"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        log(f"cannot read {img_path}"); return
    log(f"image {img_path}  {bgr.shape[1]}x{bgr.shape[0]}")
    jobs = []
    if which in ("elements", "both"):
        jobs.append(("elements", ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS))
    if which in ("walls", "both"):
        jobs.append(("walls", WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS))
    for kind, cmap, base, cthr in jobs:
        sess = load(kind)
        if sess is None:
            log(f"  {kind}: model file not present -- skipping"); continue
        t0 = time.time()
        dets = run_model(sess, bgr, cmap, base, cthr)
        log(f"  {kind}: {len(dets)} detections in {time.time()-t0:.1f}s")
        from collections import Counter
        for nm, n in Counter(d["name"] for d in dets).most_common():
            log(f"     {n:3d}  {nm}")
        cv2.imwrite(str(out_dir / f"detect_{kind}.png"), draw(bgr, dets, f"RF-DETR {kind}: {len(dets)} dets"))
        log(f"  wrote detect_{kind}.png")


if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    main(sys.argv[1], sys.argv[2], which)
