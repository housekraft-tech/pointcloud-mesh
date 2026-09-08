"""visualize_drawing_analysis.py
--------------------------------
Render a clear, detailed visualization set of the full drawing analysis:
  1_rooms.png       - detected room regions (boxes + label + score + OCR dim)
  2_walls.png       - wall / window / balcony-door detection
  3_labeled.png     - clean labeled plan: room name + nominal dim in each room
  4_schedule.png    - the room schedule as a table
Everything written to <out_dir> (a new folder).

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\visualize_drawing_analysis.py <floorplan.png> <out_dir>
"""
import sys, re
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS,
    WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}
DIM_RE = re.compile(r"(\d{3,4})\s*[xX×*,.]+\s*(\d{3,4})")
RCOL = {"Bedroom": "#e74c3c", "Bathroom": "#3498db", "Kitchen": "#e67e22",
        "Utility": "#16a085", "Walkin": "#9b59b6", "Dining Room": "#f1c40f",
        "Balcony": "#2ecc71", "Living Room": "#f39c12", "Foyer": "#1abc9c"}


def parse_dims(text):
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


def box_ocr(reader, bgr, box, scale=5.0):
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = box
    pad = 0.12 * max(x1 - x0, y1 - y0)
    cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
    crop = bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    txt = reader.readtext(big, allowlist="0123456789xX×*", detail=0,
                          text_threshold=0.4, low_text=0.3)
    c = parse_dims(" ".join(txt))
    return max(c, key=lambda wl: wl[0] * wl[1]) if c else None


def main(img_path, out_dir):
    import easyocr
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = bgr.shape[:2]

    el = run_model(load("elements"), bgr, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    ww = run_model(load("walls"), bgr, WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    rooms = [d for d in el if d["name"] in ROOM_CLASSES]

    reader = easyocr.Reader(["en"], gpu=False)
    for d in rooms:
        d["dim"] = box_ocr(reader, bgr, d["box"])

    def fig(name, title):
        f, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=110)
        ax.imshow(rgb); ax.set_title(title, fontsize=15, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        return f, ax

    # 1 rooms
    f, ax = fig("rooms", f"Room detection - {len(rooms)} rooms (RF-DETR elements)")
    for d in rooms:
        x0, y0, x1, y1 = d["box"]; c = RCOL.get(d["name"], "#555")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.16,
                               facecolor=c, edgecolor=c, lw=2.2))
        dim = f"  {d['dim'][0]}x{d['dim'][1]}" if d.get("dim") else ""
        ax.text(x0 + 4, y0 + 16, f"{d['name']} {d['score']:.2f}{dim}", fontsize=8.5,
                color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=c, ec="none", alpha=0.9))
    f.tight_layout(); f.savefig(out / "1_rooms.png", dpi=110); plt.close(f)

    # 2 walls
    wcol = {"wall": "#c0392b", "window": "#27ae60", "balcony door": "#e84393", "objects": "#7f8c8d"}
    f, ax = fig("walls", f"Wall / window / door detection - {len(ww)} elements")
    for d in ww:
        x0, y0, x1, y1 = d["box"]; c = wcol.get(d["name"], "#555")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.35,
                               facecolor=c, edgecolor=c, lw=1.2))
    from collections import Counter
    cnt = Counter(d["name"] for d in ww)
    lg = "  ".join(f"{k}:{v}" for k, v in cnt.items())
    ax.text(6, H - 10, lg, fontsize=10, color="black",
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    f.tight_layout(); f.savefig(out / "2_walls.png", dpi=110); plt.close(f)

    # 3 clean labeled
    f, ax = fig("labeled", "Labeled plan - room name + nominal dimension (mm)")
    for d in rooms:
        x0, y0, x1, y1 = d["box"]; c = RCOL.get(d["name"], "#555")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.13,
                               facecolor=c, edgecolor=c, lw=1.6))
        dim = f"\n{d['dim'][0]} x {d['dim'][1]}" if d.get("dim") else "\n(dim ?)"
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{d['name']}{dim}", fontsize=9,
                color="black", ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, alpha=0.85))
    f.tight_layout(); f.savefig(out / "3_labeled.png", dpi=110); plt.close(f)

    # 4 schedule table
    rows = [[d["name"], f"{d['score']:.2f}",
             f"{d['dim'][0]} x {d['dim'][1]}" if d.get("dim") else "-"]
            for d in sorted(rooms, key=lambda d: d["name"])]
    fh = max(2.5, 0.42 * len(rows) + 1)
    f, ax = plt.subplots(figsize=(6, fh), dpi=120); ax.axis("off")
    ax.set_title("Room schedule (detected + OCR)", fontsize=13, weight="bold")
    t = ax.table(cellText=rows, colLabels=["room", "conf", "nominal WxL (mm)"],
                 loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    for j in range(3):
        t[0, j].set_facecolor("#34495e"); t[0, j].set_text_props(color="white", weight="bold")
    f.tight_layout(); f.savefig(out / "4_schedule.png", dpi=120); plt.close(f)

    print(f"rooms={len(rooms)}  walls/windows={len(ww)}  dims read={sum(1 for d in rooms if d.get('dim'))}")
    print("wrote:", ", ".join(p.name for p in sorted(out.glob("*.png"))))
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
