"""ocr_floorplan.py
------------------
OCR an architect floorplan to extract the room schedule: for every room, its
NAME and its printed nominal dimensions (mm). Text is read with easyocr, then
dimension tokens (e.g. "3030x4890") and room-name tokens (BEDROOM, KITCHEN,
TOILET, UTILITY, WALK IN, LIVING/DINING, FOYER, BALCONY...) are paired by
proximity. Output pairs each room label with the nearest dimension token.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\ocr_floorplan.py <floorplan.png> <out_dir>
"""
import sys, re, json
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOM_WORDS = ["LIVING/DINING", "LIVING", "DINING", "BEDROOM", "BATHROOM", "TOILET",
              "KITCHEN", "UTILITY", "WALK IN", "WALKIN", "WALK-IN", "FOYER",
              "BALCONY", "BALCOA", "STORE", "POOJA", "DRESS"]
DIM_RE = re.compile(r"(\d{3,4})\s*[xX×*,.]+\s*(\d{3,4})")
DIM_CONCAT = re.compile(r"^(\d{4})(\d{4})$")   # "15502450" -> 1550 x 2450


def parse_dim(text):
    t = text.replace(" ", "")
    m = DIM_RE.search(t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = DIM_CONCAT.match(re.sub(r"\D", "", t))   # 8 pure digits -> split 4+4
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def center(box):
    b = np.array(box, float)
    return b[:, 0].mean(), b[:, 1].mean()


def main(img_path, out_dir):
    import easyocr, cv2
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"], gpu=False)
    # dimension text is tiny at native res -> upscale so small glyphs are legible,
    # and loosen the detector thresholds. Coords are divided back to native px.
    SCALE = 3.0
    img = cv2.imread(str(img_path))
    big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(big, text_threshold=0.5, low_text=0.3,
                              mag_ratio=1.5, width_ths=0.7, add_margin=0.15)

    names, dims = [], []
    for box, text, conf in results:
        box = [[p[0] / SCALE, p[1] / SCALE] for p in box]
        cx, cy = center(box)
        t = text.strip().upper()
        pd = parse_dim(text)
        if pd:
            a, b = pd
            if 200 <= a <= 12000 and 200 <= b <= 12000:
                dims.append(dict(w=min(a, b), l=max(a, b), cx=cx, cy=cy, raw=text, conf=conf))
        for kw in ROOM_WORDS:
            if kw in t:
                names.append(dict(name=kw, cx=cx, cy=cy, conf=conf))
                break

    # pair each room name with the nearest dimension token
    rooms = []
    for nm in names:
        if not dims:
            break
        d = min(dims, key=lambda d: (d["cx"] - nm["cx"]) ** 2 + (d["cy"] - nm["cy"]) ** 2)
        dist = ((d["cx"] - nm["cx"]) ** 2 + (d["cy"] - nm["cy"]) ** 2) ** 0.5
        rooms.append(dict(name=nm["name"], w=d["w"], l=d["l"], px=[nm["cx"], nm["cy"]],
                          dim_dist_px=round(dist, 1), dim_raw=d["raw"]))

    (out / "room_schedule.json").write_text(json.dumps(
        dict(rooms=rooms, all_dims=dims, all_names=names), indent=2))

    print(f"OCR: {len(results)} text boxes | {len(names)} room labels | {len(dims)} dimension tokens")
    print(f"\n{'room':16}{'nominal WxL (mm)':20}{'dim label':14}{'pair dist px'}")
    print("-" * 62)
    for r in sorted(rooms, key=lambda r: r["name"]):
        wl = "{}x{}".format(r["w"], r["l"])
        print(f"{r['name']:16}{wl:20}{r['dim_raw']:14}{r['dim_dist_px']}")
    print(f"\n-> {out}/room_schedule.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
