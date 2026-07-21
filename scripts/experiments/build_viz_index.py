"""build_viz_index.py
------------------
Contact sheets of every visualisation produced for a scan, grouped by the
question each one answers. 1500+ PNGs exist under a scan's output tree; this
makes them reviewable at a glance and prints where each tile came from.

Tiles are letterboxed, never stretched -- these are measurement figures and a
changed aspect ratio makes a wall look like a different wall.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_viz_index.py <scan_root> <out_dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2

TILE_W, TILE_H = 620, 470
COLS = 3
BG = (16, 17, 19)
FG = (235, 235, 235)
SUB = (150, 190, 245)

SHEETS = [
    ("01_model", "THE 3D MODEL - modular house from the Poisson mesh", [
        ("annotated/render/modular_perspective.png", "perspective (ceiling on)"),
        ("annotated/render/modular_cutaway.png", "cutaway - each wall its own object"),
        ("annotated/render/modular_soffit.png", "soffit - ceiling from below, by height"),
    ]),
    ("02_heights", "HEIGHTS AND FEATURES", [
        ("detailed_modular/render/ceiling_height_map.png",
         "ceiling height map - beams + dropped ceilings"),
        ("detailed_modular/walk/walk_verified_openings.png",
         "walk path vs openings"),
        ("annotated/render/openings_plan.png",
         "openings by role + exterior test"),
    ]),
    ("03_drawing", "AGAINST THE ARCHITECT DRAWING", [
        ("annotated/drawing_alignment.png",
         "alignment: green = within 100 mm"),
        ("annotated/drawing_openings.png",
         "openings found in the drawing"),
    ]),
]
ELEV_DIR = "detailed_modular/render/elevations"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit(img, w, h):
    """Letterbox into w x h. Never stretch: aspect distortion misreads a figure."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                   interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), BG, np.uint8)
    y = (h - r.shape[0]) // 2; x = (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def sheet(title, items, root, out_png):
    tiles = []
    for rel, cap in items:
        p = root / rel
        if not p.exists():
            log(f"  MISSING {rel}")
            continue
        img = cv2.imread(str(p))
        if img is None:
            log(f"  UNREADABLE {rel}")
            continue
        t = fit(img, TILE_W, TILE_H - 34)
        pane = np.full((TILE_H, TILE_W, 3), BG, np.uint8)
        pane[34:] = t
        cv2.putText(pane, cap[:62], (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1)
        cv2.putText(pane, rel[:78], (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.33, SUB, 1)
        tiles.append(pane)
    if not tiles:
        return False
    cols = min(COLS, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    H = 44 + rows * TILE_H
    canvas = np.full((H, cols * TILE_W, 3), BG, np.uint8)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, FG, 2)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[44 + r * TILE_H:44 + (r + 1) * TILE_H,
               c * TILE_W:(c + 1) * TILE_W] = t
    cv2.imwrite(str(out_png), canvas)
    log(f"wrote {out_png.name} ({len(tiles)} tiles)")
    return True


def main(scan_root, out_dir):
    root = Path(scan_root); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, title, items in SHEETS:
        p = out / f"{name}.png"
        if sheet(title, items, root, p):
            made.append(p)

    ed = root / ELEV_DIR
    if ed.exists():
        els = sorted(ed.glob("*_elevation.png"),
                     key=lambda p: -p.stat().st_size)[:12]
        items = [(f"{ELEV_DIR}/{p.name}", p.stem.replace("_elevation", ""))
                 for p in els]
        p = out / "04_elevations.png"
        if sheet("WALL ELEVATIONS - relief (grooves, pilasters, beams) + openings",
                 items, root, p):
            made.append(p)
        log(f"  ({len(list(ed.glob('*_elevation.png')))} elevations exist, "
            f"12 largest shown)")

    log("")
    log("contact sheets:")
    for p in made:
        log(f"  {p}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
