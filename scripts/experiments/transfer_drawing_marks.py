"""transfer_drawing_marks.py
-------------------------
Carry marks from the architect drawing into the model frame using the
similarity transform from align_drawing.py, and use them to name openings.

The drawing states things the LiDAR cannot infer. The main entrance is one:
geometrically it is just another ~1000 mm doorway, and width alone gave TWO
"main entrance" candidates. The drawing settles it with a marker at a specific
position, so the answer becomes a lookup instead of a guess.

Marks read from the drawing:
  entrance   the blue triangle inside the plan area (the nine 55x55 blue
             squares at the right edge are the floor selector, not marks)

Each mark is transformed to model metres and matched to the nearest opening in
annotated_model.obj. The match distance is reported, because a mark that lands
900 mm from any opening has not identified anything and should not be trusted.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\transfer_drawing_marks.py \\
      <floorplan.png> <drawing_transform.json> <annotated_model.obj> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of

MATCH_MAX = 1.20     # m: beyond this the mark identifies nothing
PANEL_X_FRAC = 0.80  # right of this fraction of the image = legend, not plan


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def px_to_model(px, tf):
    """Drawing pixel (x, y, y down) -> model metres, per align_drawing.py."""
    p = np.array([px[0], -px[1]], float) - np.array(tf["drawing_centre_px"])
    p[0] *= tf["mirror"]
    p *= tf["scale_m_per_px"]
    th = np.radians(tf["rotation_deg"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return R @ p + np.array([tf["tx"], tf["ty"]])


def find_entrance(img_path):
    """The blue triangle inside the plan. Excludes the floor-selector column."""
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = ((h > 95) & (h < 130) & (s > 60) & (v > 40)).astype(np.uint8)
    lab, n = ndimage.label(blue)
    sizes = ndimage.sum(blue, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    cands = []
    for i, (sz, sl) in enumerate(zip(sizes, objs), start=1):
        cx = (sl[1].start + sl[1].stop) / 2
        cy = (sl[0].start + sl[0].stop) / 2
        if cx > PANEL_X_FRAC * W:            # floor selector, not a plan mark
            continue
        if sz < 60:
            continue
        cands.append((sz, cx, cy, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    sz, cx, cy, w, hh = cands[0]
    log(f"entrance marker: {int(sz)} px, {w}x{hh}, at ({cx:.0f}, {cy:.0f})")
    if len(cands) > 1:
        log(f"  ({len(cands)-1} other blue marks in the plan area, ignored)")
    return np.array([cx, cy])


def main(img_path, tf_path, obj_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = json.load(open(tf_path))
    log(f"transform: {tf['scale_m_per_px']*1000:.3f} mm/px, "
        f"{tf['rotation_deg']:.2f} deg, chamfer {tf['mean_chamfer_mm']:.0f} mm")

    mark_px = find_entrance(img_path)
    if mark_px is None:
        raise SystemExit("no entrance marker found in the drawing")
    mark = px_to_model(mark_px, tf)
    log(f"entrance in model frame: ({mark[0]:.3f}, {mark[1]:.3f}) m")

    boxes = parse_named_boxes(obj_path, ("door", "balcony_door", "window",
                                         "archway", "opening"))
    rows = []
    for name, box in boxes:
        c, n, along, up, w, h = frame_of(box)
        rows.append((float(np.linalg.norm(c[:2] - mark)), name,
                     round(w * 1000), c[:2]))
    rows.sort()
    log("")
    log("openings nearest the drawing's entrance marker:")
    for d, name, wmm, c in rows[:5]:
        log(f"  {d:5.2f} m   {name:34} ({wmm} mm wide)")

    best_d, best_name, best_w, _ = rows[0]
    if best_d > MATCH_MAX:
        log(f"\nNO MATCH: nearest opening is {best_d:.2f} m away (limit "
            f"{MATCH_MAX} m). The marker does not identify an opening.")
        verdict = None
    else:
        log(f"\nMAIN ENTRANCE = {best_name}  ({best_w} mm wide, "
            f"marker {best_d*1000:.0f} mm away)")
        verdict = dict(name=best_name, width_mm=best_w,
                       match_distance_mm=round(best_d * 1000, 1))

    # every other opening that width alone had called a main entrance
    rivals = [n for _, n, _, _ in rows if n.startswith("door")
              and n != best_name]
    json.dump(dict(entrance_px=[float(mark_px[0]), float(mark_px[1])],
                   entrance_model_xy=[float(mark[0]), float(mark[1])],
                   main_entrance=verdict,
                   nearest=[dict(name=n, dist_mm=round(d * 1000, 1),
                                 width_mm=w) for d, n, w, _ in rows[:5]]),
              open(out / "drawing_marks.json", "w"), indent=1)
    log(f"wrote {out/'drawing_marks.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
