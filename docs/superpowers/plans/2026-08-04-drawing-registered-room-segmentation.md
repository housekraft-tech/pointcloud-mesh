# Drawing-Registered Room Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace wall-only room segmentation with drawing-registered segmentation so room dimensions match the architect plan to cm level (down from 1–2.4 m error).

**Architecture:** Detect rooms on the architect drawing (RF-DETR), register the drawing onto the LiDAR metric frame (footprint pose → wall refine → per-wall snap), seeded-watershed segment 1:1 using drawing seeds + LiDAR walls, then feed the labeled rooms to the existing `metrology.clear_between` / manifest / 2D / 3D. A registration quality gate falls back to today's wall-only segmentation when a good drawing registration is not available.

**Tech Stack:** Python 3.11, numpy, OpenCV (`cv2`), scipy.ndimage, onnxruntime (RF-DETR), pytest. All CPU-only, run via `venv311`.

## Global Constraints

- Python 3.11 64-bit via `.\venv311\Scripts\python.exe` (open3d/onnx need 64-bit).
- CPU-only, deterministic: fixed angle set + fixed hill-climb schedules; no `Math.random`-style nondeterminism in the segmentation path (RNG only for debug colours, seeded `default_rng(0)`).
- Reuse, do not fork: port from `scripts/experiments/register_drawing.py`, `rfdetr_infer.py`, `explain_lidar_to_3d.py` into `scripts/recon/`. Keep the experiments as-is.
- Metric raster helpers: `reconstruct(las)` returns `R` with keys `free, occ, x, y, z, xmin, ymax`; pixel↔metric via module constant `CELL` (m/px). Row = `(ymax - y)/CELL`, col = `(x - xmin)/CELL`.
- Do NOT regress `metrology.clear_between` to raster-bbox measurement.
- Keep the existing test suite green (`pytest -q` → currently 244 passed, 1 skipped).
- Model weights live at `models/rfdetr_elements.onnx`, `models/rfdetr_walls_windows.onnx`.

---

### Task 0: Verify RF-DETR models fire on the plan (de-risk gate)

**Files:**
- Create: `tests/recon/test_drawing_models_available.py`

**Interfaces:**
- Consumes: `scripts.experiments.rfdetr_infer.load`, `run_model`, `ELEMENTS_CLASSES`, `ELEMENTS_CONFIDENCE`, `ELEMENTS_CLASS_THRESHOLDS`.
- Produces: nothing (evidence gate). Confirms the model returns room boxes on `floorplan_original.png`.

- [ ] **Step 1: Write the gate test**

```python
# tests/recon/test_drawing_models_available.py
import os
import cv2
import pytest
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}
PLAN = "floorplan_original.png"

@pytest.mark.skipif(not os.path.exists("models/rfdetr_elements.onnx"),
                    reason="RF-DETR elements model weights not present")
def test_elements_model_detects_rooms_on_koushik_plan():
    bgr = cv2.imread(PLAN)
    assert bgr is not None, f"{PLAN} not found"
    dets = run_model(load("elements"), bgr, ELEMENTS_CLASSES,
                     ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    rooms = [d for d in dets if d["name"] in ROOM_CLASSES]
    assert len(rooms) >= 6, f"expected >=6 room boxes, got {len(rooms)}: {[d['name'] for d in rooms]}"
```

- [ ] **Step 2: Run it**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_drawing_models_available.py -v`
Expected: PASS (≥6 room boxes). If it FAILS or skips because weights are missing, STOP — the whole plan depends on this; report to the human before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/recon/test_drawing_models_available.py
git commit -m "test: gate RF-DETR elements model on koushik plan"
```

---

### Task 1: `recon/drawing.py` — detect rooms + walls on the drawing

**Files:**
- Create: `scripts/recon/drawing.py`
- Test: `tests/recon/test_drawing.py`

**Interfaces:**
- Consumes: `rfdetr_infer.{load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS, WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS}`.
- Produces:
  - `@dataclass DrawingModel:` `origin: tuple[int,int]` (crop x0,y0), `wall_mask: np.ndarray` (uint8, crop px), `footprint_mask: np.ndarray`, `room_seeds: list[dict]` (`{name, cx, cy}` crop px), `wall_segs: list[tuple[np.ndarray, np.ndarray]]` (p0,p1 crop px).
  - `detect_drawing(img_path: str) -> DrawingModel`.
  - `_masks_from_dets(bgr, room_dets, wall_dets) -> DrawingModel` — pure, model-free (unit-tested).

- [ ] **Step 1: Write the failing unit test (pure mask builder, no model)**

```python
# tests/recon/test_drawing.py
import numpy as np
from scripts.recon.drawing import _masks_from_dets

def _det(name, box): return {"name": name, "box": box}

def test_masks_from_dets_crops_and_builds_seeds_and_footprint():
    # 400x400 white image, two dark room rectangles side by side.
    bgr = np.full((400, 400, 3), 255, np.uint8)
    # room A box [40,40,180,360], room B box [220,40,360,360]
    for (x0, y0, x1, y1) in [(40, 40, 180, 360), (220, 40, 360, 360)]:
        bgr[y0:y1, x0:x1] = 240
        bgr[y0:y0+6, x0:x1] = 20  # thick dark wall strokes on the room edges
        bgr[y1-6:y1, x0:x1] = 20
    room_dets = [_det("Bedroom", [40, 40, 180, 360]),
                 _det("Kitchen", [220, 40, 360, 360])]
    wall_dets = [{"name": "wall", "box": [40, 40, 180, 46]}]

    dm = _masks_from_dets(bgr, room_dets, wall_dets)

    assert len(dm.room_seeds) == 2
    names = {s["name"] for s in dm.room_seeds}
    assert names == {"Bedroom", "Kitchen"}
    # seeds are in CROP coordinates and inside the crop
    for s in dm.room_seeds:
        assert 0 <= s["cx"] < dm.wall_mask.shape[1]
        assert 0 <= s["cy"] < dm.wall_mask.shape[0]
    # footprint covers both room boxes; wall mask has the dark strokes
    assert dm.footprint_mask.sum() > 0
    assert dm.wall_mask.sum() > 0
    assert len(dm.wall_segs) == 1
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_drawing.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.recon.drawing` / `_masks_from_dets` undefined.

- [ ] **Step 3: Implement `recon/drawing.py`**

Port the mask logic from `register_drawing.drawing_masks` (lines 35–88), split into a pure `_masks_from_dets` plus a thin `detect_drawing` that runs the models. Structure:

```python
# scripts/recon/drawing.py
from dataclasses import dataclass
import numpy as np
import cv2
from scipy import ndimage
from scripts.experiments.rfdetr_infer import (
    load, run_model, ELEMENTS_CLASSES, ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS,
    WALLS_WINDOWS_CLASSES, WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)

ROOM_CLASSES = {"Bedroom", "Bathroom", "Kitchen", "Utility", "Walkin",
                "Dining Room", "Balcony", "Living Room", "Foyer"}

@dataclass
class DrawingModel:
    origin: tuple           # (x0, y0) crop offset in full-image px
    wall_mask: np.ndarray   # uint8 crop
    footprint_mask: np.ndarray
    room_seeds: list        # [{name, cx, cy}] crop px
    wall_segs: list         # [(p0, p1)] crop px

def _masks_from_dets(bgr, room_dets, wall_dets) -> DrawingModel:
    H, W = bgr.shape[:2]
    rooms = [d for d in room_dets if d["name"] in ROOM_CLASSES]
    xs = [v for d in rooms for v in (d["box"][0], d["box"][2])]
    ys = [v for d in rooms for v in (d["box"][1], d["box"][3])]
    m = 40
    x0, y0 = max(0, int(min(xs)) - m), max(0, int(min(ys)) - m)
    x1, y1 = min(W, int(max(xs)) + m), min(H, int(max(ys)) + m)
    plan = bgr[y0:y1, x0:x1]; ph, pw = plan.shape[:2]
    gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
    foot = np.zeros((ph, pw), np.uint8)
    for d in rooms:
        bx0, by0, bx1, by1 = d["box"]
        foot[max(0, int(by0)-y0):int(by1)-y0, max(0, int(bx0)-x0):int(bx1)-x0] = 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    foot = ndimage.binary_fill_holes(foot).astype(np.uint8)
    wall = (gray < 110).astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & foot
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    clean = np.zeros_like(wall)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 120:
            clean[lbl == i] = 1
    wall = clean
    seeds = [dict(name=d["name"], cx=(d["box"][0]+d["box"][2])/2 - x0,
                  cy=(d["box"][1]+d["box"][3])/2 - y0) for d in rooms]
    wall_segs = []
    for d in wall_dets:
        if d["name"] != "wall":
            continue
        bx0, by0, bx1, by1 = d["box"]
        w, h = bx1-bx0, by1-by0
        if max(w, h) < 12:
            continue
        if w >= h:
            p0 = (bx0-x0, (by0+by1)/2 - y0); p1 = (bx1-x0, (by0+by1)/2 - y0)
        else:
            p0 = ((bx0+bx1)/2 - x0, by0-y0); p1 = ((bx0+bx1)/2 - x0, by1-y0)
        wall_segs.append((np.array(p0, float), np.array(p1, float)))
    return DrawingModel((x0, y0), wall, foot, seeds, wall_segs)

def detect_drawing(img_path: str) -> DrawingModel:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    room_dets = run_model(load("elements"), bgr, ELEMENTS_CLASSES,
                          ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    wall_dets = run_model(load("walls"), bgr, WALLS_WINDOWS_CLASSES,
                          WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    return _masks_from_dets(bgr, room_dets, wall_dets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_drawing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/recon/drawing.py tests/recon/test_drawing.py
git commit -m "feat(recon): drawing element detection + mask builder"
```

---

### Task 2: `recon/register.py` — align the drawing onto the LiDAR frame

**Files:**
- Create: `scripts/recon/register.py`
- Test: `tests/recon/test_register.py`

**Interfaces:**
- Consumes: `DrawingModel` (Task 1); LiDAR `free`/`occ`/`wall` uint8 rasters + `cell_m`.
- Produces:
  - `@dataclass Transform:` fields `scale, ang, mir, tx, ty, dc(tuple), lc(tuple)`, method `apply(pts_px: np.ndarray) -> np.ndarray` (→ LiDAR raster px).
  - `@dataclass Registration:` `transform: Transform`, `footprint_iou: float`, `wall_match_frac: float`, `n_snapped: int`, `snapped_walls: list[(q0,q1)]`.
  - `register(drawing, lidar_free, lidar_occ, lidar_wall) -> Registration`.

- [ ] **Step 1: Write the failing test (recover a known synthetic transform)**

```python
# tests/recon/test_register.py
import numpy as np
from scripts.recon.drawing import DrawingModel
from scripts.recon.register import register

def _rect_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, np.uint8); m[y0:y1, x0:x1] = 1; return m

def test_register_recovers_scale_and_translation_of_a_footprint():
    # LiDAR footprint: a 120x80 rectangle placed at (40,30) in a 200x200 frame.
    lfoot = _rect_mask((200, 200), 40, 30, 160, 110)
    lwall = np.zeros((200, 200), np.uint8)
    lwall[30:110, 40] = 1; lwall[30:110, 159] = 1      # two vertical walls
    lwall[30, 40:160] = 1; lwall[109, 40:160] = 1      # two horizontal walls
    lfree = (lfoot & (lwall == 0)).astype(np.uint8)
    # Drawing footprint: same rectangle at half the pixel scale, offset elsewhere.
    dfoot = _rect_mask((300, 300), 10, 10, 70, 50)     # 60x40, scale x2 to match
    dwall = np.zeros((300, 300), np.uint8)
    dwall[10:50, 10] = 1; dwall[10:50, 69] = 1
    dwall[10, 10:70] = 1; dwall[49, 10:70] = 1
    seeds = [dict(name="Room", cx=40, cy=30)]
    wall_segs = [(np.array([10., 10.]), np.array([10., 50.]))]
    dm = DrawingModel((0, 0), dwall, dfoot, seeds, wall_segs)

    reg = register(dm, lfree, lidar_occ=lfoot, lidar_wall=lwall)

    assert reg.footprint_iou > 0.9
    # a drawing corner (10,10) should map near the LiDAR rectangle corner (40,30)
    q = reg.transform.apply(np.array([[10., 10.]]))[0]
    assert abs(q[0] - 40) < 6 and abs(q[1] - 30) < 6
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_register.py -v`
Expected: FAIL (`scripts.recon.register` missing).

- [ ] **Step 3: Implement `recon/register.py`**

Port the pose search + global wall refine + per-wall snap from `register_drawing.main` (lines 110–247). Move the transform math into `Transform.apply` and return a `Registration`. Keep the exact deterministic schedules:
- footprint: angles `(0,90,180,270)` × mirror `(False,True)`, `area_scale = sqrt(lfoot.sum()/dfoot.sum())`, centroid align, translation hill-climb steps `(16,8,4,2)` maximizing footprint IoU (raster with a 7×7 close).
- global refine: hill-climb `(scale,ang,tx,ty)` over schedule `[(0.03,2.0,14),(0.015,1.0,7),(0.007,0.5,3),(0.003,0.25,1)]` maximizing fraction of drawing-wall px within `tol=5` of a LiDAR wall (distance transform of `lwall==0`); record that fraction as `wall_match_frac`.
- per-wall snap: for each `wall_seg`, search perpendicular shift `range(-42,43)`, accept if mean distance ≤ 3.5; record `n_snapped` and `snapped_walls` (transformed + snapped endpoints).

```python
# scripts/recon/register.py  (structure; port bodies from register_drawing.main)
from dataclasses import dataclass
import numpy as np, cv2
from scipy import ndimage

@dataclass
class Transform:
    scale: float; ang: float; mir: bool
    tx: float; ty: float; dc: tuple; lc: tuple
    def apply(self, pts):
        th = np.radians(self.ang); c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        p = np.asarray(pts, float) - self.dc
        if self.mir:
            p = p * [-1, 1]
        p = (p * self.scale) @ Rm.T
        return p + self.lc + [self.tx, self.ty]

@dataclass
class Registration:
    transform: Transform
    footprint_iou: float
    wall_match_frac: float
    n_snapped: int
    snapped_walls: list

def register(drawing, lidar_free, lidar_occ, lidar_wall) -> Registration:
    ...  # port lines 110-247 of register_drawing.main verbatim, returning the
         # dataclasses above (dc = drawing footprint centroid, lc = LiDAR footprint
         # centroid). No file writing here.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_register.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/recon/register.py tests/recon/test_register.py
git commit -m "feat(recon): register drawing onto LiDAR frame"
```

---

### Task 3: `recon/rooms.py` — seeded-watershed 1:1 segmentation

**Files:**
- Create: `scripts/recon/rooms.py`
- Test: `tests/recon/test_rooms.py`

**Interfaces:**
- Consumes: `DrawingModel`, `Registration` (Tasks 1–2), LiDAR `free`/`wall` rasters, `cell_m`, `xmin`, `ymax`.
- Produces:
  - `@dataclass Room:` `name: str`, `mask: np.ndarray` (bool, raster), `polygon: list[(x,y)]` (metric, wall frame), `area_m2: float`.
  - `segment_rooms(drawing, registration, free, wall, cell_m, xmin, ymax) -> list[Room]`.

- [ ] **Step 1: Write the failing test (two seeds split by a wall)**

```python
# tests/recon/test_rooms.py
import numpy as np
from scripts.recon.drawing import DrawingModel
from scripts.recon.register import Transform, Registration
from scripts.recon.rooms import segment_rooms

def test_segment_splits_two_rooms_by_a_dividing_wall():
    free = np.ones((100, 200), np.uint8)
    wall = np.zeros((100, 200), np.uint8)
    wall[:, 99:101] = 1                 # vertical divider at x=100
    free[wall > 0] = 0
    # identity transform: drawing px == raster px
    T = Transform(1.0, 0.0, False, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))
    reg = Registration(T, footprint_iou=1.0, wall_match_frac=1.0, n_snapped=0, snapped_walls=[])
    seeds = [dict(name="Left", cx=50, cy=50), dict(name="Right", cx=150, cy=50)]
    dm = DrawingModel((0, 0), wall, np.ones((100, 200), np.uint8), seeds, [])

    rooms = segment_rooms(dm, reg, free, wall, cell_m=0.02, xmin=0.0, ymax=2.0)

    assert {r.name for r in rooms} == {"Left", "Right"}
    left = next(r for r in rooms if r.name == "Left")
    right = next(r for r in rooms if r.name == "Right")
    # each room stays on its own side of the divider
    assert left.mask[:, :99].sum() > 0 and left.mask[:, 101:].sum() == 0
    assert right.mask[:, 101:].sum() > 0 and right.mask[:, :99].sum() == 0
    assert left.area_m2 > 0 and right.area_m2 > 0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_rooms.py -v`
Expected: FAIL (`scripts.recon.rooms` missing).

- [ ] **Step 3: Implement `recon/rooms.py`**

Port the watershed block from `register_drawing.main` (lines 249–300). Transform seeds with `registration.transform.apply`, burn snapped drawing walls into `free` as barriers, seed-snap into free space, `cv2.watershed`, then per-label extract mask, area (`mask.sum()*cell_m**2`), and a metric polygon (mask contour → approx → map px→metric via `x = xmin + col*cell_m`, `y = ymax - row*cell_m`). Drop labels < 0.5 m².

```python
# scripts/recon/rooms.py  (structure)
from dataclasses import dataclass
import numpy as np, cv2

@dataclass
class Room:
    name: str; mask: np.ndarray; polygon: list; area_m2: float

def _unique_names(seeds):
    cnt = {}; 
    for s in seeds: cnt[s["name"]] = cnt.get(s["name"], 0) + 1
    seen = {}
    for s in seeds:
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        s["uname"] = f'{s["name"]}-{seen[s["name"]]}' if cnt[s["name"]] > 1 else s["name"]
    return seeds

def segment_rooms(drawing, registration, free, wall, cell_m, xmin, ymax):
    LH, LW = free.shape
    free = free.copy()
    barrier = np.zeros((LH, LW), np.uint8)
    for q0, q1 in registration.snapped_walls:
        cv2.line(barrier, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), 1, 2)
    free[barrier > 0] = 0
    seeds = _unique_names(list(drawing.room_seeds))
    seed_px = registration.transform.apply(np.array([[s["cx"], s["cy"]] for s in seeds]))
    mk = np.zeros((LH, LW), np.int32); labels = []
    for i, (s, (px, py)) in enumerate(zip(seeds, seed_px), start=2):
        px, py = int(round(px)), int(round(py))
        if not (0 <= px < LW and 0 <= py < LH and free[py, px]):
            fy, fx = np.where(free > 0)
            if len(fx):
                j = np.argmin((fx - px) ** 2 + (fy - py) ** 2); px, py = int(fx[j]), int(fy[j])
        cv2.circle(mk, (px, py), 3, i, -1); labels.append(s["uname"])
    mk[free == 0] = 1
    cv2.watershed(cv2.merge([free * 200] * 3).astype(np.uint8), mk)
    rooms = []
    for i, name in enumerate(labels, start=2):
        m = (mk == i)
        a = float(m.sum() * cell_m * cell_m)
        if a < 0.5:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        poly = [(float(xmin + px * cell_m), float(ymax - py * cell_m)) for px, py in c[:, 0, :]]
        rooms.append(Room(name, m, poly, round(a, 2)))
    return rooms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_rooms.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/recon/rooms.py tests/recon/test_rooms.py
git commit -m "feat(recon): seeded-watershed 1:1 room segmentation"
```

---

### Task 4: Wire into `isolidarflow` with quality gate + wall_only fallback

**Files:**
- Modify: `scripts/isolidarflow.py` (config block near line 138; room stage near lines 409–422)
- Test: `tests/recon/test_room_source_selection.py`

**Interfaces:**
- Consumes: `detect_drawing`, `register`, `segment_rooms`; existing `reconstruct` for `free/occ/wall`.
- Produces: a `_resolve_rooms(cfg, xyz, wall_ns, ...) -> (rooms, room_source)` helper where `rooms` are shapely polygons (as `build_room_polygons` returns today) so the rest of the pipeline is unchanged; `room_source in {"drawing","wall_only"}`. `Room.polygon` → `shapely.geometry.Polygon`.

- [ ] **Step 1: Write the failing test (gate falls back when registration is poor)**

```python
# tests/recon/test_room_source_selection.py
from scripts.isolidarflow import _select_room_source

def test_low_quality_registration_falls_back_to_wall_only():
    assert _select_room_source(footprint_iou=0.30, wall_match_frac=0.20,
                               min_iou=0.6, min_wall_frac=0.5) == "wall_only"

def test_good_registration_uses_drawing():
    assert _select_room_source(footprint_iou=0.82, wall_match_frac=0.71,
                               min_iou=0.6, min_wall_frac=0.5) == "drawing"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_room_source_selection.py -v`
Expected: FAIL (`_select_room_source` undefined).

- [ ] **Step 3: Implement the gate + wiring**

Add config keys (near line 138):

```python
    "drawing_path": None,            # architect plan image; None => wall_only rooms
    "reg_min_footprint_iou": 0.6,    # registration acceptance gate
    "reg_min_wall_frac": 0.5,
```

Add the pure gate + resolver:

```python
def _select_room_source(footprint_iou, wall_match_frac, min_iou, min_wall_frac):
    ok = footprint_iou >= min_iou and wall_match_frac >= min_wall_frac
    return "drawing" if ok else "wall_only"
```

In `run()`, replace the room-polygon stage (lines 409–419): if `cfg["drawing_path"]`, build `reconstruct`-derived `free/occ/wall`, `detect_drawing` → `register` → gate; on "drawing" use `segment_rooms` and convert each `Room.polygon` to a shapely `Polygon` (carry `name`); else keep the existing `build_room_polygons` path. Set `info["room_source"]`, `info["n_rooms"]`, `info["room_areas_m2"]`. Pass room names through to `room_coords`/manifest.

- [ ] **Step 4: Run test to verify it passes + suite green**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_room_source_selection.py -q`
Then: `.\venv311\Scripts\python.exe -m pytest -q`
Expected: both PASS; full suite still 244+ passed (wall_only path unchanged when `drawing_path` is None).

- [ ] **Step 5: Commit**

```bash
git add scripts/isolidarflow.py tests/recon/test_room_source_selection.py
git commit -m "feat(isolidarflow): drawing-registered rooms with quality gate + wall_only fallback"
```

---

### Task 5: Carry room names + source into the manifest

**Files:**
- Modify: `scripts/recon/schema.py` (room_dicts, ~lines 321–326)
- Modify: `scripts/isolidarflow.py` (pass names + room_source through)
- Test: `tests/recon/test_manifest_room_names.py`

**Interfaces:**
- Consumes: labeled rooms from Task 4.
- Produces: manifest `rooms[i]` gains optional `name`; manifest top-level gains `room_source`.

- [ ] **Step 1: Write the failing test**

```python
# tests/recon/test_manifest_room_names.py
from scripts.recon.schema import build_manifest

def test_room_names_appear_in_manifest():
    rooms = [[(0, 0), (3, 0), (3, 4), (0, 4)]]
    names = ["Bedroom-1"]
    m = build_manifest([], {}, [], [], rooms, 0.0, 2.7, {},
                       room_names=names, room_source="drawing")
    assert m["rooms"][0]["name"] == "Bedroom-1"
    assert m["room_source"] == "drawing"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_manifest_room_names.py -v`
Expected: FAIL (`build_manifest` has no `room_names`/`room_source` kwargs).

- [ ] **Step 3: Implement**

Give `build_manifest` optional `room_names=None, room_source=None`; when `room_names` present, set `room_dicts[i]["name"]`; add `"room_source"` to the returned dict when provided. Update the `isolidarflow` call to pass both.

- [ ] **Step 4: Run test + suite**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_manifest_room_names.py tests/recon/test_schema.py -q`
Expected: PASS (existing schema tests unaffected — new kwargs default to None).

- [ ] **Step 5: Commit**

```bash
git add scripts/recon/schema.py scripts/isolidarflow.py tests/recon/test_manifest_room_names.py
git commit -m "feat(schema): carry room name + room_source into manifest"
```

---

### Task 6: Real-data acceptance — koushik cm-level accuracy + determinism

**Files:**
- Create: `scripts/experiments/validate_room_accuracy.py` (comparison harness)
- Test: `tests/recon/test_room_accuracy_koushik.py` (gated on model + scan presence)

**Interfaces:**
- Consumes: the full pipeline via `isolidarflow.run(..., config={"drawing_path": "floorplan_original.png", ...})`.
- Produces: a per-room error report vs the architect plan + manual survey; assertions on cm-level accuracy and determinism.

- [ ] **Step 1: Write the acceptance test (gated)**

```python
# tests/recon/test_room_accuracy_koushik.py
import os, pytest
from scripts.isolidarflow import run

GATED = not (os.path.exists("models/rfdetr_elements.onnx")
             and os.path.exists("koushikexport.las")
             and os.path.exists("floorplan_original.png"))

# Architect plan clear dims (mm), sorted (short, long).
PLAN = {
    "Living Room": (5000, 6530), "Kitchen": (2250, 5050),
    "Bedroom-a": (3450, 3540), "Bedroom-b": (3030, 4890), "Bedroom-c": (3250, 4200),
}

@pytest.mark.skipif(GATED, reason="needs RF-DETR model + koushik scan + plan")
def test_koushik_rooms_within_cm_of_plan(tmp_path):
    res = run("koushikexport.las", str(tmp_path),
              config={"drawing_path": "floorplan_original.png"})
    rooms = res["manifest"]["rooms"]
    assert res["manifest"]["room_source"] == "drawing"
    # living/dining/kitchen split exists (open-plan no longer one blob)
    names = {r.get("name", "") for r in rooms}
    assert any("Kitchen" in n for n in names) and any("Living" in n for n in names)
    # every plan room has a detected room within 60 mm on each axis
    dims = []
    for r in rooms:
        cx, cy = r.get("clear_x_m"), r.get("clear_y_m")
        if cx and cy:
            dims.append(tuple(sorted((round(cx*1000), round(cy*1000)))))
    for _, (pw, pl) in PLAN.items():
        best = min((abs(w-pw)+abs(l-pl) for (w, l) in dims), default=1e9)
        assert best <= 120, f"no room within 120mm total of plan {(pw, pl)}; dims={dims}"

@pytest.mark.skipif(GATED, reason="needs RF-DETR model + koushik scan + plan")
def test_segmentation_is_deterministic(tmp_path):
    a = run("koushikexport.las", str(tmp_path/"a"), config={"drawing_path": "floorplan_original.png"})
    b = run("koushikexport.las", str(tmp_path/"b"), config={"drawing_path": "floorplan_original.png"})
    da = sorted((r.get("name"), round(r["area_m2"], 1)) for r in a["manifest"]["rooms"])
    db = sorted((r.get("name"), round(r["area_m2"], 1)) for r in b["manifest"]["rooms"])
    assert da == db
```

- [ ] **Step 2: Run it**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_room_accuracy_koushik.py -v`
Expected: initially may FAIL on the accuracy threshold — that is the real acceptance signal. Investigate registration quality / seed placement (use `validate_room_accuracy.py` overlay) until within tolerance. Do NOT loosen the 120 mm threshold to pass; fix the pipeline. If a specific room genuinely cannot hit cm (e.g. an alcove the plan dimensions to a wardrobe face), document it and adjust that room's expected value with a comment citing why.

- [ ] **Step 3: Build the comparison/QA harness**

`validate_room_accuracy.py`: run the pipeline, print the per-room dims-vs-plan table (like the diagnosis table), and write an overlay PNG of segmented rooms for visual QA. Also compare against `validate_rooms.py`'s manual-survey numbers.

- [ ] **Step 4: Verify accuracy + determinism pass**

Run: `.\venv311\Scripts\python.exe -m pytest tests/recon/test_room_accuracy_koushik.py -q`
Expected: PASS. Capture the before/after table (1–2.4 m → cm) in the commit message.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/validate_room_accuracy.py tests/recon/test_room_accuracy_koushik.py
git commit -m "test(recon): koushik room accuracy vs plan + determinism (drawing-registered)"
```

---

## Self-Review

**Spec coverage:**
- drawing.py / register.py / rooms.py modules → Tasks 1/2/3. ✓
- isolidarflow integration + metrology on correct rooms → Task 4 (rooms replace build_room_polygons; metrology.clear_between unchanged, now fed correct polygons). ✓
- quality gate + wall_only fallback → Task 4. ✓
- room name + room_source in manifest → Task 5. ✓
- validation vs plan + survey + determinism → Task 6. ✓
- RF-DETR dependency de-risk (spec "step 0") → Task 0. ✓
- Non-goal (columns/beams/grooves/arches) → correctly absent (Spec B). ✓

**Placeholder scan:** `register()` body in Task 2 says "port lines 110–247" rather than re-pasting ~140 lines — the source is in-repo and the deterministic schedules + return types are fully specified, so an engineer has exact instructions and interfaces. All tests are concrete. No TBD/loose "add error handling".

**Type consistency:** `DrawingModel` fields (origin, wall_mask, footprint_mask, room_seeds, wall_segs) used identically in Tasks 1–4. `Transform.apply` / `Registration` fields (footprint_iou, wall_match_frac, n_snapped, snapped_walls) consistent Tasks 2–4. `Room` (name, mask, polygon, area_m2) consistent Tasks 3–4. `_select_room_source` signature matches its test. `build_manifest(room_names=, room_source=)` matches Task 5 test.

---

## Execution Handoff

Plan complete. Note Task 6's accuracy test is the real acceptance gate and may require iteration on registration/seed quality — that is expected and must be fixed in the pipeline, not by loosening the threshold.
