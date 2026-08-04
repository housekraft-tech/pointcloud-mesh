"""Detect rooms and walls on an architect floorplan image (RF-DETR).

The pipeline's room segmentation is only as good as its room boundaries, and
wall geometry alone cannot recover the sub-zones of an open-plan area (living /
dining / kitchen with no wall between them). The architect drawing *does* carry
those boundaries. This module runs the RF-DETR ``elements`` and ``walls`` models
on the drawing and returns the masks/seeds/segments the registration and
segmentation stages consume -- all in the drawing's own (cropped) pixel frame;
`register.py` maps them onto the LiDAR metric frame.
"""
from __future__ import annotations

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
    """Drawing detections in the cropped-plan pixel frame."""

    origin: tuple            # (x0, y0) crop offset in full-image px
    wall_mask: np.ndarray    # uint8 crop -- dark wall strokes within the footprint
    footprint_mask: np.ndarray   # uint8 crop -- union of room boxes, filled
    room_seeds: list         # [{name, cx, cy}] crop px (room-box centres)
    wall_segs: list          # [(p0, p1)] crop px -- clean wall segments


def _masks_from_dets(bgr, room_dets, wall_dets) -> DrawingModel:
    """Pure image processing: crop to the room-box union, build the footprint and
    wall masks, room seeds, and wall segments. Model-free so it is unit-testable.
    """
    H, W = bgr.shape[:2]
    rooms = [d for d in room_dets if d["name"] in ROOM_CLASSES]
    xs = [v for d in rooms for v in (d["box"][0], d["box"][2])]
    ys = [v for d in rooms for v in (d["box"][1], d["box"][3])]
    m = 40
    x0, y0 = max(0, int(min(xs)) - m), max(0, int(min(ys)) - m)
    x1, y1 = min(W, int(max(xs)) + m), min(H, int(max(ys)) + m)
    plan = bgr[y0:y1, x0:x1]
    ph, pw = plan.shape[:2]
    gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)

    # footprint = union of room boxes (covers the whole apartment incl. balconies;
    # robust to broken outer-wall loops at door/balcony gaps).
    foot = np.zeros((ph, pw), np.uint8)
    for d in rooms:
        bx0, by0, bx1, by1 = d["box"]
        foot[max(0, int(by0) - y0):int(by1) - y0, max(0, int(bx0) - x0):int(bx1) - x0] = 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    foot = ndimage.binary_fill_holes(foot).astype(np.uint8)

    # walls = dark thick strokes within the footprint; drop small text glyphs.
    wall = (gray < 110).astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & foot
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    clean = np.zeros_like(wall)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 120:
            clean[lbl == i] = 1
    wall = clean

    seeds = [dict(name=d["name"],
                  cx=(d["box"][0] + d["box"][2]) / 2 - x0,
                  cy=(d["box"][1] + d["box"][3]) / 2 - y0) for d in rooms]

    # clean wall segments from the walls model: each wall box -> a line along its
    # long axis at the short-axis centre (crop coords).
    wall_segs = []
    for d in wall_dets:
        if d["name"] != "wall":
            continue
        bx0, by0, bx1, by1 = d["box"]
        w, h = bx1 - bx0, by1 - by0
        if max(w, h) < 12:
            continue
        if w >= h:                       # horizontal wall
            p0 = (bx0 - x0, (by0 + by1) / 2 - y0)
            p1 = (bx1 - x0, (by0 + by1) / 2 - y0)
        else:                            # vertical wall
            p0 = ((bx0 + bx1) / 2 - x0, by0 - y0)
            p1 = ((bx0 + bx1) / 2 - x0, by1 - y0)
        wall_segs.append((np.array(p0, float), np.array(p1, float)))

    return DrawingModel((x0, y0), wall, foot, seeds, wall_segs)


def detect_drawing(img_path: str) -> DrawingModel:
    """Run the RF-DETR elements + walls models on the plan image and return a
    DrawingModel. Requires the ONNX weights under ``models/``.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    room_dets = run_model(load("elements"), bgr, ELEMENTS_CLASSES,
                          ELEMENTS_CONFIDENCE, ELEMENTS_CLASS_THRESHOLDS)
    wall_dets = run_model(load("walls"), bgr, WALLS_WINDOWS_CLASSES,
                          WALLS_WINDOWS_CONFIDENCE, WALLS_WINDOWS_CLASS_THRESHOLDS)
    return _masks_from_dets(bgr, room_dets, wall_dets)
