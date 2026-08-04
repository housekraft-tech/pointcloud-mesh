"""Seeded-watershed 1:1 room segmentation (drawing topology, LiDAR boundaries).

Ported from scripts/experiments/register_drawing.py. The drawing's room-box
centres seed a watershed on the LiDAR free-space; LiDAR walls (plus the snapped
drawing walls, which close occluded/undetected partitions) are the barriers. The
result is one labelled basin per drawing room -- so an open-plan area is split
into its labelled sub-zones (Living / Dining / Kitchen) that wall geometry alone
cannot recover. Replaces floorplan2d.build_room_polygons as the pipeline's room
source when a good drawing registration is available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class Room:
    name: str
    mask: np.ndarray        # bool raster
    polygon: list           # [(x, y)] metric, wall frame
    area_m2: float


def _unique_names(seeds):
    cnt = {}
    for s in seeds:
        cnt[s["name"]] = cnt.get(s["name"], 0) + 1
    seen = {}
    for s in seeds:
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        s["uname"] = f'{s["name"]}-{seen[s["name"]]}' if cnt[s["name"]] > 1 else s["name"]
    return seeds


def segment_rooms(drawing, registration, free, wall, cell_m, xmin, ymax):
    LH, LW = free.shape
    free = (free > 0).astype(np.uint8)

    # snapped drawing walls become barriers too -> stops basins leaking through
    # partitions the LiDAR missed.
    barrier = np.zeros((LH, LW), np.uint8)
    for q0, q1 in registration.snapped_walls:
        cv2.line(barrier, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), 1, 2)
    free[barrier > 0] = 0

    seeds = _unique_names([dict(s) for s in drawing.room_seeds])
    seed_px = registration.transform.apply(
        np.array([[s["cx"], s["cy"]] for s in seeds], float)) if seeds else np.empty((0, 2))

    mk = np.zeros((LH, LW), np.int32)
    labels = []
    for i, (s, (px, py)) in enumerate(zip(seeds, seed_px), start=2):
        px, py = int(round(px)), int(round(py))
        # snap a seed that landed on/in a wall to the nearest free pixel
        if not (0 <= px < LW and 0 <= py < LH and free[py, px]):
            fy, fx = np.where(free > 0)
            if len(fx):
                j = np.argmin((fx - px) ** 2 + (fy - py) ** 2)
                px, py = int(fx[j]), int(fy[j])
        cv2.circle(mk, (px, py), 3, i, -1)
        labels.append(s["uname"])
    mk[free == 0] = 1                       # walls = barrier label
    cv2.watershed(cv2.merge([free * 200] * 3).astype(np.uint8), mk)

    rooms = []
    for i, name in enumerate(labels, start=2):
        m = (mk == i)
        a = float(m.sum() * cell_m * cell_m)
        if a < 0.5:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = [(float(xmin + px * cell_m), float(ymax - py * cell_m)) for px, py in c[:, 0, :]]
        rooms.append(Room(name=name, mask=m, polygon=poly, area_m2=round(a, 2)))
    return rooms
