"""Register an architect drawing onto the LiDAR metric frame.

Ported from scripts/experiments/register_drawing.py. Three deterministic stages:

  1. FOOTPRINT pose -- search rotation {0,90,180,270} x mirror, scale from
     footprint areas, align centroids, hill-climb translation to maximise
     footprint IoU.
  2. GLOBAL wall refine -- the area-based scale is a few % off; hill-climb
     (scale, angle, tx, ty) to bring drawing wall pixels close to LiDAR walls.
  3. PER-WALL snap -- shift each detected drawing wall perpendicular onto the
     nearest parallel LiDAR wall, absorbing residual local drift.

Returns a Transform (drawing crop px -> LiDAR raster px) plus quality metrics
(footprint IoU, fraction of walls matched, snapped-wall count) that the pipeline
uses as an acceptance gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2
from scipy import ndimage


@dataclass
class Transform:
    scale: float
    ang: float
    mir: bool
    tx: float
    ty: float
    dc: tuple      # drawing footprint centroid (px)
    lc: tuple      # LiDAR footprint centroid (px)

    def apply(self, pts):
        th = np.radians(self.ang)
        c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        p = np.asarray(pts, float) - np.asarray(self.dc, float)
        if self.mir:
            p = p * [-1, 1]
        p = (p * self.scale) @ Rm.T
        return p + np.asarray(self.lc, float) + [self.tx, self.ty]


@dataclass
class Registration:
    transform: Transform
    footprint_iou: float
    wall_match_frac: float
    n_snapped: int
    snapped_walls: list


def _to_pts(mask):
    ys, xs = np.where(mask > 0)
    return np.column_stack([xs, ys]).astype(np.float64)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni else 0.0


def register(drawing, lidar_free, lidar_occ, lidar_wall) -> Registration:
    lfoot = ndimage.binary_fill_holes((lidar_free > 0) | (lidar_occ > 0)).astype(np.uint8)
    lw = (lidar_wall > 0).astype(np.uint8)
    dfoot = (drawing.footprint_mask > 0).astype(np.uint8)
    dwall = (drawing.wall_mask > 0).astype(np.uint8)
    LH, LW = lfoot.shape

    area_scale = np.sqrt(lfoot.sum() / max(1, dfoot.sum()))
    dpts = _to_pts(dfoot)
    dwall_pts = _to_pts(dwall)
    dc = tuple(dpts.mean(0))
    lc = tuple(_to_pts(lfoot).mean(0))

    # ---- stage 1: footprint pose (rotation x mirror, scale, translation) ----
    best = None
    for ang in (0, 90, 180, 270):
        for mir in (False, True):
            th = np.radians(ang)
            cth, sth = np.cos(th), np.sin(th)
            Rm = np.array([[cth, -sth], [sth, cth]])

            def xf(pts, tx=0, ty=0, Rm=Rm, mir=mir):
                p = pts - dc
                if mir:
                    p = p * [-1, 1]
                p = (p * area_scale) @ Rm.T
                return p + lc + [tx, ty]

            def raster(pts):
                q = np.round(pts).astype(int)
                mm = np.zeros((LH, LW), np.uint8)
                ok = (q[:, 0] >= 0) & (q[:, 0] < LW) & (q[:, 1] >= 0) & (q[:, 1] < LH)
                mm[q[ok, 1], q[ok, 0]] = 1
                return cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            tx = ty = 0.0
            cur = _iou(raster(xf(dpts)), lfoot)
            for step in (16, 8, 4, 2):
                improved = True
                while improved:
                    improved = False
                    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        v = _iou(raster(xf(dpts, tx + dx, ty + dy)), lfoot)
                        if v > cur:
                            cur, tx, ty, improved = v, tx + dx, ty + dy, True
            if best is None or cur > best["iou"]:
                best = dict(iou=cur, ang=float(ang), mir=mir, tx=tx, ty=ty, scale=area_scale)

    # ---- stage 2: global wall refine (scale, angle, tx, ty) ----
    lw_dt = ndimage.distance_transform_edt(lw == 0)

    def gxf(scale, ang, tx, ty, pts):
        th = np.radians(ang)
        c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        p = np.asarray(pts, float) - dc
        if best["mir"]:
            p = p * [-1, 1]
        return (p * scale) @ Rm.T + lc + [tx, ty]

    def gscore(scale, ang, tx, ty, tol=5.0):
        q = np.round(gxf(scale, ang, tx, ty, dwall_pts)).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < LW) & (q[:, 1] >= 0) & (q[:, 1] < LH)
        return float((lw_dt[q[ok, 1], q[ok, 0]] <= tol).mean()) if ok.sum() else 0.0

    sc, an, tx, ty = best["scale"], best["ang"], best["tx"], best["ty"]
    cur = gscore(sc, an, tx, ty)
    for ss, aa, tt in [(0.03, 2.0, 14), (0.015, 1.0, 7), (0.007, 0.5, 3), (0.003, 0.25, 1)]:
        improved = True
        while improved:
            improved = False
            for dsc in (ss, -ss, 0):
                for da in (aa, -aa, 0):
                    for dx in (tt, -tt, 0):
                        for dy in (tt, -tt, 0):
                            v = gscore(sc * (1 + dsc), an + da, tx + dx, ty + dy)
                            if v > cur + 1e-6:
                                cur, sc, an, tx, ty = v, sc * (1 + dsc), an + da, tx + dx, ty + dy
                                improved = True
    best.update(scale=sc, ang=an, tx=tx, ty=ty)
    wall_match_frac = cur

    transform = Transform(scale=best["scale"], ang=best["ang"], mir=best["mir"],
                          tx=best["tx"], ty=best["ty"], dc=dc, lc=lc)

    # ---- stage 3: per-wall perpendicular snap ----
    def sample(p0, p1, step=1.0):
        L = np.hypot(*(p1 - p0))
        n = max(2, int(L / step))
        t = np.linspace(0, 1, n)[:, None]
        return p0 + (p1 - p0) * t

    snapped = []
    MAXSHIFT = 42
    n_snapped = 0
    for p0, p1 in drawing.wall_segs:
        q0, q1 = transform.apply(np.array([p0]))[0], transform.apply(np.array([p1]))[0]
        d = q1 - q0
        L = np.hypot(*d)
        if L < 3:
            continue
        d = d / L
        nrm = np.array([-d[1], d[0]])
        pts = sample(q0, q1)
        best_sh, best_cost = 0, np.inf
        for sh in range(-MAXSHIFT, MAXSHIFT + 1):
            g = np.round(pts + sh * nrm).astype(int)
            ok = (g[:, 0] >= 0) & (g[:, 0] < LW) & (g[:, 1] >= 0) & (g[:, 1] < LH)
            if ok.sum() < 3:
                continue
            cost = lw_dt[g[ok, 1], g[ok, 0]].mean()
            if cost < best_cost:
                best_cost, best_sh = cost, sh
        if best_cost <= 3.5:
            q0 = q0 + best_sh * nrm
            q1 = q1 + best_sh * nrm
            n_snapped += 1
        snapped.append((q0, q1))

    return Registration(transform=transform, footprint_iou=float(best["iou"]),
                        wall_match_frac=float(wall_match_frac), n_snapped=n_snapped,
                        snapped_walls=snapped)
