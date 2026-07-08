"""skeleton_3d.py
--------------
Build a clean 3D straight from the wall SKELETON (centerline network). The
skeleton is the neatest wall representation -- one 1px line per wall, no doubled
faces -- so extruding it gives clean single-thickness walls following the exact
network. Optionally cut real openings from the (u,z) silhouette so it isn't a
solid slab.

  wall footprint -> skeletonize -> thicken to wall thickness -> extrude to ceiling

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\skeleton_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL
from scripts.experiments.volumetric_3d import wall_footprint, rectilinear
from scripts.experiments.build_modular_3d import footprint_polygons
from scipy import ndimage as _ndi

WALL_T = 0.11          # uniform wall thickness (m)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _union(meshes):
    """Robust manifold union -> ONE connected watertight solid (beams merge into
    walls; no doubled faces). Falls back to concatenate if the engine is absent."""
    meshes = [m for m in meshes if m is not None and len(getattr(m, "faces", []))]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    try:
        return trimesh.boolean.union(meshes, engine="manifold")
    except Exception:
        try:
            return trimesh.boolean.union(meshes)
        except Exception:
            return trimesh.util.concatenate(meshes)


def _difference(a, negs):
    """Subtract all negatives in ONE manifold op so recesses/openings are true
    intrusions of the SAME surface, not separate shells."""
    negs = [m for m in negs if m is not None and len(getattr(m, "faces", []))]
    if a is None or not negs:
        return a
    try:
        return trimesh.boolean.difference([a] + negs, engine="manifold")
    except Exception:
        for n in negs:
            try:
                a = a.difference(n)
            except Exception:
                pass
        return a


def _clean(m):
    """Weld coincident verts, drop degenerate/duplicate faces, LOSSLESS-merge
    coplanar faces (collapse only zero-error/collinear edges -> fewer triangles
    while keeping watertightness and every 90-degree edge -- no rounding), fix
    winding so the surface reads as one smooth solid."""
    if m is None:
        return None
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    try:
        import fast_simplification as _fs
        v, f = _fs.simplify(m.vertices, m.faces, target_reduction=0.99, lossless=True)
        c = trimesh.Trimesh(v, f, process=True)
        if c.is_watertight and len(c.faces):          # only accept if still watertight
            m = c
    except Exception:
        pass
    m.merge_vertices()
    try:
        m.fix_normals()
    except Exception:
        pass
    return m


def _cluster(vals, tol):
    s = sorted(vals); g = [[s[0]]]
    for v in s[1:]:
        (g[-1].append(v) if v - g[-1][-1] <= tol else g.append([v]))
    return [float(np.mean(c)) for c in g]


def rectify_ring(coords, thr=0.025):
    """Square off 45-degree chamfers into 90-degree L corners WITHOUT snapping
    coordinates (snapping would collapse the ~11cm-thin walls). Each clearly
    diagonal edge is replaced by two axis-aligned edges via an inserted corner."""
    pts = [(float(x), float(y)) for x, y in coords]
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    out = []
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        out.append((x0, y0))
        if abs(x1 - x0) > thr and abs(y1 - y0) > thr:         # a real chamfer -> square L
            out.append((x1, y0) if abs(x1 - x0) >= abs(y1 - y0) else (x0, y1))
    clean = []
    for p in out:
        if not clean or abs(clean[-1][0] - p[0]) > 1e-6 or abs(clean[-1][1] - p[1]) > 1e-6:
            clean.append(p)
    return clean


def main(las, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = reconstruct(las)
    ws = wall_footprint(R)
    xmin, ymax = R["xmin"], R["ymax"]
    H, W = R["H"], R["W"]
    x, y, z = R["x"], R["y"], R["z"]; zf = R["z_floor"]; zc = R["z_ceiling"]
    storey = zc - zf

    # ---- recover SINGLE-SIDED walls the free-space carve misses: a wall has
    # FULL-HEIGHT material (points near floor AND near ceiling at the same x,y);
    # furniture does not. Add those columns to the wall mask. ----
    def raster(mask):
        o = np.zeros((H, W), np.uint8)
        cc = np.clip(((x[mask] - xmin) / CELL).astype(int), 0, W - 1)
        rr = np.clip(((ymax - y[mask]) / CELL).astype(int), 0, H - 1)
        o[rr, cc] = 1
        return o
    lo = raster((z > zf + 0.25) & (z < zf + 0.75))
    hi = raster((z > zc - 0.75) & (z < zc - 0.25))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.10 / CELL) | 1,) * 2)
    fh = cv2.morphologyEx((cv2.dilate(lo, dk) & cv2.dilate(hi, dk)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # ---- drop COMPACT full-height BLOBS = bathroom/kitchen FIXTURES (WC, sink,
    # pipes, tall cabinet, shower) that are also floor-to-ceiling. Real walls are
    # thin (~WALL_T); fixtures are fat (>~30cm). Erosion deletes thin walls but
    # keeps fat blobs -> subtract those so only wall-like single-sided lines
    # survive, and the bathroom stops filling with clutter. ----
    fat = cv2.erode(fh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.28 / CELL) | 1,) * 2))
    fat = cv2.dilate(fat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.36 / CELL) | 1,) * 2))
    fh = ((fh > 0) & (fat == 0)).astype(np.uint8)
    # ---- recover TOP-ONLY structure (arch heads / lintels / columns that exist
    # near the ceiling but NOT at chest height) the mid-footprint misses. Take
    # near-ceiling material that ATTACHES to a known wall line (within 12cm) so a
    # top column enters the mask, while room-centre ceiling/soffit blobs (far from
    # any wall) are excluded and don't get invented as walls. ----
    top = raster((z > zc - 0.35) & (z < zc - 0.05))
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    wall_halo = cv2.dilate(((ws > 0) | (fh > 0)).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.12 / CELL) | 1,) * 2))
    top_attached = (top & wall_halo).astype(np.uint8)
    # ---- strip OPEN DOOR LEAVES: the scan caught some doors open, so a door leaf
    # (a thin ~0.8m panel that stops at ~2.0m, NOT the 2.52m ceiling) would stick
    # to the wall as a spur near an entrance. A real wall is FULL-HEIGHT; a leaf
    # has chest-height material but nothing near the ceiling. Detect thin door-
    # sized half-height panels and remove them from the mask. ----
    midb = raster((z > zf + 0.6) & (z < zf + 1.5))
    ceilb = cv2.dilate(raster((z > zc - 0.6) & (z < zc - 0.1)),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.15 / CELL) | 1,) * 2))
    halfp = cv2.morphologyEx((midb & (ceilb == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    leaves = np.zeros_like(halfp); llbl, ln = _ndi.label(halfp, structure=np.ones((3, 3)))
    nleaf = 0
    for k in range(1, ln + 1):
        cy, cx = np.where(llbl == k)
        if cx.size < 3:
            continue
        du = (np.ptp(cx) + 1) * CELL; dv = (np.ptp(cy) + 1) * CELL
        length = max(du, dv); thick = min(du, dv)
        if 0.5 <= length <= 1.3 and thick <= 0.22 and length / max(thick, CELL) >= 2.2:
            leaves |= (llbl == k).astype(np.uint8); nleaf += 1
    if nleaf:
        log(f"removed {nleaf} open-door-leaf panel(s) near entrances")
    wallmask = (((ws > 0) | (fh > 0) | (top_attached > 0)) & (leaves == 0)).astype(np.uint8)
    # keep the wall network + long thin pieces; drop compact furniture blobs
    lbl, n = _ndi.label(wallmask, structure=np.ones((3, 3)))
    if n:
        szs = _ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        big = 1 + int(np.argmax(szs)); keep = {big}
        for k, s in enumerate(szs, 1):
            if k == big or s < (0.3 / (CELL * CELL)):
                continue
            ys, xs = np.where(lbl == k)
            if max(np.ptp(xs), np.ptp(ys)) * CELL >= 0.8:
                keep.add(k)
        wallmask = np.isin(lbl, list(keep)).astype(np.uint8)
    wallmask = _ndi.binary_fill_holes(wallmask).astype(np.uint8)

    # 1. skeleton (centerline network) of the FULL wall mask
    skel = skeletonize(wallmask > 0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "skeleton.png"), skel * 255)

    # 2. vectorise the skeleton into straight, snapped wall SEGMENTS (clean
    #    axis-aligned centrelines -- no wobble, no whisker spurs).
    from shapely.geometry import Polygon as _Poly
    from shapely.geometry import Polygon as _Poly2
    from scripts.experiments.hough_vectorize import snap_and_merge
    ppm = 1.0 / CELL
    t = max(3, int(round(WALL_T / CELL)) | 1)
    lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, threshold=20,
                            minLineLength=int(0.4 * ppm), maxLineGap=int(0.3 * ppm))
    def merge_parallel(sgs, gap_px, ov_px):
        """Merge near-parallel, overlapping centrelines (the two FACES of one wall
        that skeletonised into two lines) into a single averaged wall, so a shared
        wall isn't emitted as two separate slabs."""
        sgs = [list(s) for s in sgs]
        changed = True
        while changed:
            changed = False
            for i in range(len(sgs)):
                for j in range(i + 1, len(sgs)):
                    a0, a1, c = sgs[i]; b0, b1, cj = sgs[j]
                    if abs(c - cj) <= gap_px and (min(a1, b1) - max(a0, b0)) >= ov_px:
                        w1 = a1 - a0; w2 = b1 - b0
                        sgs[i] = [min(a0, b0), max(a1, b1), (c * w1 + cj * w2) / (w1 + w2)]
                        sgs.pop(j); changed = True; break
                if changed:
                    break
        return sgs

    segs = []
    if lines is not None:
        hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
        hs = merge_parallel(hs, int(0.28 / CELL), int(0.5 * ppm))     # collapse doubled walls
        vs = merge_parallel(vs, int(0.28 / CELL), int(0.5 * ppm))
        segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL]))
                for a0, a1, yr in hs] + \
               [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL]))
                for a0, a1, xc in vs]

    # ---- measure each wall's REAL thickness + centre from the two face-planes in
    # the cloud (the scanner saw both faces, so the perp histogram has two peaks =
    # the faces; separation = thickness, midpoint = true centre). ----
    xy = np.column_stack([x, y])
    seg_geom_cache = {}

    def seg_geom(p0, p1):
        key = (round(float(p0[0]), 3), round(float(p0[1]), 3), round(float(p1[0]), 3), round(float(p1[1]), 3))
        if key in seg_geom_cache:
            return seg_geom_cache[key]
        d = p1 - p0; L = float(np.linalg.norm(d))
        res = (WALL_T, 0.0)
        if L >= 0.30:
            d = d / L; nn = np.array([-d[1], d[0]])
            rel = xy - p0; u = rel @ d; perp = rel @ nn
            near = (np.abs(perp) <= 0.28) & (u >= 0) & (u <= L)
            if near.sum() >= 200:
                pp = perp[near]
                h, edges = np.histogram(pp, bins=np.arange(-0.30, 0.301, 0.01))
                c = (edges[:-1] + edges[1:]) / 2
                neg = c < -0.005; pos = c > 0.005
                if h[neg].sum() >= 15 and h[pos].sum() >= 15:
                    fn = c[neg][int(np.argmax(h[neg]))]; ff = c[pos][int(np.argmax(h[pos]))]
                    th = ff - fn; off = (ff + fn) / 2
                    if 0.08 <= th <= 0.30:
                        res = (float(np.clip(th, 0.09, 0.28)), float(np.clip(off, -0.10, 0.10)))
        seg_geom_cache[key] = res
        return res

    # ---- REGULARISE: fold each wall's measured centre offset into its centreline,
    # then snap near-collinear / parallel walls onto a SHARED gridline so faces are
    # exactly flat & collinear (removes junction mis-alignment). Snap tolerance is
    # well below the smallest real wall gap, so distinct walls stay separate. ----
    def _regularise(segs_in):
        Hs, Vs = [], []
        for p0, p1 in segs_in:
            d = p1 - p0; L = float(np.linalg.norm(d))
            if L < 0.30:
                continue
            _, off = seg_geom(p0, p1)
            ud = d / L; nn = np.array([-ud[1], ud[0]])
            q0 = p0 + nn * off; q1 = p1 + nn * off          # shift onto measured centre
            (Hs if abs(d[0]) >= abs(d[1]) else Vs).append((q0, q1))
        hys = _cluster([(q0[1] + q1[1]) / 2 for q0, q1 in Hs], 0.08) if Hs else []
        vxs = _cluster([(q0[0] + q1[0]) / 2 for q0, q1 in Vs], 0.08) if Vs else []
        out = []
        for q0, q1 in Hs:
            sy = min(hys, key=lambda v: abs(v - (q0[1] + q1[1]) / 2))
            out.append((np.array([q0[0], sy]), np.array([q1[0], sy])))
        for q0, q1 in Vs:
            sx = min(vxs, key=lambda v: abs(v - (q0[0] + q1[0]) / 2))
            out.append((np.array([sx, q0[1]]), np.array([sx, q1[1]])))
        return out

    segs = _regularise(segs)
    seg_geom_cache.clear()                                   # segs moved -> re-measure on new coords

    # WALK-PATH doors: reconstruct the operator trajectory from gps_time and find
    # where it CROSSES a wall centreline -- the person physically walked through
    # there, so it is an entrance DOOR (works even for a shut door leaf the (u,z)
    # empty test misses, and needs no colour/RF-DETR).
    traj_hits = {}
    traj = np.zeros((0, 2))
    try:
        import laspy
        _las = laspy.read(las)
        if "gps_time" in _las.point_format.dimension_names:
            from scripts.recon.trajectory import approx_trajectory
            _xyz = np.column_stack([np.asarray(_las.x), np.asarray(_las.y), np.asarray(_las.z)])
            traj = np.asarray(approx_trajectory(np.asarray(_las.gps_time), _xyz, dt_s=0.25))[:, :2]
            for si, (p0, p1) in enumerate(segs):
                d = p1 - p0; Ls = float(np.linalg.norm(d))
                if Ls == 0:
                    continue
                hits = []
                for a, b in zip(traj[:-1], traj[1:]):
                    e = b - a; den = d[0] * e[1] - d[1] * e[0]
                    if abs(den) < 1e-12:
                        continue
                    r = a - p0
                    tt = (r[0] * e[1] - r[1] * e[0]) / den
                    ss = (r[0] * d[1] - r[1] * d[0]) / den
                    if 0.0 <= ss <= 1.0 and 0.15 <= tt * Ls <= Ls - 0.15:
                        hits.append(tt * Ls)
                if hits:
                    traj_hits[si] = sorted(hits)
            log(f"walk-path: {len(traj)} path vertices, crossings on {len(traj_hits)} walls")
    except Exception as _e:
        log(f"walk-path unavailable: {_e}")

    # 3. build CLEAN INDIVIDUAL WALLS: each straight segment -> one flat
    #    rectangular slab of uniform thickness, extended half a thickness into
    #    each junction so neighbours fuse. This removes the skeleton wobble and
    #    whisker spurs of the raw footprint.
    positives = []          # wall slabs (unioned, THEN cut)
    negatives = []          # opening + recess cutters (subtracted in one batch)
    beams = []              # header/lintel beams -- unioned AFTER cutting
    seg_mask = np.zeros((H, W), np.uint8)
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.30:
            continue
        th, off = seg_geom(p0, p1)
        ud = dd / L; nn = np.array([-ud[1], ud[0]])
        box = trimesh.creation.box(extents=(L + 3 * max(th, WALL_T), th, storey))   # bury ends in junctions
        box.apply_transform(trimesh.transformations.rotation_matrix(float(np.arctan2(dd[1], dd[0])), [0, 0, 1]))
        mid = (p0 + p1) / 2.0 + nn * off                                            # shift to the true centre
        box.apply_translation((mid[0], mid[1], zf + storey / 2.0))
        positives.append(box)
        tk = max(3, int(round(th / CELL)) | 1)
        a = (int((p0[0] - xmin) / CELL), int((ymax - p0[1]) / CELL))
        b = (int((p1[0] - xmin) / CELL), int((ymax - p1[1]) / CELL))
        cv2.line(seg_mask, a, b, 1, thickness=tk)

    # 4. fallback: sizeable footprint NOT covered by any segment = a real jog/pier
    #    Hough missed. Extrude those cleanly (small whisker spurs stay dropped).
    covered = cv2.dilate(seg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))
    residual = cv2.morphologyEx(((wallmask > 0) & (covered == 0)).astype(np.uint8),
                                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rlbl, rn = _ndi.label(residual, structure=np.ones((3, 3)))
    for k in range(1, rn + 1):
        comp = (rlbl == k).astype(np.uint8)
        if comp.sum() * CELL * CELL < 0.10:          # drop small spurs / whiskers
            continue
        comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))
        for pg in footprint_polygons(comp, xmin, ymax):
            ext = rectify_ring(pg.exterior.coords)
            if len(ext) < 4:
                continue
            try:
                g = _Poly(ext)
                g = g if g.is_valid else g.buffer(0)
                if g.area < 0.05:
                    continue
                pr = trimesh.creation.extrude_polygon(g, height=storey)
                pr.apply_translation((0, 0, zf))
                positives.append(pr)
            except Exception:
                pass

    plan = np.full((H, W, 3), 255, np.uint8)
    plan[covered > 0] = (40, 40, 40)
    cv2.imwrite(str(out_dir / "skeleton_walls_plan.png"), plan)

    wall_solid = _union(positives)          # provisional base for the cutter geometry math
    # ---- cut door/passage openings as their real (u,z) SILHOUETTE, so ARCHED
    # heads (material spanning the opening only near the top) are kept. ----
    xy = np.column_stack([x, y]); ncut = 0
    openings = []           # (center, dd, nn, width, z0, z1) of each cut opening
    UR = 0.02; z0w = zf - 0.05; z1w = zc + 0.05
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.20) & (u >= 0) & (u <= L)
        if near.sum() < 140:
            continue
        uu, zz = u[near], z[near]
        nu2 = max(int(L / UR) + 1, 4); nz2 = max(int((z1w - z0w) / UR) + 1, 4)
        mat = np.zeros((nz2, nu2), np.uint8)
        mat[np.clip(((zz - z0w) / UR).astype(int), 0, nz2 - 1),
            np.clip((uu / UR).astype(int), 0, nu2 - 1)] = 1
        mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        core = np.zeros_like(mat)
        core[int((zf + 0.08 - z0w) / UR):int((zc - 0.03 - z0w) / UR), :] = 1
        empty = cv2.morphologyEx(((core > 0) & (mat == 0)).astype(np.uint8),
                                 cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ncE, lblE, st, _ = cv2.connectedComponentsWithStats(empty, 8)
        cbot = max(int((zf + 0.08 - z0w) / UR), 0); ctop = int((zc - 0.03 - z0w) / UR)
        for i in range(1, ncE):
            bx, by, bw, bh, _ = st[i]
            wm = bw * UR; hm = bh * UR
            if not (0.4 <= wm <= 3.5) or hm < 0.5:
                continue
            # a REAL opening (door/window/balcony) is walled on >=2 sides -- material
            # below (sill), above (header) or on the flanks. This accepts windows &
            # corner/balcony openings the old "must be flanked within this segment"
            # test wrongly rejected, while dropping whole-empty (poorly scanned) walls.
            below = float(mat[cbot:by, bx:bx + bw].mean()) if by > cbot else 0.0
            above = float(mat[by + bh:ctop, bx:bx + bw].mean()) if by + bh < ctop else 0.0
            left = float(mat[cbot:ctop, max(0, bx - 3):bx].mean()) if bx > 0 else 0.0
            right = float(mat[cbot:ctop, bx + bw:min(nu2, bx + bw + 3)].mean()) if bx + bw < nu2 else 0.0
            sides = int(below >= 0.30) + int(above >= 0.20) + int(left >= 0.40) + int(right >= 0.40)
            if sides < 2:
                continue
            cs, _ = cv2.findContours((lblE == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.02 / UR, True)
            if len(cnt) < 3:
                continue
            poly = _Poly2([(c * UR, z0w + r * UR) for c, r in cnt[:, 0, :]])
            if not poly.is_valid or poly.area < 0.2:
                continue
            try:
                cutter = trimesh.creation.extrude_polygon(poly, height=WALL_T * 4)
                cutter.apply_transform(np.array(
                    [[dd[0], 0, nn[0], p0[0] - nn[0] * WALL_T * 2],
                     [dd[1], 0, nn[1], p0[1] - nn[1] * WALL_T * 2],
                     [0, 1, 0, 0], [0, 0, 0, 1]], float))
                negatives.append(cutter); ncut += 1
                u0o = bx * UR; u1o = (bx + bw) * UR
                openings.append(dict(center=p0 + dd * (u0o + u1o) / 2, dd=dd.copy(), nn=nn.copy(),
                                     width=u1o - u0o, z0=z0w + by * UR, z1=z0w + (by + bh) * UR))
            except Exception:
                pass
    log(f"cut {ncut} openings (arched heads kept)")

    # ---- DOORS = the narrow (2-3.5 ft) NECK where two rooms connect, i.e. the gap
    # you physically walk through. The watershed room map is well-segmented, so a
    # doorway is exactly where two room regions touch through a short neck (a full
    # wall keeps them apart; only a door-width gap lets them touch). Far more
    # accurate than the noisy gps-centroid path. ----
    DOOR_W = 0.9; DOOR_H = zf + 2.05
    _mk = R["mk"]; _labs = list(R["room_labels"])

    def _snap_cut_door(cx, cy, want_w):
        best = None
        q = np.array([cx, cy])
        for p0, p1 in segs:
            d = p1 - p0; L = float(np.linalg.norm(d))
            if L < 0.6:
                continue
            dv = d / L; uu = float((q - p0) @ dv); pp = abs(float((q - p0) @ np.array([-dv[1], dv[0]])))
            if 0.05 <= uu <= L - 0.05 and (best is None or pp < best[0]):
                best = (pp, uu, p0, dv, np.array([-dv[1], dv[0]]), L)
        if best is None or best[0] > 0.6:               # doorway not near any wall line
            return False
        pp, uc, p0, dv, nv, L = best
        center = p0 + dv * uc
        if any(float(np.linalg.norm(o["center"] - center)) < 0.5 for o in openings):
            return False                                # already have an opening here
        dw = float(np.clip(want_w, 0.6, 1.1))
        u0d = max(uc - dw / 2, 0.05); u1d = min(uc + dw / 2, L - 0.05)
        if u1d - u0d < 0.4:
            return False
        poly = _Poly2([(u0d, zf + 0.02), (u1d, zf + 0.02), (u1d, DOOR_H), (u0d, DOOR_H)])
        try:
            cutter = trimesh.creation.extrude_polygon(poly, height=WALL_T * 4)
            cutter.apply_transform(np.array(
                [[dv[0], 0, nv[0], p0[0] - nv[0] * WALL_T * 2],
                 [dv[1], 0, nv[1], p0[1] - nv[1] * WALL_T * 2],
                 [0, 1, 0, 0], [0, 0, 0, 1]], float))
            negatives.append(cutter)
            openings.append(dict(center=center, dd=dv.copy(), nn=nv.copy(),
                                 width=u1d - u0d, z0=zf + 0.02, z1=DOOR_H, walked=True))
            return True
        except Exception:
            return False

    # A DOORWAY is a 2-4ft GAP between two COLINEAR wall segments (the two jambs).
    # Hough only bridges <=0.3m, so a wall with a door becomes two collinear
    # segments with the doorway between their ends. Interior on both perpendicular
    # sides = an interior door. (Room-adjacency hallucinated because the watershed
    # is over-segmented; segment-gap keys off the walls themselves.)
    _inside = ndimage.binary_fill_holes((R["free"] > 0) | (R["occ"] > 0))

    def _interior(pt):
        c = int((pt[0] - xmin) / CELL); r = int((ymax - pt[1]) / CELL)
        return 0 <= r < H and 0 <= c < W and _inside[r, c]

    _Sg = []
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.35:
            continue
        if abs(dd[0]) >= abs(dd[1]):
            _Sg.append((True, (p0[1] + p1[1]) / 2, min(p0[0], p1[0]), max(p0[0], p1[0])))
        else:
            _Sg.append((False, (p0[0] + p1[0]) / 2, min(p0[1], p1[1]), max(p0[1], p1[1])))
    ndoor_gap = 0; _placed = []
    for i in range(len(_Sg)):
        for j in range(i + 1, len(_Sg)):
            hi, ci, loi, hoi = _Sg[i]; hj, cj, loj, hoj = _Sg[j]
            if hi != hj or abs(ci - cj) > 0.22:
                continue
            if hoi <= loj:
                gap = loj - hoi; gc = (hoi + loj) / 2
            elif hoj <= loi:
                gap = loi - hoj; gc = (hoj + loi) / 2
            else:
                continue
            if not (0.5 < gap < 1.2):
                continue
            cl = (ci + cj) / 2
            cen = np.array([gc, cl]) if hi else np.array([cl, gc])
            dv = np.array([1.0, 0.0]) if hi else np.array([0.0, 1.0])
            nv = np.array([0.0, 1.0]) if hi else np.array([1.0, 0.0])
            if not (_interior(cen + nv * 0.5) and _interior(cen - nv * 0.5)):
                continue                                 # one side outside = entrance/balcony (handled elsewhere)
            if any(float(np.linalg.norm(cen - pc)) < 0.4 for pc in _placed):
                continue
            if any(float(np.linalg.norm(cen - o["center"])) < 0.5 for o in openings):
                continue                                 # already have a geometric opening here
            _placed.append(cen)
            dw = float(np.clip(gap, 0.6, 1.1))
            poly = _Poly2([(-dw / 2, zf + 0.02), (dw / 2, zf + 0.02), (dw / 2, DOOR_H), (-dw / 2, DOOR_H)])
            try:
                cutter = trimesh.creation.extrude_polygon(poly, height=WALL_T * 4)
                cutter.apply_transform(np.array(
                    [[dv[0], 0, nv[0], cen[0] - nv[0] * WALL_T * 2],
                     [dv[1], 0, nv[1], cen[1] - nv[1] * WALL_T * 2],
                     [0, 1, 0, 0], [0, 0, 0, 1]], float))
                negatives.append(cutter)
                openings.append(dict(center=cen.copy(), dd=dv.copy(), nn=nv.copy(),
                                     width=dw, z0=zf + 0.02, z1=DOOR_H, walked=True))
                ndoor_gap += 1
            except Exception:
                pass
    log(f"colinear-gap doors: {ndoor_gap}")

    # ---- OFFSET-PLANE reveals / shadow-gaps / soffit faces (Ikehata-style):
    # a wall side is NOT one flat face -- it is a PROUD face plus recessed
    # plane(s). Per side build a fine (u,z) depth map of the NEAREST material,
    # take the proud face as the reference, and for every coherent recessed
    # region cut the wall back to its TRUE measured depth (uncapped, up to the
    # wall thickness) over its true (u,z) extent. This keeps a ~5cm step over the
    # top 40% of a wall, a recessed door head+jamb frame, or a shadow-gap band --
    # each preserved at real depth and real height, not as a token sliver. ----
    UB = 0.04; ZB = 0.05                       # 4cm along-wall x 5cm height cells
    MIN_DEPTH = 0.030                          # MAJOR grooves only: ignore < 3cm (minor reveals/noise)
    MAX_DEPTH = WALL_T * 0.42                  # cap so a two-sided groove can't sever the wall
    MIN_AREA = 0.15                            # >= 1500 cm2 -- only prominent recesses
    MIN_EXTENT = 0.30                          # at least 30cm in one direction
    nrev = 0
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        seg_th, seg_off = seg_geom(p0, p1)         # this wall's measured thickness + centre
        max_depth = seg_th * 0.42                  # cap relative to THIS wall's thickness
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.22) & (u >= 0) & (u <= L)
        if near.sum() < 200:
            continue
        uu, zz, pp = u[near], z[near], perp[near]
        Rz = trimesh.transformations.rotation_matrix(np.arctan2(dd[1], dd[0]), [0, 0, 1])
        for side in (+1.0, -1.0):
            sel = (pp * side) > 0.004
            if sel.sum() < 200:
                continue
            sp = pp[sel] * side; sz = zz[sel]; su = uu[sel]
            zb = np.arange(zf + 0.10, zc - 0.02, ZB); nz = len(zb)
            nu = max(int(L / UB), 2)
            ui = np.clip((su / UB).astype(int), 0, nu - 1)
            zi = np.clip(((sz - (zf + 0.10)) / ZB).astype(int), 0, nz - 1)
            depth = np.full((nz, nu), np.nan)   # nearest-face setback per cell
            for a in range(nu):
                ma = ui == a
                if not ma.any():
                    continue
                spa, zia = sp[ma], zi[ma]
                for b in np.unique(zia):
                    cm = zia == b
                    if cm.sum() >= 2:
                        depth[b, a] = np.percentile(spa[cm], 15)
            # fill SHORT along-wall gaps (sparse columns) in each height row by
            # interpolation, so a continuous recess isn't broken into pieces; leave
            # WIDE gaps (real openings) as NaN.
            MAXGAP = max(int(0.30 / UB), 2)
            for b in range(nz):
                row = depth[b]; vrow = ~np.isnan(row)
                if vrow.sum() < 2:
                    continue
                idx = np.where(vrow)[0]
                filled = np.interp(np.arange(nu), idx, row[idx])
                for g0, g1 in zip(idx[:-1], idx[1:]):
                    if 1 <= (g1 - g0 - 1) and (g1 - g0) <= MAXGAP:
                        depth[b, g0 + 1:g1] = filled[g0 + 1:g1]
            # fill SHORT VERTICAL gaps (sparse height rows) per column, so a groove's
            # true TOP->BOTTOM (or mid->bottom) extent is captured, not just the dense
            # middle band; leave tall gaps alone.
            MAXGZ = max(int(0.35 / ZB), 2)
            for a in range(nu):
                col = depth[:, a]; vcol = ~np.isnan(col)
                if vcol.sum() < 2:
                    continue
                idz = np.where(vcol)[0]
                fz = np.interp(np.arange(nz), idz, col[idz])
                for g0, g1 in zip(idz[:-1], idz[1:]):
                    if 1 <= (g1 - g0 - 1) and (g1 - g0) <= MAXGZ:
                        depth[g0 + 1:g1, a] = fz[g0 + 1:g1]
            valid = ~np.isnan(depth)
            if valid.sum() < 20:
                continue
            face = float(np.nanpercentile(depth, 10))         # proud reference plane
            noise = 1.4826 * float(np.nanmedian(np.abs(depth[valid] - np.nanmedian(depth[valid])))) + 1e-6
            thr = max(MIN_DEPTH, 3.0 * noise)
            recess = np.where(valid, depth - face, 0.0)
            groove = (recess >= thr) & valid
            # bridge intermittent detection ALONG the wall so the groove is captured
            # continuously across, not in broken chunks (u = axis 1).
            groove = ndimage.binary_closing(groove, structure=np.ones((1, 7)))   # bridge along u
            groove = ndimage.binary_closing(groove, structure=np.ones((7, 1)))   # bridge along z
            groove = ndimage.binary_closing(groove, structure=np.ones((3, 3)))
            zf_band = zf + 0.10
            lbl2, nc2 = ndimage.label(groove, structure=np.ones((3, 3)))
            for cc in range(1, nc2 + 1):
                mask = (lbl2 == cc)
                ys, xs = np.where(mask)
                area = ys.size * (UB * ZB)
                if area < MIN_AREA:                           # major grooves only
                    continue
                du = (np.ptp(xs) + 1) * UB; dz = (np.ptp(ys) + 1) * ZB
                if max(du, dz) < MIN_EXTENT:
                    continue
                # gate on depth COHERENCE, not bounding-box fill -- a recessed
                # head+jamb casing is a FRAME/L (low fill) but has uniform depth;
                # scattered noise has high depth variation. Keep coherent, drop noise.
                rv = recess[ys, xs]
                bb = (np.ptp(ys) + 1) * (np.ptp(xs) + 1)
                fill = ys.size / max(bb, 1)
                cvv = float(np.std(rv) / (np.mean(rv) + 1e-6))
                if fill < 0.4 and cvv > 0.55:
                    continue
                depth_cut = float(np.clip(np.median(rv), MIN_DEPTH, max_depth))
                box_th = depth_cut + 0.03
                # PRECISE vertical extent: take the groove's exact top/bottom from the
                # actual recessed POINTS in its u-band (not the 5cm cell grid), so the
                # cut stops exactly where the recess ends -- and snap to ceiling/floor
                # when it truly reaches them. Works for any span (top->bottom, mid->top,
                # mid->bottom).
                u0r = xs.min() * UB; u1r = (xs.max() + 1) * UB
                inb = (su >= u0r) & (su <= u1r) & (sp >= face + thr)
                rmin = float(ys.min()); rmax = float(ys.max()) + 1.0
                if int(inb.sum()) >= 8:
                    z0p = float(np.percentile(sz[inb], 2)); z1p = float(np.percentile(sz[inb], 98))
                else:
                    z0p = zf_band + rmin * ZB; z1p = zf_band + rmax * ZB
                if z1p >= zc - 0.12:
                    z1p = zc
                if z0p <= zf + 0.16:
                    z0p = zf + 0.02
                zden = max(rmax - rmin, 1.0)

                def _zmap(r, _r0=rmin, _z0=z0p, _z1=z1p, _d=zden):
                    return _z0 + (r - _r0) / _d * (_z1 - _z0)
                # cut the region's ACTUAL shape (contours in (u,z)) extruded along
                # the wall normal -- so a frame recesses its casing, not the opening.
                m8 = mask.astype(np.uint8)
                cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # local->world: x=u -> dd, y=height(z) -> world Z, z=depth -> nn*side
                M = np.array([
                    [dd[0], 0, nn[0] * side, p0[0] + nn[0] * (seg_off + side * (seg_th / 2 - depth_cut))],
                    [dd[1], 0, nn[1] * side, p0[1] + nn[1] * (seg_off + side * (seg_th / 2 - depth_cut))],
                    [0, 1, 0, 0], [0, 0, 0, 1]], float)
                for cnt in cnts:
                    ap = cv2.approxPolyDP(cnt, 0.6, True)[:, 0, :]
                    if len(ap) < 3:
                        continue
                    poly = _Poly2([(float(c) * UB, _zmap(float(r))) for c, r in ap])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty or poly.area < 0.02:
                        continue
                    try:
                        ch = trimesh.creation.extrude_polygon(poly, height=box_th)
                        ch.apply_transform(M)
                        negatives.append(ch); nrev += 1
                    except Exception:
                        pass
    log(f"cut {nrev} offset-plane reveals / soffit faces (true depth, height-mapped)")

    # ---- HEADERS / LINTELS (the "arch" over an opening): material that exists
    # only near the CEILING and BRIDGES a gap in the mid-height footprint (spans
    # the top of a doorway/passage where there is no wall below). The extruded
    # footprint can't create these -- there is no wall there to carry them. Detect
    # the bridging top-band spans and rebuild each as a beam from its measured
    # soffit up to the ceiling, so every opening gets its head/lintel. ----
    top = raster((z > zc - 0.6) & (z < zc - 0.08))
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    halo = cv2.dilate(ws, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.14 / CELL) | 1,) * 2))
    off = (top & (halo == 0)).astype(np.uint8)
    lblH, nH = _ndi.label(off, structure=np.ones((3, 3)))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.20 / CELL) | 1,) * 2)
    tkern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(WALL_T / CELL)) | 1,) * 2)
    nhead = 0
    for k in range(1, nH + 1):
        comp = (lblH == k).astype(np.uint8)
        ys, xs = np.where(comp)
        if xs.size < 4:
            continue
        du = (np.ptp(xs) + 1) * CELL; dv = (np.ptp(ys) + 1) * CELL
        if max(du, dv) < 0.5 or min(du, dv) > 1.2:            # must be a long thin span
            continue
        touch = cv2.dilate(comp, grow) & ws
        _, tn = _ndi.label(touch, structure=np.ones((3, 3)))
        if tn < 2:                                            # must bridge >=2 wall parts
            continue
        # soffit height = bottom of the lintel over this span
        cxs = xmin + xs * CELL; cys = ymax - ys * CELL
        lo = []
        for cx, cy in zip(cxs[::7], cys[::7]):
            col = (np.abs(x - cx) < 0.06) & (np.abs(y - cy) < 0.06) & (z > zf + 1.2)
            if col.sum() >= 5:
                lo.append(np.percentile(z[col], 5))
        soffit = float(np.clip(np.median(lo) if lo else zc - 0.4, zf + 0.3, zc - 0.12))
        # ---- build a CLEAN rectangular lintel: fit the span's principal axis,
        # give it uniform wall thickness, and EXTEND both ends into the adjoining
        # walls (by ~1.5x thickness) so the union fuses the beam to them -- a
        # defined, wall-connected member instead of a ragged floating blob. ----
        P = np.column_stack([xmin + xs * CELL, ymax - ys * CELL]).astype(float)
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        dvec = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)      # long axis (opening direction)
        nvec = np.array([-dvec[1], dvec[0]])
        proj = (P - c) @ dvec
        half_len = (proj.max() - proj.min()) / 2 + WALL_T * 1.5   # bury ends in the walls
        half_w = WALL_T / 2
        corners = [c + a * half_len * dvec + b * half_w * nvec
                   for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        try:
            g = _Poly(corners)
            if g.area < 0.03:
                raise ValueError
            pr = trimesh.creation.extrude_polygon(g, height=zc - soffit)
            pr.apply_translation((0, 0, soffit))
            beams.append(pr); nhead += 1
        except Exception:
            pass
    log(f"built {nhead} header/lintel beams (arches over openings)")

    # ---- DOOR / WINDOW / BALCONY-DOOR elements: classify each detected opening
    # by geometry (sill height, size, interior vs exterior) and model an element
    # in it -- a door leaf, a glass pane, or a full-height glazed balcony panel.
    # LiDAR can't tell a fitted door from an empty passage, so this is geometric
    # best-effort; a wide interior opening stays an open cased passage. ----
    inside_bldg = ndimage.binary_fill_holes((R["free"] > 0) | (R["occ"] > 0))

    def _outside(cw, off):
        px = int((cw[0] + off[0] - xmin) / CELL); py = int((ymax - (cw[1] + off[1])) / CELL)
        return not (0 <= py < H and 0 <= px < W and inside_bldg[py, px])

    GLASS = [150, 205, 230, 235]; LEAF = [150, 95, 55, 255]; FRAME = [236, 236, 240, 255]

    def _place(o, lw, th, hh, u_off, z_center):
        """A box in the opening frame: lw along the wall, th along the normal,
        hh vertical; shifted u_off along the wall and centred at height z_center."""
        b = trimesh.creation.box(extents=(max(lw, 1e-3), th, max(hh, 1e-3)))
        b.apply_transform(trimesh.transformations.rotation_matrix(float(np.arctan2(o["dd"][1], o["dd"][0])), [0, 0, 1]))
        off = o["center"] + o["dd"] * u_off
        b.apply_translation((off[0], off[1], z_center))
        return b

    def _framed(o, w, h, z0, z1, kind):
        """Build a real element in the cutout: a frame (jambs+head+sill) with an
        inset leaf (door) or glass pane + mullions (window/balcony)."""
        parts = []
        fw = 0.06; fd = 0.09; zmid = (z0 + z1) / 2.0
        parts.append((_place(o, fw, fd, h, -(w / 2 - fw / 2), zmid), FRAME))     # left jamb
        parts.append((_place(o, fw, fd, h, +(w / 2 - fw / 2), zmid), FRAME))     # right jamb
        parts.append((_place(o, w, fd, fw, 0, z1 - fw / 2), FRAME))              # head
        parts.append((_place(o, w, fd, fw, 0, z0 + fw / 2), FRAME))              # sill / threshold
        pw = w - 2 * fw; ph = h - 2 * fw
        if kind == "door":
            parts.append((_place(o, pw, 0.05, ph, 0, zmid), LEAF))               # leaf
            parts.append((_place(o, 0.05, 0.06, 0.05, pw / 2 - 0.12, zmid), FRAME))  # handle
        else:                                                                    # window / balcony
            parts.append((_place(o, pw, 0.02, ph, 0, zmid), GLASS))              # glass pane
            parts.append((_place(o, 0.04, fd * 0.8, ph, 0, zmid), FRAME))        # vertical mullion
            parts.append((_place(o, pw, fd * 0.8, 0.04, 0, zmid), FRAME))        # horizontal mullion
        return parts

    def _arch_above(o, w, z_head):
        """Recessed HEAD / transom 'arch' above a ~7ft door: a shallow recess
        spanning the door width from the head up toward the ceiling, cut into both
        faces of the wall -- reads as a cased-opening head above the door."""
        z_top = min(z_head + 0.45, zc - 0.02)
        if z_top - z_head < 0.15:
            return 0
        dc = 0.045; bt = dc + 0.03; aw = w * 0.9; n = 0
        for side in (1.0, -1.0):
            poly = _Poly2([(-aw / 2, z_head), (aw / 2, z_head), (aw / 2, z_top), (-aw / 2, z_top)])
            M = np.array([[o["dd"][0], 0, o["nn"][0] * side, o["center"][0] + o["nn"][0] * side * (WALL_T / 2 - dc)],
                          [o["dd"][1], 0, o["nn"][1] * side, o["center"][1] + o["nn"][1] * side * (WALL_T / 2 - dc)],
                          [0, 1, 0, 0], [0, 0, 0, 1]], float)
            try:
                ch = trimesh.creation.extrude_polygon(poly, height=bt); ch.apply_transform(M)
                negatives.append(ch); n += 1
            except Exception:
                pass
        return n

    elements = []          # (mesh, rgba)
    ndoor = nwin = nbal = narch = 0
    for o in openings:
        w = o["width"]; z0 = o["z0"]; z1 = min(o["z1"], zc); h = z1 - z0
        sill = z0 - zf
        ext = _outside(o["center"], o["nn"] * 0.7) or _outside(o["center"], -o["nn"] * 0.7)
        if h < 0.4 or w < 0.4:
            continue
        if o.get("walked") and sill < 0.35:           # WALK-PATH entrance DOOR
            elements.extend(_framed(o, w, h, z0, z1, "door")); ndoor += 1
            narch += bool(_arch_above(o, w, z1))       # arch above the 7ft head
        elif sill >= 0.35:                            # WINDOW -- sill above floor
            elements.extend(_framed(o, w, h, z0, z1, "window")); nwin += 1
        elif ext:                                     # BALCONY / exterior door -- full height, glazed
            bz1 = max(z1, zf + 2.0)
            elements.extend(_framed(o, w, bz1 - (zf + 0.02), zf + 0.02, bz1, "balcony")); nbal += 1
        elif w <= 1.4 and h <= 2.45:                  # DOOR -- interior leaf
            elements.extend(_framed(o, w, h, z0, z1, "door")); ndoor += 1
            narch += bool(_arch_above(o, w, z1))       # arch above the 7ft head
        # else: wide interior opening = open cased passage -> no element
    log(f"elements: {ndoor} doors, {nwin} windows, {nbal} balcony doors, {narch} door-head arches ({len(elements)} parts)")

    # ---- ASSEMBLE ONE SOLID: union all positives (walls + beams merge into one
    # connected surface), then subtract every opening/recess in a single manifold
    # difference so the 90-degree cuts are true intrusions of the SAME wall, not
    # separate shells. Weld + fix so it reads as one smooth solid. ----
    log(f"union {len(positives)} walls, subtract {len(negatives)} cutters, add {len(beams)} beams (manifold) ...")
    wall_solid = _difference(_union(positives), negatives)   # walls with openings + recesses
    wall_solid = _union([wall_solid] + beams)                # beams merge on top, uncut
    wall_solid = _clean(wall_solid)
    log(f"unified wall solid: {len(wall_solid.vertices):,}v / {len(wall_solid.faces):,}f / {wall_solid.body_count} bodies")

    # ---- SPLIT into ONE MESH PER ROOM (modular): intersect the unified solid
    # with each room's free-space column (dilated by a wall thickness), so every
    # room's enclosing walls -- with their recesses/beams -- come out as a single
    # named mesh. Shared walls are carried by both adjoining rooms. ----
    mk = R["mk"]
    room_meshes = {}
    sel_h = (zf - 0.1, zc + 0.1)
    grow_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int((WALL_T + 0.10) / CELL) | 1,) * 2)
    for ri, lab in enumerate(R["room_labels"]):
        rm = (mk == lab).astype(np.uint8)
        if rm.sum() * CELL * CELL < 1.0:                      # skip slivers < 1 m2
            continue
        band = cv2.dilate(rm, grow_r) & (rm == 0)             # the wall ring around the room
        band = cv2.dilate(band, np.ones((3, 3), np.uint8))
        sel = None
        for pg in footprint_polygons(band, xmin, ymax):
            try:
                s = trimesh.creation.extrude_polygon(pg, height=sel_h[1] - sel_h[0])
                s.apply_translation((0, 0, sel_h[0]))
                sel = s if sel is None else trimesh.util.concatenate([sel, s])
            except Exception:
                pass
        if sel is None:
            continue
        try:
            piece = trimesh.boolean.intersection([wall_solid, _union([sel])], engine="manifold")
        except Exception:
            piece = None
        if piece is not None and len(piece.faces):
            room_meshes[f"room_{ri:02d}"] = _clean(piece)
    log(f"split into {len(room_meshes)} per-room wall meshes")

    fx = R["x"].max() - R["x"].min(); fy = R["y"].max() - R["y"].min()
    floor = trimesh.creation.box(extents=(fx, fy, 0.08))
    floor.apply_translation(((R["x"].min() + R["x"].max()) / 2, (R["y"].min() + R["y"].max()) / 2, zf - 0.05))

    # whole unified model. COLOUR CONVENTION so measured vs modelled is obvious:
    #   neutral grey  = MEASURED shell (walls, openings, recesses, headers) from LiDAR
    #   amber         = SYNTHESISED fittings (door leaf/frame/handle/mullion) -- placed
    #                   by rule, not surveyed
    #   translucent cyan = SYNTHESISED glass panes
    SYNTH = [232, 150, 45, 255]; SYNTH_GLASS = [120, 200, 235, 130]
    scene = trimesh.Scene()
    wall_solid.visual.face_colors = [200, 202, 208, 255]
    floor.visual.face_colors = [150, 130, 110, 255]
    scene.add_geometry(wall_solid, geom_name="walls_measured")
    scene.add_geometry(floor, geom_name="floor")
    for ei, (em, ec) in enumerate(elements):
        is_glass = ec[2] > ec[0]
        em.visual.face_colors = SYNTH_GLASS if is_glass else SYNTH
        scene.add_geometry(em, geom_name=f"synth_{'glass' if is_glass else 'fitting'}_{ei:02d}")
    scene.export(str(out_dir / "skeleton_model.glb"))
    scene.export(str(out_dir / "skeleton_model.obj"))
    log(f"skeleton_model: {len(wall_solid.vertices):,}v / {len(wall_solid.faces):,}f + {len(elements)} elements -> {out_dir}")
    log("colour: grey=measured shell, amber=synthesised fittings, cyan=glass")

    # modular per-room model (each room's walls a distinct named mesh)
    if room_meshes:
        palette = [[210, 180, 140, 255], [150, 200, 210, 255], [200, 170, 200, 255],
                   [180, 210, 170, 255], [210, 200, 150, 255], [190, 190, 210, 255]]
        rs = trimesh.Scene()
        for i, (name, m) in enumerate(room_meshes.items()):
            m.visual.face_colors = palette[i % len(palette)]
            rs.add_geometry(m, geom_name=name)
        rs.add_geometry(floor, geom_name="floor")
        rs.export(str(out_dir / "skeleton_rooms.glb"))
        rs.export(str(out_dir / "skeleton_rooms.obj"))
        log(f"skeleton_rooms: {len(room_meshes)} room meshes -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
