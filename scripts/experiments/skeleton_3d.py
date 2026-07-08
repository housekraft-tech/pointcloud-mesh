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
    wallmask = ((ws > 0) | (fh > 0) | (top_attached > 0)).astype(np.uint8)
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
    segs = []
    if lines is not None:
        hs, vs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.25 * ppm), coord_tol_px=int(0.1 * ppm))
        segs = [(np.array([xmin + a0 * CELL, ymax - yr * CELL]), np.array([xmin + a1 * CELL, ymax - yr * CELL]))
                for a0, a1, yr in hs] + \
               [(np.array([xmin + xc * CELL, ymax - a0 * CELL]), np.array([xmin + xc * CELL, ymax - a1 * CELL]))
                for a0, a1, xc in vs]

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
        box = trimesh.creation.box(extents=(L + 3 * WALL_T, WALL_T, storey))   # bury ends in junctions
        box.apply_transform(trimesh.transformations.rotation_matrix(float(np.arctan2(dd[1], dd[0])), [0, 0, 1]))
        mid = (p0 + p1) / 2.0
        box.apply_translation((mid[0], mid[1], zf + storey / 2.0))
        positives.append(box)
        a = (int((p0[0] - xmin) / CELL), int((ymax - p0[1]) / CELL))
        b = (int((p1[0] - xmin) / CELL), int((ymax - p1[1]) / CELL))
        cv2.line(seg_mask, a, b, 1, thickness=t)

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
    UR = 0.02; z0w = zf - 0.05; z1w = zc + 0.05
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
        rel = xy - p0; u = rel @ dd; perp = rel @ nn
        near = (np.abs(perp) <= 0.20) & (u >= 0) & (u <= L)
        if near.sum() < 200:
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
        for i in range(1, ncE):
            bx, by, bw, bh, _ = st[i]
            if not (0.5 <= bw * UR <= 3.0) or bh * UR < 0.6 or bx <= 1 or bx + bw >= nu2 - 1:
                continue
            cs, _ = cv2.findContours((lblE == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.02 / UR, True)
            if len(cnt) < 3:
                continue
            poly = _Poly2([(c * UR, z0w + r * UR) for c, r in cnt[:, 0, :]])
            if not poly.is_valid or poly.area < 0.3:
                continue
            try:
                cutter = trimesh.creation.extrude_polygon(poly, height=WALL_T * 4)
                cutter.apply_transform(np.array(
                    [[dd[0], 0, nn[0], p0[0] - nn[0] * WALL_T * 2],
                     [dd[1], 0, nn[1], p0[1] - nn[1] * WALL_T * 2],
                     [0, 1, 0, 0], [0, 0, 0, 1]], float))
                negatives.append(cutter); ncut += 1
            except Exception:
                pass
    log(f"cut {ncut} openings (arched heads kept)")

    # ---- OFFSET-PLANE reveals / shadow-gaps / soffit faces (Ikehata-style):
    # a wall side is NOT one flat face -- it is a PROUD face plus recessed
    # plane(s). Per side build a fine (u,z) depth map of the NEAREST material,
    # take the proud face as the reference, and for every coherent recessed
    # region cut the wall back to its TRUE measured depth (uncapped, up to the
    # wall thickness) over its true (u,z) extent. This keeps a ~5cm step over the
    # top 40% of a wall, a recessed door head+jamb frame, or a shadow-gap band --
    # each preserved at real depth and real height, not as a token sliver. ----
    UB = 0.04; ZB = 0.05                       # 4cm along-wall x 5cm height cells
    MIN_DEPTH = 0.018                          # ignore < ~2cm (paint/scan noise)
    MAX_DEPTH = WALL_T * 0.72                  # never cut through the solid
    nrev = 0
    for p0, p1 in segs:
        dd = p1 - p0; L = float(np.linalg.norm(dd))
        if L < 0.5:
            continue
        dd = dd / L; nn = np.array([-dd[1], dd[0]])
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
            valid = ~np.isnan(depth)
            if valid.sum() < 20:
                continue
            face = float(np.nanpercentile(depth, 10))         # proud reference plane
            noise = 1.4826 * float(np.nanmedian(np.abs(depth[valid] - np.nanmedian(depth[valid])))) + 1e-6
            thr = max(MIN_DEPTH, 3.0 * noise)
            recess = np.where(valid, depth - face, 0.0)
            groove = (recess >= thr) & valid
            # close 1-cell gaps so a band split by a missing column stays one region
            groove = ndimage.binary_closing(groove, structure=np.ones((3, 3)))
            zf_band = zf + 0.10
            lbl2, nc2 = ndimage.label(groove, structure=np.ones((3, 3)))
            for cc in range(1, nc2 + 1):
                mask = (lbl2 == cc)
                ys, xs = np.where(mask)
                area = ys.size * (UB * ZB)
                if area < 0.05:                               # >= ~500 cm2 region
                    continue
                du = (np.ptp(xs) + 1) * UB; dz = (np.ptp(ys) + 1) * ZB
                if max(du, dz) < 0.15:
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
                depth_cut = float(np.clip(np.median(rv), MIN_DEPTH, MAX_DEPTH))
                box_th = depth_cut + 0.03
                # cut the region's ACTUAL shape (contours in (u,z)) extruded along
                # the wall normal -- so a frame recesses its casing, not the opening.
                m8 = mask.astype(np.uint8)
                cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # local->world: x=u -> dd, y=height(z) -> world Z, z=depth -> nn*side
                M = np.array([
                    [dd[0], 0, nn[0] * side, p0[0] + nn[0] * side * (WALL_T / 2 - depth_cut)],
                    [dd[1], 0, nn[1] * side, p0[1] + nn[1] * side * (WALL_T / 2 - depth_cut)],
                    [0, 1, 0, 0], [0, 0, 0, 1]], float)
                for cnt in cnts:
                    ap = cv2.approxPolyDP(cnt, 0.6, True)[:, 0, :]
                    if len(ap) < 3:
                        continue
                    poly = _Poly2([(float(c) * UB, zf_band + float(r) * ZB) for c, r in ap])
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

    # whole unified model (one smooth solid)
    scene = trimesh.Scene()
    wall_solid.visual.face_colors = [205, 205, 210, 255]
    floor.visual.face_colors = [150, 130, 110, 255]
    scene.add_geometry(wall_solid, geom_name="walls")
    scene.add_geometry(floor, geom_name="floor")
    scene.export(str(out_dir / "skeleton_model.glb"))
    scene.export(str(out_dir / "skeleton_model.obj"))
    log(f"skeleton_model: {len(wall_solid.vertices):,}v / {len(wall_solid.faces):,}f -> {out_dir}")

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
