"""Regularize existing wall planes, preserving rectangular openings.

Every step is either measured or a declared, bounded inference, and each
plane reports both amounts:

1. Recovery (measured): inside the rectangle of the plane's main pieces, every
   25 mm cell that has a raw return within 50 mm of the plane is restored.
   That is surface the earlier cleanup thresholds dropped, never a bridge
   across nothing.
2. Outline notches (inferred, ``notch_m``): concavities up to 0.15 m deep are
   closed; the area is reported as ``inferred_outline_area_m2``.
3. Holes: a hole completely surrounded by the plane, not rectangular and no
   larger than a scan shadow (``max_gap_m2``) is filled as inferred
   continuity; a rectangular hole is a door, window, vent or service opening
   and stays open, straightened to a rectangle.
4. Islands under 0.02 m2 are dropped (``dropped_islands_m2``).
5. Every ring is snapped to axis-aligned edges when that moves no vertex more
   than ``snap_m`` (50 mm); rings that cannot be snapped inside that bound are
   kept as measured.

No new wall run, opposite face or thickness is generated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from wall_surface_cleanup import planes_of


def is_opening(hole, min_side=.04, rectangularity=.85):
    """Rectangular holes wider than a vent slot are real openings (a circle scores 0.785)."""
    width, height = np.subtract(hole.bounds[2:], hole.bounds[:2])
    return hole.area / max(width * height, 1e-12) >= rectangularity and min(width, height) > min_side


def grid_samples(region, pitch):
    u0, v0, u1, v1 = region.bounds
    us = np.arange(u0 + pitch / 2, u1, pitch); vs = np.arange(v0 + pitch / 2, v1, pitch)
    if not len(us) or not len(vs):
        return np.zeros((0, 2))
    uv = np.stack(np.meshgrid(us, vs, indexing='ij'), -1).reshape(-1, 2)
    return uv[shapely.contains_xy(region, uv[:, 0], uv[:, 1])]


def recover_supported(region, support, pitch=.025, min_piece_m2=.1):
    """Restore, inside the rectangle of the plane's main pieces, every cell the returns support."""
    if support is None:
        return region, 0.
    main = [p for p in polygon_parts(region) if p.area >= min_piece_m2] or list(polygon_parts(region))
    if not main:
        return region, 0.
    frame = shapely.box(*shapely.union_all(main).bounds)
    candidate = frame.difference(region)
    if candidate.is_empty or candidate.area < 1e-6:
        return region, 0.
    uv = grid_samples(candidate, pitch)
    if not len(uv):
        return region, 0.
    ok = support(uv)
    if not ok.any():
        return region, 0.
    # Merge supported cells into row runs first: a union of tens of thousands
    # of single cells makes a staircase outline that later buffers choke on.
    half = pitch * .55
    keys = np.round(uv[ok] / pitch).astype(np.int64)
    order = np.lexsort((keys[:, 0], keys[:, 1]))
    keys, centres = keys[order], uv[ok][order]
    boxes = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i, 1] != keys[start, 1] or keys[i, 0] != keys[i - 1, 0] + 1:
            boxes.append(shapely.box(centres[start, 0] - half, centres[start, 1] - half, centres[i - 1, 0] + half, centres[i - 1, 1] + half))
            start = i
    cells = shapely.union_all(boxes)
    recovered = cells.intersection(candidate)
    return shapely.union_all([region, recovered]), float(recovered.area)


def _runs(points, angle_deg):
    """Classify ring edges as horizontal, vertical or diagonal and merge consecutive equals."""
    n = len(points)
    tangent = np.tan(np.radians(angle_deg))
    runs = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        du, dv = b[0] - a[0], b[1] - a[1]
        kind = 'h' if abs(dv) <= tangent * abs(du) else 'v' if abs(du) <= tangent * abs(dv) else 'd'
        length = float(np.hypot(du, dv))
        if runs and runs[-1]['kind'] == kind and kind != 'd':
            runs[-1]['idx'].append(i); runs[-1]['length'] += length
        else:
            runs.append({'kind': kind, 'idx': [i], 'length': length})
    if len(runs) > 1 and runs[0]['kind'] == runs[-1]['kind'] and runs[0]['kind'] != 'd':
        runs[-1]['idx'].extend(runs[0]['idx']); runs[-1]['length'] += runs[0]['length']; runs.pop(0)
    return runs


def orthogonalize_ring(coords, snap_m=.05, angle_deg=8.):
    """Snap a ring to axis-aligned edges; None when that needs more than ``snap_m``.

    Simplifying at the snap distance first collapses zigzag noise into long
    runs, which are then classified by direction; real steps deeper than the
    snap distance survive as their own runs.
    """
    original = np.asarray(shapely.LinearRing(coords).coords)[:-1]
    # Simplification pins the start vertex, so start at a corner (the vertex
    # farthest from the centroid) rather than wherever a clip left the seam.
    original = np.roll(original, -int(np.argmax(np.linalg.norm(original - original.mean(0), axis=1))), axis=0)
    ring = shapely.LinearRing(original).simplify(snap_m, preserve_topology=False)
    points = np.asarray(ring.coords)[:-1]
    if len(points) < 4:
        return None
    # Simplification keeps original vertices, so every kept vertex maps back to
    # its index; a run's level is then the mean of ALL original vertices along
    # it, not of the extreme points Douglas-Peucker prefers to keep.
    lookup = {tuple(np.round(p, 9)): i for i, p in enumerate(original)}
    back = [lookup.get(tuple(np.round(p, 9))) for p in points]
    if any(b is None for b in back):
        back = None
    runs = _runs(points, angle_deg)
    if len(runs) < 4 or all(r['kind'] == 'd' for r in runs):
        return None
    # Remove steps shorter than the snap distance between two runs of one kind.
    changed = True
    while changed and len(runs) > 4:
        changed = False
        order = sorted(range(len(runs)), key=lambda k: runs[k]['length'])
        for k in order:
            if runs[k]['length'] > snap_m:
                continue
            before, after = runs[(k - 1) % len(runs)], runs[(k + 1) % len(runs)]
            if before is after:
                continue
            if before['kind'] == after['kind'] and before['kind'] != 'd':
                # A step shorter than the snap distance between two runs of one kind.
                before['idx'].extend(runs[k]['idx']); before['idx'].extend(after['idx'])
                before['length'] += runs[k]['length'] + after['length']
                for gone in sorted({k, (k + 1) % len(runs)}, reverse=True):
                    runs.pop(gone)
                changed = True
                break
            if runs[k]['kind'] == 'd' and before['kind'] != 'd':
                # A tiny diagonal at a corner: absorb it into the run before it.
                before['idx'].extend(runs[k]['idx']); before['length'] += runs[k]['length']
                runs.pop(k)
                changed = True
                break

    def level(run):
        idx = run['idx']
        if back is not None:
            start, stop = back[idx[0]], back[(idx[-1] + 1) % len(points)]
            span = np.arange(start, stop + 1) if stop >= start else np.concatenate([np.arange(start, len(original)), np.arange(0, stop + 1)])
            # The two end vertices are corners shared with the neighbouring
            # runs; they bias a short run, so use the interior vertices.
            ends = original[span[1:-1]] if len(span) > 3 else original[span]
        else:
            ends = np.array([points[i] for i in idx] + [points[(idx[-1] + 1) % len(points)]])
        return float(np.mean(ends[:, 1])) if run['kind'] == 'h' else float(np.mean(ends[:, 0]))

    new = []
    for k, run in enumerate(runs):
        nxt = runs[(k + 1) % len(runs)]
        corner = points[(run['idx'][-1] + 1) % len(points)]
        if run['kind'] == 'h' and nxt['kind'] == 'v':
            new.append((level(nxt), level(run)))
        elif run['kind'] == 'v' and nxt['kind'] == 'h':
            new.append((level(run), level(nxt)))
        elif run['kind'] == 'h':
            new.append((corner[0], level(run)))
        elif run['kind'] == 'v':
            new.append((level(run), corner[1]))
        elif nxt['kind'] == 'h':
            new.append((corner[0], level(nxt)))
        elif nxt['kind'] == 'v':
            new.append((level(nxt), corner[1]))
        else:
            new.append(tuple(corner))
    if len(new) < 4:
        return None
    snapped = shapely.Polygon(new).buffer(0)
    if snapped.is_empty or snapped.geom_type != 'Polygon':
        return None
    original = shapely.Polygon(points)
    if snapped.hausdorff_distance(original) > snap_m * 1.5:
        return None
    return snapped


def orthogonalize(region, snap_m=.05):
    """Snap every ring of every piece; returns (polygon, snapped rings, kept rings)."""
    pieces, snapped, kept = [], 0, 0
    for piece in polygon_parts(region):
        outer = orthogonalize_ring(piece.exterior.coords, snap_m)
        if outer is None:
            outer = shapely.Polygon(piece.exterior.coords); kept += 1
        else:
            snapped += 1
        holes = []
        for ring in piece.interiors:
            hole = orthogonalize_ring(ring.coords, snap_m)
            if hole is None:
                hole = shapely.Polygon(ring.coords); kept += 1
            else:
                snapped += 1
            holes.append(hole)
        result = outer.difference(shapely.union_all(holes)) if holes else outer
        pieces.append(result)
    return shapely.union_all(pieces).buffer(0), snapped, kept


def rectangular_opening(hole, snap_m=.05):
    """Straight rectangle for a jagged opening.

    First the ring is snapped to axis-aligned edges (unbiased mean levels);
    when that fails the 3rd/97th percentile extents are used, and when even
    that differs from the hole by more than a third of its area the hole is
    kept as measured.
    """
    snapped = orthogonalize_ring(hole.exterior.coords, snap_m)
    if snapped is not None and snapped.area / max(np.prod(np.subtract(snapped.bounds[2:], snapped.bounds[:2])), 1e-12) > .95:
        return shapely.box(*snapped.bounds)
    xy = np.asarray(hole.exterior.coords)
    lo = np.percentile(xy, 3, axis=0); hi = np.percentile(xy, 97, axis=0)
    box = shapely.box(lo[0], lo[1], hi[0], hi[1])
    return box if box.symmetric_difference(hole).area < .35 * hole.area else hole


def regularize(region, reach=.05, max_gap_m2=1., max_gap_m=1.5, support=None, notch_m=.15, snap_m=.05, tiny_m2=.02):
    region, recovered_m2 = recover_supported(region, support)
    # Outline notches: bounded inference, reported.
    closed = region.buffer(notch_m, join_style=2).buffer(-notch_m, join_style=2)
    inferred_outline = float(closed.difference(region).area)
    # Holes of the measured region (closing swallows small ones such as vents)
    # plus holes the closing creates when a notch's mouth shuts.
    holes = [shapely.Polygon(r) for p in polygon_parts(region) for r in p.interiors]
    known = shapely.union_all(holes) if holes else None
    for p in polygon_parts(closed):
        for r in p.interiors:
            hole = shapely.Polygon(r)
            if known is None or not hole.intersects(known):
                holes.append(hole)
    protected, gaps = [], []
    for hole in holes:
        width, height = np.subtract(hole.bounds[2:], hole.bounds[:2])
        if hole.area <= .04 and max(width, height) <= .3 and not is_opening(hole):
            gaps.append(hole)          # speckle, always inferred continuity
        elif is_opening(hole) or hole.area > max_gap_m2 or max(width, height) > max_gap_m:
            protected.append(rectangular_opening(hole) if is_opening(hole) else hole)
        else:
            gaps.append(hole)          # blobby scan shadow inside one plane
    cleaned = closed.buffer(reach / 2, join_style=2).buffer(-reach / 2, join_style=2)
    cleaned = cleaned.buffer(-.0075, join_style=2).buffer(.0075, join_style=2)
    cleaned = cleaned.simplify(.015, preserve_topology=True)
    # Every hole is either a gap (filled) or a protected opening (re-cut below
    # as its straightened rectangle), so the outer rings alone remain here.
    cleaned = shapely.union_all([shapely.Polygon(poly.exterior) for poly in polygon_parts(cleaned)])
    bound = shapely.union_all([shapely.Polygon(poly.exterior) for poly in polygon_parts(closed)]).buffer(reach, join_style=2)
    cleaned = cleaned.intersection(bound)
    if gaps:
        cleaned = shapely.union_all([cleaned, *gaps])
    dropped = [p for p in polygon_parts(cleaned) if p.area < tiny_m2]
    if dropped:
        cleaned = shapely.union_all([p for p in polygon_parts(cleaned) if p.area >= tiny_m2])
    cleaned, snapped, kept = orthogonalize(cleaned, snap_m)
    if protected:
        cleaned = cleaned.difference(shapely.union_all(protected))
    # Do not turn a narrow actual wall strip into an empty result.
    if cleaned.is_empty or cleaned.area < .9 * region.area:
        return region, {'accepted': False}
    return cleaned, {'accepted': True, 'added_area_m2': cleaned.difference(region).area + recovered_m2,
                     'removed_area_m2': region.difference(cleaned).area,
                     'recovered_measured_area_m2': recovered_m2,
                     'inferred_outline_area_m2': inferred_outline,
                     'filled_gap_area_m2': float(sum(g.area for g in gaps)),
                     'dropped_islands_m2': float(sum(p.area for p in dropped)),
                     'snapped_rings': snapped, 'unsnapped_rings': kept,
                     'protected_openings': len(protected)}


def regularize_within(region, mask, reach=.05, margin=.3, support=None):
    """Regularize only the part of the plane inside ``mask`` (partly exterior faces)."""
    zone = mask.buffer(margin, join_style=2)
    inside = region.intersection(zone)
    proposed, row = regularize(inside, reach=reach, support=support)
    if not row['accepted']:
        return region, row
    return shapely.union_all([proposed, region.difference(zone)]), row


def load_selection(path):
    """Selected planes per wall.

    Accepts a plain list of wall names (every vertical plane of each wall,
    value None) or the plane-level form written by select_exterior_walls.py,
    ``{"walls": {name: [{"key", "mask_uv"}, ...]}}`` (value: plane key ->
    in-plane mask polygon or None for the whole plane), so the interior face
    of an exterior wall is left exactly as measured.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {name: None for name in data}
    selection = {}
    for name, planes in data['walls'].items():
        chosen = {}
        for entry in planes:
            key = tuple(np.round(entry['key'] if isinstance(entry, dict) else entry, 5))
            mask = entry.get('mask_uv') if isinstance(entry, dict) else None
            chosen[key] = shapely.union_all([shapely.Polygon(ring) for ring in mask]) if mask else None
        selection[name] = chosen
    return selection


def plane_selection(selected_planes, plane):
    """(selected, mask) for one plane; the mask is None for a whole-plane selection."""
    if selected_planes is None:
        return True, None
    key = tuple(np.round(plane['key'], 5))
    return key in selected_planes, selected_planes.get(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--selection', help='JSON list of wall names, or {"walls": {name: [plane keys]}} from select_exterior_walls.py')
    parser.add_argument('--evidence-cache', help='raw_evidence.py npz; enables recovery of dropped surface where the returns support it')
    parser.add_argument('--support-m', type=float, default=.025, help='band around the plane a return must lie in')
    parser.add_argument('--min-returns', type=int, default=2, help='returns within the band per 25 mm cell to count as surface')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    selected = load_selection(args.selection) if args.selection else None
    tree = None
    if args.evidence_cache:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.load(args.evidence_cache, allow_pickle=True)['p'], leafsize=64, compact_nodes=False, balanced_tree=False)

    def supporter(plane):
        if tree is None:
            return None

        def support(uv):
            # A surface in the plane gives several 10 mm evidence returns per
            # 25 mm cell; one stray return within 50 mm would also accept a
            # surface sitting 50-100 mm off the plane, so require two returns
            # within the tighter band.
            xyz = plane['origin'] + uv[:, :1] * plane['u'] + uv[:, 1:] * plane['v']
            counts = tree.query_ball_point(xyz, args.support_m, workers=-1, return_length=True)
            return np.asarray(counts) >= args.min_returns
        return support

    parts, report = [], []
    for part in json.loads(Path(args.model).read_text())['parts']:
        if selected is not None and part['name'] not in selected:
            continue
        if not part['kind'].startswith(('wall', 'parapet')):
            continue
        planes = list(planes_of(part))
        if any(p['snap'] > .0005 for p in planes):
            continue
        selected_planes = selected[part['name']] if selected is not None else None
        meshes, rows = [], []
        for plane in planes:
            poly = plane['poly']
            chosen, mask = plane_selection(selected_planes, plane)
            if chosen and abs(plane['normal'][2]) < .01 and poly.area > .1:
                support = supporter(plane)
                proposed, row = regularize(poly, support=support) if mask is None else regularize_within(poly, mask, support=support)
            else:
                proposed, row = poly, {'accepted': False}
            row['selected'] = bool(chosen)
            row['partial_mask'] = mask is not None
            row['plane_key'] = [float(x) for x in plane['key']]
            if not proposed.is_empty:
                meshes.append(mesh_from_polygon(proposed, plane['origin'], plane['u'], plane['v']))
            rows.append(row)
        if not any(r['accepted'] for r in rows) or not meshes:
            continue
        mesh = trimesh.util.concatenate(meshes)
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
        replacement = {**part, 'name': part['name'] + ' - smooth wall plane',
                       'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                       'source_group_names': [part['name']], 'merge_coplanar_faces': True,
                       'evidence_status': 'existing_wall_plane_recovered_from_returns_plus_bounded_inferred_outline',
                       'completion_is_measured': False}
        replacement.pop('planar_loops', None)
        parts.append(replacement)
        report.append({'wall': part['name'], 'planes': rows})
        print(f'Regularized {part["name"]}', flush=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    accepted = [r for p in report for r in p['planes'] if r['accepted']]
    print(json.dumps({'walls': len(parts), 'planes': len(accepted),
                      'added_area_m2': sum(r['added_area_m2'] for r in accepted),
                      'recovered_measured_area_m2': sum(r['recovered_measured_area_m2'] for r in accepted),
                      'inferred_outline_area_m2': sum(r['inferred_outline_area_m2'] for r in accepted),
                      'filled_gap_area_m2': sum(r['filled_gap_area_m2'] for r in accepted),
                      'removed_area_m2': sum(r['removed_area_m2'] for r in accepted),
                      'snapped_rings': sum(r['snapped_rings'] for r in accepted),
                      'unsnapped_rings': sum(r['unsnapped_rings'] for r in accepted)}))


if __name__ == '__main__':
    main()
