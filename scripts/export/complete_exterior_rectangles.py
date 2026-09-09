"""Complete exterior wall planes to clean rectangles (declared inference).

A drawing-clean exterior needs every outer face to run as one rectangle from
its floor datum to the slab above, with rectangular openings and nothing
else. The scan cannot see all of that, so this step fills what it cannot
see, and says so. For every whole-plane exterior selection the target
rectangle is the plane's horizontal extent by the level's datums (bottom:
the level floor; top: the next level's floor, or the plane's own top on the
last level). Inside that rectangle, every 25 mm cell not already on the
plane is classified in the raw returns:

- surface on the plane (two returns within 25 mm): measured fill;
- another surface 50-350 mm off the plane, or a parallel model face there
  (a recess, step, sunshade or the face behind an opening): left open;
- rectangular openings already cut in the plane: left open;
- nothing at all: inferred fill.

The completed outline is snapped to axis-aligned edges. Every replacement
reports ``inferred_fill_m2``; the evidence status names the inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree

from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from regularize_wall_surfaces import grid_samples, is_opening, load_selection, orthogonalize, plane_selection
from wall_surface_cleanup import planes_of


def level_datums(parts):
    floors = {}
    for part in parts:
        kind = part.get('kind', '')
        if 'floor' in kind and kind not in ('floor_single_plane', 'ground_single_plane', 'floor_junction_patch'):
            z = float(np.median(np.asarray(part['v'])[:, 2]))
            level = part.get('level', 0)
            floors.setdefault(level, []).append((z, float(trimesh.Trimesh(part['v'], part['f'], process=False).area)))
    datums = {}
    for level, rows in floors.items():
        z, a = zip(*rows)
        datums[level] = float(np.average(z, weights=a))
    return datums


def parallel_model_faces(parts, plane, band=(.02, .35), exclude_name=None, cache=None):
    """Polygons (in this plane's uv frame) of model faces parallel to the plane 20-350 mm away."""
    n = np.asarray(plane['normal']); covered = []
    cache = {} if cache is None else cache
    for part in parts:
        if part['name'] == exclude_name or not part.get('kind', '').startswith(('wall', 'parapet')):
            continue
        v = np.asarray(part['v'], float)
        d = v @ n - plane['offset']
        if np.abs(d).min() > band[1]:
            continue
        if part['name'] not in cache:
            cache[part['name']] = list(planes_of(part))
        for other in cache[part['name']]:
            if abs(float(np.asarray(other['normal']) @ n)) < .99:
                continue
            gap = abs(other['offset'] * float(np.sign(np.asarray(other['normal']) @ n)) - plane['offset'])
            if not band[0] <= gap <= band[1]:
                continue
            mesh = mesh_from_polygon(other['poly'], other['origin'], other['u'], other['v'])
            if mesh is None:
                continue
            uv = np.stack(((mesh.vertices - plane['origin']) @ plane['u'], (mesh.vertices - plane['origin']) @ plane['v']), 1)
            covered.append(shapely.union_all(shapely.polygons(uv[mesh.faces])).buffer(.03, join_style=2))
    return shapely.union_all(covered) if covered else None


def complete_plane(plane, level, datums, tree, other_faces, pitch=.025, reach_datum=.4, own_face_gaps=()):
    """``own_face_gaps``: distances from this plane to the other parallel planes of the same
    wall part (its back face, plaster steps). Returns at one of those distances are the
    wall's own other face seen through a gap in this one, not a recess, so they do not
    block the fill."""
    poly = plane['poly']
    main = [p for p in polygon_parts(poly) if p.area >= .1]
    if not main:
        return None, None
    u0, v0, u1, v1 = shapely.union_all(main).bounds
    up = np.asarray(plane['u']); vv = np.asarray(plane['v']); origin = np.asarray(plane['origin'])
    vertical_axis = 0 if abs(up[2]) > abs(vv[2]) else 1
    axis = up if vertical_axis == 0 else vv
    sign = float(np.sign(axis[2]))
    z_at = lambda t: origin[2] + t * axis[2]
    bottom = datums.get(level); top = datums.get(level + 1)
    lo, hi = (u0, u1) if vertical_axis == 0 else (v0, v1)
    z_lo, z_hi = sorted((z_at(lo), z_at(hi)))
    if bottom is not None and abs(z_lo - bottom) <= reach_datum:
        z_lo = bottom
    if top is not None and abs(z_hi - top) <= reach_datum:
        z_hi = top
    t_lo, t_hi = sorted(((z_lo - origin[2]) / axis[2], (z_hi - origin[2]) / axis[2]))
    rect = shapely.box(t_lo, v0, t_hi, v1) if vertical_axis == 0 else shapely.box(u0, t_lo, u1, t_hi)
    openings = [shapely.Polygon(r) for p in polygon_parts(poly) for r in p.interiors if is_opening(shapely.Polygon(r))]
    candidate = rect.difference(poly)
    if openings:
        candidate = candidate.difference(shapely.union_all(openings))
    if other_faces is not None:
        candidate = candidate.difference(other_faces)
    if candidate.is_empty or candidate.area < 1e-4:
        return poly, {'inferred_fill_m2': 0., 'measured_fill_m2': 0., 'left_open_other_surface_m2': 0.}
    uv = grid_samples(candidate, pitch)
    if not len(uv):
        return poly, {'inferred_fill_m2': 0., 'measured_fill_m2': 0., 'left_open_other_surface_m2': 0.}
    xyz = origin + uv[:, :1] * up + uv[:, 1:] * vv
    n = np.asarray(plane['normal'])
    near = tree.query_ball_point(xyz, .35, workers=-1)
    on_plane = np.zeros(len(uv), bool); other = np.zeros(len(uv), bool)
    pts = tree.data
    gaps = np.asarray(own_face_gaps, float)          # signed: positive towards the wall's back face
    # A recess or niche lies on the material side of the face. A surface on
    # the other side (room or street) is furniture, a car, a railing: an
    # occluder, never a reason to leave a hole in the wall.
    material_sign = float(np.sign(gaps[np.argmax(np.abs(gaps))])) if len(gaps) else 0.
    for i, idx in enumerate(near):
        if not idx:
            continue
        signed = pts[idx] @ n - plane['offset']
        d = np.abs(signed)
        on_plane[i] = (d <= .025).sum() >= 2
        if on_plane[i]:
            continue
        if material_sign:
            away = signed[(signed * material_sign >= .05)] * material_sign
        else:
            away = d[d >= .05]
        if len(away) < 2:
            continue
        if len(gaps) and np.min(np.abs(np.median(away) - np.abs(gaps))) <= .03:
            continue                       # the wall's own back face, seen through this one
        other[i] = True
    # See-through test: a doorway or window shows nothing at the wall but
    # plenty of returns beyond the wall thickness in the same patch (the room
    # or street behind). Such cells stay open whatever the 350 mm band says.
    thickness = float(np.max(np.abs(gaps))) if len(gaps) else .3
    reach = 3.
    rect_uv = candidate.bounds
    allp = tree.data
    signed_all = allp @ n - plane['offset']
    if material_sign:
        beyond = (signed_all * material_sign > thickness + .05) & (signed_all * material_sign < reach)
    else:
        beyond = (np.abs(signed_all) > thickness + .05) & (np.abs(signed_all) < reach)
    see = np.zeros(len(uv), bool)
    if beyond.any():
        buv = np.stack(((allp[beyond] - origin) @ up, (allp[beyond] - origin) @ vv), 1)
        inside = (buv[:, 0] >= rect_uv[0]) & (buv[:, 0] <= rect_uv[2]) & (buv[:, 1] >= rect_uv[1]) & (buv[:, 1] <= rect_uv[3])
        if inside.any():
            keys, counts = np.unique(np.floor(buv[inside] / pitch).astype(np.int64), axis=0, return_counts=True)
            dense = {tuple(k) for k, c in zip(keys, counts) if c >= 2}
            cell = np.floor(uv / pitch).astype(np.int64)
            see = np.array([tuple(c) in dense for c in cell]) & ~on_plane
    other |= see
    half = pitch * .55
    boxes = lambda m: shapely.union_all(shapely.box(uv[m, 0] - half, uv[m, 1] - half, uv[m, 0] + half, uv[m, 1] + half)) if m.any() else None
    fill = candidate
    other_region = boxes(other)
    if other_region is not None:
        fill = fill.difference(other_region)
    measured = boxes(on_plane)
    completed = shapely.union_all([poly, fill])
    completed, _, _ = orthogonalize(completed, .05)
    if openings:
        completed = completed.difference(shapely.union_all(openings))
    inferred = float(fill.area - (measured.intersection(fill).area if measured is not None else 0.))
    return completed, {'inferred_fill_m2': inferred, 'see_through_open_m2': float(boxes(see).area) if see.any() else 0.,
                       'measured_fill_m2': float(measured.intersection(fill).area) if measured is not None else 0.,
                       'left_open_other_surface_m2': float(other_region.area) if other_region is not None else 0.,
                       'rectangle': [float(x) for x in rect.bounds], 'z_range': [float(z_lo), float(z_hi)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='reopened_visible.build.json of the checked model')
    parser.add_argument('--patches', required=True, help='walls/patches.build.json (regularized exterior planes)')
    parser.add_argument('--recessed', help='recessed_faces.build.json, whose faces also keep their regions open')
    parser.add_argument('--selection', required=True)
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())['parts']
    patches = json.loads(Path(args.patches).read_text())['parts']
    extra = json.loads(Path(args.recessed).read_text())['parts'] if args.recessed else []
    selection = load_selection(args.selection)
    datums = level_datums(model)
    print('level datums', {k: round(v, 3) for k, v in sorted(datums.items())}, flush=True)
    tree = cKDTree(np.load(args.evidence_cache, allow_pickle=True)['p'], leafsize=64, compact_nodes=False, balanced_tree=False)
    replaced_sources = {p['source_group_names'][0]: p for p in patches}
    context = [p for p in model if p['name'] not in replaced_sources] + patches + extra
    out_parts, report, cache = [], [], {}
    for patch in patches:
        source = patch['source_group_names'][0]
        if source not in selection:
            continue
        chosen_planes = selection[source]        # None means every plane (plain list selection)
        planes = list(planes_of(patch))
        meshes, rows, changed = [], [], False
        for plane in planes:
            chosen, mask = plane_selection(chosen_planes, plane)
            proposed, row = plane['poly'], None
            if chosen and mask is None and abs(plane['normal'][2]) < .01 and plane['poly'].area > .5:
                other_faces = parallel_model_faces(context, plane, exclude_name=patch['name'], cache=cache)
                n = np.asarray(plane['normal'])
                own = [q['offset'] * float(np.sign(np.asarray(q['normal']) @ n)) - plane['offset']
                       for q in planes if q is not plane and abs(float(np.asarray(q['normal']) @ n)) > .99 and q['poly'].area > .2]
                proposed, row = complete_plane(plane, patch.get('level', 0), datums, tree, other_faces, own_face_gaps=own)
                if proposed is None:
                    proposed, row = plane['poly'], None
                elif row['inferred_fill_m2'] + row['measured_fill_m2'] > 1e-3:
                    changed = True
            if not proposed.is_empty:
                meshes.append(mesh_from_polygon(proposed, plane['origin'], plane['u'], plane['v']))
            if row is not None:
                rows.append({'plane_key': [float(x) for x in plane['key']], **row})
        if not changed or not meshes:
            continue
        mesh = trimesh.util.concatenate([m for m in meshes if m is not None])
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
        replacement = {**patch, 'name': patch['name'] + ' - completed rectangle',
                       'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                       'source_group_names': [patch['name']], 'merge_coplanar_faces': True,
                       'evidence_status': 'exterior_plane_completed_to_level_rectangle_where_scan_shows_nothing_INFERRED',
                       'completion_is_measured': False}
        replacement.pop('planar_loops', None)
        out_parts.append(replacement)
        report.append({'wall': source, 'planes': rows})
        print(f"Completed {source}: inferred {sum(r['inferred_fill_m2'] for r in rows):.2f} m2, "
              f"measured {sum(r['measured_fill_m2'] for r in rows):.2f} m2", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    rows = [r for w in report for r in w['planes']]
    print(json.dumps({'walls': len(out_parts), 'inferred_fill_m2': round(sum(r['inferred_fill_m2'] for r in rows), 2),
                      'measured_fill_m2': round(sum(r['measured_fill_m2'] for r in rows), 2),
                      'left_open_other_surface_m2': round(sum(r['left_open_other_surface_m2'] for r in rows), 2)}))


if __name__ == '__main__':
    main()
