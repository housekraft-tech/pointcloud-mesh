"""Build solid walls with thickness from the rectilinear face blocks (declared inference).

Per level:

1. every axis-aligned vertical face block is read as a segment in plan (axis,
   offset, extent along the wall, height range) with its rectangular openings
   (holes and floor-touching notches);
2. faces are paired across the wall: same axis, 60-600 mm apart, overlapping
   along the wall by at least 0.3 m. The overlap becomes a wall rectangle in
   plan with the measured thickness. Face lengths no pair covers become
   single-sided walls with the level's median measured thickness (or 230 mm),
   placed on the material side, which is the side with more returns in the
   50-400 mm band; these are tagged INFERRED thickness and coloured apart;
3. the wall rectangles of a level are unioned in plan, so corners, tees and
   crossings join exactly, and extruded between the level's floor datum and
   the slab above (a face that stops short of the slab keeps its own top);
4. openings cut the solid through its full thickness as height bands (door
   from the floor, window between sill and head); a hole with a recessed face
   behind it is a niche and is cut only to that measured depth.

Floors, ceilings, stairs and prisms are untouched. Every face block a solid
replaces is hidden as a reference.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import trimesh

from complete_exterior_rectangles import level_datums
from filter_soulace_overlap import polygon_parts, mesh_from_polygon
from wall_surface_cleanup import planes_of

MEASURED_COLOUR = [221, 190, 110]
INFERRED_COLOUR = [232, 170, 90]


def face_segments(part, min_area=.3):
    """Axis-aligned vertical planes of a block as plan segments with openings."""
    segments = []
    for plane in planes_of(part):
        n = np.asarray(plane['normal'])
        if abs(n[2]) > .01:
            continue
        axis = int(np.argmax(np.abs(n)))
        if abs(n[axis]) < .99:
            continue
        along = 1 - axis
        mesh = mesh_from_polygon(plane['poly'], plane['origin'], plane['u'], plane['v'])
        if mesh is None or mesh.area < min_area:
            continue
        tri = mesh.triangles
        flat = np.stack((tri[:, :, along], tri[:, :, 2]), -1)
        poly = shapely.union_all(shapely.polygons(flat)).buffer(0)
        offset = float(np.median(mesh.vertices[:, axis]))
        for piece in polygon_parts(poly):
            if piece.area < min_area:
                continue
            a0, z0, a1, z1 = piece.bounds
            frame = shapely.box(a0, z0, a1, z1)
            openings = []
            for hole in polygon_parts(frame.difference(piece)):
                if hole.area >= .25:
                    openings.append(hole.bounds)          # (a0, z0, a1, z1)
            segments.append({'part': part['name'], 'axis': axis, 'offset': offset, 'a0': a0, 'a1': a1,
                             'z0': z0, 'z1': z1, 'openings': openings, 'level': part.get('level', 0)})
    return segments


def pair_faces(segments, min_gap=.06, max_gap=.6, min_overlap=.3):
    """Wall rectangles from face pairs; returns (walls, uncovered intervals per segment)."""
    walls = []
    covered = defaultdict(list)
    by_axis = defaultdict(list)
    for i, s in enumerate(segments):
        by_axis[s['axis']].append(i)
    for axis, ids in by_axis.items():
        for i in ids:
            for j in ids:
                if j <= i:
                    continue
                a, b = segments[i], segments[j]
                gap = abs(a['offset'] - b['offset'])
                if not min_gap <= gap <= max_gap:
                    continue
                lo, hi = max(a['a0'], b['a0']), min(a['a1'], b['a1'])
                if hi - lo < min_overlap:
                    continue
                zlo, zhi = max(a['z0'], b['z0']), min(a['z1'], b['z1'])
                if zhi - zlo < .3:
                    continue
                walls.append({'axis': axis, 'o0': min(a['offset'], b['offset']), 'o1': max(a['offset'], b['offset']),
                              'a0': lo, 'a1': hi, 'z0': min(a['z0'], b['z0']), 'z1': max(a['z1'], b['z1']),
                              'faces': [i, j], 'thickness': gap, 'inferred': False})
                covered[i].append((lo, hi)); covered[j].append((lo, hi))
    return walls, covered


def uncovered_intervals(segment, covered, min_len=.3):
    spans = sorted(covered)
    cursor = segment['a0']; out = []
    for lo, hi in spans:
        if lo - cursor >= min_len:
            out.append((cursor, lo))
        cursor = max(cursor, hi)
    if segment['a1'] - cursor >= min_len:
        out.append((cursor, segment['a1']))
    return out


def material_side(segment, returns, band=(.05, .4)):
    """+1 or -1 along the axis: the side with more returns just behind the face."""
    axis, along = segment['axis'], 1 - segment['axis']
    inside = (returns[:, along] >= segment['a0']) & (returns[:, along] <= segment['a1']) & \
             (returns[:, 2] >= segment['z0']) & (returns[:, 2] <= segment['z1'])
    d = returns[inside, axis] - segment['offset']
    plus = ((d >= band[0]) & (d <= band[1])).sum()
    minus = ((d <= -band[0]) & (d >= -band[1])).sum()
    return 1. if plus >= minus else -1.


def build_level(segments, walls, covered, returns, thickness_default, floor_z, slab_z, recess_planes, walls_to_slab=False):
    # Single-sided walls for uncovered face lengths.
    for i, s in enumerate(segments):
        for lo, hi in uncovered_intervals(s, covered.get(i, [])):
            side = material_side(s, returns)
            o0, o1 = sorted((s['offset'], s['offset'] + side * thickness_default))
            walls.append({'axis': s['axis'], 'o0': o0, 'o1': o1, 'a0': lo, 'a1': hi, 'z0': s['z0'], 'z1': s['z1'],
                          'faces': [i], 'thickness': thickness_default, 'inferred': True})
    for w in walls:
        w['z0'] = floor_z if abs(w['z0'] - floor_z) <= .4 else w['z0']
        if slab_z is not None and (abs(w['z1'] - slab_z) <= .6 or (walls_to_slab and w['z1'] < slab_z)):
            w['z1'] = slab_z
    # Openings: slots through the walls they belong to.
    slots = []
    for w in walls:
        for i in w['faces']:
            s = segments[i]
            for a0, z0, a1, z1 in s['openings']:
                lo, hi = max(a0, w['a0']), min(a1, w['a1'])
                if hi - lo < .1:
                    continue
                depth = None
                for rp in recess_planes:
                    if rp['axis'] != w['axis']:
                        continue
                    d = rp['offset'] - s['offset']
                    if .03 <= abs(d) <= .6 and rp['a0'] <= hi and rp['a1'] >= lo and rp['z0'] <= z1 and rp['z1'] >= z0:
                        depth = d; break
                if depth is None:
                    across = (w['o0'] - .01, w['o1'] + .01)
                else:
                    across = tuple(sorted((s['offset'] - np.sign(depth) * .01, s['offset'] + depth)))
                z_lo = floor_z if z0 - floor_z <= .15 else z0
                slots.append({'axis': w['axis'], 'a': (lo, hi), 'across': across, 'z': (z_lo, z1), 'niche': depth is not None})
    return walls, slots


def plan_box(axis, o0, o1, a0, a1):
    return shapely.box(o0, a0, o1, a1) if axis == 0 else shapely.box(a0, o0, a1, o1)


def extrude_bands(walls, slots):
    """Solid mesh: plan union of active walls minus active slots per height band."""
    breaks = sorted({z for w in walls for z in (w['z0'], w['z1'])} | {z for s in slots for z in s['z']})
    meshes = []
    for z0, z1 in zip(breaks[:-1], breaks[1:]):
        if z1 - z0 < 1e-4:
            continue
        mid = (z0 + z1) / 2
        active = [plan_box(w['axis'], w['o0'], w['o1'], w['a0'], w['a1']) for w in walls if w['z0'] <= mid <= w['z1']]
        if not active:
            continue
        region = shapely.union_all(active).buffer(0)
        cuts = [plan_box(s['axis'], s['across'][0], s['across'][1], s['a'][0], s['a'][1]) for s in slots if s['z'][0] <= mid <= s['z'][1]]
        if cuts:
            region = region.difference(shapely.union_all(cuts))
        for piece in polygon_parts(region):
            if piece.area < 1e-4:
                continue
            solid = trimesh.creation.extrude_polygon(piece.simplify(1e-6), z1 - z0)
            solid.apply_translation([0, 0, z0])
            meshes.append(solid)
    if not meshes:
        return None
    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices(digits_vertex=8)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='reopened_visible.build.json (for level datums)')
    parser.add_argument('--blocks', required=True, help='rectilinear/patches.build.json')
    parser.add_argument('--recessed', help='recessed_faces.build.json')
    parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--default-thickness', type=float, default=.23)
    parser.add_argument('--walls-to-slab', action='store_true', help='Every wall reaches the slab above (completion, marked in the name)')
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())['parts']
    blocks = json.loads(Path(args.blocks).read_text())['parts']
    recessed = json.loads(Path(args.recessed).read_text())['parts'] if args.recessed else []
    datums = level_datums(model)
    top_z = max(float(np.max(np.asarray(p['v'])[:, 2])) for p in model)
    returns = np.load(args.evidence_cache, allow_pickle=True)['p']
    recess_planes = [s for p in recessed for s in face_segments(p, min_area=.05)]
    per_level = defaultdict(list)
    for b in blocks:
        per_level[b.get('level', 0)].append(b)
    out_parts, report = [], []
    for level, parts in sorted(per_level.items()):
        segments = [s for p in parts for s in face_segments(p)]
        walls, covered = pair_faces(segments)
        measured = [w['thickness'] for w in walls]
        thickness = float(np.median(measured)) if measured else args.default_thickness
        floor_z = datums.get(level, min(s['z0'] for s in segments) if segments else 0.)
        slab_z = datums.get(level + 1)
        if slab_z is None and segments:
            # Last level: the ceiling is where the tallest faces stop.
            slab_z = float(np.percentile([s['z1'] for s in segments], 90))
        walls, slots = build_level(segments, walls, covered, returns, thickness, floor_z, slab_z, recess_planes, args.walls_to_slab)
        for inferred in (False, True):
            subset = [w for w in walls if w['inferred'] == inferred]
            if not subset:
                continue
            mesh = extrude_bands(subset, slots)
            if mesh is None:
                continue
            names = sorted({segments[i]['part'] for w in subset for i in w['faces']})
            name = f"L{level} solid walls {'INFERRED thickness' if inferred else 'measured thickness'}"
            out_parts.append({'name': name, 'kind': 'wall_solid', 'level': level,
                              'colour': INFERRED_COLOUR if inferred else MEASURED_COLOUR,
                              'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist(),
                              'source_group_names': names, 'merge_coplanar_faces': True,
                              'evidence_status': 'solid_walls_from_paired_faces_' + ('single_sided_thickness_INFERRED' if inferred else 'measured_thickness_joins_INFERRED'),
                              'completion_is_measured': False})
            report.append({'level': level, 'inferred': inferred, 'walls': len(subset), 'faces': len(names),
                           'median_thickness_m': thickness, 'length_m': float(sum(w['a1'] - w['a0'] for w in subset)),
                           'openings': sum(1 for s in slots), 'niches': sum(1 for s in slots if s['niche'])})
            print(f"{name}: {len(subset)} wall rectangles, {sum(w['a1'] - w['a0'] for w in subset):.1f} m, "
                  f"thickness {thickness * 1000:.0f} mm, {len(slots)} openings", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'patches.build.json').write_text(json.dumps({'parts': out_parts}, separators=(',', ':')))
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'solids': len(out_parts), 'levels': sorted(per_level)}))


if __name__ == '__main__':
    main()
