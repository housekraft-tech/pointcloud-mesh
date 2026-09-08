"""Bounded wall faces for a level, the way the Soulace top floor was built.

Each candidate wall (both faces of a paired-plane candidate) is clipped to the
full-density 50 mm raw-return envelope with the same whole-cell bound used for
the top floor (``supported_polygon`` at 50 mm): only observed area survives.

With ``--continuity`` a bounded one-cell closing (50 mm) then bridges gaps
inside the candidate face, and a bridged region is kept only when at least
80 percent of it is strictly supported. Bridged area is emitted as a separate
amber part (``wall_face_continuity``) so it can be hidden or reviewed; it is
inferred continuity, not measurement. Candidate names listed in ``--skip``
(for instance wall hypotheses that cut through a staircase) are left out.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree

from architectural_surface_refinement import _all_planar_patches
from filter_soulace_overlap import supported_polygon, mesh_from_polygon, polygon_parts
from polygon_hygiene import clean_polygon

SUPPORT_M = .05
CLOSING_M = .05
MIN_SUPPORT_FRACTION = .8
CONTINUITY_COLOUR = [222, 160, 90]
INFERRED_COLOUR = [232, 196, 128]


def build(candidates, tree, colour, continuity, mapping, complete=False):
    parts, rows = [], []
    for cand in candidates:
        mesh = trimesh.Trimesh(cand['v'], cand['f'], process=False)
        strict_pieces, fill_pieces, inferred_pieces = [], [], []
        strict_area = fill_area = candidate_area = inferred_area = 0.
        for poly, origin, u, v in _all_planar_patches(mesh):
            candidate_area += poly.area
            bounded, _ = supported_polygon(poly, origin, u, v, tree, cutoff=SUPPORT_M, min_area=.002)
            bounded = clean_polygon(bounded.buffer(0))
            if complete:
                # The rest of the candidate face is kept as flagged inferred
                # geometry so the wall reads as a complete object with its
                # steps and niches, while every unobserved face stays marked.
                rest = clean_polygon(poly.difference(bounded).buffer(0)) if not bounded.is_empty else clean_polygon(poly.buffer(0))
                if not rest.is_empty:
                    inferred_area += rest.area
                    piece = mesh_from_polygon(rest, origin, u, v)
                    if piece is not None:
                        inferred_pieces.append(piece)
            if bounded.is_empty:
                continue
            strict_area += bounded.area
            piece = mesh_from_polygon(bounded, origin, u, v)
            if piece is not None:
                strict_pieces.append(piece)
            if not continuity or complete:
                continue
            closed = bounded.buffer(CLOSING_M, join_style='mitre').buffer(-CLOSING_M, join_style='mitre').intersection(poly)
            keep = []
            for region in polygon_parts(closed.buffer(0)):
                supported = region.intersection(bounded).area
                if region.area > 0 and supported / region.area >= MIN_SUPPORT_FRACTION:
                    keep.append(region)
            if not keep:
                continue
            fill = clean_polygon(shapely.union_all(keep).difference(bounded).buffer(0))
            if fill.is_empty:
                continue
            fill_area += fill.area
            piece = mesh_from_polygon(fill, origin, u, v)
            if piece is not None:
                fill_pieces.append(piece)
        label = re.search(r'(wall|parapet)_\d+', cand['name']).group(0)
        level = cand.get('level', 0)
        sources = mapping.get((level, label), [])
        row = {'candidate': cand['name'], 'level': level, 'candidate_area_m2': candidate_area,
               'strict_area_m2': strict_area, 'continuity_area_m2': fill_area, 'inferred_area_m2': inferred_area,
               'replaces': sources}
        rows.append(row)
        if strict_pieces:
            m = trimesh.util.concatenate(strict_pieces); m.merge_vertices(digits_vertex=8)
            m.update_faces(m.nondegenerate_faces(height=1e-8) & (m.area_faces >= 1e-12)); m.remove_unreferenced_vertices()
            parts.append({'name': f'L{level} bounded wall faces {label}', 'kind': 'wall_scan_surface', 'level': level,
                          'v': m.vertices.tolist(), 'f': m.faces.tolist(), 'colour': colour,
                          'merge_coplanar_faces': True, 'construction_solid': False, 'open_surface': True,
                          'thickness_verified': False, 'modeled_thickness_m': cand.get('modeled_thickness_m'),
                          'measured_face_positions_m': cand.get('measured_face_positions_m', []),
                          'source_group_names': sources + [cand['name']],
                          'evidence_status': 'full_density_50mm_bounded_surface'})
        if fill_pieces:
            m = trimesh.util.concatenate(fill_pieces); m.merge_vertices(digits_vertex=8)
            m.update_faces(m.nondegenerate_faces(height=1e-8) & (m.area_faces >= 1e-12)); m.remove_unreferenced_vertices()
            parts.append({'name': f'L{level} inferred continuity {label} - REVIEW', 'kind': 'wall_face_continuity', 'level': level,
                          'v': m.vertices.tolist(), 'f': m.faces.tolist(), 'colour': CONTINUITY_COLOUR,
                          'merge_coplanar_faces': True, 'construction_solid': False, 'open_surface': True,
                          'thickness_verified': False, 'source_group_names': [],
                          'evidence_status': 'one_cell_50mm_closing_inside_candidate_face_min_80pct_strict_support; inferred, not measured'})
        if inferred_pieces:
            m = trimesh.util.concatenate(inferred_pieces); m.merge_vertices(digits_vertex=8)
            m.update_faces(m.nondegenerate_faces(height=1e-8) & (m.area_faces >= 1e-12)); m.remove_unreferenced_vertices()
            parts.append({'name': f'L{level} inferred faces {label} - REVIEW', 'kind': 'wall_face_inferred', 'level': level,
                          'v': m.vertices.tolist(), 'f': m.faces.tolist(), 'colour': INFERRED_COLOUR,
                          'merge_coplanar_faces': True, 'construction_solid': False, 'open_surface': True,
                          'thickness_verified': False, 'source_group_names': [],
                          'evidence_status': 'candidate face area beyond the 50 mm raw envelope; inferred from the as-built candidate, not measured'})
        print(cand['name'], 'strict', round(strict_area, 2), 'fill', round(fill_area, 2), 'inferred', round(inferred_area, 2), 'of', round(candidate_area, 2), flush=True)
    return parts, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, help='build.json with wall_paired_planes parts (common frame)')
    parser.add_argument('--model', required=True, help='current reopened_visible.build.json, for replaced-group mapping')
    parser.add_argument('--levels', type=int, nargs='+', required=True)
    parser.add_argument('--skip', nargs='*', default=[], help='candidate labels to leave hidden, e.g. L0:wall_34')
    parser.add_argument('--evidence-spec', required=True); parser.add_argument('--evidence-cache', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--continuity', action='store_true')
    parser.add_argument('--complete', action='store_true', help='also emit the unobserved remainder of each candidate face as flagged inferred geometry')
    parser.add_argument('--colour', type=int, nargs=3, default=[205, 198, 181])
    parser.add_argument('--candidate-kinds', nargs='+', default=['wall_paired_planes'])
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from raw_evidence import load_evidence, model_bounds
    model = json.loads(Path(args.model).read_text())['parts']
    skip = set(args.skip)
    candidates = []
    for p in json.loads(Path(args.candidates).read_text())['parts']:
        if p['kind'] not in args.candidate_kinds or p.get('level', 0) not in args.levels:
            continue
        label = re.search(r'(wall|parapet)_\d+', p['name'])
        if label is None or 'L%d:%s' % (p.get('level', 0), label.group(0)) in skip:
            continue
        candidates.append(p)
    mapping = {}
    for p in model:
        if p['level'] in args.levels and p['kind'].startswith(('wall', 'parapet')) and 'observed' not in p['name']:
            m = re.search(r'(wall|parapet)_\d+', p['name'])
            if m:
                mapping.setdefault((p['level'], m.group(0)), []).append(p['name'])
    raw, provenance = load_evidence(args.evidence_spec, model_bounds(model), args.evidence_cache)
    tree = cKDTree(raw)
    parts, rows = build(candidates, tree, args.colour, args.continuity, mapping, args.complete)
    (out / 'bounded_walls.build.json').write_text(json.dumps({'parts': parts}, separators=(',', ':')))
    (out / 'bounded_walls_audit.json').write_text(json.dumps({
        'candidates': str(Path(args.candidates).resolve()), 'levels': args.levels, 'skipped': sorted(skip),
        'raw_evidence': provenance, 'support_bound_mm': SUPPORT_M * 1000,
        'continuity': {'enabled': args.continuity, 'closing_mm': CLOSING_M * 1000, 'min_strict_support_fraction': MIN_SUPPORT_FRACTION},
        'walls': rows,
        'totals': {'candidate_area_m2': sum(r['candidate_area_m2'] for r in rows),
                   'strict_area_m2': sum(r['strict_area_m2'] for r in rows),
                   'continuity_area_m2': sum(r['continuity_area_m2'] for r in rows),
                   'inferred_area_m2': sum(r['inferred_area_m2'] for r in rows)},
        'site_accuracy_certified': False}, indent=2))
    print(json.dumps({'parts': len(parts), 'strict_m2': round(sum(r['strict_area_m2'] for r in rows), 1),
                      'continuity_m2': round(sum(r['continuity_area_m2'] for r in rows), 1),
                      'candidate_m2': round(sum(r['candidate_area_m2'] for r in rows), 1)}), flush=True)


if __name__ == '__main__':
    main()
