"""Property-independent exterior wall face selection from the raw returns.

A wall face is exterior when the air in front of it is open to the sky. That
is decided in the scan, not in the reconstruction: floors may be incomplete,
terraces have no ceiling and stairwells have no walls in the model, but the
raw returns always carry the ceiling that covers a room and never carry a
ceiling over a terrace, a compound or the street.

For every near-exact vertical plane of every wall part (``planes_of``), area
weighted sample points march horizontally away from the plane on both sides
through a 50 mm occupancy grid of the raw returns. Along the free path, at
offsets from 0.35 m (past any wall thickness) to ``reach`` metres, a second
march goes straight up; if it leaves the scanned volume without meeting a
return the sample looks at open air. A plane whose samples mostly look at
open air on either side is an exterior face; a plane open along only part of
its length carries the in-plane mask of its open samples; a plane whose
samples are almost all covered on both sides is an interior face.

The output is a plane-level selection keyed by the part name and the plane
key that ``planes_of`` yields, so the interior face of an exterior wall stays
untouched when its outer face is regularized.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from filter_soulace_overlap import polygon_parts
from wall_surface_cleanup import planes_of


class Occupancy:
    """Solid 50 mm voxels of the raw returns, with vectorised ray marching."""

    def __init__(self, points, voxel_m=.05, min_returns=3, padding_m=.5):
        points = np.asarray(points, float)
        self.voxel = voxel_m
        self.lo = points.min(0) - padding_m
        self.hi = points.max(0) + padding_m
        self.shape = np.ceil((self.hi - self.lo) / voxel_m).astype(int) + 1
        idx = np.floor((points - self.lo) / voxel_m).astype(np.int64)
        flat = np.ravel_multi_index(idx.T, self.shape)
        counts = np.bincount(flat, minlength=int(np.prod(self.shape)))
        self.solid = (counts >= min_returns).reshape(self.shape)

    def _lookup(self, xyz):
        """Solid flag per point; points outside the grid are open (escaped)."""
        idx = np.floor((xyz - self.lo) / self.voxel).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        out = np.zeros(xyz.shape[:-1], bool)
        safe = np.where(inside[..., None], idx, 0)
        out[inside] = self.solid[safe[..., 0], safe[..., 1], safe[..., 2]][inside]
        return out, inside

    def free_distance(self, origins, directions, skip_m=.1, max_m=30.):
        """Distance to the first solid voxel along each ray (inf when it escapes)."""
        steps = np.arange(skip_m, max_m, self.voxel)
        xyz = origins[:, None, :] + directions[:, None, :] * steps[None, :, None]
        solid, inside = self._lookup(xyz)
        # Once a ray leaves the scanned volume it has escaped; ignore later re-entry.
        escaped = np.cumsum(~inside, axis=1) > 0
        hit = solid & ~escaped
        first = np.argmax(hit, axis=1)
        distance = np.where(hit.any(axis=1), steps[first], np.inf)
        return distance

    def open_to_sky(self, points, skip_m=.05):
        """True where a march straight up leaves the scanned volume unobstructed."""
        top = self.hi[2] + self.voxel
        steps = np.arange(skip_m, top - self.lo[2], self.voxel)
        xyz = points[:, None, :] + np.array([0, 0, 1.])[None, None, :] * steps[None, :, None]
        solid, inside = self._lookup(xyz)
        return ~(solid & inside).any(axis=1)


def sample_plane(plane, target=300, min_pitch=.05, max_pitch=.2):
    """Regular grid of in-plane samples (u, v) and their 3D positions.

    A grid rather than random points, so the outdoor samples can be turned
    back into a region of the plane (the mask used for partly exterior walls).
    """
    poly = plane['poly']
    if poly.is_empty or poly.area < 1e-6:
        return None, None, None
    pitch = float(np.clip(np.sqrt(poly.area / target), min_pitch, max_pitch))
    u0, v0, u1, v1 = poly.bounds
    us = np.arange(u0 + pitch / 2, u1, pitch)
    vs = np.arange(v0 + pitch / 2, v1, pitch)
    if not len(us) or not len(vs):
        return None, None, None
    uv = np.stack(np.meshgrid(us, vs, indexing='ij'), -1).reshape(-1, 2)
    inside = shapely.contains_xy(poly, uv[:, 0], uv[:, 1])
    uv = uv[inside]
    if len(uv) < 4:
        return None, None, None
    xyz = plane['origin'] + uv[:, :1] * plane['u'] + uv[:, 1:] * plane['v']
    return uv, xyz, pitch


def outdoor_mask(uv, outdoor, pitch):
    """Region of the plane (u, v) around the samples that look at open air."""
    if not outdoor.any():
        return None
    cells = shapely.box(uv[outdoor, 0] - pitch, uv[outdoor, 1] - pitch, uv[outdoor, 0] + pitch, uv[outdoor, 1] + pitch)
    region = shapely.union_all(cells).buffer(pitch, join_style=2).buffer(-pitch, join_style=2)
    return region.simplify(pitch / 4, preserve_topology=True)


def classify_plane(plane, occupancy, reach=4., first_offset=.35, step=.25):
    """Fraction of samples that look at open air on either side, plus the mask of those samples."""
    uv, points, pitch = sample_plane(plane)
    if points is None:
        return None, None
    normal = np.asarray(plane['normal'], float)
    normal = normal / np.linalg.norm(normal)
    offsets = np.arange(first_offset, reach + 1e-9, step)
    outdoor = {}
    for sign, label in ((1., 'positive'), (-1., 'negative')):
        direction = np.repeat((sign * normal)[None, :], len(points), axis=0)
        free = occupancy.free_distance(points, direction)
        probes = points[:, None, :] + direction[:, None, :] * offsets[None, :, None]
        sky = occupancy.open_to_sky(probes.reshape(-1, 3)).reshape(len(points), len(offsets))
        along_free_path = offsets[None, :] < free[:, None]
        outdoor[label] = (sky & along_free_path).any(axis=1)
    either = outdoor['positive'] | outdoor['negative']
    row = {'samples': int(len(points)), 'sample_pitch_m': pitch, 'open_air_fraction': float(either.mean()),
           'positive_side_fraction': float(outdoor['positive'].mean()),
           'negative_side_fraction': float(outdoor['negative'].mean())}
    return row, outdoor_mask(uv, either, pitch)


def plane_key(plane):
    return [float(x) for x in plane['key']]


def rings(region):
    return [[list(map(float, xy)) for xy in np.asarray(p.exterior.coords)] for p in polygon_parts(region)]


def select(parts, occupancy, exterior_fraction=.6, partial_fraction=.2, min_area=.05):
    """Plane-level exterior selection and an audit.

    Returns ``{'walls': {part name: [{'key', 'fraction', 'mask_uv'}, ...]}}``.
    A plane mostly open to the air is an exterior face and is selected whole
    (``mask_uv`` is null). A plane open only along part of its length, such as
    a long wall that leaves the house and continues as a compound wall, is a
    partly exterior face and is selected with the in-plane mask of the open
    samples. A plane almost never open is an interior face.
    """
    selection, audit = {}, []
    for part in parts:
        if not part.get('kind', '').startswith(('wall', 'parapet')):
            continue
        for plane in planes_of(part):
            if abs(plane['normal'][2]) > .01 or plane['poly'].area < min_area:
                continue
            row, mask = classify_plane(plane, occupancy)
            if row is None:
                continue
            fraction = row['open_air_fraction']
            if fraction >= exterior_fraction:
                classification = 'exterior_face'
                selection.setdefault(part['name'], []).append({'key': plane_key(plane), 'fraction': fraction, 'mask_uv': None})
            elif fraction > partial_fraction and mask is not None:
                classification = 'partly_exterior_face'
                selection.setdefault(part['name'], []).append({'key': plane_key(plane), 'fraction': fraction,
                                                               'mask_uv': rings(mask)})
            else:
                classification = 'interior_face'
            audit.append({'wall': part['name'], 'level': part.get('level', 0), 'plane_key': plane_key(plane),
                          'area_m2': float(plane['poly'].area), 'classification': classification,
                          'open_area_m2': float(mask.intersection(plane['poly']).area) if mask is not None else 0., **row})
    return {'walls': selection}, audit


def load_returns(cache):
    return np.load(cache, allow_pickle=True)['p']


def summary(selection, audit):
    return {'exterior_faces': sum(r['classification'] == 'exterior_face' for r in audit),
            'partly_exterior_faces': sum(r['classification'] == 'partly_exterior_face' for r in audit),
            'interior_faces': sum(r['classification'] == 'interior_face' for r in audit),
            'selected_walls': len(selection['walls']),
            'exterior_area_m2': round(sum(r['area_m2'] for r in audit if r['classification'] == 'exterior_face'), 2),
            'partly_exterior_open_area_m2': round(sum(r['open_area_m2'] for r in audit
                                                      if r['classification'] == 'partly_exterior_face'), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--evidence-cache', required=True, help='raw_evidence.py npz of the same property frame')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    parts = json.loads(Path(args.model).read_text())['parts']
    occupancy = Occupancy(load_returns(args.evidence_cache))
    selection, audit = select(parts, occupancy)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    (out / 'selection_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(summary(selection, audit)))


if __name__ == '__main__':
    main()
