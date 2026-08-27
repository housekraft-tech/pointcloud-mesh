"""Give the Poisson surface its corners back.

Poisson fits a smooth indicator function, so every crease comes out filleted:
a wall meets a floor in a soft roll instead of an edge, and a door reveal loses
its arris. The geometry either side of the crease is right -- it is the crease
itself that is missing.

The building is rectilinear, so the fix needs no guessing. The planes the
surface actually lies on are found as the peaks of where its area sits along
each axis. Any vertex whose own faces point along an axis, and which sits
within TOL of one of that axis's planes, is moved onto it. Two flattened
surfaces meeting then produce a sharp edge by construction, and anything that
is genuinely curved is left alone because it has no plane to snap to.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import trimesh


def planes_along(V, F, A, N, axis, tol, min_area):
    """Where the area piles up along this axis -- those are the planes."""
    perp = np.abs(N[:, axis]) > 0.90
    if perp.sum() < 10:
        return np.array([])
    c = V[F[perp]].mean(axis=1)[:, axis]
    w = A[perp]
    lo, hi = c.min(), c.max()
    bins = np.arange(lo, hi + 0.005, 0.005)
    h, e = np.histogram(c, bins=bins, weights=w)
    mid = 0.5*(e[:-1] + e[1:])
    keep = []
    order = np.argsort(h)[::-1]
    for i in order:
        if h[i] < min_area:
            break
        if all(abs(mid[i] - k) > tol*2 for k in keep):
            keep.append(float(mid[i]))
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.020, help="m: snap radius")
    ap.add_argument("--min-area", type=float, default=0.25, help="m2 per 5 mm bin")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--crease", action="store_true",
                    help="collapse the remaining rolled band onto the plane "
                         "intersection, which is where the edge actually is")
    ap.add_argument("--general", type=int, default=0,
                    help="how many non-axis-aligned planes to find and snap to")
    ap.add_argument("--vote", type=float, default=0.25,
                    help="share of a vertex's area that must face an axis for it "
                         "to snap on that axis; 0.55 leaves junctions rounded")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    V = np.array(m.vertices, float); F = np.array(m.faces)
    A = m.area_faces; N = m.face_normals
    print(f"in: {len(F):,} triangles, {len(V):,} vertices")

    # Iterate. A vertex that snaps in pass one changes which plane its
    # neighbours are nearest to, and small surfaces -- a door reveal, the side
    # of a niche -- only declare themselves as planes once the noise around
    # them has settled. One pass sharpens the big junctions and leaves the
    # small ones rolled.
    moved_total = 0
    for it in range(a.passes):
      if a.passes > 1:
          print(f"  pass {it+1}")
      A = trimesh.Trimesh(V, F, process=False).area_faces
      N = trimesh.Trimesh(V, F, process=False).face_normals
      for axis in (0, 1, 2):
          pl = planes_along(V, F, A, N, axis, a.tol, a.min_area)
          if not len(pl):
              continue
          # a vertex belongs to this axis if its own faces face along it
          vote = np.zeros(len(V))
          w = np.abs(N[:, axis]) * A
          np.add.at(vote, F[:, 0], w); np.add.at(vote, F[:, 1], w); np.add.at(vote, F[:, 2], w)
          tot = np.zeros(len(V))
          np.add.at(tot, F[:, 0], A); np.add.at(tot, F[:, 1], A); np.add.at(tot, F[:, 2], A)
          # A vertex in a wall-to-ceiling fillet faces partly sideways and
          # partly up, so demanding that it face MOSTLY along one axis leaves
          # every junction rounded -- which is exactly what was left over. Each
          # axis is judged on its own: if a useful share of the vertex's area
          # faces this way and it is within reach of one of this axis's planes,
          # it snaps. A fillet vertex then snaps on both axes and the junction
          # comes to a corner.
          belongs = vote > a.vote*np.maximum(tot, 1e-12)
          idx = np.flatnonzero(belongs)
          if not len(idx):
              continue
          d = np.abs(V[idx, axis][:, None] - pl[None, :])
          j = np.argmin(d, axis=1)
          near = d[np.arange(len(idx)), j] < a.tol
          V[idx[near], axis] = pl[j[near]]
          moved_total += int(near.sum())
          print(f"   axis {'xyz'[axis]}: {len(pl)} planes, {int(near.sum()):,} vertices snapped")

    # Collapse what is left of each roll onto the crease itself.
    #
    # Snapping moves a vertex onto the nearest plane, which flattens the
    # surfaces but leaves the band between them: the vertices in the middle of
    # a fillet are too far from either plane to snap, so a narrow rolled strip
    # survives every junction. But such a vertex is surrounded by faces that
    # now lie on TWO planes, and the crease is exactly where those planes
    # meet. For axis-aligned planes that intersection is trivial -- fix both
    # coordinates and the vertex lands on the line; fix three and it lands on
    # the corner.
    if a.crease:
        for _ in range(2):
            tm = trimesh.Trimesh(V, F, process=False)
            n3 = tm.face_normals
            fax = np.argmax(np.abs(n3), axis=1)
            aligned = np.max(np.abs(n3), axis=1) > 0.999
            # which plane each aligned face sits on
            foff = np.take_along_axis(V[F[:, 0]], fax[:, None], axis=1).ravel()
            want = [dict() for _ in range(len(V))]
            for fi in np.flatnonzero(aligned):
                for vi in F[fi]:
                    want[vi].setdefault(int(fax[fi]), set()).add(round(float(foff[fi]), 4))
            moved = 0
            for vi, planes in enumerate(want):
                use = {ax: next(iter(v)) for ax, v in planes.items() if len(v) == 1}
                if len(use) < 2:
                    continue
                before = V[vi].copy()
                for ax, off in use.items():
                    if abs(V[vi][ax] - off) <= a.tol*1.5:
                        V[vi][ax] = off
                if not np.allclose(before, V[vi]):
                    moved += 1
            print(f"   crease: {moved:,} vertices pulled onto the line where "
                  f"their planes meet")
            if moved == 0:
                break

    # What is left is not axis-aligned: stair soffits, chamfers, angled
    # reveals, the underside of a flight. Those have planes too -- they just do
    # not line up with the building's axes -- so they are found by RANSAC on
    # the vertices that are still rolled, and snapped the same way. A junction
    # between a slanted soffit and a wall then comes to an edge as well.
    if a.general:
        import open3d as o3d
        tm = trimesh.Trimesh(V, F, process=False)
        n2 = tm.face_normals
        loose = np.flatnonzero(np.max(np.abs(n2), axis=1) <= 0.999)
        vids = np.unique(F[loose])
        print(f"   general: {len(vids):,} vertices still off-axis")
        pts = V[vids]
        remaining = np.arange(len(vids))
        found = 0
        for _ in range(a.general):
            if len(remaining) < 400:
                break
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[remaining]))
            try:
                model, inl = pc.segment_plane(distance_threshold=a.tol/2,
                                              ransac_n=3, num_iterations=400)
            except Exception:
                break
            if len(inl) < 400:
                break
            nrm = np.array(model[:3], float); off = model[3]
            nrm /= max(np.linalg.norm(nrm), 1e-12)
            sel = remaining[np.asarray(inl)]
            d = V[vids[sel]] @ nrm + off
            V[vids[sel]] -= np.outer(d, nrm)          # onto the plane
            remaining = np.setdiff1d(remaining, sel)
            found += 1
        print(f"   general: {found} extra planes, {len(vids)-len(remaining):,} vertices snapped")

    out = trimesh.Trimesh(V, F, process=False)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    d = np.linalg.norm(np.array(m.vertices)[:len(out.vertices)] * 0 , axis=1) if False else None
    out.export(a.out)
    print(f"out: {len(out.faces):,} triangles, {moved_total:,} vertex moves -> {a.out}")


if __name__ == "__main__":
    main()
