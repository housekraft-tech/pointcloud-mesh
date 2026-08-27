"""Rebuild each plane as a polygon clipped against its neighbours.

Projection can only take the rolled edges down to about 5%: a vertex whose
faces straddle a fillet has no single correct plane to go to, and moving it
leaves the rounded strip's CONNECTIVITY intact, so the dihedral stays soft.

The fix is to stop moving the band and start deleting it. Every face already
carries a plane label (from sharp_rebuild's segmentation), so each plane's
region is a patch with a boundary. Where two patches meet, the true edge is
the line where their planes intersect -- not anywhere on either patch. So each
plane is rebuilt in 2-D: its boundary is projected into the plane, every run
of boundary that borders another plane is replaced by that pair's intersection
line, the result is triangulated, and lifted back.

Three things keep it honest:

  * the clip is WINDOWED to the stretch where the two patches are genuinely
    adjacent. Clipping by a neighbour's infinite half-plane threw away 60% of
    the surface, because a perpendicular wall's intersection line runs across
    the middle of the floor and takes half the floor with it;
  * a clip that would eat more than --max-cut of the patch is refused, because
    that means the window was wrong, not that the patch was wrong;
  * anything the rebuild does NOT produce -- unlabelled faces, planes below
    the area floor, patches whose triangulation failed -- is carried over as
    its original triangles, so the output covers the whole scan.

Corners are then welded: output vertices within --weld of each other are one
corner, and it is placed where its member planes intersect (least-squares,
with a near-parallel guard, because a thin partition meeting a ceiling gives a
system whose solution is in the next building). After welding the edges are
exactly shared, which is both what a sharp edge means and what SketchUp most
wants to be given.
"""
import argparse
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union


def frame(n):
    """An orthonormal 2-D frame on a plane with normal n."""
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    return u, v


def plane_line(n1, d1, n2, d2):
    """Where two planes meet: a point on the line and its direction."""
    dirv = np.cross(n1, n2)
    L = np.linalg.norm(dirv)
    if L < 1e-9:
        return None
    dirv = dirv / L
    A = np.vstack([n1, n2, dirv])
    b = np.array([d1, d2, 0.0])
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p, dirv


def weld(out_v, out_p, PN, PD, tol):
    """Snap coincident output vertices onto one point: the corner of their planes."""
    P = np.asarray(out_v, float)
    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(tol, output_type="ndarray"):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    root = np.array([find(i) for i in range(len(P))])

    groups = defaultdict(list)
    for i, r in enumerate(root):
        groups[r].append(i)
    moved = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        pls = sorted({out_p[i] for i in members if out_p[i] >= 0})
        c = P[members].mean(0)
        if len(pls) >= 2:
            Nn = np.array([PN[p] for p in pls])
            dd = np.array([PD[p] for p in pls])
            # drop near-duplicate planes: they add no constraint but wreck the fit
            keep = [0]
            for k in range(1, len(Nn)):
                if all(abs(Nn[k] @ Nn[j]) < 0.985 for j in keep):
                    keep.append(k)
            Nn, dd = Nn[keep], dd[keep]
            lhs = np.vstack([np.eye(3) * 1e-2, Nn])
            rhs = np.concatenate([c * 1e-2, dd])
            c, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        for i in members:
            if np.linalg.norm(P[i] - c) > 1e-9:
                moved += 1
            P[i] = c
    return P, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="npz from sharp_rebuild --save-labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reach", type=float, default=0.08,
                    help="m: an intersection line further than this from both "
                         "patches means the patches do not actually meet")
    ap.add_argument("--min-area", type=float, default=0.004, help="m2")
    ap.add_argument("--simplify", type=float, default=0.004, help="m")
    ap.add_argument("--pad", type=float, default=0.05,
                    help="m: how far past the contact stretch the clip may reach")
    ap.add_argument("--bite", type=float, default=0.05,
                    help="m: how wide the fillet band at a junction is -- only "
                         "material this close to the intersection line is fake")
    ap.add_argument("--grow", type=float, default=0.15,
                    help="m: how far a patch may be extended to reach the seam")
    ap.add_argument("--max-cut", type=float, default=0.35,
                    help="a clip removing more than this fraction of the patch "
                         "is refused: the window was wrong, not the patch")
    ap.add_argument("--weld", type=float, default=0.006, help="m: corner welding radius")
    ap.add_argument("--no-carry", action="store_true",
                    help="drop what the rebuild misses instead of carrying it over")
    a = ap.parse_args()

    z = np.load(a.labels)
    V, F, flab = z["V"], z["F"], z["flab"]
    PN, PD = z["plane_n"], z["plane_d"]
    print(f"{len(F):,} faces, {len(PN):,} planes, "
          f"{100*(flab >= 0).mean():.1f}% labelled")

    tm = trimesh.Trimesh(V, F, process=False)
    A = tm.area_faces
    total = float(A.sum())

    # Which planes touch, AND where along the shared boundary they touch.
    adj = defaultdict(set)
    contact = defaultdict(list)
    for (x, y), e in zip(tm.face_adjacency, tm.face_adjacency_edges):
        if flab[x] >= 0 and flab[y] >= 0 and flab[x] != flab[y]:
            px, py = int(flab[x]), int(flab[y])
            adj[px].add(py)
            adj[py].add(px)
            contact[(px, py)].append(e)
            contact[(py, px)].append(e)

    pa = np.zeros(len(PN))
    for p in range(len(PN)):
        pa[p] = A[flab == p].sum()
    keep = [p for p in range(len(PN)) if pa[p] >= a.min_area]
    print(f"{len(keep):,} planes carry more than {a.min_area} m2 "
          f"({100*pa[keep].sum()/total:.1f}% of area)")

    out_v, out_f, out_p = [], [], []
    done = np.zeros(len(F), bool)       # faces the rebuild actually represents
    stats = defaultdict(int)
    for p in keep:
        sel = np.flatnonzero(flab == p)
        if not len(sel):
            continue
        n, d = PN[p], PD[p]
        u, v = frame(n)
        polys = []
        for f in sel:
            t = V[F[f]]
            polys.append(Polygon([(t[i] @ u, t[i] @ v) for i in range(3)]))
        polys = [q for q in polys if q.is_valid and q.area > 1e-9]
        if not polys:
            continue
        patch = unary_union(polys).buffer(a.simplify).buffer(-a.simplify)
        if patch.is_empty:
            continue
        for q in adj[p]:
            ln = plane_line(n, d, PN[q], PD[q])
            if ln is None:
                stats["parallel"] += 1
                continue
            pt, dirv = ln
            p2 = np.array([pt @ u, pt @ v])
            d2 = np.array([dirv @ u, dirv @ v])
            L = np.linalg.norm(d2)
            if L < 1e-9:
                continue
            d2 = d2 / L
            nrm2 = np.array([-d2[1], d2[0]])
            cen = np.array(patch.representative_point().coords[0])
            s = np.sign((cen - p2) @ nrm2)
            if s == 0:
                continue
            ee = contact.get((p, q))
            if not ee:
                continue
            pts = V[np.unique(np.array(ee).ravel())]
            t = (pts - pt) @ dirv
            t0, t1 = float(t.min()) - a.pad, float(t.max()) + a.pad
            far = 1e3
            base0, base1 = p2 + d2*t0, p2 + d2*t1
            half = Polygon([base0, base1, base1 + nrm2*s*far, base0 + nrm2*s*far])
            if patch.distance(half) > a.reach:
                stats["out_of_reach"] += 1
                continue
            # The half-plane must be windowed the OTHER way too. A floor meeting
            # an interior partition has floor on both sides of the line, and an
            # unbounded half-plane deletes a full-width strip of the room next
            # door -- which is how the floor lost 70 of its 90 m2. Only the
            # fillet, within --bite of the line, is not real surface.
            strip = Polygon([base0 - nrm2*a.bite, base1 - nrm2*a.bite,
                             base1 + nrm2*a.bite, base0 + nrm2*a.bite])
            cut = strip.difference(half)                    # the wrong side of it
            # reach ACROSS the fillet, not just its width: at a floor the
            # Poisson blend can be 100 mm deep, and a patch that stops short of
            # the line leaves exactly the gap you see standing in the room.
            grown = strip.intersection(half).intersection(patch.buffer(a.grow))
            clipped = unary_union([patch.difference(cut), grown])
            if clipped.is_empty or clipped.area < (1.0 - a.max_cut) * patch.area:
                stats["clip_refused"] += 1
                continue
            patch = clipped
            stats["clipped"] += 1
        emitted = False
        for g in list(getattr(patch, "geoms", [patch])):
            if g.is_empty or g.area < 1e-6:
                continue
            g = g.simplify(a.simplify)
            if g.is_empty or not g.is_valid or g.area < 1e-6:
                continue
            vv = None
            for eng in ("earcut", "triangle", None):
                try:
                    vv, ff = (trimesh.creation.triangulate_polygon(g, engine=eng)
                              if eng else trimesh.creation.triangulate_polygon(g))
                    break
                except Exception as exc:
                    stats[f"tri_fail_{eng}"] += 1
                    last = exc
            if vv is None:
                stats["triangulate_failed"] += 1
                continue
            base = len(out_v)
            for x, y in vv:
                out_v.append(x*u + y*v + d*n)
                out_p.append(p)
            out_f.extend((np.array(ff) + base).tolist())
            stats["polys"] += 1
            emitted = True
        if emitted:
            done[sel] = True

    miss = np.flatnonzero(~done)
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"  rebuilt covers {100*A[done].sum()/total:.1f}% of the scan area; "
          f"{len(miss):,} faces ({100*A[miss].sum()/total:.1f}%) not represented")

    if len(miss) and not a.no_carry:
        # carried over verbatim, tagged so welding treats them as plane-less
        for f in miss:
            base = len(out_v)
            for vi in F[f]:
                out_v.append(V[vi])
                out_p.append(-1)
            out_f.append([base, base+1, base+2])
        print(f"  carried {len(miss):,} original faces over")

    P, moved = weld(out_v, out_p, PN, PD, a.weld)
    print(f"  welded: {moved:,} of {len(P):,} vertices snapped to a shared corner")

    out = trimesh.Trimesh(P, np.array(out_f), process=False)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"out: {len(out.faces):,} triangles, {len(out.vertices):,} vertices, "
          f"{out.area:.0f} m2 -> {a.out}")


if __name__ == "__main__":
    main()
