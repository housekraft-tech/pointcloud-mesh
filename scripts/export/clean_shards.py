"""Throw away the confetti the rebuild leaves behind, and nothing else.

The clipped-plane rebuild is accurate -- both directions agree to about 3 mm --
but it is 5,755 disconnected pieces, and most of them are junk: a scan's outer
fringe is a ragged skirt of half-seen surface, and every one of those scraps
becomes its own little polygon. Seen from a low angle the model looks shredded
even though every surface in it is in the right place.

Two things mark a scrap, and a piece has to fail neither to survive:

  * AREA. A piece under --min-area is smaller than anything a building is made
    of. This catches the speckle.
  * WIDTH, as 2*area/perimeter -- roughly the half-width of the strip. This is
    the one that matters: the fringe scraps are LONG, so they pass an area
    test, but they are a couple of centimetres wide. A wall is not.

Both are deliberately conservative. Dropping surface is a real cost: it is
better to leave a scrap standing than to shave a genuine reveal, so the
defaults keep anything that could plausibly be built.
"""
import argparse

import numpy as np
import trimesh


def pieces(m):
    return trimesh.graph.connected_components(m.face_adjacency,
                                              nodes=np.arange(len(m.faces)))


def boundary_length(m, faces):
    """Total length of the edges this piece uses only once: its outline."""
    e = np.sort(m.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    b = uniq[cnt == 1]
    if not len(b):
        return 0.0
    return float(np.linalg.norm(m.vertices[b[:, 0]] - m.vertices[b[:, 1]],
                                axis=1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-area", type=float, default=0.02,
                    help="m2: a piece smaller than this is speckle")
    ap.add_argument("--min-width", type=float, default=0.025,
                    help="m: 2*area/perimeter -- a piece thinner than this is a "
                         "fringe scrap, however long it is")
    ap.add_argument("--keep-above", type=float, default=1.0,
                    help="m2: a piece this big is kept whatever shape it is")
    a = ap.parse_args()

    m = trimesh.load(a.inp, force="mesh")
    m.merge_vertices()
    A = m.area_faces
    cc = pieces(m)
    print(f"in: {len(m.faces):,} tris, {m.area:.0f} m2, {len(cc):,} pieces")

    keep, drop_small, drop_thin = [], 0.0, 0.0
    for c in cc:
        ar = float(A[c].sum())
        if ar >= a.keep_above:
            keep.append(c)
            continue
        if ar < a.min_area:
            drop_small += ar
            continue
        L = boundary_length(m, c)
        w = 2 * ar / L if L > 0 else 0.0
        if w < a.min_width:
            drop_thin += ar
            continue
        keep.append(c)

    sel = np.concatenate(keep) if keep else np.array([], int)
    out = m.submesh([sel], append=True)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    out.export(a.out)
    print(f"  dropped {drop_small:.1f} m2 of speckle under {a.min_area} m2, "
          f"{drop_thin:.1f} m2 of strips under {a.min_width*1000:.0f} mm wide")
    print(f"out: {len(out.faces):,} tris, {out.area:.0f} m2 "
          f"({100*out.area/m.area:.1f}% kept), {len(keep):,} pieces -> {a.out}")


if __name__ == "__main__":
    main()
