"""One object per slab per level, instead of one per plateau.

The measurement grows a plateau from a seed and stops where the surface steps,
so a floor poured in one piece comes back as a part per room -- 89 objects on
this storey, sitting at a handful of levels. Nothing about that is wrong, and
all of it is tedious: a designer wants "the structural ceiling", not nineteen
pieces of it.

Slabs of the same kind at the same level are unioned. The union moves no
coordinate: what was measured is exactly what is written out, just fewer
objects to click.
"""
import argparse, json, collections, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from skp_polygons import box


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tagged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, default=0.05, help="m: same level within this")
    a = ap.parse_args()
    d = json.load(open(a.tagged))
    groups = collections.defaultdict(list)
    for b in d["boxes"]:
        groups[(b["tag"], round(b["lo"][2]/a.band)*a.band)].append(b)
    out, before = [], len(d["boxes"])
    for (tag, z), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(items) == 1:
            b = items[0]
            out.append(dict(name=b["name"], tag=tag, lo=b["lo"], hi=b["hi"], pieces=1))
            continue
        solids = [box(b["lo"], b["hi"]) for b in items]
        try:
            m = trimesh.boolean.union(solids, engine="manifold")
        except Exception:
            m = trimesh.util.concatenate(solids)
        out.append(dict(name=f"{tag}_{z:+.2f}m".replace(" ", ""), tag=tag,
                        mesh=[np.asarray(m.vertices).round(5).tolist(),
                              np.asarray(m.faces).tolist()],
                        pieces=len(items)))
    print(f"{before} objects -> {len(out)}")
    for o in sorted(out, key=lambda o: -o["pieces"])[:8]:
        print(f"   {o['name']:34} {o['pieces']:2d} pieces merged")
    json.dump(dict(faces=d["faces"], objects=out), open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
