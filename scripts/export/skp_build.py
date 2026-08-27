"""Hand a model to SketchUp as numbers, and have it build the parts itself.

SketchUp's Collada importer flattens a 74-node scene into one component of
loose edges and faces -- the named parts, which are the point of the handover,
are gone. So nothing is imported: the vertices and triangles go over as JSON
and a Ruby builder makes one named Sketchup::Group per part.
"""
import json, sys, argparse, colorsys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from to_sketchup import read_groups
from skp_client import rb

KIND_COLOUR = {"wall": (205, 197, 183), "column": (176, 160, 150),
               "floor": (168, 168, 160), "ceiling": (196, 196, 190),
               "dropped_ceiling": (186, 190, 196), "beam": (170, 160, 150),
               "parapet": (200, 192, 178)}


def kind_of(name):
    return name.rsplit("_", 1)[0] if "_" in name else name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", required=True, help="a per-part OBJ (boxes.obj)")
    ap.add_argument("--skp", required=True, help="where to save the .skp")
    ap.add_argument("--json", default=None, help="where to stage the numbers")
    a = ap.parse_args()

    parts = read_groups(a.obj)
    out = {"parts": []}
    ntri = 0
    for name, (V, F) in parts.items():
        k = kind_of(name)
        out["parts"].append(dict(name=name, kind=k,
                                 colour=list(KIND_COLOUR.get(k, (190, 190, 190))),
                                 v=np.asarray(V, float).round(6).tolist(),
                                 f=np.asarray(F, int).tolist()))
        ntri += len(F)
    js = Path(a.json or (Path(a.skp).with_suffix(".build.json")))
    js.write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(parts)} parts, {ntri:,} triangles staged in {js.name} "
          f"({js.stat().st_size/1e6:.1f} MB)")

    code = (f'load "pcm_build.rb"; PCMBuild.build('
            f'{json.dumps(str(js.resolve()).replace(chr(92), "/"))}, '
            f'{json.dumps(str(Path(a.skp).resolve()).replace(chr(92), "/"))})')
    r = rb(code, timeout=1800)
    if not r.get("ok"):
        raise SystemExit(f"SketchUp said: {r.get('error')}\n{r.get('backtrace')}")
    print(r["result"].strip('"').replace('\\"', '"'))


if __name__ == "__main__":
    main()
