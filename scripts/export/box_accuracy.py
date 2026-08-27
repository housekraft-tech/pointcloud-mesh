"""How far is the box model from what was scanned -- both ways.

One direction is not enough. Distance from the scan to the boxes says whether
anything measured was left out; distance from the boxes to the scan says
whether anything was invented. A model of paper-thin sheets scores perfectly on
the first and terribly on the second, and a model that fills the building with
solid mass does the opposite.

Reported per kind, and per part so the worst offenders can be named.
"""
import sys, json, argparse
import numpy as np, trimesh, open3d as o3d

KINDS = {"wall": ("wall", "column", "parapet"),
         "slab": ("floor", "ceiling", "dropped_ceiling", "beam")}


def scene(m):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(m.vertices), o3d.core.float32),
        o3d.core.Tensor(np.asarray(m.faces), o3d.core.uint32)))
    return s


def dist(sc, P):
    return sc.compute_distance(o3d.core.Tensor(np.asarray(P, np.float32))).numpy()*1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="the modular directory")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--boxes", required=True, help="boxes_union.glb")
    ap.add_argument("--parts", type=int, default=10, help="how many worst parts to name")
    a = ap.parse_args()

    man = json.load(open(f"{a.dir}/manifest.json"))
    zf = man["floor_z"]; zc = zf + man["modal_ceiling_height_mm"]/1000.0
    V = np.load(f"{a.dir}/verts.npy"); T = np.load(a.cache)["T"]
    lab = np.load(f"{a.dir}/labels.npy"); names = json.load(open(f"{a.dir}/names.json"))
    box = trimesh.load(a.boxes); box = box.to_mesh() if hasattr(box, "to_mesh") else box
    scan = trimesh.Trimesh(V, T[lab >= 0], process=False)
    sb, ss = scene(box), scene(scan)

    band = lambda P: P[(P[:, 2] > zf-0.7) & (P[:, 2] < zc+0.25)]
    for k, pref in KINDS.items():
        keep = np.isin(lab, [i for i, n in enumerate(names) if n.startswith(pref)])
        if keep.sum() < 500:
            continue
        P = band(trimesh.sample.sample_surface(trimesh.Trimesh(V, T[keep], process=False), 150000)[0])
        d = dist(sb, P)
        print(f"  scan->box {k:5s}: median {np.median(d):5.1f}  90th {np.percentile(d,90):6.1f} "
              f"| <25mm {100*(d<25).mean():4.1f}%")
    Q = band(trimesh.sample.sample_surface(box, 150000)[0])
    d = dist(ss, Q)
    print(f"  box->scan      : median {np.median(d):5.1f}  90th {np.percentile(d,90):6.1f} "
          f"| <25mm {100*(d<25).mean():4.1f}%   {len(box.faces)} tris, {box.volume:.1f} m3, "
          f"watertight {box.is_watertight}")

    if a.parts:
        P = {p["name"]: p for p in man["parts"]}
        rows = []
        for i, n in enumerate(names):
            if (lab == i).sum() < 400:
                continue
            m = trimesh.Trimesh(V, T[lab == i], process=False)
            d = dist(sb, trimesh.sample.sample_surface(m, 3000)[0])
            rows.append((float(np.median(d)), m.area, n, P.get(n, {}).get("kind", "?")))
        rows.sort(key=lambda r: -r[0]*r[1])
        print(f"  worst parts (median error x area):")
        for md, ar, n, k in rows[:a.parts]:
            print(f"    {md:7.1f} mm over {ar:5.1f} m2   {n} ({k})")


if __name__ == "__main__":
    main()
