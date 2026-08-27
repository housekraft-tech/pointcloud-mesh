"""Push the traced plan up, cut the openings, and check it against the scan.

This is the tracer's model: one polygon per piece of masonry, extruded from
floor to ceiling in a single move, with the doors and windows cut afterwards
from what the survey measured. No per-run boxes, no pairing -- the plan is the
plan.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, trimesh, open3d as o3d
sys.path.insert(0, str(Path(__file__).parent))
from skp_polygons import box


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--schedule", required=True, help="for the openings")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--verts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.plan))
    z0, z1 = P["z0"], P["z1"]
    from shapely.geometry import Polygon
    solids = []
    for poly in P["plan"]:
        shp = Polygon(poly["outer"], poly["holes"])
        if not shp.is_valid:
            shp = shp.buffer(0)
        m = trimesh.creation.extrude_polygon(shp, z1 - z0)
        m.apply_translation([0, 0, z0])
        solids.append(m)
    walls = trimesh.boolean.union(solids, engine="manifold") if len(solids) > 1 else solids[0]
    print(f"plan extruded: {len(walls.faces):,} triangles, watertight={walls.is_watertight}, "
          f"{walls.volume:.2f} m3")

    S = json.load(open(a.schedule))["parts"]
    cuts = [box(r["lo"], r["hi"]) for r in S if r["op"] == "opening"]
    if cuts:
        walls = trimesh.boolean.difference([walls] + cuts, engine="manifold")
        print(f"{len(cuts)} openings cut -> {len(walls.faces):,} triangles, "
              f"{walls.volume:.2f} m3")
    walls.export(a.out)

    man = json.load(open(a.manifest)); zf = man["floor_z"]
    zc = zf + man["modal_ceiling_height_mm"]/1000
    T = np.load(a.cache)["T"]; V = np.load(a.verts)
    lab = np.load(Path(a.manifest).parent/"labels.npy")
    names = json.load(open(Path(a.manifest).parent/"names.json"))
    keep = np.isin(lab, [i for i, n in enumerate(names)
                         if n.startswith(("wall", "column", "parapet"))])
    scan = trimesh.Trimesh(V, T[keep], process=False)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(walls.vertices), o3d.core.float32),
        o3d.core.Tensor(np.asarray(walls.faces), o3d.core.uint32)))
    Q, _ = trimesh.sample.sample_surface(scan, 150000)
    Q = Q[(Q[:, 2] > zf-0.2) & (Q[:, 2] < zc+0.05)]
    d = sc.compute_distance(o3d.core.Tensor(np.asarray(Q, np.float32))).numpy()*1000
    print(f"scanned walls -> the traced model: median {np.median(d):.1f} mm, "
          f"90th {np.percentile(d, 90):.1f}, <25mm {100*(d < 25).mean():.1f}%")


if __name__ == "__main__":
    main()
