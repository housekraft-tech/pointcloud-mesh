"""Turn the mesh cache back into a file other software can open.

The .npz beside each model is a NumPy archive: vertices, triangles, normals and
colour in the layout this pipeline loads fastest. It is not a 3D format, and
nothing else reads it -- not Blender, not MeshLab, not CloudCompare. When the
mesh itself is wanted, it comes back out through here.

  .ply   binary, with per-vertex colour. Opens in MeshLab, CloudCompare,
         Blender, ~half the size of the same mesh as ASCII OBJ
  .glb   binary glTF, if a browser or a quick look is what is wanted; pass
         --decimate to make one small enough to open comfortably
"""
import sys, time, argparse
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("out", help=".ply or .glb")
    ap.add_argument("--decimate", type=int, default=0,
                    help="target triangle count (0 keeps every triangle)")
    a = ap.parse_args()
    import open3d as o3d

    d = np.load(a.npz)
    V = d["V"].astype(np.float64); T = d["T"].astype(np.int32)
    log(f"{len(V):,} vertices, {len(T):,} triangles"
        + (", with colour" if "RGB" in d.files else ""))
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(T)
    if "RGB" in d.files:
        m.vertex_colors = o3d.utility.Vector3dVector(d["RGB"].astype(np.float64)/255.0)
    if a.decimate and a.decimate < len(T):
        log(f"decimating to {a.decimate:,}")
        m = m.simplify_quadric_decimation(a.decimate)
    m.compute_vertex_normals()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".glb":
        import trimesh
        tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                             vertex_normals=np.asarray(m.vertex_normals),
                             process=False)
        if m.has_vertex_colors():
            c = np.asarray(m.vertex_colors)
            rgba = np.zeros((len(c), 4), np.uint8)
            rgba[:, :3] = np.clip(c*255, 0, 255).astype(np.uint8); rgba[:, 3] = 255
            tm.visual.vertex_colors = rgba
        tm.export(str(out))
    else:
        o3d.io.write_triangle_mesh(str(out), m, write_ascii=False,
                                   compressed=False)
    log(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB, "
        f"{len(np.asarray(m.triangles)):,} triangles)")


if __name__ == "__main__":
    main()
