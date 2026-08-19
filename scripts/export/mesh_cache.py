"""Load a Poisson mesh once, put it in the Z-up aligned frame, and cache it.

The Poisson mesh is 750 MB of ASCII OBJ. Parsing it takes minutes, and every
experiment on top of it would pay that again. This caches the four arrays every
later stage actually uses -- vertices, triangles, triangle normals, triangle
centroids -- in one npz that loads in seconds.

Frame: the Poisson export is Y-up. The rest of the flow is Z-up, so the mesh is
rotated (x, y, z) -> (x, -z, y) when its thinnest extent is Y.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import open3d as o3d

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def build(src: str, cache: str):
    log(f"reading {src}")
    mesh = o3d.io.read_triangle_mesh(src)
    V = np.asarray(mesh.vertices)
    log(f"{len(V):,} verts, {len(mesh.triangles):,} tris")
    ext = V.max(0) - V.min(0)
    yup = int(np.argmin(ext)) == 1
    if yup:
        R = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
        mesh.rotate(R, center=(0, 0, 0))
        log(f"rotated Y-up -> Z-up (extent was {ext.round(2)})")
    mesh.compute_triangle_normals()
    V = np.asarray(mesh.vertices).astype(np.float32)
    T = np.asarray(mesh.triangles).astype(np.int32)
    N = np.asarray(mesh.triangle_normals).astype(np.float32)
    C = V[T].mean(axis=1).astype(np.float32)
    log(f"extent {(V.max(0)-V.min(0)).round(2)}  z {V[:,2].min():.2f}..{V[:,2].max():.2f}")
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, V=V, T=T, N=N, C=C)
    log(f"wrote {cache}")

def load(cache: str):
    d = np.load(cache)
    return d["V"], d["T"], d["N"], d["C"]

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
