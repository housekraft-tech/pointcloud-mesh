"""Look straight at one wall part, and at its depth.

This is the check that the relief survived the segmentation. A shaded view
flatters a flat wall; the depth view does not -- it maps distance from the
wall's own plane to colour, so a niche, a boxed conduit, a reveal and an arch
soffit are visible as shape rather than as shading.
"""
import sys, json
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/model/poisson_modular"
CACHE = sys.argv[2] if len(sys.argv) > 2 else "output/model/poisson_koushik.npz"
WANT = sys.argv[3:] or None

T = np.load(CACHE)["T"].astype(np.int64)
V = np.load(f"{OUT}/verts.npy").astype(np.float64)
L = np.load(f"{OUT}/labels.npy")
names = json.load(open(f"{OUT}/names.json"))
man = json.load(open(f"{OUT}/manifest.json"))
C = V[T].mean(axis=1)
zf = man["floor_z"]
parts = {p["name"]: p for p in man["parts"]}
feats = man["features"]

walls = [p for p in man["parts"] if p["kind"] in ("wall", "parapet")]
walls.sort(key=lambda p: -p["area_m2"])
pick = [w["name"] for w in walls if not WANT or w["name"] in WANT][:6]

CELL = 0.02
fig, axes = plt.subplots(len(pick), 1, figsize=(13, 2.6*len(pick)))
if len(pick) == 1:
    axes = [axes]
for ax, nm in zip(axes, pick):
    pid = names.index(nm)
    sel = np.where(L == pid)[0]
    p = C[sel]
    a = 0 if parts[nm]["axis"] == "x" else 1        # axis across the wall
    al = p[:, 1-a]; cc = p[:, a]; z = p[:, 2]
    # the near face is the densest coordinate across the wall
    h, e = np.histogram(cc, bins=np.arange(cc.min(), cc.max()+0.005, 0.005))
    face = 0.5*(e[:-1]+e[1:])[np.argmax(h)]
    keep = np.abs(cc-face) < 0.30
    al, cc, z = al[keep], cc[keep], z[keep]
    if al.size < 50:
        ax.set_title(f"{nm}: too little surface to unfold", fontsize=8)
        ax.axis("off")
        continue
    nu = max(2, int(np.ceil(np.ptp(al)/CELL))); nv = max(2, int(np.ceil(np.ptp(z)/CELL)))
    iu = np.clip(((al-al.min())/CELL).astype(int), 0, nu-1)
    iv = np.clip(((z-z.min())/CELL).astype(int), 0, nv-1)
    f = iu*nv+iv
    cnt = np.bincount(f, minlength=nu*nv).reshape(nu, nv)
    dsum = np.bincount(f, weights=(cc-face), minlength=nu*nv).reshape(nu, nv)
    d = np.full((nu, nv), np.nan)
    m = cnt > 0
    d[m] = dsum[m]/cnt[m]
    lim = np.nanpercentile(np.abs(d), 97) or 0.05
    im = ax.imshow(d.T*1000, origin="lower", cmap="RdBu_r", vmin=-lim*1000, vmax=lim*1000,
                   extent=[0, nu*CELL, (z.min()-zf), (z.min()-zf)+nv*CELL], aspect="equal")
    ax.set_title(f"{nm}  {parts[nm]['length_mm']} mm long, "
                 f"thickness {parts[nm]['thickness_mm'] or 'one face only'}  "
                 + ", ".join(f"{f['kind']} {f['width_mm']}x{f['height_mm']}"
                             for f in feats if f.get("wall") == nm)[:110],
                 fontsize=8)
    ax.set_ylabel("m over floor", fontsize=7)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, shrink=.8, label="mm off face")
plt.tight_layout()
plt.savefig(f"{OUT}/wall_elevations.png", dpi=130)
print("wrote", f"{OUT}/wall_elevations.png")
