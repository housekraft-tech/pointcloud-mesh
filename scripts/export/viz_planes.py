"""Look at what the point segmentation actually did, before building on it.

Numbers hide the two failures that matter here. A run can report a beautiful
3.8 mm RMS while having found 103 horizontal planes and 7 vertical ones -- the
fits are excellent and the segmentation is nonsense. And a coverage figure of
"4 M points on planes" says nothing about WHERE the other 28 M are: evenly
spread as clutter is fine, all in one room is a bug.

So: a plan and two elevations with every plane in its own colour, the points
that landed on no plane drawn in grey underneath, and the distributions of
orientation, size and fit quality beside them.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scatter(ax, pts, lab, unclaimed, i, j, title, s=0.4):
    if len(unclaimed):
        ax.scatter(unclaimed[:, i], unclaimed[:, j], s=s*0.6, c="#d9d9d9",
                   linewidths=0, rasterized=True)
    rng = np.random.default_rng(7)
    col = rng.permutation(np.unique(lab))
    cmap = plt.get_cmap("tab20")
    order = {p: k for k, p in enumerate(col)}
    c = np.array([cmap(order[x] % 20) for x in lab])
    ax.scatter(pts[:, i], pts[:, j], s=s, c=c, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planes", required=True, help="npz from plane_points")
    ap.add_argument("--out", required=True, help="folder")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.planes)
    n, d, rms, cnt = z["n"], z["d"], z["rms"], z["count"]
    pts, lab = z["pts"], z["lab"]
    allp = z["all_pts"]

    # which downsampled points landed on nothing
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dd, _ = tree.query(allp, distance_upper_bound=0.02, workers=-1)
    unclaimed = allp[~np.isfinite(dd)]

    nz = np.abs(n[:, 2])
    kind = np.where(nz > 0.9, "horizontal",
                    np.where(nz < 0.2, "vertical", "sloped"))
    print(f"{len(n):,} planes: "
          f"{(kind=='vertical').sum()} vertical, "
          f"{(kind=='horizontal').sum()} horizontal, "
          f"{(kind=='sloped').sum()} sloped")
    print(f"{len(pts):,} segmented points, {len(unclaimed):,} on no plane "
          f"({100*len(unclaimed)/max(len(allp),1):.0f}% of the cloud)")

    fig = plt.figure(figsize=(17, 10))
    ax = fig.add_subplot(2, 3, 1)
    scatter(ax, pts, lab, unclaimed, 0, 1, "plan  (grey = on no plane)")
    ax = fig.add_subplot(2, 3, 2)
    scatter(ax, pts, lab, unclaimed, 0, 2, "elevation, looking along +Y")
    ax = fig.add_subplot(2, 3, 3)
    scatter(ax, pts, lab, unclaimed, 1, 2, "elevation, looking along +X")

    ax = fig.add_subplot(2, 3, 4)
    ax.bar(["vertical", "horizontal", "sloped"],
           [(kind == k).sum() for k in ("vertical", "horizontal", "sloped")],
           color="#4a86c4")
    ax.set_title("planes by orientation", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(2, 3, 5)
    ax.hist(rms*1000, bins=np.arange(0, 20.5, 1), color="#4a86c4")
    ax.axvline(np.median(rms)*1000, color="#c44", lw=1.5)
    ax.set_title(f"fit RMS against the points (mm)\nmedian "
                 f"{np.median(rms)*1000:.2f} mm", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(2, 3, 6)
    ax.hist(np.log10(np.maximum(cnt, 1)), bins=30, color="#4a86c4")
    ax.set_title(f"points per plane (log10)\n{cnt.sum():,} on planes",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out / "segmentation.png", dpi=120)
    print(f"-> {out/'segmentation.png'}")

    # the offsets, which is what the model will actually be built from
    fig2, axs = plt.subplots(1, 3, figsize=(15, 3.4))
    for ax, (k, ttl) in zip(axs, [("vertical", "wall offsets"),
                                  ("horizontal", "floor/ceiling heights"),
                                  ("sloped", "sloped")]):
        s = kind == k
        if s.sum():
            ax.scatter(d[s], cnt[s], s=18, c="#4a86c4")
            ax.set_yscale("log")
        ax.set_title(f"{ttl}  ({s.sum()})", fontsize=10)
        ax.set_xlabel("offset along the normal (m)")
        ax.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out / "offsets.png", dpi=120)
    print(f"-> {out/'offsets.png'}")


if __name__ == "__main__":
    main()
