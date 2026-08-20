"""Run the whole flow over a multi-storey building, one storey at a time.

Every stage below is single-storey by design, so this is a loop, not a new
method: mesh, cache, segment, solidify, view, render -- per level, into
`<out>/L0`, `<out>/L1`, ... The point of having it as a script rather than a
shell line is that a storey takes half an hour and there are three of them: it
logs each stage, skips what is already built, and can be re-run after a crash
without redoing the meshes.
"""
import sys, os, json, time, argparse, subprocess
from pathlib import Path

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f} min] {m}", flush=True)

PY = sys.executable
S = "scripts/export"


def run(cmd, env=None, tag=""):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    log(f"$ {tag or ' '.join(cmd[1:3])}")
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  FAILED: {r.stderr.strip()[-800:]}")
        return False
    tail = [x for x in r.stdout.strip().splitlines() if x.strip()][-3:]
    for x in tail:
        log("  " + x)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="soulace_output/levels")
    ap.add_argument("--out", default="soulace_output")
    ap.add_argument("--voxel", type=float, default=0.008)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--only", default=None, help="e.g. 0,2")
    a = ap.parse_args()

    lv = json.load(open(Path(a.levels)/"storeys.json"))["storeys"]
    want = {int(x) for x in a.only.split(",")} if a.only else set(range(len(lv)))
    out = Path(a.out)

    for st in lv:
        k = st["level"]
        if k not in want:
            continue
        las = Path(a.levels)/f"L{k}.las"
        ply = out/"mesh"/f"L{k}.ply"
        npz = out/"mesh"/f"L{k}.npz"
        d = out/f"L{k}"
        ply.parent.mkdir(parents=True, exist_ok=True)
        log(f"=== L{k}: {st['points']:,} points, floor {st['floor_z']} "
            f"ceiling {st['ceiling_z']} ===")

        if not ply.exists():
            if not run([PY, f"{S}/poisson_mesh.py", str(las), str(ply)],
                       dict(PM_MASK=0, PM_CUT=-0.60, PM_VOXEL=a.voxel,
                            PM_DEPTH=a.depth, PM_TRIM=1.0), f"poisson L{k}"):
                continue
        if not npz.exists():
            if not run([PY, f"{S}/mesh_cache.py", str(ply), str(npz)], None,
                       f"cache L{k}"):
                continue
        if not (d/"manifest.json").exists():
            if not run([PY, f"{S}/modular_poisson.py", "--cache", str(npz),
                        "--out", str(d), "--las", str(las), "--no-glb",
                        "--obj"], None,
                       f"segment L{k}"):
                continue
        run([PY, f"{S}/solidify_walls.py", "--dir", str(d), "--cache", str(npz)],
            None, f"solids L{k}")
        run([PY, f"{S}/modular_viewer.py", str(d), str(npz), "--target", "3000000",
             "--raw"], None, f"viewer L{k}")
        run([PY, f"{S}/render_modular.py", str(d), str(npz)], None, f"render L{k}")
        run([PY, f"{S}/render_wall_elevation.py", str(d), str(npz)], None,
            f"elevations L{k}")
        log(f"=== L{k} done ===")

    run([PY, f"{S}/make_index.py", str(out)], None, "index")
    log("all storeys done")


if __name__ == "__main__":
    main()
