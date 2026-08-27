"""One scan, one command: raw LAS in, the whole suite out.

    raw .las
      -> isolate      the unit, axis-aligned, at FULL density
      -> poisson      the surface, its octree sized to the unit
      -> modular      that surface cut into named parts
      -> boxes        the clean solid model, one object per part
      -> sketchup     what a designer imports, plus the relief model
      -> viewer       all of it, stacked

Two things this does differently from the older `pipeline.py`, and both matter
for accuracy:

  * Isolation keeps EVERY point. pipeline.py caps the load at 4 M, which is why
    the isolated.las lying around holds 3.76 M points at 5.8 mm spacing when the
    same box in the raw export has 21.8 M at 3.0 mm.

  * Poisson runs on the isolated unit, not the raw export. Its finest octree
    cell is the bounding box over 2^depth, and a stray return 60 m away makes
    that cell 9 mm instead of 1.9 mm. Isolating first is worth more than any
    increase in depth.
"""
import os, sys, json, time, shutil, argparse, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PY_EXE = str(ROOT/"venv311"/"Scripts"/"python.exe")
EXPORT = ROOT/"scripts"/"export"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def run(args, **kw):
    log("  $ " + " ".join(str(a) for a in args[2:4]))
    r = subprocess.run([PY_EXE] + [str(a) for a in args], cwd=ROOT, **kw)
    if r.returncode:
        raise SystemExit(f"stage failed: {args}")


def poisson(iso, ply, npz, voxel, depth, ceiling_gb=None):
    """Build the surface, and never let it take the machine down with it.

    Poisson's memory is roughly (points) x (octree nodes), and octree nodes
    quadruple per level. Depth 13 at 17 M points blew 64 GB; so did depth 12.
    The failure mode is not a clean exception either -- open3d keeps asking the
    allocator for more, Windows starts paging, and the whole PC stops
    responding before anything raises. That happened once here, and once is
    enough.

    So the solve runs under a watchdog: if the child's resident memory passes
    the ceiling it is killed, and the settings step down -- first the voxel
    (fewer points, same detail per cell), then the octree depth. What is
    reported is whatever actually fitted.
    """
    import psutil
    if ceiling_gb is None:
        ceiling_gb = max(8.0, psutil.virtual_memory().total/2**30 * 0.62)
    plan = [(voxel, depth), (voxel*1.5, depth), (voxel*2, depth),
            (voxel*2, depth-1), (voxel*3, depth-1)]
    for vx, dp in plan:
        env = dict(os.environ, PM_MASK="0", PM_CUT="-0.60",
                   PM_VOXEL=f"{vx:.4f}", PM_DEPTH=str(dp), PM_TRIM="1.0")
        log(f"poisson: voxel {vx*1000:.1f} mm, depth {dp} "
            f"(memory ceiling {ceiling_gb:.0f} GB)")
        proc = subprocess.Popen([PY_EXE, str(EXPORT/"poisson_mesh.py"), str(iso), str(ply)],
                                cwd=ROOT, env=env)
        ps, peak, killed = psutil.Process(proc.pid), 0.0, False
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss/2**30
                peak = max(peak, rss)
                if rss > ceiling_gb:
                    log(f"  {rss:.1f} GB -- over the ceiling, stopping this attempt")
                    proc.kill(); killed = True
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(2)
        proc.wait()
        if not killed and proc.returncode == 0 and Path(ply).exists():
            log(f"  fitted at voxel {vx*1000:.1f} mm, depth {dp}, peak {peak:.1f} GB")
            run([EXPORT/"mesh_cache.py", ply, npz])
            return vx, dp
        log(f"  did not fit (peak {peak:.1f} GB) -- stepping down")
    raise SystemExit("poisson could not be made to fit; lower --depth or raise --voxel")


def storey_band(modular_dir):
    """The clamp for boxify: this storey's own floor and ceiling."""
    man = json.load(open(Path(modular_dir)/"manifest.json"))
    zf = man["floor_z"]
    lows = [zf + p["height_mm"]/1000 for p in man["parts"]
            if p["kind"] == "floor" and p["area_m2"] > 3]
    return min(lows + [zf]) - 0.15, zf + man["modal_ceiling_height_mm"]/1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--las", required=True, help="the raw scan")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="output_final")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--depth", type=int, default=13)
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="kill the Poisson solve if it passes this many GB "
                         "resident (default: 62%% of RAM)")
    ap.add_argument("--adopt-poisson", default=None,
                    help="a .ply already built for this unit; it is taken as "
                         "this run's poisson stage instead of rebuilding it")
    ap.add_argument("--skip-isolate", action="store_true",
                    help="the --las is already an isolated unit")
    ap.add_argument("--from-stage", default="isolate",
                    choices=["isolate", "poisson", "modular", "boxes", "pack"])
    a = ap.parse_args()
    d = Path(a.out)/a.name
    for sub in ("lidar", "poisson", "modular", "boxes", "sketchup"):
        (d/sub).mkdir(parents=True, exist_ok=True)
    stages = ["isolate", "poisson", "modular", "boxes", "pack"]
    do = stages[stages.index(a.from_stage):]

    iso = d/"lidar"/f"{a.name}.las"
    if "isolate" in do:
        if a.skip_isolate:
            src = Path(a.las).resolve()
            if src != iso.resolve():          # re-running in place is not an error
                shutil.copy(src, iso)
        else:
            isolate(Path(a.las), iso)

    ply, npz = d/"poisson"/"poisson.ply", d/"poisson"/"poisson.npz"
    if a.adopt_poisson:
        src = Path(a.adopt_poisson)
        for s_, dst in ((src, ply), (src.with_suffix(".npz"), npz)):
            if s_.exists() and not dst.exists():
                try:
                    os.link(s_, dst)
                except OSError:
                    shutil.copy2(s_, dst)
        log(f"adopted {src.name} as this run's poisson surface")
        if not npz.exists():
            run([EXPORT/"mesh_cache.py", ply, npz])
        do = [x for x in do if x != "poisson"]
    if "poisson" in do:
        poisson(iso, ply, npz, a.voxel, a.depth, a.mem_gb)

    if "modular" in do:
        # Carve first. Geometry cannot tell a 90 mm wall from two skins with a
        # cavity between them -- both are empty of returns and flat on both
        # sides. What separates them is whether the scanner ever saw THROUGH
        # the gap, and that is a measurement the trajectory carries. Without
        # this grid the pairing has no veto and invents thin walls.
        carve = d/"poisson"/"freespace.npz"
        if not carve.exists():
            log("free space: carving what the scanner looked through")
            args = [EXPORT/"freespace_carve.py", "--las", iso, "--out", carve]
            if (d/"modular"/"manifest.json").exists():
                args += ["--manifest", d/"modular"/"manifest.json"]
            run(args)
        log("modular: cutting the surface into named parts")
        run([EXPORT/"modular_poisson.py", "--cache", npz, "--out", d/"modular",
             "--freespace", carve])
        run([EXPORT/"solidify_walls.py", "--dir", d/"modular", "--cache", npz])

    if "boxes" in do:
        z0, z1 = storey_band(d/"modular")
        log(f"boxes: clamped to {z0:.3f} .. {z1:.3f} m")
        run([EXPORT/"boxify.py", "--dir", d/"modular", "--cache", npz,
             "--out", d/"boxes", "--z0", f"{z0:.4f}", "--z1", f"{z1:.4f}"])
        run([EXPORT/"box_accuracy.py", "--dir", d/"modular", "--cache", npz,
             "--boxes", d/"boxes"/"boxes_union.glb", "--parts", "0"])

    if "pack" in do:
        stem = a.name.title().replace("_", "") + "_model"
        log("sketchup: the handover files")
        run([EXPORT/"sketchup_pack.py", "--model", f"{stem}={d}", "--out", d/"sketchup"])
        run([EXPORT/"detail_pack.py", "--dir", d/"modular", "--cache", npz,
             "--out", d/"sketchup", "--stem", stem])
        run([EXPORT/"modular_viewer.py", d/"modular", npz])
        if (d/"modular"/"viewer.html").exists():
            html = (d/"modular"/"viewer.html").read_text(encoding="utf-8")
            import re
            for pat, rep in ((r"([^/])modular_view\.glb", r"\1modular/modular_view.glb"),
                             (r"([^/])modular_full\.glb", r"\1modular/modular_full.glb"),
                             (r"([^/])scan_view\.glb", r"\1poisson/scan_view.glb"),
                             (r"([^/])boxes\.glb", r"\1boxes/boxes.glb")):
                html = re.sub(pat, rep, html)
            (d/"viewer.html").write_text(html, encoding="utf-8")
    log(f"done -> {d}")


if __name__ == "__main__":
    main()
