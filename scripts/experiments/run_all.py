"""run_all.py
----------
ONE command that runs the full experimental reconstruction + diagnostic suite
for a scan and drops EVERY output into a single clearly-named folder:

  output/<name>_all/
    walk_path.png                 the operator's route (time-colored) -- the
                                  richest single signal: room-visit order,
                                  dwell vs. dash, every doorway as a crossing
    freespace_floorplan.png       the deliverable floorplan: solid free-space-
                                  carved walls + walk-path through-crossing
                                  doorways + clean rooms + furniture removed
    walls_mask.png                binary wall mask from the above
    tomographic_floorplan.png     height-signature map (wall / beam / railing)
    atlas/                        full field-mining diagnostic suite:
                                  01 height slices, 02 hough walls,
                                  03 height-colored, 04 trajectory overlay,
                                  05 gps_time heatmap, 06 intensity, 07 rgb,
                                  08 elevation profile, 09/10 vertical sections

Each stage isolates the scan independently (simple + robust); a stage that
fails is logged and skipped so one bad stage never sinks the rest.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\run_all.py <scan.las> <scan_name>
e.g.
  venv311\\Scripts\\python.exe scripts\\experiments\\run_all.py mujammelexport.las mujammel
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments import walk_path_viz, freespace_floorplan, tomographic_floorplan, diagnostic_atlas


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(las_path, scan_name):
    out_dir = ROOT / "output" / f"{scan_name}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== run_all for {scan_name} -> {out_dir} ===")

    stages = [
        ("walk path", walk_path_viz.main),
        ("free-space floorplan (+ walk-path doorways)", freespace_floorplan.main),
        ("tomographic (wall/beam/railing)", tomographic_floorplan.main),
        ("diagnostic atlas", lambda l, n, o: diagnostic_atlas.main(l, n, Path(o) / "atlas")),
    ]
    ok, failed = [], []
    for label, fn in stages:
        t0 = time.time()
        try:
            log(f"-- {label} ...")
            fn(las_path, scan_name, str(out_dir))
            ok.append(label)
            log(f"   done ({time.time()-t0:.0f}s)")
        except Exception as exc:
            failed.append((label, repr(exc)))
            log(f"   FAILED: {exc}")
            traceback.print_exc()

    log(f"=== {scan_name}: {len(ok)} stages ok, {len(failed)} failed ===")
    for label, err in failed:
        log(f"   FAILED {label}: {err}")
    log(f"all outputs in: {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
