"""Real .skp files, made by SketchUp itself.

Nothing outside SketchUp's SDK can write .skp -- but SketchUp is installed
here, and its Ruby API can import and save. So this drops a converter into the
Plugins folder, launches SketchUp once, and lets it work through every .dae in
a folder: import, explode the Collada wrapper so the parts arrive as top-level
groups, save as .skp, next. SketchUp quits itself at the end.

A window does open while it runs. There is no headless mode; the application
has to be up for its own API to exist.
"""
import os, sys, time, shutil, argparse, subprocess
from pathlib import Path

EXE_GLOB = ["C:/Program Files/SketchUp/SketchUp 2025/SketchUp/SketchUp.exe",
            "C:/Program Files/SketchUp/SketchUp 2024/SketchUp/SketchUp.exe",
            "C:/Program Files/SketchUp/SketchUp 2023/SketchUp/SketchUp.exe"]
PLUGINS = list(Path(os.environ.get("APPDATA", "")).glob("SketchUp/SketchUp */SketchUp/Plugins"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="a folder of .dae files")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep-open", action="store_true",
                    help="leave SketchUp running when it is done")
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    daes = sorted(d.rglob("*.dae"))
    if not daes:
        raise SystemExit(f"no .dae files under {d}")
    exe = next((p for p in EXE_GLOB if Path(p).exists()), None)
    if not exe:
        raise SystemExit("SketchUp not found in Program Files")
    if not PLUGINS:
        raise SystemExit("no SketchUp Plugins folder found under APPDATA")
    plug = PLUGINS[-1]
    src = Path(__file__).parent/"ruby"/"dae_to_skp.rb"
    shutil.copy(src, plug/"pcm_dae_to_skp.rb")
    log = d/"skp_convert.log"
    if log.exists():
        log.unlink()
    env = dict(os.environ, PCM_DAE_DIR=str(d), PCM_LOG=str(log),
               PCM_QUIT="0" if a.keep_open else "1")
    print(f"{len(daes)} .dae files in {d}")
    print(f"launching {exe}  (a window will open; it closes itself)")
    # Launch with a file, or SketchUp stops on its Welcome window and the
    # plugin's timer fires into a session with no model -- which is exactly
    # what happened on the first attempt: SketchUp sat there, no log, no .skp.
    # Any .skp will do to get past it, and SketchUp ships templates.
    seed = next(iter(sorted(Path(exe).parents[1].rglob("Templates/*.skp"))), None)
    cmd = [exe] + ([str(seed)] if seed else [])
    if seed:
        print(f"opening {seed.name} first, to get past the Welcome window")
    proc = subprocess.Popen(cmd, env=env)
    t0 = time.time()
    seen = 0
    while time.time() - t0 < a.timeout:
        if log.exists():
            lines = log.read_text(errors="ignore").splitlines()
            for l in lines[seen:]:
                print("   " + l)
            seen = len(lines)
            if lines and lines[-1].endswith("DONE"):
                break
        if proc.poll() is not None and seen:
            break
        time.sleep(2)
    else:
        print("timed out waiting for SketchUp")
    made = sorted(d.rglob("*.skp"))
    print(f"\n{len(made)} .skp files:")
    for m in made:
        print(f"   {m.stat().st_size/1e6:6.2f} MB  {m.relative_to(d)}")
    (plug/"pcm_dae_to_skp.rb").unlink(missing_ok=True)   # leave no plugin behind


if __name__ == "__main__":
    main()
