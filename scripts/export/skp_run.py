"""Launch SketchUp, build a model in it, save, and survive a crash.

SketchUp has gone down three times mid-session -- on save, and once between
commands. Driving it by hand costs a relaunch and a lost step each time, so the
whole cycle lives here: start it, wait for the bridge, send the build, save in
a separate call, and if the process disappears, try again from the top.
"""
import argparse, json, socket, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from skp_client import rb

EXE = "C:/Program Files/SketchUp/SketchUp 2025/SketchUp/SketchUp.exe"
BLANK = Path("C:/Users/PC/AppData/Local/Temp/pcm_blank.skp")
PLUG = Path("C:/Users/PC/AppData/Roaming/SketchUp/SketchUp 2025/SketchUp/Plugins")


def alive():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq SketchUp.exe"],
                         capture_output=True, text=True).stdout
    return "SketchUp.exe" in out


def kill():
    subprocess.run(["taskkill", "/F", "/IM", "SketchUp.exe"],
                   capture_output=True, text=True)
    time.sleep(3)


def launch(wait=180):
    if not BLANK.exists():
        tpl = next(Path(EXE).parents[1].rglob("Templates/*.skp"))
        BLANK.write_bytes(tpl.read_bytes())
    subprocess.Popen([EXE, str(BLANK)])
    # SketchUp does not load extensions while it sits on the Welcome window --
    # no document, no Ruby, no bridge. Every "bridge never came up" was that.
    # If it has not bound in half a minute, ask Windows to open the file with
    # its associated application, which forces a document into the running
    # instance and the extensions load with it.
    t0 = time.time()
    nudged = False
    while time.time() - t0 < wait:
        time.sleep(4)
        try:
            if rb("1", timeout=10).get("ok"):
                return True
        except OSError:
            pass
        if not nudged and time.time() - t0 > 30:
            subprocess.run(["cmd", "/c", "start", "", str(BLANK)], shell=False)
            nudged = True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--skp", required=True)
    ap.add_argument("--module", default="pcm_wall")
    ap.add_argument("--weld", action="store_true")
    ap.add_argument("--tries", type=int, default=2)
    a = ap.parse_args()
    js = str(Path(a.schedule).resolve()).replace("\\", "/")
    skp = str(Path(a.skp).resolve()).replace("\\", "/")
    mod = a.module
    cls = {"pcm_wall": "PCMWall", "pcm_designer": "PCMDesigner"}[mod]

    for attempt in range(1, a.tries + 1):
        if alive():
            kill()
        print(f"[{attempt}] launching SketchUp")
        if not launch():
            print("   bridge never came up"); continue
        try:
            r = rb(f'load File.join(ENV["APPDATA"],"SketchUp","SketchUp 2025",'
                   f'"SketchUp","Plugins","{mod}.rb"); '
                   f'{cls}.build("{js}", {"true" if a.weld else "false"})',
                   timeout=1500)
            if not r.get("ok"):
                print("   build error:", r.get("error")); continue
            print("   build:", r["result"].strip('"').replace('\\"', '"'))
            r = rb(f'{cls}.save_as("{skp}")', timeout=900)
            print("   save:", r.get("result", r.get("error")))
            p = Path(a.skp)
            if p.exists() and p.stat().st_size > 8_000:
                print(f"OK {p} ({p.stat().st_size/1e6:.2f} MB)")
                return
        except OSError as e:
            print(f"   SketchUp went away ({e.__class__.__name__}); retrying")
    raise SystemExit("could not produce the .skp")


if __name__ == "__main__":
    main()
