"""Summarise slam_calib.yaml: key names, value shapes, truncated values.
Read-only probe. Usage: python probe_calib.py [path]"""
import sys, re
p = sys.argv[1] if len(sys.argv)>1 else "data/Soulace/slam_calib.yaml"
for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
    line = line.rstrip("\n")
    indent = len(line) - len(line.lstrip())
    key, _, val = line.lstrip().partition(":")
    val = val.strip()
    n = None
    if val.startswith("["):
        n = val.count(",") + 1
    show = val if len(val) <= 160 else val[:160] + " ...<TRUNC>"
    print(f"L{i:>3} indent={indent:<3} key={key!r} len={len(val)} nelem={n}")
    if val:
        print(f"      {show}")
