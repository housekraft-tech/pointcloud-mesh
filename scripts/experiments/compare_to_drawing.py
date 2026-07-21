"""compare_to_drawing.py
---------------------
Check the reconstruction against the architect's floor plan.

Drawing values are transcribed from floorplan_original.png (the room boxes carry
explicit mm dimensions, and the panel gives carpet / carpet+balcony / built-up
areas), so this is a comparison against stated design intent, not against
another measurement of mine.

Only comparisons that can be made WITHOUT guessing a correspondence are
reported as numbers:

  totals      carpet area vs the sum of interior ceiling plateaus
  wet rooms   the drawing has exactly 3 toilets; the scan finds the dropped
              ceilings independently, so these match by construction, not by
              my choosing which is which
  counts      rooms, balconies, and door roles

Where a one-to-one room match would need me to decide which plateau is which
bedroom, the pairing is done by area rank and LABELLED as such, because a rank
pairing can silently swap two similar rooms.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\compare_to_drawing.py \\
      <modular_manifest.json> <openings_classified.json> <out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np

SQFT_M2 = 0.09290304

# --- transcribed from floorplan_original.png (mm) ---
DRAWING_ROOMS = [
    ("Living/Dining", 6530, 5000), ("Bedroom-2",     3030, 4890),
    ("Bedroom-3",     4200, 3250), ("Bedroom-1",     3450, 3540),
    ("Kitchen",       2250, 5050), ("Toilet-1",      1550, 2450),
    ("Toilet-2",      1550, 2450), ("Toilet-3",      2500, 1550),
    ("Walk-in",       1650, 1600), ("Utility",       1580, 1600),
    ("Foyer",         1300, 1650),
]
DRAWING_BALCONIES = [
    ("Balcony-1", 2100, 3400), ("Balcony-2", 1500, 2100),
    ("Balcoa-3",   900, 2300), ("Balcoa-4",   950, 1200),
]
PANEL = {"carpet_sqft": 1143, "carpet_balcony_sqft": 1327,
         "rera_sqft": 1182, "builtup_sqft": 1463}
# doors implied by the plan: 1 main + 3 bedroom + 3 toilet + kitchen/utility
DRAWING_DOORS = {"main entrance": 1, "bedroom": 3, "bathroom / WC": 3}
WET_MAX_H_MM = 2300          # dropped ceilings = the wet rooms


def area(w, l): return w * l / 1e6


def main(manifest, cls_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(manifest))
    cls = json.load(open(cls_path))

    ceil = [o for o in man["objects"] if o["name"].startswith("ceiling")
            and "height_mm" in o]
    interior = [c for c in ceil if c["area_m2"] >= 0.4]
    wet = [c for c in interior if c["height_mm"] < WET_MAX_H_MM]
    dry = [c for c in interior if c["height_mm"] >= WET_MAX_H_MM]

    d_carpet = sum(area(w, l) for _, w, l in DRAWING_ROOMS)
    d_balc = sum(area(w, l) for _, w, l in DRAWING_BALCONIES)
    m_total = sum(c["area_m2"] for c in interior)

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 78)
    P("AREA TOTALS")
    P("=" * 78)
    P(f"{'quantity':34} {'drawing':>10} {'scan':>10} {'delta':>10}")
    carpet_panel = PANEL["carpet_sqft"] * SQFT_M2
    P(f"{'carpet area (panel, sq ft->m2)':34} {carpet_panel:10.1f} "
      f"{'-':>10} {'-':>10}")
    P(f"{'carpet area (sum of room boxes)':34} {d_carpet:10.1f} "
      f"{'-':>10} {'-':>10}")
    P(f"{'ceiling plateaus measured':34} {'-':>10} {m_total:10.1f} {'-':>10}")
    P(f"{'  vs carpet+balcony panel':34} "
      f"{PANEL['carpet_balcony_sqft']*SQFT_M2:10.1f} {m_total:10.1f} "
      f"{m_total - PANEL['carpet_balcony_sqft']*SQFT_M2:+10.1f}")
    P(f"{'  vs built-up panel':34} {PANEL['builtup_sqft']*SQFT_M2:10.1f} "
      f"{m_total:10.1f} {m_total - PANEL['builtup_sqft']*SQFT_M2:+10.1f}")

    P("")
    P("=" * 78)
    P("WET ROOMS  (independent match: the drawing says 3 toilets; the scan")
    P("           finds dropped ceilings without being told where they are)")
    P("=" * 78)
    d_wet = sorted([(n, area(w, l)) for n, w, l in DRAWING_ROOMS
                    if n.startswith("Toilet")], key=lambda t: -t[1])
    m_wet = sorted([(c["name"], c["area_m2"], c["height_mm"]) for c in wet],
                   key=lambda t: -t[1])
    P(f"drawing toilets: {len(d_wet)}    scan dropped-ceiling plateaus: {len(m_wet)}")
    P(f"{'drawing':14} {'m2':>7}   {'scan':14} {'m2':>7} {'h_mm':>6} {'delta':>8} {'%':>7}")
    for i in range(max(len(d_wet), len(m_wet))):
        if i < len(d_wet) and i < len(m_wet):
            dn, da = d_wet[i]; mn, ma, mh = m_wet[i]
            P(f"{dn:14} {da:7.2f}   {mn:14} {ma:7.2f} {mh:6.0f} "
              f"{ma-da:+8.2f} {100*(ma-da)/da:+6.1f}%")
        elif i < len(d_wet):
            dn, da = d_wet[i]
            P(f"{dn:14} {da:7.2f}   {'(not found)':14}")
        else:
            mn, ma, mh = m_wet[i]
            P(f"{'(no match)':14} {'':>7}   {mn:14} {ma:7.2f} {mh:6.0f}")

    P("")
    P("=" * 78)
    P("LARGE ROOMS  (paired by AREA RANK -- a rank pairing can swap two rooms")
    P("              of similar size, so treat the names as indicative)")
    P("=" * 78)
    d_big = sorted([(n, area(w, l)) for n, w, l in DRAWING_ROOMS
                    if area(w, l) >= 10], key=lambda t: -t[1])
    m_big = sorted([(c["name"], c["area_m2"]) for c in dry
                    if c["area_m2"] >= 10], key=lambda t: -t[1])
    P(f"{'drawing':16} {'m2':>7}   {'scan':14} {'m2':>7} {'delta':>8} {'%':>7}")
    for i in range(max(len(d_big), len(m_big))):
        dn, da = d_big[i] if i < len(d_big) else ("(none)", float("nan"))
        mn, ma = m_big[i] if i < len(m_big) else ("(none)", float("nan"))
        if np.isnan(da) or np.isnan(ma):
            P(f"{dn:16} {da:7.2f}   {mn:14} {ma:7.2f}")
        else:
            P(f"{dn:16} {da:7.2f}   {mn:14} {ma:7.2f} {ma-da:+8.2f} "
              f"{100*(ma-da)/da:+6.1f}%")

    P("")
    P("=" * 78)
    P("COUNTS")
    P("=" * 78)
    ext = [r for r in cls if r.get("leads", "").startswith("EXTERIOR")]
    roles = {}
    for r in cls:
        if r.get("inferred_role"):
            roles[r["inferred_role"]] = roles.get(r["inferred_role"], 0) + 1
    P(f"{'item':30} {'drawing':>9} {'scan':>9}   note")
    P(f"{'rooms (excl. balconies)':30} {len(DRAWING_ROOMS):9} "
      f"{len(interior):9}   scan counts ceiling plateaus")
    P(f"{'balconies':30} {len(DRAWING_BALCONIES):9} {len(ext):9}   "
      f"scan = openings with no ceiling beyond")
    for role, n in sorted(DRAWING_DOORS.items()):
        P(f"{'doors: '+role:30} {n:9} {roles.get(role, 0):9}")
    for role, n in sorted(roles.items()):
        if role not in DRAWING_DOORS:
            P(f"{'doors: '+role:30} {'-':>9} {n:9}   no drawing equivalent")

    (out / "drawing_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    json.dump(dict(drawing_carpet_m2=round(d_carpet, 2),
                   drawing_balcony_m2=round(d_balc, 2),
                   panel_m2={k: round(v * SQFT_M2, 2) for k, v in PANEL.items()},
                   scan_plateau_total_m2=round(m_total, 2),
                   n_wet_drawing=len(d_wet), n_wet_scan=len(m_wet)),
              open(out / "drawing_comparison.json", "w"), indent=1)
    print(f"\nwrote {out/'drawing_comparison.txt'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
