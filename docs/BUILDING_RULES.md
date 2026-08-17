# Building Rules — the logic the pipeline runs on

Every rule the pipeline uses to turn your scan into geometry. Some you gave me,
some I inferred from the data, some are assumptions I made that you may want to
overrule. **Mark up anything wrong and hand it back.**

Each rule has a **source** and a **status**:

| Source | Meaning |
|---|---|
| **SITE** | You told me this about the building |
| **MEASURED** | Derived from your scan and cross-checked |
| **ASSUMED** | My assumption — most likely to be wrong |

| Status | Meaning |
|---|---|
| ✅ | Implemented and validated against the scan |
| 🔶 | Implemented, not yet validated |
| ❌ | Not implemented yet |

---

## 1. The building

| # | Rule | Source | Status |
|---|---|---|---|
| 1.1 | Bare-shell concrete apartment. No furniture, no fittings, no finishes. | SITE | ✅ |
| 1.2 | Loose items (chairs, broomsticks, construction debris) are placed **in the middle of rooms**, deliberately clear of walls. | SITE | ✅ |
| 1.3 | Bathroom fixtures (commode, washbasin) exist and must be removed. | SITE | ✅ |
| 1.4 | **No false ceilings.** | SITE | ✅ |
| 1.5 | **Nothing is mounted or hung overhead.** Everything above standing height is structure — beams, arches, columns, ceiling drops. | SITE | ✅ |
| 1.6 | The building is rectilinear; walls run on two perpendicular axes. | SITE | ✅ |
| 1.7 | Building yaw vs the scan axes: **5.17°**, corrected during alignment. | MEASURED | ✅ |
| 1.8 | Floor is levelled to z = 0; walls are plumb to a median **0.33°**. | MEASURED | ✅ |
| 1.9 | Overall footprint **13433 × 12405 mm**; X extent holds within **17 mm** across every height band. | MEASURED | ✅ |

## 2. Heights

| # | Rule | Source | Status |
|---|---|---|---|
| 2.1 | Clear height is in the **2730–2770 mm** range. | SITE | ✅ |
| 2.2 | Measured clear height **2747 mm** (mujammel), **2732.4 mm** (koushik). Two independent scans agree to **4.1 mm**. | MEASURED | ✅ |
| 2.3 | Several distinct ceiling levels exist, one per room, not one flat slab. Detected: 2732.5, 2702.5, 2657.5, 2142.5 mm. | MEASURED | ✅ |
| 2.4 | Wet areas have dropped ceilings — site report gives MBR wash **2423 / 2407 mm**; found in the 2198–2472 mm slice. | SITE + MEASURED | ✅ |
| 2.5 | The 2142.5 mm level matches **nothing** in the site report. Unresolved — either an uncovered room or a false detection. | MEASURED | ❌ |

## 3. Walls

| # | Rule | Source | Status |
|---|---|---|---|
| 3.1 | A wall cell is proven by a **floor-to-ceiling column of points**. Nothing that fails to span is wall. | ASSUMED | ✅ |
| 3.2 | Wall thickness **varies in steps**, not continuously: thick where a pillar overlaps, thin where it does not. | SITE | ✅ |
| 3.3 | Measured thickness **244.0 mm** across 37 bins = 230 mm brick + plaster. | MEASURED | ✅ |
| 3.4 | Thickness is only trustworthy where the wall was seen from **both sides** (a "paired" wall). Single-sided walls have no measured thickness. | ASSUMED | ✅ |
| 3.5 | Walls carry rectangular intrusions/extrusions of about **75 mm** — sharp square edges, never rounded. | SITE | 🔶 |
| 3.6 | **All walls connect** into a closed network. Any break is an opening (door/window), never a gap. | SITE | ❌ |
| 3.7 | Occlusion gaps up to **600 mm** along a wall run are bridged; never invent wall where no points exist. | ASSUMED | ✅ |
| 3.8 | L-cuts, grooves and steps are real and must be kept; irregular concrete lumps are not. | SITE | 🔶 |

## 4. Cleanup (declutter)

| # | Rule | Source | Status |
|---|---|---|---|
| 4.1 | **Clean below standing height (1300 mm); keep everything above it.** This is the whole rule. | SITE (1.5) | ✅ |
| 4.2 | Below 1300 mm, a point survives only within **100 mm** of a proven wall cell. | ASSUMED | ✅ |
| 4.3 | Floor (below 50 mm) is always kept. | ASSUMED | ✅ |
| 4.4 | Result: **1.29%** of points removed, all below 1300 mm, median height 500 mm. **Zero** removed above. | MEASURED | ✅ |
| 4.5 | Zero free-standing objects survive in the 400–1200 mm band. | MEASURED | ✅ |
| 4.6 | ⚠️ Earlier versions deleted **62%** of removed points from *above* 1.8 m — they were beams and arches. Fixed in v6. | — | ✅ |

## 5. Doorways

| # | Rule | Source | Status |
|---|---|---|---|
| 5.1 | **Open below, wall above ~2100 mm ⇒ doorway**, with an arch/lintel over it. | SITE | ✅ |
| 5.2 | The **arch above the doorway is crucial** and must be modelled as its own part. | SITE | 🔶 |
| 5.3 | Door leaf width is **2'6" (762 mm) or 3'0" (914 mm)** only. | SITE | 🔶 |
| 5.4 | A doorway has **wall jambs parallel on both sides**. | SITE | 🔶 |
| 5.5 | If the scanner **walked through it**, it is a real opening, not a scan hole. | SITE | ✅ |
| 5.6 | Anything open to the floor but much wider than a door leaf is a **cased passage**, not a door. | ASSUMED | 🔶 |
| 5.7 | Measured head heights cluster at **2000–2250 mm**; arch depths **375–750 mm**. | MEASURED | ✅ |
| 5.8 | An opening is a **gap along a wall line** — width is measured along the wall, not from a 2D blob. | ASSUMED | 🔶 |

## 6. Windows

| # | Rule | Source | Status |
|---|---|---|---|
| 6.1 | **Wall up to the sill (~900 mm), open to ~2100 mm, wall above** ⇒ window cutout. | SITE | 🔶 |
| 6.2 | Sill height, head height and width are all measured from the void, never assumed. | ASSUMED | 🔶 |
| 6.3 | ⚠️ An earlier sill measurement read the *band edge* rather than the void and returned ~1025 mm for everything. Discard any sill number from before this fix. | — | ✅ |

## 7. Balcony doors

| # | Rule | Source | Status |
|---|---|---|---|
| 7.1 | Balcony doors are distinct from internal doors and from plain arched openings. | SITE | 🔶 |
| 7.2 | Beyond a balcony there is **no ceiling overhead**. | ASSUMED | 🔶 |
| 7.3 | Beyond a balcony the return is **daylight-bright** in RGB (mujammel has 16-bit colour). | ASSUMED | 🔶 |
| 7.4 | Open doors were present during the scan and must not be mistaken for walls. | SITE | ❌ |

## 8. Other elements

| # | Rule | Source | Status |
|---|---|---|---|
| 8.1 | Columns/pillars are their own parts, and are where walls step thicker. | SITE | 🔶 |
| 8.2 | Beams and arches are their own parts. Measured drops **510–932 mm**. | SITE + MEASURED | 🔶 |
| 8.3 | Electrical points are wall features to be located and counted. | SITE | 🔶 |
| 8.4 | Every point is **assigned** — to a part, or explicitly to clutter. Nothing unaccounted. | SITE | 🔶 |

## 9. Output

| # | Rule | Source | Status |
|---|---|---|---|
| 9.1 | Each wall, door, window, beam, column is a **separate named part**. | SITE | 🔶 |
| 9.2 | Targets: SketchUp / Rhino / 3ds Max, Blender / glTF, AutoCAD 2D. **Not** Revit/IFC. | SITE | ❌ |
| 9.3 | Rasters decide topology; **raw points decide distance**. | SITE | ✅ |
| 9.4 | Accuracy target is millimetre-level. Current: **−1, −4, +13 mm** against three site-report clear heights. | SITE | ✅ |

---

## Open questions

1. **2142.5 mm ceiling level** (2.5) — which room, or is it spurious?
2. **Minimum door width** (5.3) — 600–700 mm gaps are detected. Are those real narrow doors, or artefacts to reject?
3. **Passage vs door** (5.6) — openings of 1500–2400 mm exist. Cased openings, or missing walls?
4. **Window sill** (6.1) — is 900 mm exact, or does it vary by room?
5. **Balcony count** — how many balconies, and on which elevations?
6. **Wall thickness steps** (3.2) — what are the actual thin and thick values? 244 mm is a global median.

## Known defects

- Density gate rejects 53 of 90 faces; three real-scan tests fail (`assert 37 >= 60`). Likely discarding real faces.
- Walls are still disconnected boxes — rule 3.6 not implemented.
- Opening widths measured from 2D blobs merged one window run into a single 7900 mm "window". Being replaced by rule 5.8.
