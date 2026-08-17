# Integrating `bbox_data.json` with the LiDAR

## Division of labour

`bbox_data.json` is a 2D floorplan lifted to 3D. It supplies **identity and
topology**; it does not supply measurements.

| From bbox_data | From LiDAR |
|---|---|
| Which walls exist, and how they connect | Where each wall actually is |
| Which rooms exist, and their names | Every dimension, to the millimetre |
| Which openings exist, and their type | True sill / head / width per opening |
| `wall_uuid`, `wall_semantic_id` — the keys that tie parts to the webapp | **Arches** — cannot exist in a 2D-derived model |
| | **Beams, columns, niches** — same |
| | Wall thickness (bbox has a 100 mm placeholder) |
| | Per-room ceiling heights, including soffits |

**Every number in bbox_data is nominal.** The 2750 mm wall height is a default,
so its agreement with our measured 2747 mm is coincidence, not confirmation.
The 2100 mm door head is the standard leaf height and would read 2100 whether
or not the building was ever measured.

## What bbox_data contains

- `dimensions`: 1056 × 835 px, scale 16.88 mm/px — same image as `image.png`
- `global_walls`: 24, each `height: 275`, `thickness: 10` (cm)
- `global_openings`: 14 — 4 balcony doors, 4 doors, 6 windows
  - doors: `height: 210`, `elevation: 0`
  - windows: `height: 120`, `elevation: 80`, `top_wall_height: 75`
- `areas`: 15 named rooms, each with `walls`, `doors`, `ceilings`,
  `placeable_surfaces`, `paintable_surfaces`
- Stable `wall_uuid` / `wall_semantic_id` / `opening_uuid` per element

Door widths are 917 / 758 / 754 mm — i.e. 3'0" and 2'6", matching the site rule.

## Known corrections the LiDAR makes

| Quantity | bbox | LiDAR | Note |
|---|---|---|---|
| Wall thickness | 100 mm | 190–200 mm | 2× out; halves every BOM quantity |
| Ceiling height | 2750 mm everywhere | 4 distinct levels; site report gives 2690–2739 per room | bbox has one flat value |
| Wet-area ceilings | 2750 mm | ~2410 mm | bbox has no soffits |
| Arch depth | absent | 510–932 mm measured on beams | only in 3D |

## Registration status — NOT YET GOOD ENOUGH

Best fit: **rot 270°, mirrored, ×0.982 (16.57 mm/px), shift (+140, +420) mm**,
giving **71 mm median** drawn-to-scanned distance with **57% within 100 mm**.

That is not accurate enough to snap openings. Evidence it is failing: measured
door sills come out at 1650–1783 mm when a door sill must be 0, and balcony arch
depths come out at 8–44 mm when beams measure 510–932 mm. Both mean some
openings are being sampled against the wall *beside* the intended one.

**Do not use `output/openings_verified.json` for anything yet.**

### Why a global fit stalls here

One rigid transform has to compromise across the whole flat. The fix is per-wall
registration: match each bbox wall to its own scanned wall using the semantic
ids to keep identity, then measure openings in that wall's local frame. A wall
matched individually does not care what the rest of the flat does.

## Outstanding — the parts that matter most

1. **Niche walls** — recesses in a wall face. Present in the LiDAR relief maps
   (the per-wall elevations show them as blue), absent from bbox.
2. **Columns** — where walls step thicker. The site rule says thickness varies in
   steps at columns; bbox has a single 100 mm value so it cannot express this.
3. **Arches** — measured 510–932 mm drops, and required as their own parts.
