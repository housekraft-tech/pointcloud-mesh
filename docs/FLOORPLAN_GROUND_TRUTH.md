# Floorplan ground truth

Transcribed from the architect's floorplan (`image.png`), supplied 2026-08-17.
All dimensions in mm as printed on the drawing. **Check the transcription** —
these become the reference every pipeline measurement is scored against.

## Rooms

| Room | Dimensions (mm) | As printed | Notes |
|---|---|---|---|
| Bedroom 1 | 3030 × 4890 | 9'11" × 16'0" | top right on the drawing |
| Bedroom 2 | 3450 × 3540 | 11'3" × 11'7" | top left |
| Bedroom 3 | 4200 × 3250 | 13'9" × 10'7" | bottom left |
| Living / Dining | 6530 × 4275 | 21'5" × 14'0" | central |
| Kitchen | 2250 × 2625 | 7'4" × 8'7" | |
| Kitchen (second run) | 3080 × 1700 | 10'1" × 5'6" | |
| Foyer | 1300 × 1650 | 4'3" × 5'4" | entry |
| Walk-in | 2550 × 1600 | 8'4" × 5'2" | |
| Toilet 1 | 1550 × 2450 | 5'1" × 8'0" | |
| Toilet 2 | 1550 × 2450 | 5'1" × 8'0" | |
| Toilet 3 | 2450 × 1550 | 8'0" × 5'1" | |
| Utility | 3080 × 1550 | 10'1" × 4'11" | |

## Balconies — 4 total

| Balcony | Dimensions (mm) | As printed |
|---|---|---|
| Balcony A | 950 × 1200 | 3'1" × 3'11" |
| Balcony B | 900 × 2300 | 2'11" × 7'6" |
| Balcony C | 2100 × 2675 | 6'10" × 8'9" |
| Balcony D | 1500 × 2100 | 4'11" × 6'10" |

This matches the stated count of 4 balconies exactly.

## Cross-checks available

- **12 rooms + 4 balconies** — the pipeline currently produces no room extraction
  at all, so this is the target for the wall-network work.
- **Toilets 1 and 2 are identical** (1550 × 2450) — a useful symmetry check.
- Site report clear heights already matched to −1 / −4 / +13 mm; room dimensions
  are the next, and much stronger, test because there are 32 of them (2 per room).
- The drawing's north arrow points up-and-right; the scan is in its own frame.
  See `docs/FLOORPLAN_ALIGNMENT.md` for the transform.

## Known orientation issue

The floorplan as supplied is rotated relative to the aligned scan
(`output/mujammel_structural_v6.las`). The transform is solved by matching wall
masks, not assumed — see the alignment output.
