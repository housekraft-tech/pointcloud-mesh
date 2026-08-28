Soulace -- one closed solid per storey.

Import: File > Import > Collada (.dae). Tick 'Merge coplanar faces'
in the importer options, or every box face arrives split into two
triangles. Units are metres, Z up. File > Save As gives you a .skp.

viewer.html  serve this folder and open it to see these exact
        files, storey by storey, with every edge drawn

*.dae         the storey as ONE watertight solid
*_parts.dae   the same geometry with its parts intact: one named
              group per wall, slab, beam and column, each already
              cut where it meets its neighbours. Import THIS to
              edit wall by wall.
*_detail.stl  the scanned surface itself at 60k triangles, ~1 mm
              from the Poisson mesh -- all the relief the boxes
              flatten. A surface, not a solid.
*.stl   the same, for SketchUp Web, which takes STL
*.obj   the same, for anything that is not SketchUp
*_plan.dxf  the 2D plan cut at 1.2 m, layers WALLS / OPENINGS /
        SLABS / DIMENSIONS / LABELS, to trace or to underlay

| storey | triangles | masonry | floor -> ceiling | clear height |
|---|---|---|---|---|
| Soulace_L0_ground | 11,030 | 132.8 m3 | -0.205 -> 3.709 m | 3200 mm |
| Soulace_L1_first | 9,142 | 118.5 m3 | -0.089 -> 3.885 m | 3350 mm |
| Soulace_L2_second | 4,686 | 76.6 m3 | -0.114 -> 3.637 m | 3040 mm |

Each storey sits in its own frame with its floor near z = 0.
To stack them, raise L1 by the height of L0 and L2 by the height of
L0 + L1; the exact figures are in soulace_output/storeys.json.
